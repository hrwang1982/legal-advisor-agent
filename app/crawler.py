"""国家法律法规数据库（flk.npc.gov.cn）动态差分核对与时效性同步管线：
1. 真实差分核对：向国家法律法规数据库官方接口实时核对已入库外规的最新时效（sxx）与施行状态；
2. 废止状态同步：发现官网最新版已废止后，自动将本地库内旧版本标记为「已废止」并推导失效日期；
3. 新版本发现与演进：发现官网发布新版本或修正案时自动提示，并自动检测本地多版本法规；
4. 条文级 Diff 与修改日志：对本地收录的新旧版本自动运行条文级差异比对，沉淀条文修改日志（law_article_revision_log）；
5. 新旧法对照自动推导：通过条文内容相似度对齐算法，自动推导新旧条文承继关系并生成待审提案（rule_proposals）；
6. 严格按需触发：无任何后台定时任务，完全由用户在界面手动发起；遇到网络不可达或 WAF 拦截时如实报错，绝不伪造假数据。
"""
from __future__ import annotations

import difflib
import json
import logging
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from app.audit import audit_store
from app.flk_downloader import SXX_STATUS_MAP, flk_downloader
from app.ingest import invalidate_cache
from app.kb import kb_store, normalize_article_id

logger = logging.getLogger(__name__)


def _day_before(d_str: str) -> str:
    """计算指定日期的前一日（YYYY-MM-DD）。"""
    try:
        return (date.fromisoformat(str(d_str)[:10]) - timedelta(days=1)).isoformat()
    except Exception:
        return d_str


def _summarize_article_diff(old_content: str, new_content: str, ratio: float) -> str:
    """提取新旧条文修订的实质变动说明：优先尝试调用大模型提炼，离线/未配置时规则差异比对兜底。"""
    pct = int(ratio * 100)
    try:
        from app.llm import get_llm

        llm = get_llm()
        if llm:
            prompt = (
                "你是一名资深中国法务专家。请对比以下同一部法规新旧版本的条文修改，"
                "用一句话（15至25字以内）概括实质修改要点，直接输出要点，不要任何前缀客套话：\n"
                f"【旧版条文】：{old_content[:200]}\n"
                f"【新版条文】：{new_content[:200]}"
            )
            resp = llm.invoke(prompt)
            summary = getattr(resp, "content", str(resp)).strip().replace("\n", " ")
            if summary and len(summary) >= 4 and len(summary) <= 60:
                return f"条文实质修订（内容匹配度 {pct}%）：{summary}"
    except Exception:
        pass

    # 规则差异比对降级：提取差异增删片段
    diff = list(difflib.ndiff(old_content[:120], new_content[:120]))
    added_chars = [d[2:] for d in diff if d.startswith("+ ") and d[2:].strip()]
    removed_chars = [d[2:] for d in diff if d.startswith("- ") and d[2:].strip()]

    hint = "条款实质修订，优化适用要件与法理表述"
    if added_chars and not removed_chars:
        add_str = "".join(added_chars[:8]).strip()
        hint = f"新增限定与规范要件（增加“{add_str}”等表述）" if add_str else hint
    elif removed_chars and not added_chars:
        rem_str = "".join(removed_chars[:8]).strip()
        hint = f"精简规范要件（删除“{rem_str}”等表述）" if rem_str else hint
    elif added_chars and removed_chars:
        rem_str = "".join(removed_chars[:6]).strip()
        add_str = "".join(added_chars[:6]).strip()
        if rem_str and add_str:
            hint = f"调整要件表述（变更“{rem_str}”为“{add_str}”等）"

    return f"条文实质修订（内容匹配度 {pct}%）：{hint}"


def align_articles_for_counterparts(
    new_doc_id: str,
    old_doc_id: str,
    similarity_threshold: float = 0.80,
) -> List[Dict[str, Any]]:
    """基于条文内容相似度对同一部法规的新旧两个版本进行条文级自动对齐推导，生成新旧法对照映射。
    规则阈值：
    - >= 90%：条文文字承继（文字基本无实质变动）；
    - 80% ~ 90%：条文实质修订，提取实质修改说明；
    - < 80%：视为新增条款或废止条款，不建立新旧对照（不沉淀至新旧法条文对照表）。
    """
    from app.mysql_store import mysql_store

    res_new = kb_store.get_document_chunks(new_doc_id)
    res_old = kb_store.get_document_chunks(old_doc_id)
    if res_new.get("status") != "ok" or res_old.get("status") != "ok":
        return []

    doc_new = res_new["document"]
    doc_old = res_old["document"]
    new_title = doc_new.get("title", "")
    old_title = doc_old.get("title", "")
    switch_date = doc_new.get("effective_date") or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # 提取非目录切片的条文内容
    old_articles: List[Tuple[str, str]] = []
    for c in res_old.get("chunks", []):
        if c.get("chunk_type") == "hierarchical_parent":
            continue
        art = normalize_article_id(str(c.get("article") or ""))
        cnt = (c.get("content") or "").strip()
        if art and cnt:
            old_articles.append((art, cnt))

    new_articles: List[Tuple[str, str]] = []
    for c in res_new.get("chunks", []):
        if c.get("chunk_type") == "hierarchical_parent":
            continue
        art = normalize_article_id(str(c.get("article") or ""))
        cnt = (c.get("content") or "").strip()
        if art and cnt:
            new_articles.append((art, cnt))

    counterparts: List[Dict[str, Any]] = []
    seen = set()

    for o_art, o_cnt in old_articles:
        best_n_art = ""
        best_n_cnt = ""
        best_ratio = 0.0

        for n_art, n_cnt in new_articles:
            ratio = difflib.SequenceMatcher(None, o_cnt, n_cnt).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_n_art = n_art
                best_n_cnt = n_cnt

        # 匹配度低于 80% 视为新增或废止条款，不建立对照
        if best_ratio >= similarity_threshold and best_n_art:
            pair_key = (old_title, o_art, new_title, best_n_art)
            if pair_key not in seen:
                seen.add(pair_key)
                if best_ratio >= 0.90:
                    note = f"条文文字承继（内容匹配度 {int(best_ratio * 100)}%），表述微调承继，无实质变动"
                else:
                    note = _summarize_article_diff(o_cnt, best_n_cnt, best_ratio)

                item = {
                    "old_law": old_title,
                    "old_article": o_art,
                    "new_law": new_title,
                    "new_article": best_n_art,
                    "switch_date": switch_date,
                    "diff_note": note,
                    "relation": "one_to_one",
                    "authority": "国家法规库多版本条文对齐引擎",
                    "similarity": round(best_ratio, 3),
                }
                counterparts.append(item)

    return counterparts


class NationalLawDatabaseMonitor:
    """国家法律法规数据库动态差分比对与时效性同步器（无任何本地写死假数据）。"""

    def check_and_sync_announcements(
        self,
        mock_bulletins: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """执行全库外规与国家法律法规数据库官方实时状态的差分比对与同步更新。"""
        from app.mysql_store import mysql_store

        now_str = datetime.now(timezone.utc).isoformat()

        # 1. 显式传入 mock_bulletins 时的测试回退通道（仅供单元测试）
        if mock_bulletins is not None:
            return self._handle_mock_bulletins(mock_bulletins, now_str)

        # 2. 真实动态差分主流程：读取本地所有现行有效外规文档
        local_docs = mysql_store.query(
            """
            SELECT doc_id, title, effective_date, validity_status, version_name, kb_type
            FROM law_documents
            WHERE is_latest = 1 AND (kb_type LIKE '外规知识库%' OR kb_type = '外规知识库')
            """
        )

        if not local_docs:
            return {
                "status": "success",
                "sync_time": now_str,
                "scanned_laws_count": 0,
                "updated_laws_count": 0,
                "updated_laws": [],
                "new_versions_found_count": 0,
                "new_versions_found": [],
                "revision_logs_count": 0,
                "revision_logs": [],
                "counterpart_proposals_count": 0,
                "message": "本地外规知识库暂无可核对的现行有效法规。",
            }

        updated_laws: List[str] = []
        new_versions_found: List[str] = []
        revision_logs_created: List[str] = []
        counterpart_proposals_created = 0

        # 逐部核对本地法规在官网的实时状态
        for idx, doc in enumerate(local_docs):
            clean_title = (doc.get("title") or "").strip().strip("《》")
            if not clean_title:
                continue

            # 调用真实接口请求 flk.npc.gov.cn 检索该法规所有版本
            search_res = flk_downloader.search_laws(keyword=clean_title, sxx=[], size=10)
            if search_res.get("status") == "network_unreachable":
                # 网络不可达或官方网关阻断时如实报错，绝不伪造假数据
                logger.warning(f"[FLKSync] 连接国家法律法规数据库官网不可达: {search_res.get('message')}")
                return {
                    "status": "network_unreachable",
                    "sync_time": now_str,
                    "scanned_laws_count": idx,
                    "updated_laws_count": len(updated_laws),
                    "updated_laws": updated_laws,
                    "new_versions_found_count": len(new_versions_found),
                    "new_versions_found": new_versions_found,
                    "revision_logs_count": len(revision_logs_created),
                    "revision_logs": revision_logs_created,
                    "counterpart_proposals_count": counterpart_proposals_created,
                    "message": search_res.get("message") or "网络不可达：无法连接国家法律法规数据库官网进行差分比对。",
                }

            laws = search_res.get("laws") or []
            matched_laws = [l for l in laws if l.get("title") == clean_title]

            if matched_laws:
                # 寻找官网返回的最新版本（按公布日或施行日降序）
                def _sort_key(it):
                    return (str(it.get("effective_date") or ""), str(it.get("publish") or ""))

                latest_off = max(matched_laws, key=_sort_key)
                off_sxx = latest_off.get("sxx")
                off_eff_date = str(latest_off.get("effective_date") or latest_off.get("publish") or "")[:10]
                off_sxx_label = SXX_STATUS_MAP.get(off_sxx, str(off_sxx))

                # 差分 A：官网最新版已标记为废止（sxx == 1）
                if off_sxx == 1 and doc.get("validity_status") == "有效":
                    calc_eff_to = _day_before(off_eff_date) if off_eff_date else now_str[:10]
                    mysql_store.execute(
                        "UPDATE law_documents SET validity_status = '已废止', valid_until = ? WHERE title = ?",
                        (calc_eff_to, clean_title),
                    )
                    mysql_store.execute(
                        "UPDATE law_chunks SET validity = '已废止', effective_to = ? WHERE law_name = ?",
                        (calc_eff_to, clean_title),
                    )
                    mysql_store.execute(
                        "UPDATE kb_articles SET validity = '已废止', effective_to = ?, updated_at = ? WHERE law_name = ?",
                        (calc_eff_to, now_str, clean_title),
                    )
                    mysql_store.execute(
                        "UPDATE sample_laws SET validity = '已废止', effective_to = ? WHERE law_name = ?",
                        (calc_eff_to, clean_title),
                    )
                    updated_laws.append(f"《{clean_title}》官网已标记废止（官方失效日: {calc_eff_to}）")

                # 差分 B：官网存在更新版本（施行日晚于本地最新版，或官方时效为尚未生效/现行有效的新修正案）
                local_eff = str(doc.get("effective_date") or "")[:10]
                if off_eff_date and local_eff and off_eff_date > local_eff and off_sxx in (2, 3, 4):
                    new_versions_found.append(
                        f"《{clean_title}》官网已发布更新版本 [施行日: {off_eff_date}, 官方时效: {off_sxx_label}]"
                    )
                    if off_sxx == 2 and doc.get("validity_status") == "有效":
                        mysql_store.execute(
                            "UPDATE law_documents SET validity_status = '已修正' WHERE doc_id = ?",
                            (doc["doc_id"],),
                        )

            # 单次请求间隔 0.5 秒抖动，严格防范官方频控
            time.sleep(0.5)

        # 3. 本地多版本法规条文级差异核对与新旧法对照推导
        # 针对本地已有新旧两版的同名法规，自动生成条文级修改日志与新旧对照建议
        all_same_names = mysql_store.query(
            """
            SELECT title FROM law_documents
            WHERE (kb_type LIKE '外规知识库%' OR kb_type = '外规知识库')
            GROUP BY title HAVING COUNT(doc_id) > 1
            """
        )

        for row in all_same_names:
            t_name = row["title"]
            v_rows = mysql_store.query(
                "SELECT doc_id, is_latest, effective_date, version_name FROM law_documents WHERE title = ? ORDER BY is_latest DESC, created_at DESC",
                (t_name,),
            )
            if len(v_rows) >= 2:
                new_doc = v_rows[0]
                old_doc = v_rows[1]
                # A. 运行条文级版本 Diff 比对
                diff_res = kb_store.compare_document_versions(new_doc["doc_id"], old_doc["doc_id"])
                for mod_art in diff_res.get("modified_articles", []):
                    art_name = mod_art.get("article", "")
                    old_text = mod_art.get("old_content", "")
                    new_text = mod_art.get("new_content", "")
                    eff_from = new_doc.get("effective_date") or now_str[:10]
                    # 写入条文修改日志
                    kb_store.record_article_revision(
                        law_name=t_name,
                        article=art_name,
                        amendment_name=new_doc.get("version_name") or "最新修正案",
                        effective_from=eff_from,
                        new_content=new_text,
                        old_content=old_text,
                        diff_summary=f"官方版本比对修订（{old_doc.get('version_name', '旧版')} → {new_doc.get('version_name', '新版')}）",
                    )
                    revision_logs_created.append(f"《{t_name}》{art_name}（自动比对写入修改日志）")

                # B. 运行条文相似度对齐算法推导新旧法对照并沉淀至待审规则提案池
                counterparts = align_articles_for_counterparts(new_doc["doc_id"], old_doc["doc_id"])
                for cp in counterparts:
                    prop_key = f"{cp['old_law']}|{cp['old_article']}->{cp['new_law']}|{cp['new_article']}"
                    try:
                        audit_store.record_rule_proposal(
                            thread_id="flk_version_counterparts_deducer",
                            proposal_type="article_succession",
                            proposal_key=prop_key,
                            proposal_data=cp,
                            query=f"版本对齐推导: 《{cp['new_law']}》{cp['new_article']}",
                        )
                        counterpart_proposals_created += 1
                    except Exception:
                        pass

        # 刷新全系统缓存
        if updated_laws or revision_logs_created or new_versions_found:
            invalidate_cache()
            kb_store.version_counter += 1

        return {
            "status": "success",
            "sync_time": now_str,
            "scanned_laws_count": len(local_docs),
            "updated_laws_count": len(updated_laws),
            "updated_laws": updated_laws,
            "new_versions_found_count": len(new_versions_found),
            "new_versions_found": new_versions_found,
            "revision_logs_count": len(revision_logs_created),
            "revision_logs": revision_logs_created,
            "counterpart_proposals_count": counterpart_proposals_created,
        }

    def _handle_mock_bulletins(self, bulletins: List[Dict[str, Any]], now_str: str) -> Dict[str, Any]:
        """处理显式传入的 mock 公告（仅供测试环境回退校验）。"""
        from app.mysql_store import mysql_store

        updated_laws: List[str] = []
        revision_logs_created: List[str] = []

        for b in bulletins:
            action = b.get("action")
            target_law = b.get("target_law", "").strip().strip("《》")
            new_law = b.get("new_law", "").strip().strip("《》")
            effective_date = b.get("effective_date", "").strip()
            amendment_name = b.get("amendment_name", "").strip() or new_law

            if action == "repeal" and target_law and effective_date:
                calc_effective_to = _day_before(effective_date)
                mysql_store.execute(
                    "UPDATE kb_articles SET validity = '已废止', effective_to = ?, updated_at = ? WHERE law_name = ? AND validity = '现行有效'",
                    (calc_effective_to, now_str, target_law),
                )
                mysql_store.execute(
                    "UPDATE law_chunks SET validity = '已废止', effective_to = ? WHERE law_name = ? AND validity = '现行有效'",
                    (calc_effective_to, target_law),
                )
                mysql_store.execute(
                    "UPDATE law_documents SET validity_status = '已废止', valid_until = ? WHERE title = ?",
                    (calc_effective_to, target_law),
                )
                mysql_store.execute(
                    "UPDATE sample_laws SET validity = '已废止', effective_to = ? WHERE law_name = ? AND validity = '现行有效'",
                    (calc_effective_to, target_law),
                )
                updated_laws.append(f"《{target_law}》已同步标记为已废止（失效日: {calc_effective_to}）")

            elif action == "amend" and target_law:
                revised_articles = b.get("revised_articles", [])
                for rev in revised_articles:
                    art_no = normalize_article_id(rev.get("article", ""))
                    eff_from = rev.get("effective_from") or effective_date
                    new_content = rev.get("new_content", "")
                    diff_note = rev.get("diff_note", "国家法规库修正公告")
                    if art_no and eff_from and new_content:
                        kb_store.record_article_revision(
                            law_name=target_law,
                            article=art_no,
                            amendment_name=amendment_name,
                            effective_from=eff_from,
                            new_content=new_content,
                            diff_summary=diff_note,
                        )
                        revision_logs_created.append(f"《{target_law}》{art_no}（修正案: {amendment_name}，生效日: {eff_from}）")

        if updated_laws or revision_logs_created:
            invalidate_cache()
            kb_store.version_counter += 1

        return {
            "status": "success",
            "sync_time": now_str,
            "scanned_laws_count": len(bulletins),
            "updated_laws_count": len(updated_laws),
            "updated_laws": updated_laws,
            "new_versions_found_count": 0,
            "new_versions_found": [],
            "revision_logs_count": len(revision_logs_created),
            "revision_logs": revision_logs_created,
            "counterpart_proposals_count": 0,
        }


# 全局国家法规库监听同步单例
flk_monitor = NationalLawDatabaseMonitor()
