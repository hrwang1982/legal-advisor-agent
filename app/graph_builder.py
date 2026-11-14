"""LLM-Graph Builder 与司法解释「理解与适用」引申抽取流水线（离线批处理知识图谱构建）：
1. 离线大模型批处理：全量扫描法律法规，离线抽取所有关联、参照、兜底引致关系，经专业法务抽检后固化为知识图谱 JSON，避免在线检索时的幻觉风险；
2. 司法解释引申抽取：利用最高院司法解释的“理解与适用”文本，自动提取法条对应引致关系。
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from app.audit import audit_store
from app.config import settings
from app.knowledge import knowledge_center
from app.llm import _call, _extract_json

logger = logging.getLogger(__name__)


# ————————————————————————————————————————————————
# 1. 离线大模型批处理（LLM-Graph Builder）
# ————————————————————————————————————————————————
def build_cross_reference_prompt(law_name: str, article: str, content: str) -> str:
    """构建引致条款抽取的标准化提示词。"""
    return (
        "你是一名权威立法专家与法学教授。请深入分析以下条文内容，识别其法条引致与规范连接关系：\n"
        f"【法规全称】《{law_name}》\n"
        f"【条文编号】{article}\n"
        f"【条文内容】{content}\n\n"
        "引致关系分类：\n"
        "1. explicit（明示引致）：条文明确出现“依照本法第X条”、“适用《XX法》有关规定”等；\n"
        "2. substantive_dependent（实体法效果依赖）：本条主张成立后，其具体的法律后果、赔偿计算方式、救济途径需依赖其他专门条款（如解除劳动合同依赖经济补偿计算条款）；\n"
        "3. general_special（特别/一般法衔接）：本条明确规定在某前提下优先或补充适用其他法律。\n\n"
        "请严格按以下 JSON 数组格式输出，不要有额外解释：\n"
        "[\n"
        "  {\n"
        f'    "source_law": "{law_name}",\n'
        f'    "source_article": "{article}",\n'
        '    "target_law": "目标法规全称（若为本法请直接填法规全称）",\n'
        '    "target_article": "目标条号（标准格式如：第46条）",\n'
        '    "relation_type": "explicit / substantive_dependent / general_special",\n'
        '    "reason": "引致依据与法理关系简述"\n'
        "  }\n"
        "]"
    )


_LEGAL_DEPENDENCY_KEYWORDS = (
    "赔偿", "补偿", "参照", "依照", "按照", "适用", "承担", "责任",
    "解除", "无效", "违约", "追偿", "连带", "优先", "另行规定", "有关规定",
)


_ART_NUM_PAT = r"第[零〇一二两三四五六七八九十百千0-9]+条"
_INVALID_LAW_PREFIXES = (
    "本", "该", "前款", "前条", "有关", "相关", "其他", "另行",
    "上位", "上述", "下列", "依照", "按照", "根据", "参照", "适用",
    "依据", "违反", "符合", "属于", "认定为", "执行", "遵守", "于",
)
_INVALID_LAW_CONTAINS = ("本法", "本条例", "本规定", "本办法", "本细则", "适用", "符合", "情形", "情况", "规定之")


def extract_quick_regex_references(
    law_name: str,
    article: str,
    content: str,
) -> List[Dict[str, Any]]:
    """正则快速抽取单条条文中的明示引致关系（毫秒级、零幻觉）。"""
    from app.kb import normalize_article_id

    clean_law = (law_name or "").strip().strip("《》")
    raw_art = (article or "").strip()
    norm_art = normalize_article_id(raw_art) if raw_art else ""
    if not clean_law or not norm_art or not content:
        return []

    results: List[Dict[str, Any]] = []
    seen = set()

    def _add_ref(t_law: str, t_art_raw: str, reason: str) -> None:
        t_art = normalize_article_id(t_art_raw)
        if not t_law or not t_art or (t_law == clean_law and t_art == norm_art):
            return
        k = (clean_law, norm_art, t_law, t_art)
        if k not in seen:
            seen.add(k)
            results.append(
                {
                    "source_law": clean_law,
                    "source_article": norm_art,
                    "target_law": t_law,
                    "target_article": t_art,
                    "relation_type": "explicit",
                    "reason": reason,
                }
            )

    # 1) 匹配本法/本条例/本规定等内部条文引致（支持顿号连续引用如「本法第二十六条、第二十七条」）
    for m in re.finditer(
        rf"(?:本法|本条例|本规定|本办法|本细则|本解释)\s*({_ART_NUM_PAT}(?:\s*[、和及]\s*{_ART_NUM_PAT})*)",
        content,
    ):
        for art_m in re.finditer(_ART_NUM_PAT, m.group(1)):
            t_art = normalize_article_id(art_m.group(0))
            _add_ref(clean_law, t_art, f"正文明确引致本法{t_art}")

    # 2) 匹配带书名号的跨法引致，如「《XX法》第XX条」
    for m in re.finditer(
        rf"《([^》]{{2,80}})》\s*({_ART_NUM_PAT}(?:\s*[、和及]\s*{_ART_NUM_PAT})*)",
        content,
    ):
        t_law = m.group(1).strip()
        for art_m in re.finditer(_ART_NUM_PAT, m.group(2)):
            t_art = normalize_article_id(art_m.group(0))
            _add_ref(t_law, t_art, f"正文明确引致《{t_law}》{t_art}")

    # 3) 匹配不带书名号的常用法律法规引致，如「按照监察法第五十三条」「根据民事诉讼法第一百条」
    for m in re.finditer(
        rf"(?:依照|按照|根据|参照|适用|依据|违反|符合|属于|认定为|遵守)\s*([\u4e00-\u9fa5]{{2,20}}(?:法典|法|条例|规定|办法|解释|细则))\s*({_ART_NUM_PAT}(?:\s*[、和及]\s*{_ART_NUM_PAT})*)",
        content,
    ):
        t_law = m.group(1).strip()
        if (
            any(t_law.startswith(p) for p in _INVALID_LAW_PREFIXES)
            or any(c in t_law for c in _INVALID_LAW_CONTAINS)
            or (len(t_law) < 3 and t_law != "刑法")
        ):
            continue
        for art_m in re.finditer(_ART_NUM_PAT, m.group(2)):
            t_art = normalize_article_id(art_m.group(0))
            _add_ref(t_law, t_art, f"正文明确引致《{t_law}》{t_art}")

    return results


def _record_graph_proposals(refs: List[Dict[str, Any]], query_prefix: str = "离线图谱构建") -> None:
    """将抽取的引致关系批量登记至 audit_store 待审规则提案池。"""
    for ref in refs:
        key = f"{ref['source_law']}|{ref['source_article']}->{ref['target_law']}|{ref['target_article']}"
        try:
            audit_store.record_rule_proposal(
                thread_id="offline_llm_graph_builder",
                proposal_type="cross_reference_graph",
                proposal_key=key,
                proposal_data=ref,
                query=f"{query_prefix}: 《{ref['source_law']}》{ref['source_article']}",
            )
        except Exception:
            pass


def extract_article_cross_references(
    law_name: str,
    article: str,
    content: str,
) -> List[Dict[str, Any]]:
    """抽取单条法律条文的引致关系三元组。"""
    clean_law = law_name.strip().strip("《》")
    norm_art = article.strip()

    # 1) 正则快速明示抽取（零依赖离线保证）
    quick_results = extract_quick_regex_references(clean_law, norm_art, content)

    if settings.mock:
        return quick_results

    # 2) 真实模式下，调用大模型抽取深层实体法依赖与衔接关系
    prompt = build_cross_reference_prompt(clean_law, norm_art, content)
    try:
        data = _extract_json(_call(prompt))
        if isinstance(data, list):
            llm_results = []
            for item in data:
                if isinstance(item, dict) and item.get("target_law") and item.get("target_article"):
                    llm_results.append(
                        {
                            "source_law": item.get("source_law") or clean_law,
                            "source_article": item.get("source_article") or norm_art,
                            "target_law": item.get("target_law", clean_law),
                            "target_article": item.get("target_article", ""),
                            "relation_type": item.get("relation_type", "explicit"),
                            "reason": item.get("reason", "大模型离线批处理提取"),
                        }
                    )
            # 合并去重
            seen = set()
            merged = []
            for r in quick_results + llm_results:
                k = (r["source_law"], r["source_article"], r["target_law"], r["target_article"])
                if k not in seen:
                    seen.add(k)
                    merged.append(r)
            return merged
    except Exception as e:
        logger.warning(f"大模型离线抽取引致关系异常: {e}")

    return quick_results


def batch_build_law_graph(
    articles: List[Dict[str, Any]],
    auto_propose: bool = True,
) -> List[Dict[str, Any]]:
    """批量扫描多条条文，构建引致图谱，并自动沉淀至待审规则提案池。"""
    all_refs: List[Dict[str, Any]] = []
    for art in articles:
        ln = art.get("law_name", "")
        a_no = art.get("article", "")
        cnt = art.get("content", "")
        if ln and a_no and cnt:
            refs = extract_article_cross_references(ln, a_no, cnt)
            all_refs.extend(refs)

    if auto_propose and all_refs:
        _record_graph_proposals(all_refs)

    return all_refs


def _batch_llm_extract_references(
    law_name: str,
    articles_batch: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    """合并多条重点条文一次性调用大模型批量抽取引致与实体法依赖关系，减少调用次数与 Token 消耗。"""
    if not articles_batch or settings.mock:
        return []

    from app.kb import normalize_article_id

    clean_law = (law_name or "").strip().strip("《》")
    items_text = "\n".join(
        f"- 【{a['article']}】{a['content'][:600]}"
        for a in articles_batch
    )
    prompt = (
        "你是一名权威立法专家与法学教授。请批量分析以下同一部法规的多条重点条文，识别每条的法条引致与实体法规范连接关系：\n"
        f"【法规全称】《{clean_law}》\n"
        f"【待分析条文列表】\n{items_text}\n\n"
        "引致关系分类：\n"
        "1. explicit（明示引致）：条文明确提及本法其他条文或其他法律规定；\n"
        "2. substantive_dependent（实体法效果依赖）：本条权利主张成立后，其赔偿计算、经济补偿、救济途径或责任承担需依赖本法或其他法律的专门条款；\n"
        "3. general_special（特别/一般法衔接）：优先或补充适用其他法律。\n\n"
        "注意：请仅输出确定存在的真实法条编号，不要编造不存在的条号。严格输出 JSON 数组：\n"
        "[\n"
        "  {\n"
        f'    "source_law": "{clean_law}",\n'
        '    "source_article": "来源条号（如 第38条）",\n'
        '    "target_law": "目标法规全称（本法请填原法规全称）",\n'
        '    "target_article": "目标条号（标准格式如 第46条）",\n'
        '    "relation_type": "explicit / substantive_dependent / general_special",\n'
        '    "reason": "引致依据与法理关系简述"\n'
        "  }\n"
        "]"
    )
    out: List[Dict[str, Any]] = []
    try:
        data = _extract_json(_call(prompt))
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                s_art = normalize_article_id(str(item.get("source_article") or ""))
                t_law = str(item.get("target_law") or clean_law).strip().strip("《》")
                t_art = normalize_article_id(str(item.get("target_article") or ""))
                if not s_art or not t_law or not t_art:
                    continue
                if t_law == clean_law and s_art == t_art:
                    continue
                out.append(
                    {
                        "source_law": str(item.get("source_law") or clean_law).strip().strip("《》"),
                        "source_article": s_art,
                        "target_law": t_law,
                        "target_article": t_art,
                        "relation_type": item.get("relation_type", "substantive_dependent"),
                        "reason": item.get("reason", "大模型批量漏斗抽取"),
                    }
                )
    except Exception as e:
        logger.warning(f"大模型批量合并抽取引致关系异常: {e}")
    return out


def _heuristic_cross_references(
    law_name: str,
    chunks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """基于同法条文内容的确定性实体依赖启发式抽取（确保离线/Mock环境下仍能产出核心赔偿/补偿衔接关系）。"""
    from app.kb import normalize_article_id

    clean_law = (law_name or "").strip().strip("《》")
    if not clean_law or not chunks:
        return []

    norm_chunks: List[Tuple[str, str]] = []
    for c in chunks:
        art = normalize_article_id(str(c.get("article") or ""))
        cnt = (c.get("content") or "").strip()
        if art and cnt:
            norm_chunks.append((art, cnt))

    # 识别同法中专门定义计算标准的核心条文（仅基于真实存在的条文，绝不虚构条号）
    anchor_targets: List[Tuple[str, str, str, str]] = []
    for art, cnt in norm_chunks:
        if "经济补偿" in cnt and ("每满一年" in cnt or "月工资" in cnt or "标准" in cnt):
            anchor_targets.append(("经济补偿", art, "经济补偿金计算标准", "substantive_dependent"))
            break
    for art, cnt in norm_chunks:
        if "赔偿金" in cnt and ("二倍" in cnt or "经济补偿标准" in cnt):
            anchor_targets.append(("赔偿金", art, "违法行为赔偿金计算标准", "substantive_dependent"))
            break
    for art, cnt in norm_chunks:
        if "违约金" in cnt and ("过分高于" in cnt or "适当减少" in cnt or "损失" in cnt):
            anchor_targets.append(("违约金", art, "违约金调整与损失赔偿规则", "substantive_dependent"))
            break

    results: List[Dict[str, Any]] = []
    if not anchor_targets:
        return results

    for art, cnt in norm_chunks:
        for kw, target_art, desc, rel_type in anchor_targets:
            if art == target_art or kw not in cnt:
                continue
            # 仅针对明确提出主张/责任但条文本身未给出计算标准的条款建立关联
            if any(trigger in cnt for trigger in ("支付", "承担", "赔偿", "解除", "终止", "依照")):
                results.append(
                    {
                        "source_law": clean_law,
                        "source_article": art,
                        "target_law": clean_law,
                        "target_article": target_art,
                        "relation_type": rel_type,
                        "reason": f"本条涉及「{kw}」主张，实体计算/救济依据衔接本法{target_art}（{desc}）",
                    }
                )
    return results


def batch_extract_from_kb_docs(
    doc_ids: Optional[List[str]] = None,
    kb_type: Optional[str] = None,
    scan_mode: str = "smart",
    auto_propose: bool = True,
    auto_apply: bool = False,
) -> Dict[str, Any]:
    """从已有知识库（law_documents / law_chunks）自动读取法规条文并执行漏斗式批量图谱构建。"""
    from app.mysql_store import mysql_store
    from app.kb import normalize_article_id

    clean_doc_ids = [str(d).strip() for d in (doc_ids or []) if str(d).strip()]
    if clean_doc_ids:
        placeholders = ",".join(["?"] * len(clean_doc_ids))
        docs = mysql_store.query(
            f"SELECT doc_id, title, kb_type FROM law_documents WHERE doc_id IN ({placeholders})",
            tuple(clean_doc_ids),
        )
    elif kb_type and kb_type.strip() and kb_type.strip().upper() != "ALL":
        clean_kb = kb_type.strip().rstrip("/")
        docs = mysql_store.query(
            "SELECT doc_id, title, kb_type FROM law_documents WHERE (kb_type = ? OR kb_type LIKE ?) AND is_latest = 1",
            (clean_kb, clean_kb + "/%"),
        )
    else:
        docs = mysql_store.query("SELECT doc_id, title, kb_type FROM law_documents WHERE is_latest = 1")

    if not docs:
        return {
            "status": "ok",
            "scanned_docs": 0,
            "scanned_chunks": 0,
            "extracted_count": 0,
            "applied_count": 0,
            "doc_titles": [],
            "references": [],
            "message": "所选范围内暂无已解析的有效法规文档",
        }

    all_refs: List[Dict[str, Any]] = []
    seen_keys = set()
    total_chunks = 0
    doc_titles: List[str] = []

    def _append_ref(ref: Dict[str, Any]) -> None:
        k = (ref["source_law"], ref["source_article"], ref["target_law"], ref["target_article"])
        if k not in seen_keys:
            seen_keys.add(k)
            all_refs.append(ref)

    for doc in docs:
        did = doc["doc_id"]
        title = (doc.get("title") or "").strip().strip("《》")
        if not title:
            continue
        doc_titles.append(title)
        chunks = mysql_store.query(
            "SELECT article, content, chunk_type FROM law_chunks WHERE doc_id = ? AND chunk_type != 'hierarchical_parent'",
            (did,),
        )
        total_chunks += len(chunks)

        # 第一阶段：全量毫秒级正则明示扫描
        smart_candidates: List[Dict[str, str]] = []
        for c in chunks:
            art = normalize_article_id(str(c.get("article") or "")) or str(c.get("article") or "").strip()
            cnt = (c.get("content") or "").strip()
            if not art or not cnt:
                continue
            for r in extract_quick_regex_references(title, art, cnt):
                _append_ref(r)
            if scan_mode == "smart" and any(kw in cnt for kw in _LEGAL_DEPENDENCY_KEYWORDS):
                smart_candidates.append({"article": art, "content": cnt})

        # 第二阶段：智能深层依赖扫描（启发式兜底 + 重点条文合并批处理调用 LLM）
        if scan_mode == "smart":
            for r in _heuristic_cross_references(title, chunks):
                _append_ref(r)

            if not settings.mock and smart_candidates:
                # 最多抽取前 18 条高价值重点条文，分批（每批 6 条）打包调用 LLM 防止超时
                capped_candidates = smart_candidates[:18]
                batch_size = 6
                for idx in range(0, len(capped_candidates), batch_size):
                    batch = capped_candidates[idx : idx + batch_size]
                    for r in _batch_llm_extract_references(title, batch):
                        _append_ref(r)

    if auto_propose and all_refs:
        _record_graph_proposals(all_refs, query_prefix="知识库批量图谱构建")

    applied_count = 0
    if auto_apply and all_refs:
        applied_count = apply_cross_references_to_config(all_refs)

    return {
        "status": "ok",
        "scanned_docs": len(docs),
        "scanned_chunks": total_chunks,
        "extracted_count": len(all_refs),
        "applied_count": applied_count,
        "doc_titles": doc_titles,
        "references": all_refs,
    }


# ————————————————————————————————————————————————
# 2. 司法解释「理解与适用」引申抽取引擎
# ————————————————————————————————————————————————
_RE_INTERP_ANCHOR = re.compile(
    r"(?:系|根据|依照|依据|结合|承继|对照|落实|基于)\s*《([^》]{2,80})》\s*(第[零〇一二两三四五六七八九十百千0-9]+条)"
)


def extract_interpretation_and_application_relations(
    interpretation_name: str,
    article: str,
    text_content: str,
) -> List[Dict[str, Any]]:
    """从最高院《某某司法解释理解与适用》论述文本中抽取法条锚定与协同适用关系。"""
    clean_interp = interpretation_name.strip().strip("《》")
    norm_art = article.strip()
    relations: List[Dict[str, Any]] = []

    # 1) 正则快速捕获最高院论述中的宿主法条锚定
    for m in _RE_INTERP_ANCHOR.finditer(text_content):
        from app.kb import normalize_article_id
        target_law = m.group(1).strip()
        target_art = normalize_article_id(m.group(2).strip())
        relations.append(
            {
                "interpretation_name": clean_interp,
                "interpretation_article": norm_art,
                "anchor_law": target_law,
                "anchor_article": target_art,
                "relation": "interpretation_anchor",
                "authority_source": "最高人民法院司法解释理解与适用",
                "notes": f"《{clean_interp}》{norm_art}系对照《{target_law}》{target_art}裁判规则之具体释义与展开",
            }
        )

    # 2) 大模型深层语义抽取
    if not settings.mock and len(text_content) > 100:
        prompt = (
            "你是一名最高人民法院司法解释研究学者。请从以下《理解与适用》论述中抽取该司法解释条款与实体法律的配套依据关系：\n"
            f"司法解释：《{clean_interp}》{norm_art}\n"
            f"论述文本：{text_content[:2000]}\n\n"
            "输出 JSON 数组：\n"
            "[\n"
            '  {"anchor_law": "被解释法律全称", "anchor_article": "具体条号", "relation": "interpretation_anchor", "notes": "适用要旨说明"}\n'
            "]"
        )
        try:
            data = _extract_json(_call(prompt))
            if isinstance(data, list):
                for d in data:
                    if isinstance(d, dict) and d.get("anchor_law") and d.get("anchor_article"):
                        relations.append(
                            {
                                "interpretation_name": clean_interp,
                                "interpretation_article": norm_art,
                                "anchor_law": d.get("anchor_law"),
                                "anchor_article": d.get("anchor_article"),
                                "relation": d.get("relation", "interpretation_anchor"),
                                "authority_source": "最高人民法院司法解释理解与适用（LLM精炼）",
                                "notes": d.get("notes", "司法解释配套裁判指引"),
                            }
                        )
        except Exception:
            pass

    # 去重
    seen = set()
    uniq = []
    for r in relations:
        k = (r["interpretation_name"], r["interpretation_article"], r["anchor_law"], r["anchor_article"])
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    return uniq


def apply_cross_references_to_config(references: List[Dict[str, Any]]) -> int:
    """将抽检通过的引致关系写入 knowledge_center 配置文件中热更新生效。"""
    graph = knowledge_center.get_cross_reference_graph()
    count = 0
    for r in references:
        s_law = r.get("source_law") or r.get("interpretation_name")
        s_art = r.get("source_article") or r.get("interpretation_article")
        t_law = r.get("target_law") or r.get("anchor_law")
        t_art = r.get("target_article") or r.get("anchor_article")
        reason = r.get("reason") or r.get("notes") or "引致关联"
        if not s_law or not s_art or not t_law or not t_art:
            continue
        key = (s_law, s_art)
        item = (t_law, t_art, reason)
        if key not in graph:
            graph[key] = []
        if item not in graph[key]:
            graph[key].append(item)
            count += 1

    if count > 0:
        cfg = knowledge_center.get_all()
        raw_list = []
        for (s_law, s_art), t_list in graph.items():
            raw_list.append(
                {
                    "source_law": s_law,
                    "source_article": s_art,
                    "targets": [
                        {"target_law": t[0], "target_article": t[1], "reason": t[2]}
                        for t in t_list
                    ],
                }
            )
        cfg["cross_reference_graph"] = raw_list
        knowledge_center.save(cfg)
    return count
