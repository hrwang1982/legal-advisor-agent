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


def extract_article_cross_references(
    law_name: str,
    article: str,
    content: str,
) -> List[Dict[str, Any]]:
    """抽取单条法律条文的引致关系三元组。"""
    clean_law = law_name.strip().strip("《》")
    norm_art = article.strip()

    # 1) 正则快速明示抽取（零依赖离线保证）
    quick_results: List[Dict[str, Any]] = []
    # 匹配「依照本法第XX条」
    for m in re.finditer(r"依照本法第([零〇一二两三四五六七八九十百千0-9]+)条", content):
        from app.kb import normalize_article_id
        t_art = normalize_article_id(f"第{m.group(1)}条")
        quick_results.append(
            {
                "source_law": clean_law,
                "source_article": norm_art,
                "target_law": clean_law,
                "target_article": t_art,
                "relation_type": "explicit",
                "reason": f"正文明确规定依照本法{t_art}",
            }
        )

    # 匹配「《XX法》第XX条」
    for m in re.finditer(r"《([^》]{2,80})》\s*(第[零〇一二两三四五六七八九十百千0-9]+条)", content):
        from app.kb import normalize_article_id
        t_law = m.group(1).strip()
        t_art = normalize_article_id(m.group(2).strip())
        quick_results.append(
            {
                "source_law": clean_law,
                "source_article": norm_art,
                "target_law": t_law,
                "target_article": t_art,
                "relation_type": "explicit",
                "reason": f"正文明确引致《{t_law}》{t_art}",
            }
        )

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
        # 将抽取的引致关系以提案形式记录至 audit_store
        for ref in all_refs:
            key = f"{ref['source_law']}|{ref['source_article']}->{ref['target_law']}|{ref['target_article']}"
            try:
                audit_store.record_rule_proposal(
                    thread_id="offline_llm_graph_builder",
                    proposal_type="cross_reference_graph",
                    proposal_key=key,
                    proposal_data=ref,
                    query=f"离线图谱构建: 《{ref['source_law']}》{ref['source_article']}",
                )
            except Exception:
                pass

    return all_refs


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
