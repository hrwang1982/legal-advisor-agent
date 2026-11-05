"""LangGraph 节点函数（P0 + P1 + P2 完整可交付版）：
事实抽取与合规守卫（PII脱敏 + 利益冲突检索）→ 问题拆解 → 法言法语改写
→ 混合检索（含引致条款扩展 + 新旧法条对照回填）→ 有效性打标
→ 多子问题重排与噪声截断 → 位阶排序 → 规则判定（含规则缺口自动归集）
→ 六段式结构化合成 + 三重后验校验 → 增强型人在环工作台（支持增删引用、改稿留痕 Diff、补充重检索、分级权限与防篡改审计）。
"""
from __future__ import annotations

import re
from datetime import date
from typing import Dict, List, Optional, Tuple

try:
    from langgraph.types import interrupt
except ImportError:
    def interrupt(payload):  # type: ignore
        # P0 安全加固：无 langgraph 环境下禁止静默返回 approve 自动放行！返回空字典表示挂起待审
        return {}

from app.audit import audit_store, compute_answer_diff
from app.compliance import (
    build_compliance_disclaimer,
    check_conflict_of_interest,
    mask_pii,
    verify_reviewer_permission,
)
from app.facts import analyze_facts, check_regional_applicability
from app.ingest import (
    build_retriever,
    build_version_timelines,
    extract_inline_references,
    find_counterpart_article,
    get_cross_reference_graph,
    load_nodes,
    resolve_canonical_law_name,
)
from app.kb import normalize_article_id
from app.knowledge import knowledge_center
from app.llm import (
    decompose_query,
    generate_answer,
    generate_structured_queries,
    get_and_clear_last_llm_error,
    rewrite_queries,
)
from app.models import Citation, SentenceTrace, TwoLayerAnswer
from app.trace import align_answer_to_citations
from app.rerank import filter_by_relevance, rerank
from app.rules import (
    Confidence,
    ReviewLevel,
    RuleOutcome,
    RuleStatus,
    aggregate_outcomes,
    fallback_plan,
    hard_constraints,
    identify_special_general,
    locate_article,
    prompt_footer_from_plan,
    resolve_hierarchy,
    temporal_locate,
)
from app.state import AgentState


def _node_to_citation(node, score: float) -> Citation:
    m = node.metadata
    return Citation(
        law_name=m.get("law_name", ""),
        document_number=m.get("document_number", ""),
        hierarchy=m.get("hierarchy", ""),
        host_law=m.get("host_law", ""),
        host_hierarchy=m.get("host_hierarchy", ""),
        host_effective_date=m.get("host_effective_date", ""),
        promulgation_date=m.get("promulgation_date", ""),
        effective_date=m.get("effective_date", ""),
        article_effective_date=m.get("article_effective_date", ""),
        effective_to=m.get("effective_to", ""),
        validity=m.get("validity", "现行有效"),
        superseded_by=m.get("superseded_by", ""),
        article=m.get("article", ""),
        paragraph=m.get("paragraph", ""),
        content=m.get("content", ""),
        source=m.get("source", ""),
        category=m.get("category", ""),
        scope=m.get("scope", ""),
        norm_type=m.get("norm_type", ""),
        partially_invalid=bool(m.get("partially_invalid", False)),
        invalid_parts=m.get("invalid_parts", ""),
        trial=bool(m.get("trial", False)),
        verified=bool(m.get("verified", True)),
        note=m.get("note", ""),
        score=round(float(score), 4),
    )


def _ensure_fact_summary(state: AgentState) -> Dict:
    """获取或即时计算结构化事实摘要（保证单步测试与完整图执行均具备事实层）。"""
    fs = state.get("fact_summary")
    if fs and isinstance(fs, dict):
        return fs
    summary = analyze_facts(
        state.get("query", ""),
        explicit_act_date=state.get("act_date"),
        explicit_end_date=state.get("act_end_date"),
        explicit_region=state.get("region"),
    )
    return summary.to_dict()


# 0) 事实要素抽取 + 合规安全检查（P0 事实层 + P2 PII脱敏与利益冲突检索）
def extract_facts_node(state: AgentState) -> Dict:
    raw_query = state.get("query", "")
    masked_q, pii_types = mask_pii(raw_query)
    coi = check_conflict_of_interest(raw_query)

    summary = analyze_facts(
        masked_q,
        explicit_act_date=state.get("act_date"),
        explicit_end_date=state.get("act_end_date"),
        explicit_region=state.get("region"),
    )
    out: Dict = {
        "masked_query": masked_q,
        "compliance_info": {
            "pii_masked": bool(pii_types),
            "pii_types": pii_types,
            "has_conflict": coi["has_conflict"],
            "entities": coi["entities"],
            "conflicts": coi["conflicts"],
        },
        "fact_summary": summary.to_dict(),
    }
    if summary.act_date_source in ("explicit", "extracted"):
        out["act_date"] = summary.act_date
    if summary.act_end_date:
        out["act_end_date"] = summary.act_end_date
    if summary.region:
        out["region"] = summary.region
    return out


# 1) 问题拆解（优先使用脱敏后文本防止敏感信息外发）
def decompose(state: AgentState) -> Dict:
    q = state.get("masked_query") or state["query"]
    subs = decompose_query(q)
    return {"decomposed": subs}


# 2) 问题改写与三维检索式生成（融合 ks-lma-chat 检索理念）
def rewrite(state: AgentState) -> Dict:
    queries: List[str] = []
    for sub in state.get("decomposed", []):
        queries.extend(rewrite_queries(sub))

    # 生成三维法律检索式：最精准、找例外、查后果
    structured = generate_structured_queries(
        state.get("masked_query") or state.get("query", ""),
        sub_questions=state.get("decomposed"),
        fact_summary=state.get("fact_summary"),
    )
    for dim_q_list in structured.values():
        queries.extend(dim_q_list)

    seen, uniq = set(), []
    for q in queries:
        if q not in seen:
            seen.add(q)
            uniq.append(q)
    return {
        "rewritten_queries": uniq,
        "structured_queries": structured,
    }


# 3) RAG 混合检索（BM25 + 概念扩展 + P1 引致条款图谱扩展 + P1 新旧法条文对照回填）
def retrieve(state: AgentState) -> Dict:
    nodes = load_nodes()
    node_index: Dict[Tuple[str, str, str], object] = {}
    for n in nodes:
        k = (
            n.metadata.get("law_name", ""),
            n.metadata.get("article", ""),
            n.metadata.get("paragraph", ""),
        )
        node_index[k] = n

    retriever = build_retriever(nodes)
    collected: Dict[Tuple[str, str, str], Citation] = {}
    for q in state.get("rewritten_queries", []):
        for r in retriever.retrieve(q):
            node = r.node if hasattr(r, "node") else r
            key = (
                node.metadata.get("law_name", ""),
                node.metadata.get("article", ""),
                node.metadata.get("paragraph", ""),
            )
            cit = _node_to_citation(node, getattr(r, "score", 0.0))
            if key not in collected or cit.score > collected[key].score:
                collected[key] = cit

    # 3.1 P1 引致条款图谱扩展（实时读取外置配置） + 正文内嵌条号自动关联
    cross_graph = get_cross_reference_graph()
    seed_cites = sorted(collected.values(), key=lambda c: c.score, reverse=True)[:5]
    for sc in seed_cites:
        refs = list(cross_graph.get((sc.law_name, sc.article), []))
        for inline_ref in extract_inline_references(sc.law_name, sc.content):
            if inline_ref not in refs:
                refs.append(inline_ref)
        for target_law, target_art, ref_reason in refs:
            t_key = (target_law, normalize_article_id(target_art), "")
            if t_key not in collected and t_key in node_index:
                ref_cit = _node_to_citation(node_index[t_key], round(sc.score * 0.85, 4))
                prefix = f"［引致关联：{ref_reason}］"
                ref_cit.note = f"{prefix} {ref_cit.note}".strip()
                collected[t_key] = ref_cit

    # 3.2 P1 新旧法条文级对照标注与历史纠纷自动回填
    fs = _ensure_fact_summary(state)
    act_date = state.get("act_date") or fs.get("act_date") or date.today().isoformat()
    for c in list(collected.values()):
        cp = find_counterpart_article(c.law_name, c.article)
        if not cp:
            continue
        tag = f"［新旧法对照：{cp['diff_note']}］"
        if tag not in (c.note or ""):
            c.note = f"{tag} {c.note}".strip()
        if cp["role"] == "current" and act_date < cp["switch_date"]:
            old_key = (cp["counterpart_law"], cp["counterpart_article"], "")
            if old_key not in collected and old_key in node_index:
                old_cit = _node_to_citation(node_index[old_key], round(c.score * 0.95, 4))
                old_cit.note = f"［新旧法对照回填：{cp['diff_note']}］ {old_cit.note}".strip()
                collected[old_key] = old_cit

    cites = sorted(collected.values(), key=lambda c: c.score, reverse=True)
    return {"retrieved": cites}


# 4) 有效性校验（时间感知）：只打标留痕，不提前删除失效条文
def validate(state: AgentState) -> Dict:
    notes: List[str] = []
    kept: List[Citation] = []
    for c in state.get("retrieved", []):
        if c.validity == "已废止":
            notes.append(
                f"已废止（保留待时间定位）：{c.brief()}；"
                f"替代依据：{c.superseded_by or '未标注'}。"
                f"仅当行为发生日不晚于失效日时才适用。"
            )
        elif c.validity == "已被修订":
            notes.append(
                f"已被修订（保留待时间定位）：{c.brief()}；现行文本：{c.superseded_by or '未标注'}。"
            )
        elif c.validity == "部分失效":
            notes.append(
                f"部分失效：{c.brief()}；失效部分：{c.invalid_parts or '未标注'}，适用时必须排除该部分。"
            )
        elif c.validity == "尚未施行":
            notes.append(
                f"尚未施行：{c.brief()}；施行日 {c.effective_date or '未标注'}，"
                f"仅当行为发生日不早于施行日时才适用。"
            )
        if c.trial:
            notes.append(f"试行/暂行规范：{c.brief()}，效力稳定性较低，须关注后续转正或废止。")
        if not c.verified:
            notes.append(f"待核验条文：{c.brief()}，内容未经权威源核验，不得进入自动对外通道。")
        kept.append(c)
    return {"validated": kept, "validation_notes": notes}


# 5) 重排（多子问题 Max-Pooling 相关性精排 + 相关性阈值过滤噪声）
def rerank_node(state: AgentState) -> Dict:
    candidates = state.get("validated", [])
    sub_queries = state.get("decomposed") or []
    scored = rerank(state["query"], candidates, sub_queries=sub_queries)
    filtered = filter_by_relevance(scored)
    dropped_cnt = len(scored) - len(filtered)
    notes = [f"重排：{c.brief()} → 相关性 {c.rerank_score:.3f}" for c in filtered]
    if dropped_cnt > 0:
        notes.append(f"相关性截断：已过滤 {dropped_cnt} 条低于相关性阈值的边缘噪声条文。")
    return {"reranked": filtered, "rerank_scores": notes}


# 6) 位阶排序
def rank_hierarchy(state: AgentState) -> Dict:
    items = state.get("reranked") or state.get("validated", [])
    ranked = sorted(items, key=lambda c: (c.rank(), -c.rerank_score, -c.score))
    return {"ranked": ranked}


def _infer_norm_type(c: Citation) -> str:
    """推导条文的规范类型（实体法 / 程序法 / 司法解释 / 行政处罚）。"""
    if c.norm_type:
        return c.norm_type
    if c.hierarchy == "司法解释":
        return "judicial_interpretation"
    if c.category == "诉讼" or any(k in c.law_name for k in ("诉讼法", "复议法", "仲裁法")):
        return "procedural"
    if "行政处罚法" in c.law_name:
        return "administrative_penalty"
    return "substantive"


def _are_successor_laws(a: Citation, b: Citation) -> bool:
    """判断两部同位阶法律是否属于新旧替代/修订演进关系。"""
    sa, sb = a.superseded_by or "", b.superseded_by or ""
    return (a.law_name in sb) or (b.law_name in sa)


# 7) 规则判定：事实前提核验 + 合规利益冲突核验 + 条文级时空效力过滤 + 冲突识别与位阶裁决
def rules_node(state: AgentState) -> Dict:
    cites = state.get("ranked", [])
    fs = _ensure_fact_summary(state)
    comp = state.get("compliance_info") or {}
    act_date = state.get("act_date") or fs.get("act_date") or date.today().isoformat()
    act_end_date = state.get("act_end_date") or fs.get("act_end_date") or ""
    act_date_source = "explicit" if state.get("act_date") else fs.get("act_date_source", "assumed_today")
    user_region = state.get("region") or fs.get("region") or ""

    notes: List[str] = []
    active_outcomes: List[RuleOutcome] = []

    # 7.0a P2 合规预警：PII 脱敏留痕 & 律所利益冲突升级
    if comp.get("pii_masked"):
        notes.append(f"合规保护：已自动对输入中的敏感个人信息（{'、'.join(comp.get('pii_types', []))}）执行脱敏保护。")
    if comp.get("has_conflict"):
        for c_item in comp.get("conflicts", []):
            w_msg = c_item["warning"]
            notes.append(f"合规预警：{w_msg}")
            active_outcomes.append(
                RuleOutcome(
                    status=RuleStatus.PARTIAL,
                    reason=w_msg,
                    confidence=Confidence.B,
                    review=ReviewLevel.EXPERT,
                    llm_allowed=True,
                    constraints=hard_constraints(),
                    verify_actions=["由律所合伙人/风控主管完成《律师法》第39条利益冲突检索与豁免审批"],
                )
            )

    # 7.0b 事实时间前提检查
    if act_date_source == "assumed_today":
        active_outcomes.append(
            RuleOutcome(
                status=RuleStatus.PARTIAL,
                reason=f"当事人未提供法律事实发生日期，暂按当前日期（{act_date}）定位现行有效法；若系历史纠纷，适用依据可能不同",
                confidence=Confidence.B,
                review=ReviewLevel.MUST,
                llm_allowed=True,
                constraints=hard_constraints(),
                verify_actions=["向当事人核实法律事实发生的具体日期（或持续起止期间）"],
            )
        )
        notes.append(
            f"事实前提：未提供法律事实发生日，暂按 {act_date} 现行有效法推演（已降级为 B 级置信度/人工必审）。"
        )
    elif fs.get("date_granularity") in ("year", "month"):
        active_outcomes.append(
            RuleOutcome(
                status=RuleStatus.PARTIAL,
                reason=f"行为发生时间仅精确到{fs.get('date_granularity')}（按 {act_date} 推算），需确认具体日期以防跨新旧法分界点",
                confidence=Confidence.B,
                review=ReviewLevel.MUST,
                llm_allowed=True,
                constraints=hard_constraints(),
                verify_actions=["向当事人确认法律事实发生的具体年月日"],
            )
        )

    # 7.1 法律级版本定位留痕
    timelines = build_version_timelines()
    for law in sorted({c.law_name for c in cites if c.law_name}):
        o = temporal_locate(act_date, timelines.get(law, []))
        if o.status == RuleStatus.CONFLICT:
            active_outcomes.append(o)
        if o.status == RuleStatus.HIT:
            ver = o.detail.get("applicable_version")
            if ver:
                notes.append(f"版本定位：《{law}》依 {act_date} 适用 {ver}。")
            elif o.detail.get("applicable") is False:
                notes.append(f"版本定位：《{law}》于 {act_date} 尚未生效，原则上不溯及既往。")
        else:
            notes.append(f"版本定位：《{law}》{o.status.value} — {o.reason}")

    # 7.2 条文级时空适用性判定
    kept: List[Citation] = []
    dropped = 0
    for c in cites:
        label = f"《{c.law_name}》{c.article}{c.paragraph}"
        ntype = _infer_norm_type(c)

        reg = check_regional_applicability(c.law_name, c.hierarchy, c.source, user_region)
        if not reg["applicable"]:
            dropped += 1
            notes.append(f"地域效力：{label} — {reg['note']}，已排除。")
            continue

        o = locate_article(
            act_date,
            c.effective_date,
            timelines.get(c.law_name, []),
            c.effective_to,
            label,
            article_effective_from=c.article_effective_date,
            norm_type=ntype,
            host_effective_from=c.host_effective_date,
            act_end_date=act_end_date,
        )
        if o.detail.get("applicable") is False:
            dropped += 1
            notes.append(f"时间效力：{label} — {o.reason}，已排除。")
            continue

        active_outcomes.append(o)
        c.applicable_at = True
        note_parts = [o.detail.get("note", "")]
        if reg.get("regional_warning"):
            note_parts.append(reg["note"])
            active_outcomes.append(
                RuleOutcome(
                    status=RuleStatus.PARTIAL,
                    reason=reg["note"],
                    confidence=Confidence.B,
                    review=ReviewLevel.MUST,
                    llm_allowed=True,
                    constraints=hard_constraints(),
                    verify_actions=[f"核实案件用工地/合同履行地/行为发生地是否位于{reg['law_region']}"],
                )
            )
            notes.append(f"地域效力：{label} — {reg['note']}。")

        if not c.verified:
            active_outcomes.append(
                RuleOutcome(
                    status=RuleStatus.PARTIAL,
                    reason=f"{label}未经权威来源核验（verified=false），不得作为正式对外依据",
                    confidence=Confidence.B,
                    review=ReviewLevel.MUST,
                    llm_allowed=True,
                    constraints=hard_constraints(),
                    verify_actions=[f"对照官方权威文本核验{label}条文原文与现行有效性"],
                )
            )

        c.applicable_note = "；".join(p for p in note_parts if p)
        if o.detail.get("historical"):
            notes.append(
                f"时间效力：{label} — {o.reason}；输出必须标注该条现已失效，不得以现行法主张。"
            )
        kept.append(c)

    if dropped:
        notes.append(f"效力过滤：已排除 {dropped} 条在 {act_date}（或当前地域）不适用的条文。")
    if cites and not kept:
        active_outcomes.append(
            RuleOutcome(
                status=RuleStatus.MISS,
                reason="全部候选条文在行为发生日或所属地域均不适用，无法给出实体结论",
                confidence=Confidence.C,
                review=ReviewLevel.EXPERT,
                llm_allowed=True,
                constraints=hard_constraints(),
                verify_actions=["确认法律事实发生日、地域，或排查是否遗漏当时有效的法律"],
            )
        )

    # 7.3 冲突识别与位阶裁决
    precedence: List[dict] = []
    seen_pairs = set()
    pairs = kept[:6]
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            a, b = pairs[i], pairs[j]
            if a.law_name == b.law_name:
                continue
            pair_key = tuple(sorted((a.law_name, b.law_name)))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            is_interp_pair = (
                (a.hierarchy == "司法解释" and a.host_law == b.law_name)
                or (b.hierarchy == "司法解释" and b.host_law == a.law_name)
            )
            if not is_interp_pair and a.category and b.category and a.category != b.category:
                continue
            if not a.hierarchy or not b.hierarchy:
                continue

            if not user_region and (
                a.hierarchy in ("地方性法规", "地方政府规章")
                or b.hierarchy in ("地方性法规", "地方政府规章")
            ):
                notes.append(
                    f"位阶提示：《{a.law_name}》（{a.hierarchy}）与《{b.law_name}》（{b.hierarchy}）并存；"
                    "因用户未指明地域，优先适用全国性规定，若案件位于地方性法规辖区内且规定不一致，依《立法法》第106条处理。"
                )
                continue

            a_dict = {
                "law_name": a.law_name,
                "hierarchy": a.hierarchy,
                "host_law": a.host_law,
                "host_hierarchy": a.host_hierarchy,
                "promulgation_date": a.promulgation_date,
                "authority": a.source,
                "scope": a.scope,
            }
            b_dict = {
                "law_name": b.law_name,
                "hierarchy": b.hierarchy,
                "host_law": b.host_law,
                "host_hierarchy": b.host_hierarchy,
                "promulgation_date": b.promulgation_date,
                "authority": b.source,
                "scope": b.scope,
            }

            if a.hierarchy == b.hierarchy:
                sg = identify_special_general(a_dict, b_dict)
                if sg is None and not _are_successor_laws(a, b):
                    notes.append(
                        f"规范关系：《{a.law_name}》与《{b.law_name}》同属{a.hierarchy}，调整不同制度环节，并行协同适用。"
                    )
                    continue

            o = resolve_hierarchy(a_dict, b_dict)
            active_outcomes.append(o)
            if o.status == RuleStatus.HIT and o.detail.get("precedence"):
                precedence.append(
                    {
                        "precedence": o.detail["precedence"],
                        "secondary": o.detail.get("secondary", ""),
                        "relation": o.detail.get("relation", ""),
                        "reason": o.reason,
                    }
                )
                notes.append(f"位阶裁决：{o.reason}")
            elif not o.llm_allowed:
                notes.append(f"位阶裁决：{o.status.value} — {o.reason}")

    # 7.4 把裁决结论固化到引用顺序
    prio: Dict[str, int] = {}
    for p in precedence:
        name = p.get("precedence")
        if name and name not in prio:
            prio[name] = len(prio)
    if prio:
        kept = sorted(
            kept,
            key=lambda c: (prio.get(c.law_name, len(prio)), c.rank(), -c.rerank_score),
        )

    overall = aggregate_outcomes(active_outcomes, fallback_reason="本次未产生可判定的规则条目")
    plan = fallback_plan(overall)
    for mf in fs.get("missing_facts", [])[:3]:
        if mf not in plan["verify_actions"]:
            plan["verify_actions"].append(mf)

    # P2：自动将非确定性（PARTIAL/MISS/CONFLICT）规则缺口归集至持久化规则缺口表（绑定真实会话 thread_id）
    try:
        audit_store.record_rule_gap(
            thread_id=state.get("thread_id") or "rule_engine",
            plan=plan,
            query=state.get("masked_query") or state.get("query", ""),
        )
    except Exception:
        pass

    notes.append(
        f"规则汇总：{overall.status.value}｜mode={plan['mode']}"
        f"｜置信度={plan['confidence']}｜人审={plan['review']}"
    )
    return {
        "act_date": act_date,
        "fact_summary": fs,
        "ranked": kept,
        "rule_plan": plan,
        "rule_notes": notes,
        "precedence": precedence,
    }


def _precedence_block(precedence: List[dict], need_interpretation_fallback: bool = False) -> str:
    """把位阶与适用关系结论转化为精准提示词指令（区分解释协同、特别法补充、上下位排他、法理涵摄推演兜底）。"""
    lines = []
    if precedence:
        lines.append("【规范适用与位阶关系规则｜必须严格遵循】")
        for p in precedence:
            rel = p.get("relation") or ""
            pri = p.get("precedence")
            sec = p.get("secondary")
            reason = p.get("reason")
            if rel == "interpretation":
                lines.append(f"- 解释协同适用：以《{pri}》为基础法律依据，并结合其配套司法解释的具体裁判规则协同分析（{reason}）")
            elif rel == "special_general":
                sec_str = f"《{sec}》" if sec else "一般法"
                lines.append(f"- 特别法优于一般法：优先适用《{pri}》；仅在《{pri}》未作具体规定之处，补充适用{sec_str}（{reason}）")
            else:
                lines.append(f"- 上位法/新法优先：优先适用《{pri}》，下位法或旧法与之抵触的部分不得作为依据（{reason}）")
    if need_interpretation_fallback:
        lines.append("【规范竞合与法理涵摄推演要求（法律适用关系未配置时的兜底约束）】")
        lines.append("- 法律解释学涵摄：若涉及跨部门法或多部法律规范竞合且无绝对位阶排他，须严格遵循法律解释学（文义解释、体系解释、立法目的解释），将案件具体事实逐项对应代入规范要件进行涵摄推演；")
        lines.append("- 审慎抗辩与双向争议提示：明确向用户披露潜在的法律适用分歧点与司法裁决倾向，禁止给出单向绝对化胜诉承诺，提供审慎的诉辩应对策略。")
    return "\n".join(lines)


# ————————————————————————————————————————————————
# 生成后三重后验校验器（条号级严格引用闭包 + 失效部分/待核验披露 + 兜底模式措辞合规）
# ————————————————————————————————————————————————
_DEFAULT_CERTAINTY_PATTERNS = [
    (r"必然胜诉", "在证据充分前提下有较大可能获得支持"),
    (r"一定能(?:够)?获得", "在符合法定要件前提下倾向于可主张"),
    (r"肯定能(?:够)?", "在满足前提条件下倾向于可"),
    (r"毫无疑问", "结合现有依据倾向于认为"),
    (r"绝对无效", "存在被认定为无效的法律风险"),
]

# 匹配《法规名》第X条 或 无书名号的常见法典简称（如 劳动合同法第38条、民法典第五百七十七条）
_BRACKET_CITE_RE = re.compile(
    r"[《〈]([^》〉]{2,80})[》〉]\s*(?:（[^）]{1,20}）)?\s*(第[零〇一二两三四五六七八九十百千万0-9]+条)"
)
_UNBRACKET_CITE_RE = re.compile(
    r"(?<![《〈\w\u4e00-\u9fa5])"
    r"((?:中华人民共和国)?(?:民法典|合同法|劳动合同法|公司法|消费者权益保护法|消保法|道路交通安全法|道交法|个人信息保护法|个保法|行政处罚法|行政复议法|行政诉讼法|民事诉讼法|劳动争议调解仲裁法|刑法|立法法|律师法))"
    r"\s*(第[零〇一二两三四五六七八九十百千万0-9]+条)"
)


def _norm_law_and_art(raw_law: str, raw_art: str) -> Tuple[str, str]:
    """将法规名（含简称/别名/年份后缀）与条号（含中文数字）统一归一化为 (标准法规全称, '第N条')。"""
    clean_law = re.sub(r"（\d{4}年(?:修正|修订)）", "", (raw_law or "").strip().strip("《》〈〉"))
    canon_law = resolve_canonical_law_name(clean_law)
    norm_art = normalize_article_id(raw_art)
    return canon_law, norm_art


def verify_citation_closure(answer: str, citations: List[Citation]) -> List[str]:
    """P0 严格条号级引用闭包核验：
    核验生成回答中引用的每一处「《法规名》第X条」或「XX法第X条」是否真实存在于第一层候选集（或程序元法条级白名单）中。
    彻底堵死：
    1) 某部法律虽被召回（如召回《劳动合同法》第38条），但模型编造同部法律其他未召回条号（如《劳动合同法》第999条）；
    2) 《民法典》整部在白名单导致模型编造《民法典》第8888条不告警；
    3) 中文数字条号（如“第三十八条”）、超长司法解释名称、或不加书名号引用漏检。
    """
    if not answer:
        return []

    # 1) 构建允许的 (标准法规名, 标准条号) 集合：第一层召回条文 + 外置程序/立法元法条级白名单
    allowed_pairs = set(knowledge_center.get_allowed_meta_citations())
    for c in citations:
        ln_raw = (c.law_name or "").strip().strip("《》〈〉")
        canon_ln, norm_art = _norm_law_and_art(ln_raw, c.article or "")
        if canon_ln and norm_art:
            allowed_pairs.add((canon_ln, norm_art))
            allowed_pairs.add((ln_raw, norm_art))
            allowed_pairs.add((canon_ln.replace("中华人民共和国", ""), norm_art))

        # 条文原文及官方元数据备注（c.content / c.note）中提及的对照/引致法条属于受信任元数据
        for meta_txt in (c.content or "", c.note or ""):
            for mm in _BRACKET_CITE_RE.finditer(meta_txt):
                m_law, m_art = _norm_law_and_art(mm.group(1), mm.group(2))
                allowed_pairs.add((m_law, m_art))
                allowed_pairs.add((m_law.replace("中华人民共和国", ""), m_art))

    # 仅对实体分析正文执行引用闭包核验（排除【五、诉讼/仲裁时效与程序路径提示】及【六、建议法务/律师核实引证的补充条文】）
    check_text = answer.split("【五、诉讼/仲裁时效与程序路径提示】")[0]
    check_text = check_text.split("【六、建议法务/律师核实引证的补充条文】")[0]

    extracted_mentions: List[Tuple[str, str, str]] = []
    for m in _BRACKET_CITE_RE.finditer(check_text):
        extracted_mentions.append((m.group(1).strip(), m.group(2).strip(), f"《{m.group(1).strip()}》{m.group(2).strip()}"))
    for m in _UNBRACKET_CITE_RE.finditer(check_text):
        extracted_mentions.append((m.group(1).strip(), m.group(2).strip(), f"《{m.group(1).strip()}》{m.group(2).strip()}"))

    warnings: List[str] = []
    for raw_law, raw_art, display_str in extracted_mentions:
        canon_law, norm_art = _norm_law_and_art(raw_law, raw_art)
        short_law = canon_law.replace("中华人民共和国", "")
        if (
            (canon_law, norm_art) not in allowed_pairs
            and (short_law, norm_art) not in allowed_pairs
            and (raw_law, norm_art) not in allowed_pairs
        ):
            msg = f"引用闭包告警：回答提及未在第一层召回的法条 {display_str}（归一：{canon_law} {norm_art}），请人工核删"
            if msg not in warnings:
                warnings.append(msg)
    return warnings


def extract_suggested_citations(answer: str, fs_suggested: Optional[List[str]] = None) -> List[str]:
    """从回答末尾的建议补充章节中抽取条文，并与事实层动态推导的建议条文合并去重。"""
    res = list(fs_suggested or [])
    if "【六、建议法务/律师核实引证的补充条文】" in (answer or ""):
        sec = answer.split("【六、建议法务/律师核实引证的补充条文】")[-1]
        for m in _BRACKET_CITE_RE.finditer(sec):
            c = f"《{m.group(1).strip()}》{m.group(2).strip()}"
            if c not in res:
                res.append(c)
        for m in _UNBRACKET_CITE_RE.finditer(sec):
            c = f"《{m.group(1).strip()}》{m.group(2).strip()}"
            if c not in res:
                res.append(c)
    return res


def post_verify_answer(
    answer: str,
    citations: List[Citation],
    plan: Dict,
) -> Tuple[str, List[str]]:
    """对生成的第二层回答执行三重后验校验与自动纠偏。"""
    notes: List[str] = []
    fixed_answer = answer or ""

    closure_warns = verify_citation_closure(fixed_answer, citations)
    notes.extend(closure_warns)

    for c in citations:
        if c.partially_invalid and c.invalid_parts and c.invalid_parts not in fixed_answer:
            warn_line = f"\n【部分失效排除警示】《{c.law_name}》{c.article}中「{c.invalid_parts}」已失效，适用时必须予以排除。"
            fixed_answer += warn_line
            notes.append(f"后验补齐：已自动追加《{c.law_name}》{c.article}部分失效（{c.invalid_parts}）排除警示。")
        if not c.verified and "待核验" not in fixed_answer:
            warn_line = f"\n【待核验警示】《{c.law_name}》{c.article}未经权威源核验，不得作为正式对外依据。"
            fixed_answer += warn_line
            notes.append(f"后验补齐：已自动追加《{c.law_name}》{c.article}待核验警示。")

    if plan.get("llm_callable"):
        patterns = knowledge_center.get_certainty_patterns() or _DEFAULT_CERTAINTY_PATTERNS
        for pat, repl in patterns:
            if re.search(pat, fixed_answer):
                fixed_answer = re.sub(pat, repl, fixed_answer)
                notes.append(f"后验措辞降级：兜底模式下检测到绝对化表述「{pat}」，已自动替换为谨慎表述。")

    return fixed_answer, notes


# 8) 两层结构合成：第一层=引用（真实可追溯），第二层=六段式分析 + 三重后验校验 + AI 合规标识
def synthesize(state: AgentState) -> Dict:
    cites = state.get("ranked", [])
    plan = dict(state.get("rule_plan") or {})
    mode = plan.get("mode", "deterministic")
    fs = _ensure_fact_summary(state)

    # 规范关系未配置或涉及跨部门法规范竞合时的法理涵摄推演约束
    distinct_laws = {c.law_name for c in cites if c.law_name}
    need_interp_fallback = len(distinct_laws) >= 2 or not state.get("precedence")
    precedence_note = _precedence_block(
        state.get("precedence") or [],
        need_interpretation_fallback=need_interp_fallback,
    )

    rule_notes = list(state.get("rule_notes") or [])
    conf = plan.get("confidence", "A")
    disclaimer = build_compliance_disclaimer(is_human_approved=False, answer_confidence=conf)

    struct_queries = state.get("structured_queries") or {}

    if mode == "escalate":
        answer = (
            "【规则判定：需法定机关或执业律师人工裁决，本系统不出具实体结论】\n"
            f"原因：{plan.get('reason', '规则未覆盖或存在冲突')}\n"
            "依《中华人民共和国立法法》（2023年修正）第105条、第106条相关规定，此类规范冲突情形应由有权机关裁决或由执业律师核实特别法转致条款后按程序处理。"
        )
        escalate_traces = align_answer_to_citations(answer, cites)
        return {
            "final_output": TwoLayerAnswer(
                citations=cites,
                answer=answer,
                sentence_traces=escalate_traces,
                structured_queries=struct_queries,
                missing_facts=fs.get("missing_facts", []),
                clarification_questions=fs.get("clarification_questions", []),
                procedural_notes=fs.get("procedural_notes", []),
                disclaimer=disclaimer,
                suggested_citations=[],
            ),
            "sentence_traces": [t.model_dump() for t in escalate_traces],
            "answer_confidence": plan.get("confidence", "D"),
        }

    constraints = prompt_footer_from_plan(plan)
    raw_answer = generate_answer(
        state.get("masked_query") or state["query"],
        cites,
        state.get("human_feedback"),
        extra_constraints=constraints,
        precedence_note=precedence_note,
        fact_summary=fs,
        sub_questions=state.get("decomposed"),
    )

    verified_answer, post_notes = post_verify_answer(raw_answer, cites, plan)
    llm_err = get_and_clear_last_llm_error()
    if llm_err:
        post_notes.append(f"LLM调用告警：外部模型调用异常（{llm_err}），已降级为规则骨架兜底并强制人工必审。")
        plan["review"] = ReviewLevel.MUST.value
        if plan.get("confidence") == Confidence.A.value:
            plan["confidence"] = Confidence.B.value

    if post_notes:
        rule_notes.extend(post_notes)
        if any("引用闭包告警" in n for n in post_notes):
            if plan.get("review") in (None, ReviewLevel.SAMPLE.value):
                plan["review"] = ReviewLevel.MUST.value
            if plan.get("confidence") == Confidence.A.value:
                plan["confidence"] = Confidence.B.value

    conf = plan.get("confidence", "A")
    disclaimer = build_compliance_disclaimer(is_human_approved=False, answer_confidence=conf)

    # 建议2：提取大模型推荐补充条文候选池
    suggested_cits = extract_suggested_citations(verified_answer, fs.get("suggested_citations", []))

    # 建议4：知识飞轮——大模型推导成果自动归集为待审规则补丁（rule_proposals）
    thread_id = state.get("thread_id") or "session_dynamic"
    raw_q = state.get("query", "")
    dyn_prop = fs.get("dynamic_proposal")
    if dyn_prop and isinstance(dyn_prop, dict):
        try:
            audit_store.record_rule_proposal(
                thread_id=thread_id,
                proposal_type="case_type_rules",
                proposal_key=dyn_prop.get("case_type", ""),
                proposal_data=dyn_prop,
                query=raw_q,
            )
        except Exception:
            pass
    if suggested_cits:
        try:
            audit_store.record_rule_proposal(
                thread_id=thread_id,
                proposal_type="suggested_citations",
                proposal_key=f"cite_patch_{thread_id}",
                proposal_data={"citations": suggested_cits},
                query=raw_q,
            )
        except Exception:
            pass

    # 融合 ks-lma-chat：计算大模型分析回答各句子对齐法条的细粒度溯源
    sentence_traces = align_answer_to_citations(verified_answer, cites)

    return {
        "final_output": TwoLayerAnswer(
            citations=cites,
            answer=verified_answer,
            sentence_traces=sentence_traces,
            structured_queries=struct_queries,
            missing_facts=fs.get("missing_facts", []),
            clarification_questions=fs.get("clarification_questions", []),
            procedural_notes=fs.get("procedural_notes", []),
            disclaimer=disclaimer,
            suggested_citations=suggested_cits,
        ),
        "sentence_traces": [t.model_dump() for t in sentence_traces],
        "rule_plan": plan,
        "rule_notes": rule_notes,
        "answer_confidence": conf,
    }


# ————————————————————————————————————————————————
# P2 升级：律师工作台引用条文增删辅助函数
# ————————————————————————————————————————————————
def apply_citation_edits(
    citations: List[Citation],
    add_specs: Optional[List[str]] = None,
    remove_specs: Optional[List[str]] = None,
) -> Tuple[List[Citation], List[str]]:
    """支持执业律师在审核台增删第一层引用条文。

    规格格式：`"中华人民共和国民法典|第1191条"` 或 `"民法典|第1191条"` 或 `"第1191条"`。
    """
    log: List[str] = []
    cur = list(citations)

    # 1) 删除指定条文
    for spec in remove_specs or []:
        s = (spec or "").strip()
        if not s:
            continue
        if "|" in s:
            r_law, r_art = [x.strip().strip("《》") for x in s.split("|", 1)]
        else:
            r_law, r_art = "", s.strip()
        before_len = len(cur)
        cur = [
            c
            for c in cur
            if not (
                c.article == r_art
                and (not r_law or r_law in c.law_name or c.law_name in r_law)
            )
        ]
        if len(cur) < before_len:
            log.append(f"律师剔除引用：{s}")

    # 2) 从知识库增补指定条文
    if add_specs:
        nodes = load_nodes()
        existing_keys = {(c.law_name, c.article) for c in cur}
        for spec in add_specs:
            s = (spec or "").strip()
            if not s:
                continue
            if "|" in s:
                a_law, a_art = [x.strip().strip("《》") for x in s.split("|", 1)]
            else:
                a_law, a_art = "", s.strip()
            for n in nodes:
                m = n.metadata
                ln, art = m.get("law_name", ""), m.get("article", "")
                if art == a_art and (not a_law or a_law in ln or ln in a_law):
                    if (ln, art) not in existing_keys:
                        added = _node_to_citation(n, score=1.0)
                        added.rerank_score = 1.0
                        added.applicable_note = "律师人工增补引用"
                        cur.append(added)
                        existing_keys.add((ln, art))
                        log.append(f"律师增补引用：《{ln}》{art}")
                    break

    return cur, log


# 9) P2 增强型人在环工作台：支持增删引用、直接改稿留存 Diff、补充重检索、分级权限门禁与防篡改审计
def human_review(state: AgentState) -> Dict:
    plan = state.get("rule_plan") or {}
    fs = _ensure_fact_summary(state)
    fo = state["final_output"]
    review_lvl = plan.get("review", "must")
    conf = plan.get("confidence", state.get("answer_confidence", "B"))

    payload = {
        "message": "请执业律师审核以下 AI 生成的法律回答。",
        "citations": [c.model_dump() for c in fo.citations],
        "answer": fo.answer,
        "missing_facts": fo.missing_facts or fs.get("missing_facts", []),
        "clarification_questions": fo.clarification_questions or fs.get("clarification_questions", []),
        "procedural_notes": fo.procedural_notes or fs.get("procedural_notes", []),
        "compliance_info": state.get("compliance_info", {}),
        "validation_notes": state.get("validation_notes", []),
        "rerank_scores": state.get("rerank_scores", []),
        "rule_notes": state.get("rule_notes", []),
        "rule_plan": {
            "mode": plan.get("mode"),
            "confidence": conf,
            "review": review_lvl,
            "reason": plan.get("reason", ""),
            "verify_actions": plan.get("verify_actions", []),
        },
    }
    feedback = state.get("_mock_review_feedback") or interrupt(payload) or {}
    decision = (feedback.get("decision") or "").strip().lower()
    comments = (feedback.get("comments") or "").strip()
    reviewer_name = (feedback.get("reviewer_name") or "值班执业律师").strip()
    license_no = (feedback.get("license_no") or "").strip()
    reviewer_role = (feedback.get("reviewer_role") or "lawyer").strip()
    reviewer_token = (feedback.get("reviewer_token") or "").strip()
    edited_answer = (feedback.get("edited_answer") or "").strip()
    add_citations = feedback.get("add_citations") or []
    remove_citations = feedback.get("remove_citations") or []
    extra_query = (feedback.get("extra_query") or "").strip()
    new_act_date = (feedback.get("act_date") or "").strip()
    new_region = (feedback.get("region") or "").strip()

    rule_notes = list(state.get("rule_notes") or [])
    draft_answer = fo.answer

    # P0 安全加固：若未收到有效审核决策（或 decision 非法），绝不默认 approve 自动放行！
    if decision not in ("approve", "revise", "re_retrieve"):
        return {
            "human_approved": False,
            "review_action": "pending",
            "rule_notes": rule_notes + ["人在环状态：等待执业律师提交明确审核决策（approve / revise / re_retrieve）。"],
        }

    # 1) 应用律师对第一层引用的增删操作
    updated_cites, cite_logs = apply_citation_edits(fo.citations, add_citations, remove_citations)
    if cite_logs:
        rule_notes.extend(cite_logs)

    # 2) 若律师选择「补充检索重跑 (re_retrieve)」：更新查询/时间/地域并路由回 extract_facts
    if decision == "re_retrieve":
        updated_q = f"{state['query']}；{extra_query}" if extra_query else state["query"]
        out_re: Dict = {
            "query": updated_q,
            "human_approved": False,
            "human_feedback": comments,
            "review_action": "re_retrieve",
            "rule_notes": rule_notes,
        }
        if new_act_date:
            out_re["act_date"] = new_act_date
        if new_region:
            out_re["region"] = new_region
        audit_rec = audit_store.record_audit_event(
            thread_id=feedback.get("thread_id", state.get("thread_id", "session")),
            event_type="human_review",
            decision="re_retrieve",
            reviewer_name=reviewer_name,
            license_no=license_no,
            reviewer_role=reviewer_role,
            comments=comments or f"补充检索：{extra_query}",
            confidence=conf,
            review_level=review_lvl,
            citations=[c.model_dump() for c in updated_cites],
            draft_answer=draft_answer,
            final_answer=draft_answer,
        )
        out_re["review_audit"] = audit_rec
        return out_re

    # 3) 若律师选择「通过 (approve)」，先执行分级审核权限与执业证注册表校验（expert/immediate 需资深律师/合伙人）
    if decision == "approve":
        perm_ok, perm_msg = verify_reviewer_permission(
            review_lvl,
            reviewer_role,
            license_no,
            reviewer_token=reviewer_token,
        )
        if not perm_ok:
            rule_notes.append(perm_msg)
            audit_rec = audit_store.record_audit_event(
                thread_id=feedback.get("thread_id", "session"),
                event_type="permission_denied",
                decision="reject_approve",
                reviewer_name=reviewer_name,
                license_no=license_no,
                reviewer_role=reviewer_role,
                comments=perm_msg,
                confidence=conf,
                review_level=review_lvl,
                citations=[c.model_dump() for c in updated_cites],
                draft_answer=draft_answer,
                final_answer=draft_answer,
            )
            return {
                "human_approved": False,
                "human_feedback": f"{comments}（{perm_msg}）".strip(),
                "review_action": "revise",
                "rule_notes": rule_notes,
                "review_audit": audit_rec,
            }

        # 确定终稿内容：若律师直接在线改稿（edited_answer），采用改稿后正文并计算 Diff
        final_ans = edited_answer if edited_answer else fo.answer
        if comments and f"【执业律师审核与补充意见】{comments}" not in final_ans:
            final_ans = f"{final_ans}\n\n【执业律师审核与补充意见】{comments}"

        diff_str = compute_answer_diff(draft_answer, final_ans)
        if diff_str:
            rule_notes.append("律师改稿留痕：已记录 AI 初稿与律师终稿的 Unified Diff。")

        approved_disclaimer = build_compliance_disclaimer(
            is_human_approved=True,
            reviewer_name=reviewer_name,
            license_no=license_no,
            answer_confidence=conf,
        )
        updated_fo = TwoLayerAnswer(
            citations=updated_cites,
            answer=final_ans,
            missing_facts=fo.missing_facts,
            clarification_questions=fo.clarification_questions,
            procedural_notes=fo.procedural_notes,
            disclaimer=approved_disclaimer,
        )
        audit_rec = audit_store.record_audit_event(
            thread_id=feedback.get("thread_id", "session"),
            event_type="human_review",
            decision="approve",
            reviewer_name=reviewer_name,
            license_no=license_no,
            reviewer_role=reviewer_role,
            comments=comments,
            confidence=conf,
            review_level=review_lvl,
            citations=[c.model_dump() for c in updated_cites],
            draft_answer=draft_answer,
            final_answer=final_ans,
        )
        return {
            "ranked": updated_cites,
            "human_approved": True,
            "human_feedback": comments,
            "review_action": "approve",
            "final_output": updated_fo,
            "rule_notes": rule_notes,
            "review_audit": audit_rec,
        }

    # 4) revise：带着律师增删后的 citations 与修改意见回到 synthesize 重生成
    audit_rec = audit_store.record_audit_event(
        thread_id=feedback.get("thread_id", "session"),
        event_type="human_review",
        decision="revise",
        reviewer_name=reviewer_name,
        license_no=license_no,
        reviewer_role=reviewer_role,
        comments=comments,
        confidence=conf,
        review_level=review_lvl,
        citations=[c.model_dump() for c in updated_cites],
        draft_answer=draft_answer,
        final_answer=edited_answer or draft_answer,
    )
    return {
        "ranked": updated_cites,
        "human_approved": False,
        "human_feedback": comments,
        "review_action": "revise",
        "rule_notes": rule_notes,
        "review_audit": audit_rec,
    }
