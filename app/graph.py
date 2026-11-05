"""LangGraph 编排（P2 增强版）：
支持三种人在环后路由分支：
1. approve → END（审核通过并固化审计哈希链）
2. revise → synthesize（结合律师增删后的引用与意见重新合成）
3. re_retrieve → extract_facts（结合律师补充的事实/时间/地域/关键词重新检索并重跑规则引擎）

P0 安全加固：即便在无 langgraph 的轻量环境下，_FallbackCompiledGraph 也严格执行人在环中断（pending_review）与恢复（resume），
绝不允许未通过 human_review 即标记为已完成（done）。
"""
from __future__ import annotations

from types import SimpleNamespace

from app.nodes import (
    decompose,
    extract_facts_node,
    human_review,
    rank_hierarchy,
    rerank_node,
    retrieve,
    rewrite,
    rules_node,
    synthesize,
    validate,
)
from app.state import AgentState

try:
    try:
        from langgraph.checkpoint.memory import MemorySaver
    except ImportError:
        from langgraph.checkpoint.memory import InMemorySaver as MemorySaver
    from langgraph.graph import END, START, StateGraph
    _HAS_LANGGRAPH = True
except ImportError:
    _HAS_LANGGRAPH = False
    START = "__start__"
    END = "__end__"
    MemorySaver = None  # type: ignore


def route_after_review(state: AgentState) -> str:
    """人在环后的三路条件路由。"""
    if state.get("human_approved"):
        return END
    action = state.get("review_action", "revise")
    if action == "re_retrieve":
        return "extract_facts"
    return "synthesize"


class _FallbackCompiledGraph:
    """无 langgraph 环境下的状态图执行器（严格支持 interrupt 挂起待审与 Command(resume=...) 恢复）。"""

    def __init__(self):
        self._states = {}
        self._pending_interrupts = {}

    def _build_interrupt_payload(self, state: dict) -> dict:
        plan = state.get("rule_plan") or {}
        fs = state.get("fact_summary") or {}
        fo = state.get("final_output")
        conf = plan.get("confidence", state.get("answer_confidence", "B"))
        review_lvl = plan.get("review", "must")
        traces = [t.model_dump() for t in (fo.sentence_traces if fo and getattr(fo, "sentence_traces", None) else [])]
        if not traces and state.get("sentence_traces"):
            traces = state.get("sentence_traces")

        sq = (fo.structured_queries if fo and getattr(fo, "structured_queries", None) else None) or state.get("structured_queries", {})

        return {
            "message": "请执业律师审核以下 AI 生成的法律回答。",
            "citations": [c.model_dump() for c in (fo.citations if fo else [])],
            "answer": fo.answer if fo else "",
            "sentence_traces": traces,
            "structured_queries": sq,
            "missing_facts": (fo.missing_facts if fo else None) or fs.get("missing_facts", []),
            "clarification_questions": (fo.clarification_questions if fo else None) or fs.get("clarification_questions", []),
            "procedural_notes": (fo.procedural_notes if fo else None) or fs.get("procedural_notes", []),
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

    def invoke(self, payload, config=None):
        tid = ((config or {}).get("configurable") or {}).get("thread_id", "default")
        state = dict(self._states.get(tid, {}))
        state["thread_id"] = tid

        # 检查是否为 Command(resume=...) 恢复审核
        resume_data = getattr(payload, "resume", None)
        if resume_data is None and isinstance(payload, dict) and "_resume" in payload:
            resume_data = payload["_resume"]

        if resume_data is not None:
            state["_mock_review_feedback"] = resume_data
            rev_out = human_review(state)
            state.pop("_mock_review_feedback", None)
            state.update(rev_out)
            next_step = route_after_review(state)
            if next_step == END:
                self._pending_interrupts.pop(tid, None)
                self._states[tid] = state
                return state
            if next_step == "extract_facts":
                for fn in (
                    extract_facts_node,
                    decompose,
                    rewrite,
                    retrieve,
                    validate,
                    rerank_node,
                    rank_hierarchy,
                    rules_node,
                    synthesize,
                ):
                    state.update(fn(state))
            elif next_step == "synthesize":
                state.update(synthesize(state))
            self._pending_interrupts[tid] = self._build_interrupt_payload(state)
            self._states[tid] = state
            return state

        if isinstance(payload, dict):
            state.update(payload)
        state["human_approved"] = False
        for fn in (
            extract_facts_node,
            decompose,
            rewrite,
            retrieve,
            validate,
            rerank_node,
            rank_hierarchy,
            rules_node,
            synthesize,
        ):
            state.update(fn(state))
        self._pending_interrupts[tid] = self._build_interrupt_payload(state)
        self._states[tid] = state
        return state

    def get_state(self, config):
        tid = ((config or {}).get("configurable") or {}).get("thread_id", "default")
        st = dict(self._states.get(tid, {}))
        intr = self._pending_interrupts.get(tid)
        tasks = [SimpleNamespace(interrupts=[SimpleNamespace(value=intr)])] if intr else []
        return SimpleNamespace(values=st, tasks=tasks)


def build_graph(checkpointer=None):
    if not _HAS_LANGGRAPH:
        return _FallbackCompiledGraph()

    g = StateGraph(AgentState)

    g.add_node("extract_facts", extract_facts_node)
    g.add_node("decompose", decompose)
    g.add_node("rewrite", rewrite)
    g.add_node("retrieve", retrieve)
    g.add_node("validate", validate)
    g.add_node("rerank", rerank_node)
    g.add_node("rank_hierarchy", rank_hierarchy)
    g.add_node("rules", rules_node)
    g.add_node("synthesize", synthesize)
    g.add_node("human_review", human_review)

    g.add_edge(START, "extract_facts")
    g.add_edge("extract_facts", "decompose")
    g.add_edge("decompose", "rewrite")
    g.add_edge("rewrite", "retrieve")
    g.add_edge("retrieve", "validate")
    g.add_edge("validate", "rerank")
    g.add_edge("rerank", "rank_hierarchy")
    g.add_edge("rank_hierarchy", "rules")
    g.add_edge("rules", "synthesize")
    g.add_edge("synthesize", "human_review")

    g.add_conditional_edges(
        "human_review",
        route_after_review,
        {
            "extract_facts": "extract_facts",
            "synthesize": "synthesize",
            END: END,
        },
    )

    return g.compile(checkpointer=checkpointer or MemorySaver())


# 默认单例
graph = build_graph()
