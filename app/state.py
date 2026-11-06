"""LangGraph 共享状态定义（P0 + P1 + P2 完整版）。"""
from __future__ import annotations

from typing import Annotated, List, Optional, TypedDict

try:
    from langgraph.graph import add_messages
except ImportError:
    def add_messages(left, right):  # type: ignore
        return (left or []) + (right or [])

from app.models import Citation, TwoLayerAnswer


class AgentState(TypedDict, total=False):
    thread_id: str
    query: str
    # 多轮对话上下文历史（[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]）
    history: List[dict]
    # 知识库检索范围（all / none / 不使用知识库 / 自定义知识库或文档路径）
    kb_scope: str
    # 当前问答指定模型与深度思考开关
    model: str
    deep_thinking: bool
    # 脱敏后的查询文本（用于外发 LLM 与日志留存）
    masked_query: str
    # 合规检查摘要（PII 脱敏项、律所利益冲突预警）
    compliance_info: dict
    # 法律事实发生日（YYYY-MM-DD），用于时间效力定位
    act_date: str
    # 持续性法律事实结束日（YYYY-MM-DD，可选）
    act_end_date: str
    # 案件所属省级地域（如 上海/北京，可选）
    region: str
    # 建议向当事人追问的澄清问题
    # 结构化三维检索式（精准式 / 找例外 / 查后果，源自 ks-lma-chat 检索理念）
    structured_queries: dict
    # 细粒度句级溯源对照（回答各句对齐法条原文）
    sentence_traces: list
    # 结构化事实摘要（含 act_date_source / missing_facts / procedural_notes / clarification_questions）
    fact_summary: dict
    # 问题拆解得到的子问题
    decomposed: List[str]
    # 问题改写得到的检索 query（多路）
    rewritten_queries: List[str]
    # 检索召回的原始引用
    retrieved: List[Citation]
    # 有效性校验说明（被打标的失效/待核验条文）
    validation_notes: List[str]
    # 有效性过滤后的引用
    validated: List[Citation]
    # 重排后的引用（含 rerank_score）
    reranked: List[Citation]
    # 重排打分说明
    rerank_scores: List[str]
    # 位阶排序后的引用
    ranked: List[Citation]
    # 规则判定说明（时间效力 / 地域效力 / 位阶裁决 / 引用闭包核验 / 合规预警）
    rule_notes: List[str]
    # 兜底策略（fallback_plan 产出的可序列化 dict）
    rule_plan: dict
    # 位阶裁决结论（precedence 排序后的法规名 → 裁决理由），供提示词与前端使用
    precedence: List[dict]
    # 本次回答的置信度等级 A/B/C/D
    answer_confidence: str
    # 两层结构最终回答
    final_output: Optional[TwoLayerAnswer]
    # 人在环审核结果与路由动作（approve / revise / re_retrieve）
    human_approved: bool
    human_feedback: str
    review_action: str
    review_audit: dict
    # 聊天消息（预留，便于扩展）
    messages: Annotated[list, add_messages]
