"""数据模型：引用（Citation）、两层回答（TwoLayerAnswer）、位阶排序。"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from pydantic import BaseModel, Field
except ImportError:  # 无 pydantic 环境下回退轻量模型，确保全模块可零依赖单测
    class _FieldDefault:
        def __init__(self, default: Any = None, default_factory=None):
            self.default = default
            self.default_factory = default_factory

    def Field(default: Any = None, default_factory=None, **kwargs) -> Any:  # type: ignore
        return _FieldDefault(default=default, default_factory=default_factory)

    class BaseModel:  # type: ignore
        def __init__(self, **kwargs: Any):
            for cls in reversed(self.__class__.__mro__):
                for k, v in cls.__dict__.items():
                    if k.startswith("_") or callable(v):
                        continue
                    if isinstance(v, _FieldDefault):
                        val = v.default_factory() if v.default_factory is not None else v.default
                        setattr(self, k, val)
                    else:
                        setattr(self, k, v)
            for k, v in kwargs.items():
                setattr(self, k, v)

        def model_dump(self) -> Dict[str, Any]:
            out: Dict[str, Any] = {}
            for k, v in self.__dict__.items():
                if isinstance(v, BaseModel):
                    out[k] = v.model_dump()
                elif isinstance(v, list):
                    out[k] = [x.model_dump() if isinstance(x, BaseModel) else x for x in v]
                else:
                    out[k] = v
            return out

from app.rules import BASE_RANK, rank_of

# 静态位阶表（不含司法解释）：司法解释位阶随被解释法律浮动，见 rules.rank_of
HIERARCHY_ORDER = dict(BASE_RANK)


class Citation(BaseModel):
    """一条法规引用。所有字段来自知识库元数据，保证可追溯、可校验。"""
    law_name: str = Field(default="", description="法规全称，如《中华人民共和国民法典》")
    document_number: str = Field(default="", description="发文字号")
    hierarchy: str = Field(default="", description="位阶：宪法/法律/行政法规/地方性法规/部门规章/地方政府规章/司法解释/规范性文件")
    host_law: str = Field(default="", description="司法解释所解释的法律全称（位阶依附对象）")
    host_hierarchy: str = Field(default="", description="被解释法律的位阶，用于计算司法解释位阶")
    host_effective_date: str = Field(default="", description="被解释法律的施行日期（用于判断司法解释时间效力边界）")
    promulgation_date: str = Field(default="", description="颁布日期")
    effective_date: str = Field(default="", description="法律/修正案施行日期")
    article_effective_date: str = Field(default="", description="本条文首次施行日期（未改动条文优先按此判定时间效力）")
    effective_to: str = Field(default="", description="失效日期；为空表示未失效")
    validity: str = Field(
        default="现行有效",
        description="有效性：现行有效/已废止/已被修订/部分失效/尚未施行",
    )
    superseded_by: str = Field(default="", description="若已失效，被何条文替代")
    article: str = Field(default="", description="条号，如 第577条")
    paragraph: str = Field(default="", description="款/项定位，如 第1款 / 第（一）项")
    content: str = Field(default="", description="条文正文（含但书与例外，不得截断）")
    source: str = Field(default="", description="权威来源/制定机关，便于溯源核验")
    category: str = Field(default="", description="业务分类，便于检索与展示")
    scope: str = Field(default="", description="规范性质：特别/一般，用于「特别法优于一般法」判定")
    norm_type: str = Field(
        default="",
        description="规范类型：substantive(实体法)/procedural(程序法)/judicial_interpretation(司法解释)/administrative_penalty(行政处罚)",
    )
    partially_invalid: bool = Field(default=False, description="是否部分条款失效")
    invalid_parts: str = Field(default="", description="已失效的具体款/项，如 第2款")
    trial: bool = Field(default=False, description="是否试行/暂行规范（效力稳定性较低）")
    verified: bool = Field(default=True, description="条文内容是否已经权威来源核验；false 不得进入自动对外通道")
    note: str = Field(default="", description="数据侧备注（如摘要提示、适用限制）")
    score: float = Field(default=0.0, description="BM25 检索得分")
    rerank_score: float = Field(default=0.0, description="重排相关性得分（0~1）")

    # ——— 由 rules_node 回填的时间/地域适用性结论 ———
    applicable_at: bool = Field(default=True, description="在行为发生日与地域下是否适用")
    applicable_note: str = Field(default="", description="时间/地域适用性说明（如「适用行为时旧法」）")

    def rank(self) -> float:
        """位阶数值（越小越高）。司法解释按被解释法律位阶 +0.5 计算。"""
        return rank_of(self.hierarchy, self.host_hierarchy)

    def brief(self) -> str:
        loc = f"{self.article}{self.paragraph}" if self.paragraph else self.article
        mark = "" if self.verified else "［待核验］"
        return f"{self.law_name}{loc}（{self.hierarchy}·{self.validity}）{mark}"

    def applicable_hint(self) -> str:
        """输出给律师/LLM 的适用性提示，避免旧法被当作现行法使用。"""
        if not self.applicable_note:
            return ""
        hint = self.applicable_note
        if self.superseded_by:
            hint += f"；现替代依据：{self.superseded_by}"
        if self.partially_invalid and self.invalid_parts:
            hint += f"；注意该条 {self.invalid_parts} 已失效，适用时须排除"
        return hint


class SentenceTrace(BaseModel):
    """回答中单个句子/观点的细粒度法条溯源对齐信息（源自 ks-lma-chat 工业级 Trace 思想）。"""
    sentence_idx: int = Field(default=0, description="回答中句子索引序号")
    sentence_text: str = Field(default="", description="回答中的单句或观点内容")
    matched_law: str = Field(default="", description="所依据的法规全称")
    matched_article: str = Field(default="", description="所依据的具体条号")
    confidence_score: float = Field(default=0.0, description="溯源匹配置信度/相似度（0~1）")
    highlight_snippet: str = Field(default="", description="法条原文中对应的支撑高亮片段")
    trace_source: str = Field(default="ngram_overlap", description="溯源算法来源，如 ngram_overlap / direct_citation")


class TwoLayerAnswer(BaseModel):
    """两层结构回答：
    第一层 = citations（引用的法规，真实可追溯）
    第二层 = answer（基于引用法规给出的结构化分析）"""
    citations: List[Citation] = Field(default_factory=list, description="第一层：引用的法规")
    answer: str = Field(default="", description="第二层：基于引用法规的回答")
    sentence_traces: List[SentenceTrace] = Field(default_factory=list, description="细粒度句级溯源对照清单（句子对齐法条原文）")
    structured_queries: Dict[str, List[str]] = Field(default_factory=dict, description="三维专业法律检索式（最精准/找例外/查后果）")
    missing_facts: List[str] = Field(default_factory=list, description="待核实的关键事实要素（小前提清单）")
    clarification_questions: List[str] = Field(default_factory=list, description="建议向当事人追问的澄清问题")
    procedural_notes: List[str] = Field(default_factory=list, description="诉讼/仲裁时效与程序前置提示")
    suggested_citations: List[str] = Field(default_factory=list, description="大模型建议法务/律师核实引证的补充条文（待采纳候选池）")
    disclaimer: str = Field(
        default="本回答由 AI 生成，不构成正式法律意见，须经执业律师审核。",
        description="免责声明",
    )
