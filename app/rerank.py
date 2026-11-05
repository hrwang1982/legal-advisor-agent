"""重排（Rerank，P1+P2 可交付升级版）：
1. 支持「整句查询 + 拆解子问题（sub_queries）+ 法言法语扩展词 + BGE-M3 向量语义余弦相似度」的多粒度 Max-Pooling 相关性精排，
   防止复合问题中的次要争点被整句稀释；
2. 支持相关性阈值截断（filter_by_relevance），防止无关的上位法混入第一层引用并压过相关的司法解释；
3. 独立于 LangGraph，纯函数，零第三方依赖可离线单测。
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from app.config import settings
from app.ingest import _tokenize, expand_legal_query
from app.kb import BGEM3Embedder, kb_store
from app.llm import rerank_citations_llm

if TYPE_CHECKING:
    from app.models import Citation

# 最低相关性阈值：低于该分值且非引致补齐的噪声条文将被过滤（至少保留 min_keep 条）
MIN_RELEVANCE_SCORE = 0.018


def _overlap(query_tokens: List[str], content_tokens: List[str]) -> float:
    """Jaccard 重叠度：交集 / 并集，范围 0~1。"""
    if not query_tokens:
        return 0.0
    qset, cset = set(query_tokens), set(content_tokens)
    if not cset:
        return 0.0
    inter = len(qset & cset)
    union = len(qset | cset)
    return inter / union if union else 0.0


def rerank(
    query: str,
    citations: List[Citation],
    sub_queries: Optional[List[str]] = None,
) -> List[Citation]:
    """对候选条文重排，返回按相关性降序的新列表（原地写入 rerank_score）。

    P1/P2 增强：
    1) 对整句查询、每个拆解子问题及法言法语扩展分别计算词面重叠度并取最大值；
    2) 融合 BGE-M3 语义特征余弦相似度（解决同义不同词的语义精排）；
    3) 真实大模型模式下，将 sub_queries 一并传入 LLM 精排打分，调用失败时回退至本地词面+向量混合分。
    """
    q_tokens = _tokenize(query)
    exp_q = expand_legal_query(query)
    exp_tokens = _tokenize(exp_q)
    sub_expanded_list = [expand_legal_query(sq) for sq in (sub_queries or []) if sq]
    sub_token_list = [_tokenize(sq_exp) for sq_exp in sub_expanded_list]

    embedder = kb_store.embedder
    q_vec = embedder._local_semantic_embed(exp_q)
    sub_vecs = [embedder._local_semantic_embed(sq_exp) for sq_exp in sub_expanded_list]

    # 真实模式：批量 LLM 相关性打分（1~10），失败则返回 None 回退到词面+BGE-M3分
    llm_scores: Optional[List[float]] = None
    if not settings.mock:
        try:
            llm_scores = rerank_citations_llm(query, citations, sub_queries=sub_queries)
        except Exception:
            llm_scores = None

    scored: List[Citation] = []
    for i, c in enumerate(citations):
        doc_text = f"{c.content} {c.article} {c.law_name} {getattr(c, 'category', '')}"
        doc_tokens = _tokenize(doc_text)
        base_kw = _overlap(q_tokens, doc_tokens)
        exp_kw = _overlap(exp_tokens, doc_tokens) if exp_tokens != q_tokens else base_kw
        sub_kw = max((_overlap(st, doc_tokens) for st in sub_token_list), default=0.0)
        kw = max(base_kw, 0.9 * exp_kw, 0.92 * sub_kw)

        # 计算 BGE-M3 语义相似度（整句与子争点 Max-Pooling）
        doc_vec = embedder._local_semantic_embed(doc_text)
        sem_sim = BGEM3Embedder.cosine_similarity(q_vec, doc_vec)
        if sub_vecs:
            sub_sem = max((BGEM3Embedder.cosine_similarity(sv, doc_vec) for sv in sub_vecs), default=0.0)
            sem_sim = max(sem_sim, 0.95 * sub_sem)

        # 融合词面重叠与语义相似度（保持 0~1 区间，且确保不相关性条文仍低于阈值）
        local_rel = 0.75 * kw + (0.25 * sem_sim if kw > 0 or sem_sim >= 0.35 else 0.0)

        if llm_scores is not None and i < len(llm_scores):
            rel = 0.4 * local_rel + 0.6 * (llm_scores[i] / 10.0)
        else:
            rel = local_rel
        c.rerank_score = round(min(1.0, max(0.0, rel)), 4)
        scored.append(c)

    scored.sort(key=lambda c: c.rerank_score, reverse=True)
    return scored


def filter_by_relevance(
    citations: List[Citation],
    min_score: float = MIN_RELEVANCE_SCORE,
    min_keep: int = 2,
) -> List[Citation]:
    """过滤低于相关性阈值的噪声条文（引致关联补齐条文及新旧法回填条文豁免过滤）。"""
    if len(citations) <= min_keep:
        return list(citations)
    kept = [
        c
        for c in citations
        if c.rerank_score >= min_score
        or "［引致关联" in (getattr(c, "note", "") or "")
        or "［新旧法对照" in (getattr(c, "note", "") or "")
    ]
    if len(kept) < min_keep:
        return list(citations[:min_keep])
    return kept
