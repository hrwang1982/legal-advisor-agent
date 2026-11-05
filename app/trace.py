"""工业级法律知识溯源与文本对齐引擎（Trace Engine）：
源自 ks-lma-chat（skbllmqa）的细粒度溯源机制，将第二层大模型生成的分析回答切分为单句/论断，
利用多尺度 N-gram 滑窗、法言法语关键词重合度与法条显式锚定算法，
实现「句子 → 法条条文 → 法条高亮片段」的双向精准绑定。
"""
from __future__ import annotations

import re
from typing import List, Optional, Sequence, Set, Tuple

from app.models import Citation, SentenceTrace


def split_sentences(text: str) -> List[str]:
    """将文本切分为有独立语义的句子列表（过滤空行与纯标点行）。"""
    if not text:
        return []
    # 按照换行、句号、分号、感叹号、问号切分，但保留序号标题行如 1. / (1) / 【xxx】
    lines = text.splitlines()
    sentences: List[str] = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # 若是明显的标题或结构标记行，整体作为一个分析单元
        if re.match(r"^【.+?】|^第[一二三四五六七八九十]+部分|^[0-9]+[、.．]", line):
            # 去除前缀标题后若有长内容则切分，否则作为独立单元
            sub_sents = re.split(r"(?<=[。；;！？!?])\s*", line)
            for s in sub_sents:
                s = s.strip()
                if len(s) >= 4:
                    sentences.append(s)
            continue

        parts = re.split(r"(?<=[。；;！？!?])\s*", line)
        for p in parts:
            p = p.strip()
            # 过滤过短无意义的标点片段
            if len(p) >= 4:
                sentences.append(p)
    return sentences


def _clean_text_for_matching(text: str) -> str:
    """去除无实义标点与空白，保留汉字、英文字母与数字。"""
    return re.sub(r"[^\w\u4e00-\u9fa5]", "", text.lower())


def _extract_char_ngrams(text: str, n: int) -> Set[str]:
    """提取字符级 n-gram。"""
    cleaned = _clean_text_for_matching(text)
    if len(cleaned) < n:
        return {cleaned} if cleaned else set()
    return {cleaned[i : i + n] for i in range(len(cleaned) - n + 1)}


def compute_ngram_similarity(
    s1: str,
    s2: str,
    weights: Sequence[float] = (0.2, 0.3, 0.5),
) -> float:
    """计算多尺度 N-gram (2-gram, 3-gram, 4-gram) 综合重叠度（Jaccard 变体）。"""
    c1 = _clean_text_for_matching(s1)
    c2 = _clean_text_for_matching(s2)
    if not c1 or not c2:
        return 0.0
    if c1 in c2 or c2 in c1:
        # 存在直接子串包含
        return 0.85 + 0.15 * (min(len(c1), len(c2)) / max(len(c1), len(c2)))

    scores = []
    ns = [2, 3, 4]
    for n in ns:
        ng1 = _extract_char_ngrams(c1, n)
        ng2 = _extract_char_ngrams(c2, n)
        if not ng1 or not ng2:
            continue
        inter = len(ng1 & ng2)
        union = len(ng1 | ng2)
        scores.append(inter / union if union > 0 else 0.0)

    if not scores:
        return 0.0
    # 加权平均
    w_sum = sum(weights[: len(scores)])
    if w_sum <= 0:
        return 0.0
    return sum(s * w for s, w in zip(scores, weights)) / w_sum


def extract_best_matching_snippet(sentence: str, full_content: str) -> Tuple[str, float]:
    """从法条正文中找出与句子语义最贴近的具体子句/段落作为高亮支撑。"""
    if not full_content:
        return "", 0.0
    # 将法条切分为自然款项/分句
    clauses = re.split(r"[。；;\n]+", full_content)
    clauses = [c.strip() for c in clauses if len(c.strip()) >= 4]
    if not clauses:
        return full_content[:80], compute_ngram_similarity(sentence, full_content)

    best_clause = clauses[0]
    best_score = -1.0
    for cl in clauses:
        score = compute_ngram_similarity(sentence, cl)
        if score > best_score:
            best_score = score
            best_clause = cl

    return best_clause, best_score


def align_answer_to_citations(
    answer: str,
    citations: List[Citation],
    ngram_threshold: float = 0.08,
) -> List[SentenceTrace]:
    """将大模型生成的回答逐句对齐到第一层法条引用上（融合 ks-lma-chat Trace 机制）。"""
    if not answer or not citations:
        return []

    sentences = split_sentences(answer)
    traces: List[SentenceTrace] = []

    for idx, sent in enumerate(sentences):
        best_cite: Optional[Citation] = None
        best_score = 0.0
        best_snippet = ""
        trace_source = "ngram_overlap"

        # 1. 优先检查显式条号/法律名称引证（硬锚定）
        for c in citations:
            matched_explicit = False
            # 条号如 "第577条" 或 "577条"
            if c.article and (c.article in sent or c.article.replace("第", "") in sent):
                matched_explicit = True
            # 法规简称或全称
            if c.law_name and (c.law_name in sent or c.law_name.replace("中华人民共和国", "") in sent):
                matched_explicit = True

            snippet, score = extract_best_matching_snippet(sent, c.content)
            if matched_explicit:
                # 显式命中有加权奖励
                score = min(1.0, score + 0.35)
                trace_source = "explicit_anchor"

            if score > best_score:
                best_score = score
                best_cite = c
                best_snippet = snippet

        # 2. 如果未显式命中，则通过 N-Gram 检索最匹配的法条
        if not best_cite:
            for c in citations:
                snippet, score = extract_best_matching_snippet(sent, c.content)
                if score > best_score:
                    best_score = score
                    best_cite = c
                    best_snippet = snippet
                    trace_source = "ngram_overlap"

        # 3. 如果最高匹配度达到阈值，则生成溯源条目
        if best_cite and best_score >= ngram_threshold:
            traces.append(
                SentenceTrace(
                    sentence_idx=idx + 1,
                    sentence_text=sent,
                    matched_law=best_cite.law_name,
                    matched_article=best_cite.article,
                    confidence_score=round(float(best_score), 4),
                    highlight_snippet=best_snippet,
                    trace_source=trace_source,
                )
            )

    return traces

