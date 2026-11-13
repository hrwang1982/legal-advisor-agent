"""知识库载入与 BGE-M3 + BM25 混合检索（P1+P2 可交付升级版）：
1. 支持从外置知识中心（`knowledge_center`）热加载「口语→法言法语」同义词、引致图谱、新旧法对照表，改知识无需改代码；
2. 融合静态法规库（`sample_laws.json`）与动态导入的外规向量库（`kb_store` SQLite）；
3. 实现 Okapi BM25 + BGE-M3 稠密语义向量的双路混合检索（Hybrid Retrieval with RRF / 加权融合），并内置内存缓存防止重复构建；
4. 引致条款图谱 + 条文正文内嵌引致解析（支持 `本法第X条`、`《XX法》第X条` 等多种引致句式）。
"""
from __future__ import annotations

import json
import math
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from app.config import settings
from app.kb import BGEM3Embedder, cn_to_arabic_int, kb_store, normalize_article_id
from app.knowledge import knowledge_center

try:
    from llama_index.core import TextNode
except Exception:  # 无 LlamaIndex 环境下回退为轻量节点，保证模块可被纯逻辑导入
    class _FallbackNode:
        def __init__(self, text: str = "", metadata: dict | None = None, **kwargs):
            self.text = text
            self.metadata = metadata or {}

        def get_content(self) -> str:
            return self.text

    TextNode = _FallbackNode


def _text_node_class():
    """返回可用 TextNode 类（LlamaIndex 存在则用真实类，否则回退轻量节点）。"""
    return TextNode


class RetrievalResult(NamedTuple):
    node: TextNode
    score: float


# ————————————————————————————————————————————————
# P1 升级 1：口语 → 法言法语同义词扩展（由外置 knowledge_center 驱动，保留兼容常量引用）
# ————————————————————————————————————————————————
def get_legal_synonym_map() -> Dict[str, str]:
    return knowledge_center.get_synonym_map()


# 向下兼容旧代码直接导入 LEGAL_SYNONYM_MAP
LEGAL_SYNONYM_MAP: Dict[str, str] = get_legal_synonym_map()


def expand_legal_query(query: str) -> str:
    """将用户口语查询扩展为含标准法言法语的增强查询（实时读取外置同义词表）。"""
    text = query or ""
    syn_map = knowledge_center.get_synonym_map() or LEGAL_SYNONYM_MAP
    additions: List[str] = []
    for colloq, legal_terms in syn_map.items():
        if colloq in text:
            additions.append(legal_terms)
    if not additions:
        return text
    return f"{text} {' '.join(additions)}"


# ————————————————————————————————————————————————
# P1 升级 2：引致条款图谱 + 条文正文内嵌引致解析（由外置 knowledge_center 驱动）
# ————————————————————————————————————————————————
def get_cross_reference_graph() -> Dict[Tuple[str, str], List[Tuple[str, str, str]]]:
    return knowledge_center.get_cross_reference_graph()


CROSS_REFERENCE_GRAPH: Dict[Tuple[str, str], List[Tuple[str, str, str]]] = get_cross_reference_graph()


def _cn_to_int(cn: str) -> Optional[int]:
    return cn_to_arabic_int(cn)


def resolve_canonical_law_name(raw_name: str) -> str:
    """将法规简称或别名解析为标准法规全称。"""
    s = (raw_name or "").strip().strip("《》〈〉")
    aliases = knowledge_center.get_law_aliases()
    if s in aliases:
        return aliases[s]
    short = s.replace("中华人民共和国", "")
    if short in aliases:
        return aliases[short]
    return s


def extract_inline_references(law_name: str, content: str) -> List[Tuple[str, str, str]]:
    """从条文正文中自动解析「本法第X条」或「《XX法》第X条」或「民法典第X条」等内嵌引致关系。"""
    results: List[Tuple[str, str, str]] = []
    text = content or ""
    # 1) 匹配「本法第三十八条」「本法第四十七条」
    for m in re.finditer(r"本法第([零〇一二两三四五六七八九十百千0-9]+)条", text):
        n = _cn_to_int(m.group(1))
        if n:
            art = f"第{n}条"
            results.append((law_name, art, f"条文正文直接引致《{law_name}》{art}"))
    # 2) 匹配「《XX法》第X条」
    for m in re.finditer(r"《([^》]{2,40})》\s*第([零〇一二两三四五六七八九十百千0-9]+)条", text):
        target_law = resolve_canonical_law_name(m.group(1))
        n = _cn_to_int(m.group(2))
        if n and target_law:
            art = f"第{n}条"
            results.append((target_law, art, f"条文正文直接引致《{target_law}》{art}"))
    # 3) 匹配无书名号的「民法典第五百八十四条」
    for m in re.finditer(r"(?<![《〈])民法典第([零〇一二两三四五六七八九十百千0-9]+)条", text):
        n = _cn_to_int(m.group(1))
        if n:
            art = f"第{n}条"
            results.append(("中华人民共和国民法典", art, f"条文正文直接引致《中华人民共和国民法典》{art}"))
    return results


# ————————————————————————————————————————————————
# P1 升级 3：新旧法条文级双向对照表（由外置 knowledge_center 驱动）
# ————————————————————————————————————————————————
ARTICLE_SUCCESSION_LIST = knowledge_center.get_article_succession_list()


def find_counterpart_article(law_name: str, article: str) -> Optional[Dict]:
    """查询某条文的新旧法对照信息（实时读取外置配置）。"""
    succ_list = knowledge_center.get_article_succession_list() or ARTICLE_SUCCESSION_LIST
    norm_art = normalize_article_id(article)
    for old_law, old_art, new_law, new_art, switch_date, diff_note in succ_list:
        if law_name == old_law and norm_art == normalize_article_id(old_art):
            return {
                "role": "historical",
                "counterpart_law": new_law,
                "counterpart_article": normalize_article_id(new_art),
                "switch_date": switch_date,
                "diff_note": diff_note,
            }
        if law_name == new_law and norm_art == normalize_article_id(new_art):
            return {
                "role": "current",
                "counterpart_law": old_law,
                "counterpart_article": normalize_article_id(old_art),
                "switch_date": switch_date,
                "diff_note": diff_note,
            }
    return None


# —————————————————— 中文/英文 轻量分词 ——————————————————
def _tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    tokens: List[str] = re.findall(r"[a-z0-9]+", text)
    cn = re.findall(r"[一-龥]", text)
    # 字 uni-gram + bi-gram，兼顾中文召回
    tokens.extend(cn)
    tokens.extend(cn[i] + cn[i + 1] for i in range(len(cn) - 1))
    return tokens


class _BM25:
    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_count = len(corpus)
        self.avgdl = (sum(len(d) for d in corpus) / self.doc_count) if self.doc_count else 1.0
        df: dict = defaultdict(int)
        for doc in corpus:
            for t in set(doc):
                df[t] += 1
        self.idf = {
            t: math.log(1 + (self.doc_count - n + 0.5) / (n + 0.5))
            for t, n in df.items()
        }

    def score(self, query: List[str], doc: List[str]) -> float:
        freq = defaultdict(int)
        for t in doc:
            freq[t] += 1
        dl = len(doc)
        total = 0.0
        for t in query:
            idf = self.idf.get(t)
            if not idf:
                continue
            f = freq.get(t, 0)
            if f == 0:
                continue
            denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            total += idf * (f * (self.k1 + 1)) / denom
        return total


def _load_rows() -> List[dict]:
    """读取法规原始条目：完全从 MySQL 数据库加载基础法规与外规知识库条文（支持多版本并存隔离）。"""
    merged: Dict[Tuple[str, str, str, str], dict] = {}

    # 1. 从 MySQL 数据库读取 sample_laws 表
    try:
        from app.mysql_store import mysql_store
        db_rows = mysql_store.query("SELECT * FROM sample_laws")
        if db_rows:
            for r in db_rows:
                meta = {}
                if r.get("metadata_json"):
                    try:
                        meta = json.loads(r["metadata_json"])
                    except Exception:
                        pass
                item = dict(meta) if meta else dict(r)
                item["law_name"] = r.get("law_name", item.get("law_name", ""))
                item["article"] = r.get("article", item.get("article", ""))
                item["paragraph"] = r.get("paragraph", item.get("paragraph", ""))
                item["content"] = r.get("content", item.get("content", ""))
                item["hierarchy"] = r.get("hierarchy", item.get("hierarchy", "法律"))
                item["validity"] = r.get("validity", item.get("validity", "现行有效"))
                ver_key = item.get("version_code") or item.get("effective_date") or "v1"
                item.setdefault("version_code", ver_key)
                item.setdefault("is_latest", 1 if item["validity"] == "现行有效" else 0)
                key = (
                    item.get("law_name", ""),
                    ver_key,
                    normalize_article_id(item.get("article", "")),
                    item.get("paragraph", ""),
                )
                merged[key] = item
    except Exception:
        pass

    # 2. 合并动态 RAG 知识库中导入的外规条文（来自 MySQL law_chunks，按 law_name + version_code 隔离）
    try:
        for kb_row in kb_store.load_all_kb_rows():
            ver_key = kb_row.get("version_code") or kb_row.get("effective_date") or "v1"
            key = (
                kb_row.get("law_name", ""),
                ver_key,
                normalize_article_id(kb_row.get("article", "")),
                kb_row.get("paragraph", ""),
            )
            merged[key] = kb_row
    except Exception:
        pass

    return list(merged.values())


def _day_before(d: str) -> str:
    """返回给定日期的前一天（用于推导旧版本的失效日）。失败则原样返回。"""
    from datetime import date, timedelta

    try:
        return (date.fromisoformat(str(d)[:10]) - timedelta(days=1)).isoformat()
    except (ValueError, TypeError):
        return d


# 内存级节点与版本时间线缓存（避免每次请求重复读盘与重复计算向量）
_CACHE_STATE: Dict[str, Any] = {
    "signature": None,
    "nodes": None,
    "timelines": None,
}


def _current_data_signature() -> Tuple[Any, ...]:
    return (kb_store.version_counter,)


def invalidate_cache() -> None:
    """手动清空法规节点与时间线缓存（在导入/删除外规后自动触发）。"""
    _CACHE_STATE["signature"] = None
    _CACHE_STATE["nodes"] = None
    _CACHE_STATE["timelines"] = None


def build_version_timelines() -> dict:
    """从法规库推导每部法律的「版本时间线」，供时间效力定位使用。"""
    sig = _current_data_signature()
    if _CACHE_STATE["signature"] == sig and _CACHE_STATE["timelines"] is not None:
        return _CACHE_STATE["timelines"]

    grouped: dict = defaultdict(dict)
    for r in _load_rows():
        law = r.get("law_name", "")
        if not law:
            continue
        eff = (r.get("effective_date") or "").strip()
        val = (r.get("validity") or "现行有效").strip()
        key = (eff, val)
        if key not in grouped[law]:
            grouped[law][key] = {
                "version": eff or val,
                "effective_from": eff,
                "effective_to": (r.get("effective_to") or "").strip() or None,
                "validity": val,
                "superseded_by": r.get("superseded_by", "") or "",
            }

    timelines: dict = {}
    for law, vers in grouped.items():
        items = sorted(vers.values(), key=lambda v: v.get("effective_from") or "0000-00-00")
        for i, v in enumerate(items):
            declared = v.get("effective_to")
            if v.get("validity") == "现行有效":
                v["effective_to"] = None
            elif declared:
                v["effective_to"] = declared
            elif i + 1 < len(items):
                v["effective_to"] = _day_before(items[i + 1].get("effective_from"))
        timelines[law] = items

    # 第二遍：已废止且失效日未知 → 用被替代法的生效日前一天
    for law, items in timelines.items():
        for v in items:
            if v.get("validity") == "现行有效" or v.get("effective_to") is not None:
                continue
            succ = v.get("superseded_by", "")
            resolved = None
            if succ:
                for other_law, other_items in timelines.items():
                    if other_law == law:
                        continue
                    short = other_law.replace("中华人民共和国", "")
                    if other_law in succ or (short and short in succ):
                        starts = [x["effective_from"] for x in other_items if x.get("effective_from")]
                        if starts:
                            resolved = _day_before(min(starts))
                            break
            v["effective_to"] = resolved if resolved else (v.get("effective_from") or "")

    _CACHE_STATE["timelines"] = timelines
    return timelines


def load_nodes() -> List[TextNode]:
    """读取静态与动态法规库，构造带完整元数据与 BGE-M3 向量的 LlamaIndex 节点（带缓存）。"""
    sig = _current_data_signature()
    if _CACHE_STATE["signature"] == sig and _CACHE_STATE["nodes"] is not None:
        return _CACHE_STATE["nodes"]

    rows = _load_rows()
    embedder = kb_store.embedder

    TextNode = _text_node_class()
    nodes: List[TextNode] = []
    for r in rows:
        text = (
            f"{r.get('law_name', '')} {r.get('article', '')} {r.get('paragraph', '')} "
            f"{r.get('content', '')} {r.get('category', '')}"
        )
        vec = r.get("_embedding")
        if not vec or not isinstance(vec, list):
            vec = embedder._local_semantic_embed(text)
        nodes.append(
            TextNode(
                text=text,
                metadata={
                    "law_name": r.get("law_name", ""),
                    "document_number": r.get("document_number", ""),
                    "hierarchy": r.get("hierarchy", ""),
                    "host_law": r.get("host_law", ""),
                    "host_hierarchy": r.get("host_hierarchy", ""),
                    "host_effective_date": r.get("host_effective_date", ""),
                    "promulgation_date": r.get("promulgation_date", ""),
                    "effective_date": r.get("effective_date", ""),
                    "article_effective_date": r.get("article_effective_date", ""),
                    "effective_to": r.get("effective_to", ""),
                    "validity": r.get("validity", "现行有效"),
                    "superseded_by": r.get("superseded_by", ""),
                    "article": normalize_article_id(r.get("article", "")),
                    "paragraph": r.get("paragraph", ""),
                    "content": r.get("content", ""),
                    "source": r.get("source", ""),
                    "category": r.get("category", ""),
                    "scope": r.get("scope", ""),
                    "norm_type": r.get("norm_type", ""),
                    "partially_invalid": bool(r.get("partially_invalid", False)),
                    "invalid_parts": r.get("invalid_parts", ""),
                    "trial": bool(r.get("trial", False)),
                    "verified": bool(r.get("verified", True)),
                    "note": r.get("note", ""),
                    "version_code": r.get("version_code", "v1"),
                    "version_name": r.get("version_name", ""),
                    "is_latest": int(r.get("is_latest", 1)),
                    "kb_type": r.get("kb_type", ""),
                    "doc_id": r.get("doc_id", ""),
                    "_embedding": vec,
                },
            )
        )

    _CACHE_STATE["signature"] = sig
    _CACHE_STATE["nodes"] = nodes
    return nodes


def build_retriever(nodes: List[TextNode], top_k: int | None = None) -> "_Retriever":
    return _Retriever(nodes, top_k or settings.top_k)


class _Retriever:
    """Okapi BM25 + BGE-M3 稠密语义向量的双路混合检索器。"""

    def __init__(self, nodes: List[TextNode], top_k: int):
        self.nodes = nodes
        self.top_k = top_k
        self._doc_tokens = [_tokenize(n.text) for n in nodes]
        self._bm25 = _BM25(self._doc_tokens)
        self._embedder = kb_store.embedder

    def retrieve(self, query: str) -> List[RetrievalResult]:
        qt = _tokenize(query)
        expanded = expand_legal_query(query)
        has_expansion = expanded != query
        exp_t = _tokenize(expanded) if has_expansion else qt

        # 计算查询的 BGE-M3 向量（若开启 embed_enabled）
        q_vec: Optional[List[float]] = None
        if settings.embed_enabled:
            q_vecs, _ = self._embedder.embed_texts([expanded])
            if q_vecs:
                q_vec = q_vecs[0]

        # 1. 优先尝试调用 Elasticsearch 8.x 原生多字段 BM25 与 kNN 向量联合粗排 (Coarse Ranking)
        try:
            from app.es_store import es_vector_store
            if es_vector_store.ping():
                es_hits = es_vector_store.coarse_rank(
                    query=expanded,
                    query_vector=q_vec,
                    top_k=max(self.top_k * 2, 20),
                )
                if es_hits:
                    hit_map = {}
                    for h in es_hits:
                        k = (
                            h.get("law_name", ""),
                            normalize_article_id(h.get("article", "")),
                        )
                        hit_map[k] = max(hit_map.get(k, 0.0), float(h.get("_score", 1.0)))

                    es_scored: List[RetrievalResult] = []
                    for n in self.nodes:
                        meta = n.metadata or {}
                        k = (
                            meta.get("law_name", ""),
                            normalize_article_id(meta.get("article", "")),
                        )
                        if k in hit_map:
                            es_scored.append(RetrievalResult(n, hit_map[k]))
                    if es_scored:
                        es_scored.sort(key=lambda x: x.score, reverse=True)
                        return es_scored[: self.top_k]
        except Exception:
            pass

        # 2. 回退机制：使用内置内存 BM25 + 本地余弦相似度粗排引擎
        scored: List[RetrievalResult] = []
        for n, dt in zip(self.nodes, self._doc_tokens):
            base_s = self._bm25.score(qt, dt)
            if has_expansion:
                exp_s = self._bm25.score(exp_t, dt)
                bm25_s = 0.7 * base_s + 0.3 * exp_s
            else:
                bm25_s = base_s

            # 融合 BGE-M3 向量余弦相似度
            dense_s = 0.0
            if q_vec is not None:
                doc_vec = n.metadata.get("_embedding")
                if doc_vec and len(doc_vec) == len(q_vec):
                    dense_s = BGEM3Embedder.cosine_similarity(q_vec, doc_vec)

            # 仅当词面有命中或向量语义高度相关（> 0.35）时计入候选，防止无关噪声
            if bm25_s > 0 or dense_s >= 0.35:
                s = bm25_s + (dense_s * 3.5 if settings.embed_enabled else 0.0)
                scored.append(RetrievalResult(n, s))

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[: self.top_k]
