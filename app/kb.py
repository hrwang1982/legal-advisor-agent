"""RAG 法规知识库管理与向量解析引擎（Knowledge Base & BGE-M3 Vector Engine）：
1. 支持动态创建知识库集合、导入外规文档（支持 JSON 条文数组、Markdown/纯文本自动按「第X条」切分解析并提取元数据）；
2. 支持 BAAI/bge-m3 向量解析与混合检索：
   - 在线模式：对接任意 OpenAI `/v1/embeddings` 兼容接口（如本地 Ollama/vLLM/Xinference 部署的 `BAAI/bge-m3`、SiliconFlow 或阿里百炼千问 `text-embedding-v3`）；
   - 离线/免密钥回退模式：内置基于法律字符 n-gram 与法言法语语义簇的确定性特征哈希向量器（BGE-M3 兼容维度/余弦相似度接口），确保无外部 GPU 服务时也能完成向量构建、持久化与混合检索闭环；
3. 基于 SQLite 持久化存储知识库文档、解析后的法规条文元数据与稠密向量（Dense Embeddings），与静态 `sample_laws.json` 无缝融合。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from app.config import settings
from app.knowledge import knowledge_center

# 中文数字转阿拉伯数字（用于解析外部法规文档中的“第一条”“第五百七十七条”）
_CN_NUM = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CN_UNIT = {"十": 10, "百": 100, "千": 1000}


def cn_to_arabic_int(cn: str) -> Optional[int]:
    """将中文数字或阿拉伯数字字符串转为整数（如 '五百七十七' -> 577）。"""
    s = (cn or "").strip()
    if not s:
        return None
    if s.isdigit():
        return int(s)
    total = 0
    current = 0
    for ch in s:
        if ch in _CN_NUM:
            current = _CN_NUM[ch]
        elif ch in _CN_UNIT:
            unit = _CN_UNIT[ch]
            if current == 0:
                current = 1
            total += current * unit
            current = 0
        else:
            return None
    return total + current


def normalize_article_id(art: str) -> str:
    """将任意条号写法统一归一化为 '第X条'（阿拉伯数字），如 '第三十八条' -> '第38条'。"""
    s = (art or "").strip()
    m = re.search(r"第\s*([零〇一二两三四五六七八九十百千0-9]+)\s*条", s)
    if not m:
        return s
    n = cn_to_arabic_int(m.group(1))
    return f"第{n}条" if n is not None else s


class BGEM3Embedder:
    """BGE-M3 向量解析器（支持远程 OpenAI 兼容 `/v1/embeddings` 接口 + 本地确定性语义特征向量回退）。"""

    # 内置法律核心语义簇（用于在无外部 Embedding API 时增强跨词汇语义对齐能力）
    _SEMANTIC_CLUSTERS = [
        ("劳动解除赔偿", ("辞退", "开除", "炒鱿鱼", "解雇", "解除劳动合同", "违法解除", "赔偿金", "经济补偿", "第87条", "第47条", "第38条")),
        ("欠薪与报酬", ("欠薪", "拖欠工资", "不发工资", "克扣", "未及时足额支付劳动报酬", "工资支付")),
        ("未签书面合同", ("没签合同", "未签劳动合同", "未订立书面劳动合同", "二倍工资", "双倍工资", "第82条", "第10条")),
        ("消费欺诈三倍", ("假货", "假一赔三", "退一赔三", "欺诈", "虚假宣传", "消费者", "增加赔偿", "三倍", "第55条")),
        ("格式条款无效", ("霸王条款", "概不负责", "格式条款", "减轻或者免除", "排除或者限制", "无效")),
        ("合同违约与调减", ("违约", "不履行合同", "继续履行", "赔偿损失", "违约金过高", "适当减少", "百分之三十", "第577条", "第584条", "第65条")),
        ("股东出资与加速", ("认缴", "五年内缴足", "股东不交钱", "不能清偿到期债务", "提前缴纳出资", "出资加速到期", "第47条", "第54条", "第26条")),
        ("职务侵权与交通", ("员工撞人", "送货途中", "执行工作任务", "交通事故", "机动车", "侵权", "追偿", "第1191条", "第76条")),
        ("行政首违不罚", ("初犯", "轻微违法", "首违不罚", "初次违法", "及时改正", "不予行政处罚", "第33条")),
        ("程序与时效", ("诉讼时效", "仲裁时效", "三年", "一年", "仲裁前置", "行政复议", "第188条", "第27条")),
    ]

    def __init__(self, dim: int = 128):
        self.local_dim = dim

    def embed_texts(self, texts: List[str]) -> Tuple[List[List[float]], str]:
        """批量生成文本向量。返回 (vectors, backend_model_name)。"""
        if not texts:
            return [], "none"

        # 若非 MOCK 且配置了 Embedding/LLM API Key 或本地接口，优先调用 OpenAI 兼容 /v1/embeddings（如 BAAI/bge-m3）
        api_base = (settings.embed_api_base or settings.llm_api_base or "").strip().rstrip("/")
        api_key = (settings.embed_api_key or settings.llm_api_key or "").strip()
        model_name = (settings.embed_model or "BAAI/bge-m3").strip()

        if not settings.mock and api_base and (api_key or "localhost" in api_base or "127.0.0.1" in api_base):
            try:
                vecs = self._call_remote_embeddings(texts, api_base, api_key, model_name)
                if vecs and len(vecs) == len(texts):
                    return vecs, f"remote:{model_name}"
            except Exception:
                pass

        # 本地确定性 BGE-M3 兼容语义特征哈希向量（支持中文字符 uni/bi/tri-gram + 法律同义词扩展 + 语义簇对齐）
        vecs = [self._local_semantic_embed(t) for t in texts]
        return vecs, f"local-bge-m3-compat({self.local_dim}d)"

    def _call_remote_embeddings(
        self,
        texts: List[str],
        api_base: str,
        api_key: str,
        model: str,
    ) -> List[List[float]]:
        url = f"{api_base}/embeddings"
        payload = json.dumps({"model": model, "input": texts}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=settings.llm_timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        data_items = sorted(body.get("data", []), key=lambda x: x.get("index", 0))
        out: List[List[float]] = []
        for item in data_items:
            vec = [float(v) for v in item.get("embedding", [])]
            out.append(self._l2_normalize(vec))
        return out

    def _local_semantic_embed(self, text: str) -> List[float]:
        """构建 L2 归一化的稠密语义特征向量（融合法言法语同义词与法律概念簇特征）。"""
        raw = (text or "").lower()
        # 融合外置同义词扩展，使口语查询与法典条文在向量空间高维对齐
        syn_map = knowledge_center.get_synonym_map()
        expanded_parts = [raw]
        for colloq, legal_terms in syn_map.items():
            if colloq in raw:
                expanded_parts.append(legal_terms)
        full_text = " ".join(expanded_parts)

        vec = [0.0] * self.local_dim

        # 1) 前 16 维：法律专业语义簇激活通道（显式语义对齐）
        for idx, (_, kw_group) in enumerate(self._SEMANTIC_CLUSTERS):
            if idx >= 16:
                break
            hits = sum(1 for kw in kw_group if kw in full_text)
            if hits > 0:
                vec[idx] += 2.2 * math.log1p(hits)

        # 2) 其余维度：基于 SHA-256 特征哈希的字符 uni/bi/tri-gram 稠密投影
        cn_chars = re.findall(r"[一-龥]", full_text)
        alnum_words = re.findall(r"[a-z0-9]+", full_text)

        features: List[Tuple[str, float]] = []
        for c in cn_chars:
            features.append((c, 1.0))
        for i in range(len(cn_chars) - 1):
            features.append((cn_chars[i] + cn_chars[i + 1], 1.6))
        for i in range(len(cn_chars) - 2):
            features.append((cn_chars[i] + cn_chars[i + 1] + cn_chars[i + 2], 1.8))
        for w in alnum_words:
            features.append((f"#{w}", 1.5))

        span = self.local_dim - 16
        for feat, weight in features:
            h = int(hashlib.md5(feat.encode("utf-8")).hexdigest()[:8], 16)
            pos = 16 + (h % span)
            sign = 1.0 if ((h >> 8) & 1) == 0 else -1.0
            vec[pos] += sign * weight

        return self._l2_normalize(vec)

    @staticmethod
    def _l2_normalize(vec: List[float]) -> List[float]:
        norm = math.sqrt(sum(v * v for v in vec))
        if norm <= 1e-12:
            return vec
        return [round(v / norm, 6) for v in vec]

    @staticmethod
    def cosine_similarity(v1: List[float], v2: List[float]) -> float:
        if not v1 or not v2 or len(v1) != len(v2):
            return 0.0
        dot = sum(a * b for a, b in zip(v1, v2))
        return max(0.0, min(1.0, float(dot)))


def parse_external_regulation_text(
    raw_text: str,
    law_name: str,
    hierarchy: str = "法律",
    document_number: str = "",
    promulgation_date: str = "",
    effective_date: str = "",
    effective_to: str = "",
    validity: str = "现行有效",
    source: str = "全国人民代表大会常务委员会",
    category: str = "民事",
    scope: str = "一般",
    host_law: str = "",
    host_hierarchy: str = "",
    host_effective_date: str = "",
    verified: bool = True,
) -> List[Dict[str, Any]]:
    """智能解析外部法规文档（支持 JSON 格式或整篇法规纯文本/Markdown 自动按「第X条」切分入库）。"""
    text = (raw_text or "").strip()
    if not text:
        return []

    # 1) 若输入本身是 JSON 数组或包含 laws 的 JSON 对象，直接规范化字段后返回
    if text.startswith("[") or text.startswith("{"):
        try:
            parsed = json.loads(text)
            rows = parsed if isinstance(parsed, list) else parsed.get("laws", [parsed])
            out_json: List[Dict[str, Any]] = []
            for r in rows:
                if not isinstance(r, dict):
                    continue
                ln = (r.get("law_name") or law_name or "").strip().strip("《》")
                art_raw = (r.get("article") or "").strip()
                art = normalize_article_id(art_raw) if art_raw else ""
                content = (r.get("content") or "").strip()
                if not ln or not art or not content:
                    continue
                out_json.append(
                    {
                        "law_name": ln,
                        "document_number": r.get("document_number", document_number),
                        "hierarchy": r.get("hierarchy", hierarchy),
                        "host_law": r.get("host_law", host_law),
                        "host_hierarchy": r.get("host_hierarchy", host_hierarchy),
                        "host_effective_date": r.get("host_effective_date", host_effective_date),
                        "promulgation_date": r.get("promulgation_date", promulgation_date),
                        "effective_date": r.get("effective_date", effective_date),
                        "article_effective_date": r.get("article_effective_date", r.get("effective_date", effective_date)),
                        "effective_to": r.get("effective_to", effective_to),
                        "validity": r.get("validity", validity),
                        "superseded_by": r.get("superseded_by", ""),
                        "article": art,
                        "paragraph": r.get("paragraph", ""),
                        "content": content,
                        "source": r.get("source", source),
                        "category": r.get("category", category),
                        "scope": r.get("scope", scope),
                        "norm_type": r.get("norm_type", "judicial_interpretation" if r.get("hierarchy", hierarchy) == "司法解释" else "substantive"),
                        "partially_invalid": bool(r.get("partially_invalid", False)),
                        "invalid_parts": r.get("invalid_parts", ""),
                        "trial": bool(r.get("trial", False)),
                        "verified": bool(r.get("verified", verified)),
                        "note": r.get("note", ""),
                    }
                )
            if out_json:
                return out_json
        except Exception:
            pass

    # 2) 非结构化法规全文：按“第X条”正则切分条文，自动转换阿拉伯数字条号
    clean_law = (law_name or "未命名法规").strip().strip("《》")
    pattern = re.compile(
        r"(?:^|\n)\s*(第[零〇一二两三四五六七八九十百千0-9]+条)\s*(.*?)(?=(?:\n\s*第[零〇一二两三四五六七八九十百千0-9]+条\s*)|\Z)",
        re.DOTALL,
    )
    matches = list(pattern.finditer(text))
    articles: List[Dict[str, Any]] = []

    if matches:
        for m in matches:
            raw_art = m.group(1).strip()
            norm_art = normalize_article_id(raw_art)
            body = re.sub(r"\s+", " ", m.group(2).strip())
            if not body:
                continue
            articles.append(
                {
                    "law_name": clean_law,
                    "document_number": document_number,
                    "hierarchy": hierarchy,
                    "host_law": host_law,
                    "host_hierarchy": host_hierarchy,
                    "host_effective_date": host_effective_date,
                    "promulgation_date": promulgation_date,
                    "effective_date": effective_date,
                    "article_effective_date": effective_date,
                    "effective_to": effective_to,
                    "validity": validity,
                    "superseded_by": "",
                    "article": norm_art,
                    "paragraph": "",
                    "content": body,
                    "source": source,
                    "category": category,
                    "scope": scope,
                    "norm_type": "judicial_interpretation" if hierarchy == "司法解释" else "substantive",
                    "partially_invalid": False,
                    "invalid_parts": "",
                    "trial": "试行" in clean_law or "暂行" in clean_law,
                    "verified": verified,
                    "note": "",
                }
            )
    else:
        # 若整段文本未包含“第X条”字样，作为第1条单条录入
        articles.append(
            {
                "law_name": clean_law,
                "document_number": document_number,
                "hierarchy": hierarchy,
                "host_law": host_law,
                "host_hierarchy": host_hierarchy,
                "host_effective_date": host_effective_date,
                "promulgation_date": promulgation_date,
                "effective_date": effective_date,
                "article_effective_date": effective_date,
                "effective_to": effective_to,
                "validity": validity,
                "superseded_by": "",
                "article": "第1条",
                "paragraph": "",
                "content": re.sub(r"\s+", " ", text),
                "source": source,
                "category": category,
                "scope": scope,
                "norm_type": "judicial_interpretation" if hierarchy == "司法解释" else "substantive",
                "partially_invalid": False,
                "invalid_parts": "",
                "trial": "试行" in clean_law or "暂行" in clean_law,
                "verified": verified,
                "note": "",
            }
        )
    return articles


class KnowledgeBaseStore:
    """持久化 RAG 法规知识库与 BGE-M3 向量索引存储（MySQL）。"""

    def __init__(self, db_path: Optional[str] = None):
        self.embedder = BGEM3Embedder()
        self.version_counter: int = 0
        self._manually_pinned_doc_ids: Set[str] = set()
        self._init_db()

    def _init_db(self) -> None:
        from app.mysql_store import mysql_store
        mysql_store.init_schema()

        # 仅在知识库目录树完全为空（全新部署首次启动）时初始化顶层根知识库；已有数据时严格保留，不执行覆盖注入
        count_row = mysql_store.query_one("SELECT COUNT(*) AS c FROM kb_tree_nodes")
        if not count_row or (count_row.get("c") or 0) == 0:
            default_nodes = [
                ("kb_internal", "", "内规知识库", "kb", "企业内部规章制度与合规政策", 1),
                ("kb_external", "", "外规知识库", "kb", "国家法律法规、司法解释与部委规章", 2),
            ]
            now_ts = datetime.now(timezone.utc).isoformat()
            for nid, pid, nname, ntype, ndesc, nsort in default_nodes:
                mysql_store.execute(
                    """
                    INSERT INTO kb_tree_nodes (node_id, parent_id, name, node_type, description, sort_order, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (nid, pid, nname, ntype, ndesc, nsort, now_ts),
                )
        now = datetime.now(timezone.utc).isoformat()
        mysql_store.execute(
            """
            REPLACE INTO kb_collections (collection_id, name, description, embed_model, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                "default_external_laws",
                "外部导入法规与司法解释库",
                "支持动态创建、导入外规文档并基于 BAAI/bge-m3 进行向量化解析",
                settings.embed_model,
                now,
            ),
        )

    def create_collection(
        self,
        collection_id: str,
        name: str,
        description: str = "",
        embed_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """创建一个新的 RAG 法规知识库集合。"""
        from app.mysql_store import mysql_store
        cid = (collection_id or "").strip() or f"kb_{int(datetime.now().timestamp())}"
        now = datetime.now(timezone.utc).isoformat()
        em = (embed_model or settings.embed_model or "BAAI/bge-m3").strip()
        mysql_store.execute(
            """
            REPLACE INTO kb_collections (collection_id, name, description, embed_model, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (cid, name, description, em, now),
        )
        existing = mysql_store.query_one("SELECT node_id FROM kb_tree_nodes WHERE name = ?", (name,))
        if not existing:
            mysql_store.execute(
                """
                REPLACE INTO kb_tree_nodes (node_id, parent_id, name, node_type, description, sort_order, created_at)
                VALUES (?, '', ?, 'kb', ?, 99, ?)
                """,
                (cid, name, description, now),
            )
        return {
            "collection_id": cid,
            "name": name,
            "description": description,
            "embed_model": em,
            "created_at": now,
        }

    def list_collections(self) -> List[Dict[str, Any]]:
        from app.mysql_store import mysql_store
        rows = mysql_store.query(
            """
            SELECT c.collection_id, c.name, c.description, c.embed_model, c.created_at,
                   COUNT(a.id) AS article_count
            FROM kb_collections c
            LEFT JOIN kb_articles a ON c.collection_id = a.collection_id
            GROUP BY c.collection_id
            ORDER BY c.created_at ASC
            """
        )
        return [
            {
                "collection_id": r["collection_id"],
                "name": r["name"],
                "description": r.get("description") or "",
                "embed_model": r["embed_model"],
                "created_at": r["created_at"],
                "article_count": r.get("article_count", 0),
            }
            for r in rows
        ]

    def ingest_articles(
        self,
        articles: List[Dict[str, Any]],
        collection_id: str = "default_external_laws",
    ) -> Dict[str, Any]:
        """将解析后的法规条文批量生成 BGE-M3 向量并入库（同法规+同条号+同款项自动 Upsert 更新）。"""
        if not articles:
            return {"ingested_count": 0, "collection_id": collection_id, "embed_backend": "none"}

        texts_for_embed = [
            f"{a.get('law_name', '')} {a.get('article', '')} {a.get('paragraph', '')} "
            f"{a.get('content', '')} {a.get('category', '')}"
            for a in articles
        ]
        vectors, backend = self.embedder.embed_texts(texts_for_embed)
        now = datetime.now(timezone.utc).isoformat()

        from app.mysql_store import mysql_store
        ingested = 0
        for art, vec in zip(articles, vectors):
            law_name = (art.get("law_name") or "").strip()
            article_no = normalize_article_id(art.get("article") or "")
            para = (art.get("paragraph") or "").strip()
            if not law_name or not article_no or not art.get("content"):
                continue
            art["article"] = article_no
            mysql_store.execute(
                """
                REPLACE INTO kb_articles (
                    collection_id, law_name, article, paragraph, hierarchy, validity,
                    effective_date, effective_to, category, content, metadata_json,
                    embedding_json, embed_backend, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    collection_id,
                    law_name,
                    article_no,
                    para,
                    art.get("hierarchy", "法律"),
                    art.get("validity", "现行有效"),
                    art.get("effective_date", ""),
                    art.get("effective_to", ""),
                    art.get("category", "民事"),
                    art.get("content", ""),
                    json.dumps(art, ensure_ascii=False),
                    json.dumps(vec),
                    backend,
                    now,
                ),
            )
            ingested += 1

        self.version_counter += 1
        return {
            "ingested_count": ingested,
            "collection_id": collection_id,
            "embed_backend": backend,
            "vector_dim": len(vectors[0]) if vectors else 0,
        }

    def ingest_document(
        self,
        raw_text: str,
        law_name: str,
        collection_id: str = "default_external_laws",
        **meta_kwargs: Any,
    ) -> Dict[str, Any]:
        """一步完成：外部法规文档切分解析 + BGE-M3 向量计算 + SQLite 落库。"""
        parsed_articles = parse_external_regulation_text(raw_text, law_name=law_name, **meta_kwargs)
        res = self.ingest_articles(parsed_articles, collection_id=collection_id)
        res["articles_preview"] = [
            {
                "law_name": a["law_name"],
                "article": a["article"],
                "hierarchy": a["hierarchy"],
                "validity": a["validity"],
                "content_preview": a["content"][:80] + ("..." if len(a["content"]) > 80 else ""),
            }
            for a in parsed_articles[:10]
        ]
        return res

    def load_all_kb_rows(self) -> List[Dict[str, Any]]:
        """加载动态知识库中存储的全部法规条文（含 kb_articles 与 law_chunks，携带完整版本元数据）。"""
        from app.mysql_store import mysql_store
        out: List[Dict[str, Any]] = []

        # 1. 兼容原 kb_articles
        rows_art = mysql_store.query("SELECT metadata_json, embedding_json FROM kb_articles ORDER BY id ASC")
        for r in rows_art:
            try:
                item = json.loads(r["metadata_json"])
                item["_embedding"] = json.loads(r["embedding_json"])
                item.setdefault("version_code", "v1")
                item.setdefault("version_name", item.get("effective_date") or "现行有效版")
                item.setdefault("is_latest", 1)
                out.append(item)
            except Exception:
                continue

        # 2. 载入 law_chunks (只取非 parent 的条文块，携带版本与最新状态)
        try:
            rows_chunk = mysql_store.query(
                """
                SELECT c.doc_id, c.version_code, c.is_latest, c.effective_date, c.effective_to,
                       c.validity, c.metadata_json, c.embedding_json, d.version_name, d.kb_type
                FROM law_chunks c
                LEFT JOIN law_documents d ON c.doc_id = d.doc_id
                WHERE c.chunk_type != 'hierarchical_parent'
                ORDER BY c.created_at ASC
                """
            )
        except Exception:
            rows_chunk = mysql_store.query(
                "SELECT metadata_json, embedding_json FROM law_chunks WHERE chunk_type != 'hierarchical_parent' ORDER BY created_at ASC"
            )

        for r in rows_chunk:
            try:
                item = json.loads(r["metadata_json"])
                item["_embedding"] = json.loads(r["embedding_json"])
                if r.get("doc_id"):
                    item["doc_id"] = r["doc_id"]
                if r.get("version_code"):
                    item["version_code"] = r["version_code"]
                else:
                    item.setdefault("version_code", "v1")
                if r.get("version_name"):
                    item["version_name"] = r["version_name"]
                else:
                    item.setdefault("version_name", item.get("version_code", "现行有效版"))
                if r.get("is_latest") is not None:
                    item["is_latest"] = int(r["is_latest"])
                else:
                    item.setdefault("is_latest", 1)
                if r.get("kb_type"):
                    item["kb_type"] = r["kb_type"]
                if r.get("effective_date"):
                    item["effective_date"] = r["effective_date"]
                if r.get("effective_to") is not None:
                    item["effective_to"] = r["effective_to"]
                if r.get("validity"):
                    item["validity"] = r["validity"]
                out.append(item)
            except Exception:
                continue

        today_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        for item in out:
            if not item.get("effective_date"):
                item["effective_date"] = today_iso
            if not item.get("article_effective_date"):
                item["article_effective_date"] = item["effective_date"]

        return out

    def sniff_document_version(self, filename: str, raw_text: str = "") -> Dict[str, Any]:
        """智能正文与文件名版本嗅探器：
        自动扫描文档前 3500 字题注与文件名，提取修正/修订/通过日期与推荐版本号。
        """
        fname = (filename or "").strip()
        base_title = fname.rsplit(".", 1)[0] if "." in fname else fname
        # 剥离文件名中常见的括号版本后缀，提取干净法名
        clean_law = re.sub(r"[\(（\[【][^\)）\]】]*(?:修正|修订|版|通过|施行|\d{4})[^\)）\]】]*[\)）\]】]", "", base_title).strip().strip("《》")
        if not clean_law:
            clean_law = base_title.strip().strip("《》")

        head_text = (raw_text or "")[:3500]
        # 优先截取正文「第一条」之前的题注区，避免条文内部引用的历史日期干扰版本嗅探
        first_art_match = re.search(r"(?:^|\n)\s*第一条[\s　]", head_text)
        caption_zone = head_text[: first_art_match.start()] if first_art_match and first_art_match.start() > 20 else head_text
        combined = f"{fname}\n{caption_zone}"

        # 1. 匹配标准题注中的精确日期+动作（如：2018年12月29日...修正 / 1994年7月5日...通过）
        date_action_matches = re.findall(
            r"((?:19|20)\d{2})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日[^\n。；;]{0,65}?(修正|修订|通过|公布|施行|制定)",
            combined,
        )
        if not date_action_matches and caption_zone != head_text:
            date_action_matches = re.findall(
                r"((?:19|20)\d{2})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日[^\n。；;]{0,65}?(修正|修订|通过|公布|施行|制定)",
                f"{fname}\n{head_text}",
            )
        if date_action_matches:
            # 按真实日期 (年, 月, 日) 升序排列，取题注中时间最晚（即最新一次）的日期记录
            sorted_matches = sorted(
                date_action_matches,
                key=lambda m: (int(m[0]), int(m[1]), int(m[2])),
            )
            y, m, d, action = sorted_matches[-1]
            iso_date = f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
            # 若同一年份在题注中明确带有「修正」或「修订」，优先标注为修正版/修订版
            same_year_actions = [act for yr, _mo, _da, act in sorted_matches if yr == y]
            if "修订" in same_year_actions or "修订" in combined:
                act_label = "修订版"
                is_rev = True
            elif "修正" in same_year_actions or "修正" in combined:
                act_label = "修正版"
                is_rev = True
            elif action in ("通过", "制定"):
                act_label = "制定版"
                is_rev = False
            else:
                act_label = f"{action}版"
                is_rev = False
            ver_name = f"{y}年{act_label}"
            ver_code = f"{y}-{'revised' if is_rev else 'original'}"
            return {
                "law_name": clean_law,
                "version_name": ver_name,
                "version_code": ver_code,
                "effective_date": iso_date,
                "sniffed": True,
                "reason": f"从文档题注自动识别：{y}年{int(m)}月{int(d)}日{action}",
            }

        # 2. 匹配年份+修正/修订/版（如：2023修正、2018年版）
        year_matches = re.findall(r"((?:19|20)\d{2})\s*年?\s*(修正|修订|通过|制定|版)", combined)
        if year_matches:
            y, action = year_matches[-1]
            act_label = f"{action}版" if not action.endswith("版") else action
            ver_name = f"{y}年{act_label}"
            ver_code = f"{y}-{'revised' if '修' in action else 'edition'}"
            return {
                "law_name": clean_law,
                "version_name": ver_name,
                "version_code": ver_code,
                "effective_date": f"{y}-01-01",
                "sniffed": True,
                "reason": f"从文件名/正文识别年份特征：{ver_name}",
            }

        # 3. 匹配 ISO 日期（如 2024-07-01）
        iso_matches = re.findall(r"((?:19|20)\d{2}-\d{2}-\d{2})", combined)
        if iso_matches:
            iso_d = iso_matches[-1]
            yr = iso_d[:4]
            return {
                "law_name": clean_law,
                "version_name": f"{yr}年施行版",
                "version_code": f"v{yr}",
                "effective_date": iso_d,
                "sniffed": True,
                "reason": f"识别到日期标记：{iso_d}",
            }

        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return {
            "law_name": clean_law,
            "version_name": "现行有效版",
            "version_code": "v1",
            "effective_date": today_str,
            "sniffed": False,
            "reason": "未检测到特定历史年份题注，默认设为现行有效版",
        }

    def check_version_conflict(
        self,
        filename: str,
        law_name: str = "",
        kb_type: str = "",
        content_bytes: Optional[bytes] = None,
        raw_text: str = "",
    ) -> Dict[str, Any]:
        """上传前检测同名法规冲突并执行智能版本嗅探。"""
        from app.mysql_store import mysql_store
        from app.splitter import extract_text_from_file_bytes

        self._normalize_legacy_duplicate_versions()

        sample_text = raw_text or ""
        if not sample_text and content_bytes:
            try:
                sample_text = extract_text_from_file_bytes(filename, content_bytes)[:4000]
            except Exception:
                sample_text = ""

        sniff_res = self.sniff_document_version(filename, sample_text)
        target_law = (law_name or sniff_res["law_name"] or filename.rsplit(".", 1)[0]).strip().strip("《》")

        existing = mysql_store.query(
            """
            SELECT doc_id, title, kb_type, version_code, version_name, is_latest,
                   effective_date, valid_until, validity_status, chunks_count, uploader, created_at
            FROM law_documents
            WHERE title = ?
            ORDER BY is_latest DESC, created_at DESC
            """,
            (target_law,),
        )

        # 如果存在同名法规，且未嗅探到具体年份版本（仍为默认 v1），自动递增版本号推荐 v{N+1}
        rec_ver_code = sniff_res["version_code"]
        rec_ver_name = sniff_res["version_name"]
        existing_codes = {r.get("version_code") or "v1" for r in existing}
        if existing and (not sniff_res["sniffed"] or rec_ver_code in existing_codes):
            next_num = len(existing) + 1
            while f"v{next_num}" in existing_codes:
                next_num += 1
            if not sniff_res["sniffed"]:
                rec_ver_code = f"v{next_num}"
                rec_ver_name = f"第{next_num}版 (最新修订)"
            elif rec_ver_code in existing_codes:
                rec_ver_code = f"{rec_ver_code}-v{next_num}"

        return {
            "has_conflict": len(existing) > 0,
            "law_name": target_law,
            "recommended_version_code": rec_ver_code,
            "recommended_version_name": rec_ver_name,
            "recommended_effective_date": sniff_res["effective_date"],
            "sniffed": sniff_res["sniffed"],
            "sniff_reason": sniff_res["reason"],
            "existing_count": len(existing),
            "existing_versions": existing,
        }

    def import_advanced_document(
        self,
        filename: str,
        content_bytes: Optional[bytes] = None,
        raw_text: Optional[str] = None,
        law_name: str = "",
        kb_type: str = "外规知识库/法律法规",
        department: str = "法律合规部",
        uploader: str = "执业律师",
        split_mode: str = "article",  # auto / article / hierarchical
        validity_option: str = "permanent",  # permanent / custom
        valid_until: str = "",
        hierarchy: str = "法律",
        promulgation_date: str = "",
        effective_date: str = "",
        host_law: str = "",
        version_code: str = "",
        version_name: str = "",
        conflict_action: str = "append_new",  # append_new / overwrite_latest / rename_new
        custom_new_title: str = "",
        official_sxx: Optional[int] = None,
    ) -> Dict[str, Any]:
        """对标「AI法盾」企业级上传、智能版本管理与三模式切分入库流水线：
        1. 抽取纯文本与自动嗅探版本元数据
        2. 同名法规版本演进控制（追加新版本归档旧版 / 覆盖当前版 / 独立重命名）
        3. 根据 split_mode 执行切分算法 (自动 / 法条 / 分层)
        4. 批量生成 BGE-M3 向量并写入 law_chunks (MySQL，携带版本标签)
        5. 同步至 Elasticsearch 8.x 混合索引库
        6. 登记至 law_documents 文档总表
        """
        import uuid
        from app.splitter import (
            extract_text_from_file_bytes,
            split_auto,
            split_by_articles,
            split_hierarchical,
        )
        from app.es_store import es_vector_store
        from app.mysql_store import mysql_store

        self._normalize_legacy_duplicate_versions()

        text = ""
        doc_size = "0 KB"
        ext = (filename.rsplit(".", 1)[-1] if "." in filename else "TXT").upper()

        import base64

        if content_bytes is not None:
            doc_size = f"{len(content_bytes) / 1024:.1f} KB"
            text = extract_text_from_file_bytes(filename, content_bytes)
        elif raw_text:
            # 容错：如果 raw_text 实际上是 Base64 DataURL (如 data:application/pdf;base64,...)
            if raw_text.startswith("data:") and ";base64," in raw_text:
                try:
                    b64_data = raw_text.split(";base64,", 1)[1]
                    b_bytes = base64.b64decode(b64_data)
                    doc_size = f"{len(b_bytes) / 1024:.1f} KB"
                    text = extract_text_from_file_bytes(filename, b_bytes)
                except Exception:
                    text = raw_text
                    doc_size = f"{len(raw_text.encode('utf-8')) / 1024:.1f} KB"
            # 容错：如果是 PDF 文件但传入了未处理的二进制流
            elif filename.lower().endswith(".pdf") and ("%PDF" in raw_text[:50] or "0 obj" in raw_text[:100]):
                try:
                    b_bytes = raw_text.encode("latin1", errors="ignore")
                    text = extract_text_from_file_bytes(filename, b_bytes)
                    doc_size = f"{len(b_bytes) / 1024:.1f} KB"
                except Exception:
                    text = raw_text
                    doc_size = f"{len(raw_text.encode('utf-8')) / 1024:.1f} KB"
            else:
                text = raw_text
                doc_size = f"{len(raw_text.encode('utf-8')) / 1024:.1f} KB"

        # 智能嗅探版本与法规名称
        sniff = self.sniff_document_version(filename, text)
        if conflict_action == "rename_new" and custom_new_title.strip():
            clean_law = custom_new_title.strip().strip("《》")
        else:
            clean_law = (law_name or sniff["law_name"] or filename.rsplit(".", 1)[0]).strip().strip("《》")

        eff_date = (effective_date or "").strip() or sniff["effective_date"] or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        ver_name = (version_name or "").strip() or sniff["version_name"] or "现行有效版"
        ver_code = (version_code or "").strip() or sniff["version_code"] or "v1"

        # 查询库中已有的同名法规版本
        existing_docs = mysql_store.query(
            "SELECT doc_id, version_code, version_name, is_latest, effective_date, validity_status, created_at FROM law_documents WHERE title = ? ORDER BY is_latest DESC, created_at DESC",
            (clean_law,),
        )
        for ed in existing_docs:
            self._manually_pinned_doc_ids.discard(ed["doc_id"])

        doc_id = f"doc_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc).isoformat()
        val_status = "有效"
        if official_sxx == 1:
            val_status = "已废止"
        elif official_sxx == 2 and existing_docs:
            val_status = "已修正"
        elif valid_until and valid_until < datetime.now().strftime("%Y-%m-%d"):
            val_status = "已过期"

        archived_count = 0
        overwritten_doc_id = ""
        new_doc_is_latest = 1
        new_doc_valid_until = valid_until if validity_option == "custom" else ""

        if existing_docs:
            if conflict_action == "overwrite_latest":
                # 选项 B：覆盖已有最新版本（彻底清除原最新版本的文档与切片）
                target_old = existing_docs[0]
                overwritten_doc_id = target_old["doc_id"]
                mysql_store.execute("DELETE FROM law_documents WHERE doc_id = ?", (overwritten_doc_id,))
                mysql_store.execute("DELETE FROM law_chunks WHERE doc_id = ?", (overwritten_doc_id,))
            else:
                # 选项 A（默认 append_new）：基于施行日期/版本年份与官方时效状态智能判定新旧次序
                existing_codes = {d.get("version_code") or "v1" for d in existing_docs}
                if ver_code in existing_codes:
                    ver_code = f"{ver_code}-v{len(existing_docs) + 1}"
                if ver_name == "现行有效版":
                    ver_name = f"第{len(existing_docs) + 1}版 ({eff_date[:4]}修订)"

                candidate_doc = {
                    "doc_id": doc_id,
                    "version_code": ver_code,
                    "version_name": ver_name,
                    "effective_date": eff_date,
                    "validity_status": val_status,
                    "official_sxx": official_sxx,
                    "created_at": now,
                }
                max_existing_doc = max(existing_docs, key=self._extract_version_sort_key)
                if self._extract_version_sort_key(candidate_doc) < self._extract_version_sort_key(max_existing_doc):
                    # 本次入库的文档施行日期/版本年份早于库中已有的最新有效版本（如先入库 2025 版，后入库 2021 版）
                    new_doc_is_latest = 0
                    if val_status == "有效":
                        val_status = "已修正"
                    archived_count = 1
                    try:
                        from datetime import date, timedelta
                        newer_eff = (max_existing_doc.get("effective_date") or "")[:10]
                        if newer_eff:
                            new_doc_valid_until = (date.fromisoformat(newer_eff) - timedelta(days=1)).isoformat()
                    except Exception:
                        pass
                else:
                    # 本次入库的文档确实是更新的版本，将原同名文档沉淀为历史版本 (is_latest=0)
                    new_doc_is_latest = 1
                    prev_end = ""
                    try:
                        from datetime import date, timedelta
                        prev_end = (date.fromisoformat(eff_date[:10]) - timedelta(days=1)).isoformat()
                    except Exception:
                        prev_end = eff_date

                    for old_d in existing_docs:
                        old_id = old_d["doc_id"]
                        mysql_store.execute(
                            """
                            UPDATE law_documents
                            SET is_latest = 0,
                                validity_status = CASE WHEN validity_status = '有效' THEN '已修正' ELSE validity_status END,
                                valid_until = CASE WHEN valid_until = '' OR valid_until IS NULL THEN ? ELSE valid_until END
                            WHERE doc_id = ?
                            """,
                            (prev_end, old_id),
                        )
                        mysql_store.execute(
                            """
                            UPDATE law_chunks
                            SET is_latest = 0,
                                validity = CASE WHEN validity = '现行有效' THEN '已修正' ELSE validity END,
                                effective_to = CASE WHEN effective_to = '' OR effective_to IS NULL THEN ? ELSE effective_to END
                            WHERE doc_id = ?
                            """,
                            (prev_end, old_id),
                        )
                        archived_count += 1

        chunk_validity = "现行有效" if (new_doc_is_latest == 1 and val_status == "有效") else ("已废止" if val_status in ("已废止", "已过期") else "已修正")

        # 根据分段模式执行切分
        raw_chunks: List[Dict[str, Any]] = []
        if split_mode == "auto":
            raw_chunks = split_auto(text, doc_id=doc_id, law_name=clean_law)
        elif split_mode == "hierarchical":
            p_chunks, c_chunks = split_hierarchical(text, doc_id=doc_id, law_name=clean_law)
            raw_chunks = p_chunks + c_chunks
        else:  # article 模式
            raw_chunks = split_by_articles(
                text,
                doc_id=doc_id,
                law_name=clean_law,
                hierarchy=hierarchy,
                effective_date=eff_date,
                effective_to=new_doc_valid_until,
                validity=chunk_validity,
            )

        # 为分段批量计算 BGE-M3 稠密语义向量
        texts_for_embed = [
            f"{c.get('law_name', '')} {ver_name} {c.get('article', '')} {c.get('chapter_path', '')} {c.get('content', '')}"
            for c in raw_chunks
        ]
        vectors, backend = self.embedder.embed_texts(texts_for_embed)

        # 写入 law_documents 主表（含版本控制字段）
        mysql_store.execute(
            """
            REPLACE INTO law_documents (
                doc_id, title, kb_type, department, uploader, doc_size, doc_format,
                split_mode, chunks_count, validity_status, doc_status, valid_until,
                version_code, version_name, is_latest, effective_date, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doc_id,
                clean_law,
                kb_type,
                department,
                uploader,
                doc_size,
                ext,
                split_mode,
                len(raw_chunks),
                val_status,
                "已解析",
                new_doc_valid_until,
                ver_code,
                ver_name,
                new_doc_is_latest,
                eff_date,
                now,
            ),
        )

        # 准备分块写入 law_chunks
        for c, vec in zip(raw_chunks, vectors):
            meta = {
                "doc_id": doc_id,
                "law_name": clean_law,
                "version_code": ver_code,
                "version_name": ver_name,
                "is_latest": new_doc_is_latest,
                "kb_type": kb_type,
                "document_number": "",
                "hierarchy": c.get("hierarchy", hierarchy),
                "host_law": host_law,
                "host_hierarchy": "法律" if hierarchy == "司法解释" else "",
                "host_effective_date": "",
                "promulgation_date": promulgation_date or eff_date,
                "effective_date": eff_date,
                "article_effective_date": eff_date,
                "effective_to": new_doc_valid_until,
                "validity": chunk_validity,
                "superseded_by": "",
                "article": c.get("article", "第1条"),
                "paragraph": c.get("paragraph", ""),
                "content": c.get("content", ""),
                "source": "全国人民代表大会常务委员会" if hierarchy == "法律" else "最高人民法院",
                "category": "内规" if "内规" in kb_type else "民事",
                "scope": "一般",
                "norm_type": "judicial_interpretation" if hierarchy == "司法解释" else "substantive",
                "partially_invalid": False,
                "invalid_parts": "",
                "trial": "试行" in clean_law or "暂行" in clean_law,
                "verified": True,
                "note": f"［{ver_name}·{split_mode}切分］",
            }
            c["content_vector"] = vec
            c["version_code"] = ver_code
            c["version_name"] = ver_name
            c["is_latest"] = new_doc_is_latest
            mysql_store.execute(
                """
                REPLACE INTO law_chunks (
                    chunk_id, doc_id, law_name, version_code, is_latest, chapter_path,
                    article, paragraph, content, chunk_type, parent_chunk_id, hierarchy,
                    effective_date, effective_to, validity, metadata_json, embedding_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    c["chunk_id"],
                    doc_id,
                    clean_law,
                    ver_code,
                    new_doc_is_latest,
                    c.get("chapter_path", ""),
                    c.get("article", "第1条"),
                    c.get("paragraph", ""),
                    c.get("content", ""),
                    c.get("chunk_type", "article"),
                    c.get("parent_chunk_id", ""),
                    hierarchy,
                    eff_date,
                    new_doc_valid_until,
                    meta["validity"],
                    json.dumps(meta, ensure_ascii=False),
                    json.dumps(vec),
                    now,
                ),
            )

        # 同步写入 Elasticsearch 8.x
        es_indexed_count = 0
        if es_vector_store.ping():
            es_vector_store.init_index(dims=len(vectors[0]) if vectors else 128)
            es_indexed_count = es_vector_store.index_chunks(raw_chunks)

        # 文档入库即自动静默抽取条文明示引致关系，沉淀至待审规则提案池
        auto_refs_count = 0
        if new_doc_is_latest == 1:
            try:
                from app.graph_builder import extract_quick_regex_references, _record_graph_proposals

                auto_refs: List[Dict[str, Any]] = []
                seen_auto = set()
                for c in raw_chunks:
                    if c.get("chunk_type") == "hierarchical_parent":
                        continue
                    for r in extract_quick_regex_references(clean_law, c.get("article", ""), c.get("content", "")):
                        k = (r["source_law"], r["source_article"], r["target_law"], r["target_article"])
                        if k not in seen_auto:
                            seen_auto.add(k)
                            auto_refs.append(r)
                if auto_refs:
                    _record_graph_proposals(auto_refs, query_prefix="文档入库自动抽取")
                    auto_refs_count = len(auto_refs)
            except Exception:
                pass

        self.version_counter += 1
        return {
            "doc_id": doc_id,
            "title": clean_law,
            "version_code": ver_code,
            "version_name": ver_name,
            "effective_date": eff_date,
            "is_latest": new_doc_is_latest,
            "conflict_action": conflict_action,
            "archived_history_count": archived_count,
            "overwritten_doc_id": overwritten_doc_id,
            "split_mode": split_mode,
            "chunks_count": len(raw_chunks),
            "es_indexed": es_indexed_count > 0,
            "embed_backend": backend,
            "auto_extracted_graph_refs": auto_refs_count,
            "preview_chunks": [
                {
                    "chunk_id": c["chunk_id"],
                    "article": c.get("article", ""),
                    "chapter_path": c.get("chapter_path", ""),
                    "chunk_type": c.get("chunk_type", ""),
                    "content_preview": c["content"][:100] + ("..." if len(c["content"]) > 100 else ""),
                }
                for c in raw_chunks[:8]
            ],
        }

    def get_node_path(self, node_id: str) -> str:
        """递归获取节点的完整层级路径（如: '外规知识库/法律法规'）。"""
        if not node_id:
            return ""
        from app.mysql_store import mysql_store
        parts = []
        cur_id = node_id
        while cur_id:
            row = mysql_store.query_one("SELECT parent_id, name FROM kb_tree_nodes WHERE node_id = ?", (cur_id,))
            if not row:
                break
            p_id = row["parent_id"]
            n_name = row["name"]
            parts.insert(0, n_name)
            cur_id = p_id
        return "/".join(parts)

    def get_kb_tree(self) -> List[Dict[str, Any]]:
        """获取层级知识库与目录树（含各级节点下的文档统计数）。"""
        from app.mysql_store import mysql_store
        rows = mysql_store.query("SELECT node_id, parent_id, name, node_type, description, sort_order, created_at FROM kb_tree_nodes ORDER BY sort_order ASC, created_at ASC")

        doc_counts_rows = mysql_store.query("SELECT kb_type, COUNT(*) AS cnt FROM law_documents GROUP BY kb_type")
        doc_counts = {r["kb_type"]: r["cnt"] for r in doc_counts_rows}

        nodes_map: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            nid = r["node_id"]
            nodes_map[nid] = {
                "node_id": nid,
                "parent_id": r.get("parent_id") or "",
                "name": r["name"],
                "node_type": r["node_type"],
                "description": r.get("description") or "",
                "sort_order": r.get("sort_order", 0),
                "created_at": r.get("created_at") or "",
                "children": [],
                "doc_count": 0,
            }

        for nid, node in nodes_map.items():
            path_parts = []
            cid = nid
            while cid and cid in nodes_map:
                path_parts.insert(0, nodes_map[cid]["name"])
                cid = nodes_map[cid]["parent_id"]
            node_path = "/".join(path_parts)
            node["path"] = node_path
            count = sum(cnt for kb_path, cnt in doc_counts.items() if kb_path == node_path or kb_path.startswith(node_path + "/"))
            node["doc_count"] = count

        tree: List[Dict[str, Any]] = []
        for nid, node in nodes_map.items():
            pid = node["parent_id"]
            if not pid or pid not in nodes_map:
                tree.append(node)
            else:
                nodes_map[pid]["children"].append(node)
        return tree

    def create_kb_tree_node(
        self,
        name: str,
        parent_id: str = "",
        node_type: str = "folder",
        description: str = "",
        owner_dept_id: str = "dept_hq",
        visibility: str = "dept",
    ) -> Dict[str, Any]:
        """创建知识库（parent_id 为空）或层级子目录（支持绑定主责管理部门与可见性）。"""
        import uuid
        from app.mysql_store import mysql_store
        clean_name = (name or "").strip()
        if not clean_name:
            raise ValueError("知识库或目录名称不能为空")

        nid = f"kb_{uuid.uuid4().hex[:10]}" if not parent_id else f"folder_{uuid.uuid4().hex[:10]}"
        actual_type = "kb" if not parent_id else node_type
        now_ts = datetime.now(timezone.utc).isoformat()
        clean_owner_dept = (owner_dept_id or "dept_hq").strip()
        clean_visibility = (visibility or "dept").strip()

        row = mysql_store.query_one("SELECT MAX(sort_order) AS max_sort FROM kb_tree_nodes WHERE parent_id = ?", (parent_id or "",))
        max_sort = (row["max_sort"] if row and row.get("max_sort") is not None else 0) or 0

        mysql_store.execute(
            """
            REPLACE INTO kb_tree_nodes (node_id, parent_id, name, node_type, owner_dept_id, visibility, description, sort_order, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (nid, parent_id or "", clean_name, actual_type, clean_owner_dept, clean_visibility, description or "", max_sort + 1, now_ts),
        )
        if actual_type == "kb":
            mysql_store.execute(
                """
                REPLACE INTO kb_collections (collection_id, name, description, embed_model, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (nid, clean_name, description or "", settings.embed_model, now_ts),
            )

        full_path = self.get_node_path(nid)
        return {
            "node_id": nid,
            "parent_id": parent_id or "",
            "name": clean_name,
            "node_type": actual_type,
            "owner_dept_id": clean_owner_dept,
            "visibility": clean_visibility,
            "path": full_path,
            "description": description or "",
        }

    def rename_kb_tree_node(self, node_id: str, new_name: str) -> bool:
        """重命名知识库或层级目录，并同步更新关联文档的 kb_type 路径。"""
        from app.mysql_store import mysql_store
        clean_name = (new_name or "").strip()
        if not clean_name:
            raise ValueError("新名称不能为空")
        old_path = self.get_node_path(node_id)

        mysql_store.execute("UPDATE kb_tree_nodes SET name = ? WHERE node_id = ?", (clean_name, node_id))
        new_path = self.get_node_path(node_id)
        if old_path and new_path and old_path != new_path:
            mysql_store.execute(
                "UPDATE law_documents SET kb_type = ? WHERE kb_type = ?",
                (new_path, old_path),
            )
            docs = mysql_store.query("SELECT doc_id, kb_type FROM law_documents WHERE kb_type LIKE ?", (old_path + "/%",))
            for d in docs:
                sub_path = new_path + d["kb_type"][len(old_path):]
                mysql_store.execute("UPDATE law_documents SET kb_type = ? WHERE doc_id = ?", (sub_path, d["doc_id"]))

        return True

    def delete_kb_tree_node(self, node_id: str) -> bool:
        """删除知识库或层级目录（递归删除所有子目录和挂在该目录下的文档与切片）。"""
        from app.mysql_store import mysql_store

        def _collect_descendants(nid: str) -> List[str]:
            children_rows = mysql_store.query("SELECT node_id FROM kb_tree_nodes WHERE parent_id = ?", (nid,))
            children = [r["node_id"] for r in children_rows]
            all_ids = list(children)
            for c in children:
                all_ids.extend(_collect_descendants(c))
            return all_ids

        all_to_del = [node_id] + _collect_descendants(node_id)
        node_path = self.get_node_path(node_id)

        for nid in all_to_del:
            mysql_store.execute("DELETE FROM kb_tree_nodes WHERE node_id = ?", (nid,))

        if node_path:
            doc_rows = mysql_store.query(
                "SELECT doc_id FROM law_documents WHERE kb_type = ? OR kb_type LIKE ?",
                (node_path, node_path + "/%"),
            )
            doc_ids = [r["doc_id"] for r in doc_rows]
            for did in doc_ids:
                mysql_store.execute("DELETE FROM law_documents WHERE doc_id = ?", (did,))
                mysql_store.execute("DELETE FROM law_chunks WHERE doc_id = ?", (did,))

        # 级联清理角色权限表中的已删知识库记录
        try:
            from app.rbac import rbac_manager
            rbac_manager.cleanup_kb_permissions_for_deleted_kb(node_id)
        except Exception:
            pass

        self.version_counter += 1
        return True

    def _extract_version_sort_key(self, doc: Dict[str, Any]) -> Tuple[int, str, int, str, str]:
        """计算同名法规多版本的时效演进排序权重 (返回值越大代表版本越新、越优先作为现行主版本)：
        1. 是否未被显式废止/过期（已废止/已过期=0，正常/有效/已修正=1）
        2. 法规版本生效日期/题注年份构成的标准化日期键 (YYYY-MM-DD)
        3. 官方时效码辅助权重 (sxx=3 现行有效 > 未指定 > sxx=2 已修改 > sxx=1 已废止)
        4. 入库创建时间 (created_at) — 当生效日期完全相同时，后上传的修正版优先
        5. 文档唯一标识 (doc_id)
        """
        val_status = (doc.get("validity_status") or "").strip()
        sxx = doc.get("official_sxx")
        status_rank = 0 if (sxx == 1 or val_status in ("已废止", "已过期")) else 1

        eff = (doc.get("effective_date") or "").strip()[:10]
        created_day = (doc.get("created_at") or "").strip()[:10]
        v_text = f"{doc.get('version_code') or ''} {doc.get('version_name') or ''}"
        year_matches = re.findall(r"(?:19|20)\d{2}", v_text)
        sniff_year = max(year_matches) if year_matches else ""

        if eff and re.match(r"^\d{4}-\d{2}-\d{2}$", eff):
            # 若 effective_date 恰好是上传当天默认值，而题注/版本名中明确抽出了不同的历史年份，则以题注年份为准
            if sniff_year and eff == created_day and not eff.startswith(sniff_year):
                date_key = f"{sniff_year}-01-01"
            else:
                date_key = eff
        elif sniff_year:
            date_key = f"{sniff_year}-01-01"
        else:
            date_key = created_day or "1970-01-01"

        if sxx == 3:
            sxx_rank = 2
        elif sxx in (1, 2):
            sxx_rank = 0
        else:
            sxx_rank = 1

        return (
            status_rank,
            date_key,
            sxx_rank,
            str(doc.get("created_at") or ""),
            str(doc.get("doc_id") or ""),
        )

    def _normalize_legacy_duplicate_versions(self) -> None:
        """自动规范化存量同名文档的版本链与新旧时效次序：
        1. 若同一法规 (title) 存在多条记录，优先按施行日期/版本年份 (_extract_version_sort_key) 升序排列；
        2. 确保施行日期/版本年份最新的一条为现行有效主版本 (is_latest=1, validity_status='有效')，
           其余旧版本自动归档为历史版本 (is_latest=0, validity_status='已修正') 并自动补齐截止日期 (valid_until / effective_to)；
        3. 若用户通过『设为当前版』手动锁定了某版本 (_manually_pinned_doc_ids)，则保留用户的手动锁定。
        """
        from app.mysql_store import mysql_store
        try:
            all_docs = mysql_store.query(
                "SELECT doc_id, title, kb_type, version_code, version_name, is_latest, effective_date, valid_until, validity_status, chunks_count, created_at FROM law_documents ORDER BY created_at ASC, doc_id ASC"
            )
        except Exception:
            return

        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for d in all_docs:
            t = (d.get("title") or "").strip()
            if t:
                grouped[t].append(d)

        updated_any = False
        for title, items in grouped.items():
            # 确保单版本也具备非空默认值
            if len(items) == 1:
                single = items[0]
                v_code = single.get("version_code") or "v1"
                v_name = single.get("version_name") or "现行有效版"
                is_lat = 1 if single.get("is_latest") is None else int(single.get("is_latest"))
                eff_d = single.get("effective_date") or (single.get("created_at") or "")[:10]
                if not single.get("version_code") or not single.get("version_name") or single.get("is_latest") is None:
                    mysql_store.execute(
                        "UPDATE law_documents SET version_code = ?, version_name = ?, is_latest = ?, effective_date = ? WHERE doc_id = ?",
                        (v_code, v_name, is_lat, eff_d, single["doc_id"]),
                    )
                    updated_any = True
                continue

            # 若该同名法规组存在用户显式手动置顶的版本，则尊重手动指定
            if any(x["doc_id"] in self._manually_pinned_doc_ids for x in items):
                continue

            # 先尝试为尚未嗅探过年份的默认记录补齐题注嗅探信息，以便准确按年份排序
            for item in items:
                cur_code = (item.get("version_code") or "").strip()
                if not cur_code or cur_code == "v1":
                    try:
                        c_rows = mysql_store.query("SELECT content FROM law_chunks WHERE doc_id = ? LIMIT 3", (item["doc_id"],))
                        sample_txt = "\n".join(r.get("content", "") for r in c_rows)
                        if sample_txt:
                            sn = self.sniff_document_version(title, sample_txt)
                            if sn.get("sniffed"):
                                item["_sniffed_code"] = sn["version_code"]
                                item["_sniffed_name"] = sn["version_name"]
                                if sn.get("effective_date"):
                                    item["effective_date"] = sn["effective_date"]
                    except Exception:
                        pass

            # 按版本生效日期/年份权重升序排列（排在最后的一条即为最新有效版本）
            sorted_items = sorted(items, key=self._extract_version_sort_key)
            expected_latest_id = sorted_items[-1]["doc_id"]
            current_latest_ids = [
                x["doc_id"] for x in items
                if (1 if x.get("is_latest") is None else int(x.get("is_latest"))) == 1
            ]
            codes = [x.get("version_code") or "v1" for x in items]
            if (
                len(current_latest_ids) == 1
                and current_latest_ids[0] == expected_latest_id
                and len(set(codes)) == len(items)
            ):
                continue

            total_ver = len(sorted_items)
            for idx, item in enumerate(sorted_items, 1):
                did = item["doc_id"]
                is_last = (idx == total_ver)
                cur_code = (item.get("version_code") or "").strip()
                cur_name = (item.get("version_name") or "").strip()
                eff_d = (item.get("effective_date") or "").strip() or (item.get("created_at") or "")[:10]

                sniffed_code = item.get("_sniffed_code") or ""
                sniffed_name = item.get("_sniffed_name") or ""

                new_code = sniffed_code or (cur_code if cur_code and cur_code != "v1" else f"v{idx}")
                # 防止同一组内 code 碰撞
                if any(other.get("_assigned_code") == new_code for other in sorted_items[:idx - 1]):
                    new_code = f"{new_code}-v{idx}"
                item["_assigned_code"] = new_code

                if sniffed_name:
                    new_name = sniffed_name if is_last else f"{sniffed_name} (v{idx})"
                elif cur_name and cur_name not in ("现行有效版", "v1"):
                    new_name = cur_name
                else:
                    new_name = f"最新版 v{idx} ({item.get('chunks_count', 0)}分段)" if is_last else f"历史版 v{idx} ({item.get('chunks_count', 0)}分段)"

                new_latest = 1 if is_last else 0
                new_val_status = "有效" if is_last else "已修正"

                # 计算历史版本的截止日期（下一版本的生效日前一天），最新有效版本则清空截止日期
                new_valid_until = ""
                if not is_last:
                    next_eff = (sorted_items[idx].get("effective_date") or "").strip()[:10]
                    if next_eff:
                        try:
                            from datetime import date, timedelta
                            new_valid_until = (date.fromisoformat(next_eff) - timedelta(days=1)).isoformat()
                        except Exception:
                            new_valid_until = next_eff

                mysql_store.execute(
                    """
                    UPDATE law_documents
                    SET version_code = ?, version_name = ?, is_latest = ?, effective_date = ?, valid_until = ?, validity_status = ?
                    WHERE doc_id = ?
                    """,
                    (new_code, new_name, new_latest, eff_d, new_valid_until, new_val_status, did),
                )
                mysql_store.execute(
                    """
                    UPDATE law_chunks
                    SET version_code = ?, is_latest = ?, effective_to = ?, validity = ?
                    WHERE doc_id = ?
                    """,
                    (new_code, new_latest, new_valid_until, "现行有效" if is_last else "已修正", did),
                )
                updated_any = True

        if updated_any:
            self.version_counter += 1

    def list_law_documents(self, kb_type: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """获取法规文档列表（支持根据所属知识库或层级目录路径筛选，携带完整版本链元数据）。"""
        from app.mysql_store import mysql_store
        self._normalize_legacy_duplicate_versions()

        if kb_type and kb_type != "all":
            clean_kb = kb_type.strip()
            return mysql_store.query(
                """
                SELECT doc_id, title, kb_type, department, uploader, doc_size, doc_format,
                       split_mode, chunks_count, validity_status, doc_status, valid_until,
                       version_code, version_name, is_latest, effective_date, created_at
                FROM law_documents
                WHERE kb_type = ? OR kb_type LIKE ?
                ORDER BY is_latest DESC, created_at DESC LIMIT ?
                """,
                (clean_kb, clean_kb + "/%", limit),
            )
        else:
            return mysql_store.query(
                """
                SELECT doc_id, title, kb_type, department, uploader, doc_size, doc_format,
                       split_mode, chunks_count, validity_status, doc_status, valid_until,
                       version_code, version_name, is_latest, effective_date, created_at
                FROM law_documents
                ORDER BY is_latest DESC, created_at DESC LIMIT ?
                """,
                (limit,),
            )

    def set_document_as_latest(self, doc_id: str) -> Dict[str, Any]:
        """将指定历史版本切换为当前最新有效版本，并将同名其他版本归档为历史版本。"""
        from app.mysql_store import mysql_store
        target = mysql_store.query_one("SELECT * FROM law_documents WHERE doc_id = ?", (doc_id,))
        if not target:
            raise ValueError(f"未找到文档 {doc_id}")

        title = target["title"]
        # 1. 将同名法规的所有其他文档置为历史版本
        mysql_store.execute(
            "UPDATE law_documents SET is_latest = 0, validity_status = '已修正' WHERE title = ? AND doc_id != ?",
            (title, doc_id),
        )
        other_docs = mysql_store.query("SELECT doc_id FROM law_documents WHERE title = ? AND doc_id != ?", (title, doc_id))
        for od in other_docs:
            self._manually_pinned_doc_ids.discard(od["doc_id"])
            mysql_store.execute(
                "UPDATE law_chunks SET is_latest = 0, validity = '已修正' WHERE doc_id = ?",
                (od["doc_id"],),
            )

        # 2. 将当前文档置为最新有效版本并记录手动锁定
        self._manually_pinned_doc_ids.add(doc_id)
        mysql_store.execute(
            "UPDATE law_documents SET is_latest = 1, validity_status = '有效' WHERE doc_id = ?",
            (doc_id,),
        )
        mysql_store.execute(
            "UPDATE law_chunks SET is_latest = 1, validity = '现行有效' WHERE doc_id = ?",
            (doc_id,),
        )
        self.version_counter += 1
        return {
            "status": "ok",
            "doc_id": doc_id,
            "title": title,
            "version_name": target.get("version_name") or target.get("version_code") or "最新版",
        }

    def compare_document_versions(self, doc_id_new: str, doc_id_old: str) -> Dict[str, Any]:
        """对比同一部法规两个版本之间的条文级差异 (Diff)：新增、删减、修订条文对照。"""
        res_new = self.get_document_chunks(doc_id_new)
        res_old = self.get_document_chunks(doc_id_old)
        if res_new.get("status") != "ok" or res_old.get("status") != "ok":
            return {"status": "error", "message": "待对比的文档版本不存在"}

        doc_new = res_new["document"]
        doc_old = res_old["document"]

        map_new: Dict[str, str] = {}
        order_new: List[str] = []
        for c in res_new.get("chunks", []):
            if c.get("chunk_type") == "hierarchical_parent":
                continue
            art_key = normalize_article_id(c.get("article") or "") or c.get("chunk_id", "")
            if art_key not in map_new:
                order_new.append(art_key)
                map_new[art_key] = (c.get("content") or "").strip()
            else:
                map_new[art_key] += "\n" + (c.get("content") or "").strip()

        map_old: Dict[str, str] = {}
        order_old: List[str] = []
        for c in res_old.get("chunks", []):
            if c.get("chunk_type") == "hierarchical_parent":
                continue
            art_key = normalize_article_id(c.get("article") or "") or c.get("chunk_id", "")
            if art_key not in map_old:
                order_old.append(art_key)
                map_old[art_key] = (c.get("content") or "").strip()
            else:
                map_old[art_key] += "\n" + (c.get("content") or "").strip()

        added = []
        removed = []
        modified = []
        unchanged_count = 0

        for k in order_new:
            if k not in map_old:
                added.append({"article": k, "new_content": map_new[k]})
            elif map_new[k] != map_old[k]:
                modified.append({"article": k, "old_content": map_old[k], "new_content": map_new[k]})
            else:
                unchanged_count += 1

        for k in order_old:
            if k not in map_new:
                removed.append({"article": k, "old_content": map_old[k]})

        return {
            "status": "ok",
            "doc_new": {
                "doc_id": doc_new["doc_id"],
                "title": doc_new["title"],
                "version_name": doc_new.get("version_name") or doc_new.get("version_code") or "新版本",
                "chunks_count": doc_new.get("chunks_count", 0),
                "effective_date": doc_new.get("effective_date") or "",
            },
            "doc_old": {
                "doc_id": doc_old["doc_id"],
                "title": doc_old["title"],
                "version_name": doc_old.get("version_name") or doc_old.get("version_code") or "历史版本",
                "chunks_count": doc_old.get("chunks_count", 0),
                "effective_date": doc_old.get("effective_date") or "",
            },
            "summary": {
                "added_count": len(added),
                "modified_count": len(modified),
                "removed_count": len(removed),
                "unchanged_count": unchanged_count,
            },
            "added_articles": added[:50],
            "modified_articles": modified[:50],
            "removed_articles": removed[:50],
        }

    def delete_law_document(self, doc_id: str) -> bool:
        """删除指定法规文档及其所有条文分段。若删除的是当前最新版且存在历史版本，自动将次新版提拔为最新版。"""
        from app.mysql_store import mysql_store
        target = mysql_store.query_one("SELECT title, is_latest FROM law_documents WHERE doc_id = ?", (doc_id,))
        mysql_store.execute("DELETE FROM law_documents WHERE doc_id = ?", (doc_id,))
        mysql_store.execute("DELETE FROM law_chunks WHERE doc_id = ?", (doc_id,))

        if target and int(target.get("is_latest") or 0) == 1:
            rem = mysql_store.query(
                "SELECT doc_id FROM law_documents WHERE title = ? ORDER BY created_at DESC LIMIT 1",
                (target["title"],),
            )
            if rem:
                self.set_document_as_latest(rem[0]["doc_id"])

        self.version_counter += 1
        return True

    def get_document_chunks(self, doc_id: str) -> Dict[str, Any]:
        """获取指定法规文档的元数据与全量切片分段列表。"""
        from app.mysql_store import mysql_store
        clean_id = (doc_id or "").strip()
        doc = mysql_store.query_one("SELECT * FROM law_documents WHERE doc_id = ?", (clean_id,))
        if not doc:
            return {"status": "not_found", "message": f"未找到文档 {clean_id}", "chunks": []}

        chunks = mysql_store.query(
            """
            SELECT chunk_id, doc_id, law_name, chapter_path, article, paragraph, content,
                   chunk_type, parent_chunk_id, hierarchy, effective_date, effective_to, validity, created_at
            FROM law_chunks
            WHERE doc_id = ?
            ORDER BY chunk_id ASC
            """,
            (clean_id,),
        )

        # 按法律条文自然序（第1条 -> 第2条 -> ...）智能排序
        def _art_sort_key(c):
            art = c.get("article") or ""
            m = re.search(r"第\s*(\d+)\s*条", art)
            num = int(m.group(1)) if m else 999999
            type_prio = 0 if c.get("chunk_type") == "hierarchical_parent" else 1
            return (num, type_prio, c.get("chunk_id", ""))

        chunks.sort(key=_art_sort_key)

        return {
            "status": "ok",
            "document": doc,
            "total_chunks": len(chunks),
            "chunks": chunks,
        }

    def record_article_revision(
        self,
        law_name: str,
        article: str,
        amendment_name: str,
        effective_from: str,
        new_content: str,
        effective_to: str = "",
        paragraph: str = "",
        revision_type: str = "修改",
        old_content: str = "",
        diff_summary: str = "",
    ) -> int:
        """记录细粒度（条、款、项）法律修改日志。"""
        from app.mysql_store import mysql_store
        now = datetime.now(timezone.utc).isoformat()
        mysql_store.execute(
            """
            INSERT INTO law_article_revision_log (
                law_name, article, paragraph, revision_type, amendment_name,
                effective_from, effective_to, old_content, new_content, diff_summary, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                law_name.strip().strip("《》"),
                normalize_article_id(article),
                paragraph.strip(),
                revision_type,
                amendment_name,
                effective_from,
                effective_to,
                old_content,
                new_content,
                diff_summary,
                now,
            ),
        )
        max_row = mysql_store.query_one("SELECT MAX(id) AS id FROM law_article_revision_log")
        return int(max_row["id"]) if max_row and max_row.get("id") else 0

    def list_article_revisions(
        self,
        law_name: Optional[str] = None,
        article: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """查询条文粒度的修改日志列表。"""
        from app.mysql_store import mysql_store
        if law_name and article:
            return mysql_store.query(
                """
                SELECT id, law_name, article, paragraph, revision_type, amendment_name,
                       effective_from, effective_to, old_content, new_content, diff_summary, created_at
                FROM law_article_revision_log
                WHERE law_name = ? AND article = ?
                ORDER BY effective_from DESC
                """,
                (law_name.strip().strip("《》"), normalize_article_id(article)),
            )
        elif law_name:
            return mysql_store.query(
                """
                SELECT id, law_name, article, paragraph, revision_type, amendment_name,
                       effective_from, effective_to, old_content, new_content, diff_summary, created_at
                FROM law_article_revision_log
                WHERE law_name = ?
                ORDER BY effective_from DESC
                """,
                (law_name.strip().strip("《》"),),
            )
        else:
            return mysql_store.query(
                """
                SELECT id, law_name, article, paragraph, revision_type, amendment_name,
                       effective_from, effective_to, old_content, new_content, diff_summary, created_at
                FROM law_article_revision_log
                ORDER BY id DESC LIMIT 50
                """
            )

    def delete_law(self, law_name: str, article: Optional[str] = None) -> int:
        """删除某部法规或某条具体条文。"""
        from app.mysql_store import mysql_store
        deleted = 0
        if article:
            deleted += mysql_store.execute(
                "DELETE FROM kb_articles WHERE law_name = ? AND article = ?",
                (law_name.strip().strip("《》"), normalize_article_id(article)),
            )
            deleted += mysql_store.execute(
                "DELETE FROM law_chunks WHERE law_name = ? AND article = ?",
                (law_name.strip().strip("《》"), normalize_article_id(article)),
            )
        else:
            deleted += mysql_store.execute(
                "DELETE FROM kb_articles WHERE law_name = ?",
                (law_name.strip().strip("《》"),),
            )
            deleted += mysql_store.execute(
                "DELETE FROM law_chunks WHERE law_name = ?",
                (law_name.strip().strip("《》"),),
            )
        if deleted > 0:
            self.version_counter += 1
        return deleted


# 全局知识库存储单例
kb_store = KnowledgeBaseStore()

