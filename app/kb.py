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
import sqlite3
import urllib.request
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
    """持久化 RAG 法规知识库与 BGE-M3 向量索引存储（SQLite）。"""

    def __init__(self, db_path: Optional[str] = None):
        self._explicit_db_path = db_path
        self._mem_conn: Optional[sqlite3.Connection] = None
        self.embedder = BGEM3Embedder()
        self.version_counter: int = 0
        self._init_db()

    @property
    def db_path(self) -> str:
        return self._explicit_db_path or settings.kb_db_path

    def _conn(self) -> sqlite3.Connection:
        if self.db_path == ":memory:":
            if self._mem_conn is None:
                self._mem_conn = sqlite3.connect(":memory:", check_same_thread=False)
            return self._mem_conn
        parent = os.path.dirname(os.path.abspath(self.db_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _init_db(self) -> None:
        conn = self._conn()
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS kb_collections (
                collection_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                embed_model TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS kb_articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                collection_id TEXT NOT NULL,
                law_name TEXT NOT NULL,
                article TEXT NOT NULL,
                paragraph TEXT DEFAULT '',
                hierarchy TEXT NOT NULL,
                validity TEXT NOT NULL,
                effective_date TEXT DEFAULT '',
                effective_to TEXT DEFAULT '',
                category TEXT DEFAULT '',
                content TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                embedding_json TEXT NOT NULL,
                embed_backend TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(law_name, article, paragraph)
            )
            """
        )
        # 确保默认集合存在
        now = datetime.now(timezone.utc).isoformat()
        cur.execute(
            """
            INSERT OR IGNORE INTO kb_collections (collection_id, name, description, embed_model, created_at)
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
        conn.commit()
        if self._mem_conn is None:
            conn.close()

    def create_collection(
        self,
        collection_id: str,
        name: str,
        description: str = "",
        embed_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """创建一个新的 RAG 法规知识库集合。"""
        cid = (collection_id or "").strip() or f"kb_{int(datetime.now().timestamp())}"
        now = datetime.now(timezone.utc).isoformat()
        em = (embed_model or settings.embed_model or "BAAI/bge-m3").strip()
        conn = self._conn()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO kb_collections (collection_id, name, description, embed_model, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(collection_id) DO UPDATE SET
                name=excluded.name,
                description=excluded.description,
                embed_model=excluded.embed_model
            """,
            (cid, name, description, em, now),
        )
        conn.commit()
        if self._mem_conn is None:
            conn.close()
        return {
            "collection_id": cid,
            "name": name,
            "description": description,
            "embed_model": em,
            "created_at": now,
        }

    def list_collections(self) -> List[Dict[str, Any]]:
        conn = self._conn()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT c.collection_id, c.name, c.description, c.embed_model, c.created_at,
                   COUNT(a.id) AS article_count
            FROM kb_collections c
            LEFT JOIN kb_articles a ON c.collection_id = a.collection_id
            GROUP BY c.collection_id
            ORDER BY c.created_at ASC
            """
        )
        rows = cur.fetchall()
        if self._mem_conn is None:
            conn.close()
        return [
            {
                "collection_id": r[0],
                "name": r[1],
                "description": r[2],
                "embed_model": r[3],
                "created_at": r[4],
                "article_count": r[5],
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

        conn = self._conn()
        cur = conn.cursor()
        ingested = 0
        for art, vec in zip(articles, vectors):
            law_name = (art.get("law_name") or "").strip()
            article_no = normalize_article_id(art.get("article") or "")
            para = (art.get("paragraph") or "").strip()
            if not law_name or not article_no or not art.get("content"):
                continue
            art["article"] = article_no
            cur.execute(
                """
                INSERT INTO kb_articles (
                    collection_id, law_name, article, paragraph, hierarchy, validity,
                    effective_date, effective_to, category, content, metadata_json,
                    embedding_json, embed_backend, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(law_name, article, paragraph) DO UPDATE SET
                    collection_id=excluded.collection_id,
                    hierarchy=excluded.hierarchy,
                    validity=excluded.validity,
                    effective_date=excluded.effective_date,
                    effective_to=excluded.effective_to,
                    category=excluded.category,
                    content=excluded.content,
                    metadata_json=excluded.metadata_json,
                    embedding_json=excluded.embedding_json,
                    embed_backend=excluded.embed_backend,
                    updated_at=excluded.updated_at
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

        conn.commit()
        if self._mem_conn is None:
            conn.close()
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
        """加载动态知识库中存储的全部法规条文（含预计算的 embedding 向量）。"""
        conn = self._conn()
        cur = conn.cursor()
        cur.execute("SELECT metadata_json, embedding_json FROM kb_articles ORDER BY id ASC")
        rows = cur.fetchall()
        if self._mem_conn is None:
            conn.close()
        out: List[Dict[str, Any]] = []
        for meta_str, emb_str in rows:
            try:
                item = json.loads(meta_str)
                item["_embedding"] = json.loads(emb_str)
                out.append(item)
            except Exception:
                continue
        return out

    def delete_law(self, law_name: str, article: Optional[str] = None) -> int:
        """删除某部法规或某条具体条文。"""
        conn = self._conn()
        cur = conn.cursor()
        if article:
            cur.execute(
                "DELETE FROM kb_articles WHERE law_name = ? AND article = ?",
                (law_name.strip().strip("《》"), normalize_article_id(article)),
            )
        else:
            cur.execute(
                "DELETE FROM kb_articles WHERE law_name = ?",
                (law_name.strip().strip("《》"),),
            )
        deleted = cur.rowcount
        conn.commit()
        if self._mem_conn is None:
            conn.close()
        if deleted > 0:
            self.version_counter += 1
        return deleted


# 全局知识库存储单例
kb_store = KnowledgeBaseStore()

