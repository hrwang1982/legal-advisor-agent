"""Elasticsearch 8.x 混合检索与向量生命周期管理引擎（BM25 全文倒排 + Dense Vector 稠密语义向量）。
承担法规文档分块与向量的写入、维护、更新、删除及首轮工业级混合粗排（Coarse Ranking）。
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from app.config import settings

logger = logging.getLogger(__name__)


class ESVectorStore:
    """Elasticsearch 8.x 稠密向量与全文检索集成管理器。"""

    def __init__(
        self,
        hosts: Optional[str] = None,
        index_name: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.hosts = (hosts or settings.es_hosts).strip().rstrip("/")
        self.index_name = (index_name or settings.es_index).strip()
        self.api_key = api_key or settings.es_api_key
        self._is_available: Optional[bool] = None

    def _get_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"ApiKey {self.api_key}"
        return headers

    def ping(self) -> bool:
        """检查 ES 服务是否连通且可用。"""
        if not settings.es_enabled:
            return False
        try:
            req = urllib.request.Request(
                f"{self.hosts}/",
                headers=self._get_headers(),
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                self._is_available = resp.status == 200
                return self._is_available
        except Exception as e:
            logger.debug(f"ES 连接检测失败: {e}")
            self._is_available = False
            return False

    def init_index(self, dims: int = 1024) -> bool:
        """初始化 Elasticsearch 8.x 索引与 mapping（BM25 文本 + Dense Vector HNSW 向量）。"""
        if not self.ping():
            return False

        mapping = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
            },
            "mappings": {
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "doc_id": {"type": "keyword"},
                    "law_name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                    "hierarchy": {"type": "keyword"},
                    "host_law": {"type": "keyword"},
                    "article": {"type": "keyword"},
                    "paragraph": {"type": "keyword"},
                    "chapter_path": {"type": "text"},
                    "content": {"type": "text"},
                    "chunk_type": {"type": "keyword"},
                    "parent_chunk_id": {"type": "keyword"},
                    "validity": {"type": "keyword"},
                    "effective_date": {"type": "date", "format": "yyyy-MM-dd||yyyy-MM-dd HH:mm:ss||epoch_millis"},
                    "effective_to": {"type": "date", "format": "yyyy-MM-dd||yyyy-MM-dd HH:mm:ss||epoch_millis"},
                    "content_vector": {
                        "type": "dense_vector",
                        "dims": dims,
                        "index": True,
                        "similarity": "cosine",
                        "index_options": {
                            "type": "hnsw",
                            "m": 16,
                            "ef_construction": 100,
                        },
                    },
                }
            },
        }

        url = f"{self.hosts}/{self.index_name}"
        data = json.dumps(mapping).encode("utf-8")
        try:
            req = urllib.request.Request(url, data=data, headers=self._get_headers(), method="PUT")
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                return resp.status in (200, 201)
        except urllib.error.HTTPError as e:
            if e.code == 400:
                body = e.read().decode("utf-8", errors="ignore")
                if "resource_already_exists_exception" in body:
                    return True
            logger.warning(f"ES 索引初始化异常: {e}")
            return False
        except Exception as e:
            logger.warning(f"ES 索引初始化失败: {e}")
            return False

    def index_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """通过 Bulk API 批量索引条文分段与向量数据。"""
        if not chunks or not self.ping():
            return 0

        lines: List[str] = []
        for c in chunks:
            chunk_id = c.get("chunk_id") or c.get("id") or ""
            if not chunk_id:
                continue
            action = json.dumps({"index": {"_index": self.index_name, "_id": chunk_id}})
            doc = {
                "chunk_id": chunk_id,
                "doc_id": c.get("doc_id", ""),
                "law_name": c.get("law_name", ""),
                "hierarchy": c.get("hierarchy", "法律"),
                "host_law": c.get("host_law", ""),
                "article": c.get("article", ""),
                "paragraph": c.get("paragraph", ""),
                "chapter_path": c.get("chapter_path", ""),
                "content": c.get("content", ""),
                "chunk_type": c.get("chunk_type", "article"),
                "parent_chunk_id": c.get("parent_chunk_id", ""),
                "validity": c.get("validity", "现行有效"),
                "effective_date": c.get("effective_date") or None,
                "effective_to": c.get("effective_to") or None,
                "content_vector": c.get("content_vector") or [],
            }
            # 清理空字段，防止时间格式不合法报错
            clean_doc = {}
            for k, v in doc.items():
                if v is not None:
                    if k in ("effective_date", "effective_to") and not str(v).strip():
                        continue
                    clean_doc[k] = v

            lines.append(action)
            lines.append(json.dumps(clean_doc, ensure_ascii=False))

        body = "\n".join(lines) + "\n"
        req = urllib.request.Request(
            f"{self.hosts}/_bulk",
            data=body.encode("utf-8"),
            headers=self._get_headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=15.0) as resp:
                res = json.loads(resp.read().decode("utf-8"))
                errors = res.get("errors", False)
                items = res.get("items", [])
                return len(items) if not errors else sum(1 for it in items if "error" not in it.get("index", {}))
        except Exception as e:
            logger.warning(f"ES Bulk 写入失败: {e}")
            return 0

    def delete_by_doc_id(self, doc_id: str) -> int:
        """从 ES 索引中根据文档 doc_id 物理删除其全部向量切片。"""
        if not doc_id or not self.ping():
            return 0

        url = f"{self.hosts}/{self.index_name}/_delete_by_query"
        payload = {
            "query": {
                "term": {
                    "doc_id": doc_id
                }
            }
        }
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers=self._get_headers(),
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                res = json.loads(resp.read().decode("utf-8"))
                return res.get("deleted", 0)
        except Exception as e:
            logger.warning(f"ES 按 doc_id ({doc_id}) 删除失败: {e}")
            return 0

    def delete_chunk(self, chunk_id: str) -> bool:
        """从 ES 索引中按单个 chunk_id 删除。"""
        if not chunk_id or not self.ping():
            return False

        url = f"{self.hosts}/{self.index_name}/_doc/{chunk_id}"
        try:
            req = urllib.request.Request(url, headers=self._get_headers(), method="DELETE")
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                return resp.status in (200, 204)
        except Exception as e:
            logger.warning(f"ES 删除 chunk ({chunk_id}) 失败: {e}")
            return False

    def count(self) -> int:
        """获取 ES 索引中当前的法规分块与向量总数。"""
        if not self.ping():
            return 0
        try:
            req = urllib.request.Request(
                f"{self.hosts}/{self.index_name}/_count",
                headers=self._get_headers(),
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("count", 0)
        except Exception:
            return 0

    def coarse_rank(
        self,
        query: str,
        query_vector: Optional[List[float]] = None,
        top_k: int = 15,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """执行工业级第一轮多路召回与粗排：
        利用 Elasticsearch 原生 BM25 多字段文本加权与 kNN 余弦语义向量，返回 Top-K 候选法规条文。
        """
        return self.hybrid_search(
            query=query,
            query_vector=query_vector,
            top_k=top_k,
            filters=filters,
        )

    def hybrid_search(
        self,
        query: str,
        query_vector: Optional[List[float]] = None,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """执行 BM25 全文检索 + kNN 稠密向量余弦混合检索。"""
        if not self.ping():
            return []

        search_body: Dict[str, Any] = {
            "size": top_k,
            "query": {
                "bool": {
                    "should": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": [
                                    "content^2.5",
                                    "law_name^3.5",
                                    "chapter_path^1.5",
                                    "article^2.0",
                                ],
                            }
                        }
                    ]
                }
            },
        }

        # 若提供了查询向量且 ES 支持，添加 kNN 稠密向量检索
        if query_vector:
            search_body["knn"] = {
                "field": "content_vector",
                "query_vector": query_vector,
                "k": top_k,
                "num_candidates": max(50, top_k * 5),
                "boost": 1.5,
            }

        # 过滤项（如生效状态等）
        if filters:
            filter_clauses = []
            for k, v in filters.items():
                if v:
                    filter_clauses.append({"term": {k: v}})
            if filter_clauses:
                search_body["query"]["bool"]["filter"] = filter_clauses

        req = urllib.request.Request(
            f"{self.hosts}/{self.index_name}/_search",
            data=json.dumps(search_body).encode("utf-8"),
            headers=self._get_headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                hits = (data.get("hits") or {}).get("hits") or []
                out = []
                for h in hits:
                    item = h.get("_source") or {}
                    item["_score"] = h.get("_score", 0.0)
                    out.append(item)
                return out
        except Exception as e:
            logger.warning(f"ES Hybrid Search 执行异常: {e}")
            return []


es_vector_store = ESVectorStore()
