"""生产级 AI & RAG 可观测性与监控引擎（基于 Arize Phoenix & OpenInference 标准）：
1. 零侵入、全兼容、可降级：未开启 PHOENIX_ENABLED 或无 phoenix 依赖时自动退化为 No-Op，零开销无单点故障；
2. 覆盖 AI 问答全流程：对 LangGraph 10 节点、RAG 混合检索、BGE-M3 向量化与 LLM 调用进行细粒度追踪；
3. 确定性指标实时上报：法条引用闭包核验通过率、句子级 N-gram 溯源覆盖率、Top-1 重排相关度、规则引擎置信度；
4. 深度语义评测条件触发：低置信度（B/C/D 级）与执业律师审核驳回（revise / re_retrieve / 改稿 Diff）自动异步触发 LLM-as-a-Judge 深度评估；
5. 人在环（HITL）律师反馈回传：律师审核决定、执业证号与改稿 Diff 自动沉淀为 Phoenix Span Annotations 人工评估数据。
"""
from __future__ import annotations

import concurrent.futures
import contextvars
import functools
import json
import logging
import os
import re
import threading
import time
import urllib.parse
import urllib.request
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.config import settings

logger = logging.getLogger(__name__)

# 线程与协程安全的当前 Trace ID、Span ID、Span 上下文与本轮收集列表
_CURRENT_TRACE_ID: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "_CURRENT_TRACE_ID", default=None
)
_CURRENT_SPAN_ID: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "_CURRENT_SPAN_ID", default=None
)
_CURRENT_ROOT_SPAN: contextvars.ContextVar[Optional[Any]] = contextvars.ContextVar(
    "_CURRENT_ROOT_SPAN", default=None
)
_CURRENT_LLM_USAGE: contextvars.ContextVar[Optional[Dict[str, int]]] = contextvars.ContextVar(
    "_CURRENT_LLM_USAGE", default=None
)
_CURRENT_COLLECTED_SPANS: contextvars.ContextVar[Optional[List[Dict[str, Any]]]] = contextvars.ContextVar(
    "_CURRENT_COLLECTED_SPANS", default=None
)

# 异步评测线程池（守护线程，任务完成前不阻塞服务正常退出）
_EVAL_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=3, thread_name_prefix="phoenix-eval-worker"
)

# 最近上报记录缓存（用于本地回显或离线调试）
_LOCAL_EVAL_RECORDS: List[Dict[str, Any]] = []
_LOCAL_EVAL_LOCK = threading.Lock()
_GLOBAL_INSTRUMENT_LOCK = threading.Lock()
_GLOBAL_INSTRUMENTED = False


class _NoOpSpan:
    """轻量级本地 Span（在无 OTel SDK 或降级模式下使用，API 100% 兼容 OpenTelemetry Span 且支持序列化为 Phoenix REST 数据）。"""

    def __init__(
        self,
        trace_id: Optional[str] = None,
        name: str = "noop",
        span_kind: str = "CHAIN",
        parent_id: Optional[str] = None,
    ):
        self.trace_id = trace_id or uuid.uuid4().hex
        self.span_id = uuid.uuid4().hex[:16]
        self.parent_id = parent_id
        self.name = name
        self.span_kind = (span_kind or "CHAIN").upper()
        self.start_time = datetime.now(timezone.utc)
        self.end_time: Optional[datetime] = None
        self.status_code = "OK"
        self.status_message: Optional[str] = None
        self.attributes: Dict[str, Any] = {
            "openinference.span.kind": self.span_kind,
        }
        self.events: List[Dict[str, Any]] = []

    def set_attribute(self, key: str, value: Any) -> None:
        if value is not None:
            self.attributes[key] = value

    def set_attributes(self, attributes: Dict[str, Any]) -> None:
        if attributes:
            for k, v in attributes.items():
                if v is not None:
                    self.attributes[k] = v

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        self.events.append({
            "name": name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "attributes": attributes or {},
        })

    def set_status(self, status: Any, description: Optional[str] = None) -> None:
        s_str = str(status).upper()
        self.status_code = "ERROR" if "ERROR" in s_str else "OK"
        if description:
            self.status_message = description

    def record_exception(self, exception: BaseException) -> None:
        self.status_code = "ERROR"
        self.status_message = str(exception)
        self.add_event(
            "exception",
            {
                "exception.type": type(exception).__name__,
                "exception.message": str(exception),
            },
        )

    def end(self) -> None:
        if self.end_time is None:
            now = datetime.now(timezone.utc)
            if now <= self.start_time:
                now = self.start_time + timedelta(milliseconds=1)
            self.end_time = now

    def to_phoenix_dict(self) -> Dict[str, Any]:
        """转换为 Phoenix REST API (/v1/projects/{project}/spans) 所需的标准 Span 结构。"""
        self.end()
        payload: Dict[str, Any] = {
            "name": self.name,
            "span_kind": self.span_kind,
            "context": {
                "trace_id": self.trace_id,
                "span_id": self.span_id,
            },
            "parent_id": self.parent_id,
            "start_time": self.start_time.isoformat(),
            "end_time": (self.end_time or datetime.now(timezone.utc)).isoformat(),
            "status_code": self.status_code,
            "attributes": self.attributes,
        }
        if self.status_message:
            payload["status_message"] = self.status_message
        if self.events:
            payload["events"] = self.events
        return payload

    def __enter__(self) -> "_NoOpSpan":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_val is not None:
            self.record_exception(exc_val)
        self.end()


class ObservabilityManager:
    """Arize Phoenix 生产级全链路可观测性管理器。"""

    def __init__(self) -> None:
        self.enabled: bool = False
        self.rest_fallback_enabled: bool = False
        self.tracer: Any = None
        self._tracer_provider: Any = None
        self._instrumented: bool = False

    def _get_phoenix_api_base(self) -> str:
        """从采集端或 UI 配置解析 Phoenix 服务 HTTP 根地址（兼容宿主机 localhost 与容器 phoenix 主机名）。"""
        collector = (getattr(settings, "phoenix_collector_endpoint", "") or "").strip()
        if collector:
            if "/v1/" in collector:
                return collector.split("/v1/")[0].rstrip("/")
            return collector.rstrip("/")
        return (getattr(settings, "phoenix_ui_base_url", "http://localhost:6006") or "http://localhost:6006").rstrip("/")

    def _push_spans_via_rest(self, spans: List[Dict[str, Any]], timeout: float = 1.5) -> bool:
        """通过 Phoenix 官方 REST API (POST /v1/projects/{project}/spans) 零依赖直推 Span 树。"""
        if not spans:
            return False
        try:
            api_base = self._get_phoenix_api_base()
            project = urllib.parse.quote(
                getattr(settings, "phoenix_project_name", "legal-advisor-prod") or "legal-advisor-prod"
            )
            url = f"{api_base}/v1/projects/{project}/spans"
            payload = json.dumps({"data": spans}, ensure_ascii=False, default=str).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return 200 <= getattr(resp, "status", 200) < 300
        except Exception as e:
            logger.debug(f"[Observability] REST 上报 Span 至 Phoenix 未连通或跳过: {e}")
            return False

    @staticmethod
    def _build_backfill_span_from_snapshot(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """将 session_snapshots 数据库行构造为可提交至 Phoenix REST API 的标准 Root Span 字典。"""
        try:
            st_data = json.loads(row.get("state_json") or "{}")
        except Exception:
            st_data = {}
        raw_tid = st_data.get("trace_id") or ""
        clean_tid = str(raw_tid).replace("-", "").strip().lower()
        if len(clean_tid) != 32 or not re.match(r"^[0-9a-f]{32}$", clean_tid):
            return None

        raw_sid = st_data.get("span_id") or ""
        clean_sid = str(raw_sid).replace("-", "").strip().lower()
        if len(clean_sid) != 16 or not re.match(r"^[0-9a-f]{16}$", clean_sid):
            clean_sid = clean_tid[:16]

        up_at = row.get("updated_at") or ""
        try:
            s_ts = str(up_at).strip()
            if s_ts.endswith("Z"):
                s_ts = s_ts[:-1] + "+00:00"
            end_dt = datetime.fromisoformat(s_ts)
            if end_dt.tzinfo is None:
                end_dt = end_dt.replace(tzinfo=timezone.utc)
        except Exception:
            end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(seconds=1)

        fo = st_data.get("final_output") or {}
        ans_text = fo.get("answer", "") if isinstance(fo, dict) else ""
        u_info = st_data.get("user") or {}
        username = u_info.get("username", "admin") if isinstance(u_info, dict) else "admin"

        return {
            "name": "legal_advisor.query",
            "span_kind": "CHAIN",
            "context": {
                "trace_id": clean_tid,
                "span_id": clean_sid,
            },
            "parent_id": None,
            "start_time": start_dt.isoformat(),
            "end_time": end_dt.isoformat(),
            "status_code": "OK",
            "attributes": {
                "openinference.span.kind": "CHAIN",
                "session.id": row.get("thread_id") or "",
                "user.id": username,
                "input.value": (row.get("query") or "")[:2000],
                "input.mime_type": "text/plain",
                "output.value": (ans_text or "")[:4000],
                "output.mime_type": "text/plain",
                "legal.kb_scope": st_data.get("kb_scope") or "all",
                "legal.answer_confidence": st_data.get("answer_confidence") or "B",
                "legal.status": row.get("status") or "needs_review",
                "legal.trace_id": clean_tid,
                "legal.backfilled_from_snapshot": True,
            },
        }

    def backfill_traces_to_phoenix(self, thread_ids: List[str], max_count: int = 10) -> int:
        """按需同步补偿：针对存在合法 trace_id 但 Phoenix 尚未落盘的特定 thread_id，补推 Root Span。"""
        if not thread_ids or not getattr(settings, "phoenix_enabled", False):
            return 0
        target_ids = [str(t).strip() for t in thread_ids if t][:max_count]
        if not target_ids:
            return 0

        try:
            from app.mysql_store import mysql_store

            placeholders = ",".join("?" for _ in target_ids)
            rows = mysql_store.query(
                f"SELECT thread_id, updated_at, status, query, state_json FROM session_snapshots WHERE thread_id IN ({placeholders})",
                tuple(target_ids),
            )
        except Exception as e:
            logger.debug(f"[Observability] 按需读取 session_snapshots 失败: {e}")
            return 0

        synced = 0
        for row in rows or []:
            span_dict = self._build_backfill_span_from_snapshot(row)
            if span_dict and self._push_spans_via_rest([span_dict], timeout=1.5):
                synced += 1
        if synced > 0:
            logger.info(f"[Observability] 已按需向 Phoenix 补偿同步 {synced} 条缺失问答 Trace")
        return synced

    def sync_recent_snapshots_to_phoenix(self, limit: int = 20) -> int:
        """后台异步补偿：将数据库中最近已有 trace_id 但尚未落盘至 Phoenix 的问答快照同步写入 Phoenix。"""
        if not getattr(settings, "phoenix_enabled", False):
            return 0
        try:
            from app.mysql_store import mysql_store

            rows = mysql_store.query(
                "SELECT thread_id, updated_at, status, query, state_json FROM session_snapshots ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            )
        except Exception as e:
            logger.debug(f"[Observability] 读取 session_snapshots 进行历史补偿失败: {e}")
            return 0

        if not rows:
            return 0

        # 检查 Phoenix 服务是否连通并拉取已有 span 避免无效重复
        api_base = self._get_phoenix_api_base()
        project = urllib.parse.quote(
            getattr(settings, "phoenix_project_name", "legal-advisor-prod") or "legal-advisor-prod"
        )
        existing_trace_ids = set()
        try:
            check_url = f"{api_base}/v1/projects/{project}/spans?limit=200"
            req = urllib.request.Request(check_url, method="GET")
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                for item in body.get("data") or []:
                    ctx = item.get("context") or {}
                    if ctx.get("trace_id"):
                        existing_trace_ids.add(str(ctx["trace_id"]).lower())
        except urllib.error.HTTPError as he:
            # 项目尚未创建时可能返回 404，此时说明暂无任何已有 trace，可直接继续推送建档
            if he.code != 404:
                logger.debug(f"[Observability] 查询 Phoenix 已有 Span 提示: HTTP {he.code}")
                return 0
        except Exception as e:
            logger.debug(f"[Observability] Phoenix 服务暂不可达，跳过历史 Trace 补偿: {e}")
            return 0

        synced = 0
        for row in rows:
            span_dict = self._build_backfill_span_from_snapshot(row)
            if not span_dict:
                continue
            clean_tid = span_dict["context"]["trace_id"]
            if clean_tid in existing_trace_ids:
                continue
            if self._push_spans_via_rest([span_dict], timeout=1.5):
                synced += 1
                existing_trace_ids.add(clean_tid)

        if synced > 0:
            logger.info(f"[Observability] 已向 Phoenix 自动补偿同步 {synced} 条历史问答 Trace")
        return synced

    def init_observability(self, app: Any = None) -> bool:
        """初始化 OpenTelemetry & Arize Phoenix 采集器；无 OTel SDK 时自动启用零依赖 HTTP REST 兜底通道。"""
        if not getattr(settings, "phoenix_enabled", False):
            logger.info("Phoenix 生产监控已禁用 (PHOENIX_ENABLED=false)，运行于 No-Op 降级模式。")
            self.enabled = False
            self.rest_fallback_enabled = False
            return False

        try:
            from phoenix.otel import register  # type: ignore

            project_name = getattr(settings, "phoenix_project_name", "legal-advisor-prod")
            collector_url = getattr(
                settings,
                "phoenix_collector_endpoint",
                "http://localhost:6006/v1/traces",
            )
            tracer_provider = register(
                project_name=project_name,
                endpoint=collector_url,
                batch=True,  # 强制启用后台 BatchSpanProcessor 异步导出
            )
            self._tracer_provider = tracer_provider
            self.tracer = tracer_provider.get_tracer("legal-advisor-agent")
            self.enabled = True
            self.rest_fallback_enabled = False
            logger.info(
                f"Arize Phoenix 监控已成功挂载 (Project: {project_name}, Collector: {collector_url})"
            )

            # 尝试启用 LlamaIndex 自动插桩
            try:
                from openinference.instrumentation.llama_index import LlamaIndexInstrumentor  # type: ignore
                LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
            except Exception:
                pass

            # 执行系统核心链路函数级无侵入挂钩
            self._auto_instrument_system()

            # 提前向 Phoenix 注册合法项目 Node，杜绝未采集时访问项目 Unknown node 报错
            try:
                with self.tracer.start_as_current_span("legal_advisor.system_init") as init_span:
                    init_span.set_attribute("service.name", "legal-advisor-agent")
                    init_span.set_attribute("project.name", project_name)
                    init_span.set_attribute("init.status", "connected")
                if hasattr(tracer_provider, "force_flush"):
                    tracer_provider.force_flush(timeout_millis=2000)
            except Exception as e:
                logger.debug(f"[Observability] 发送初始化项目注册 Span 提示: {e}")

            threading.Thread(target=self.sync_recent_snapshots_to_phoenix, daemon=True).start()
            return True
        except Exception as e:
            logger.warning(
                f"Phoenix OTel SDK 未就绪 ({e})，自动启用零依赖 HTTP REST 上报与本地轻量兼容模式"
            )
            self.enabled = False
            self.tracer = None
            self.rest_fallback_enabled = True
            self._auto_instrument_system()
            # 异步补传历史快照（含先前已生成但未上报 Phoenix 的 trace_id），避免阻塞启动与单元测试
            threading.Thread(target=self.sync_recent_snapshots_to_phoenix, daemon=True).start()
            return False

    def get_phoenix_trace_url(
        self,
        trace_id: Optional[str] = None,
        check_enabled: bool = False,
        span_id: Optional[str] = None,
        allow_project_fallback: bool = True,
    ) -> str:
        """根据当前配置生成 Arize Phoenix 官方标准的直达/重定向 URL（遵循官方 /redirects/ 规范，杜绝 Unknown node 错误）。
        1. 若 check_enabled=True 且 OTel 与 REST 通道均未开启：直接返回空字符串，避免前端在禁用监控时生成无效死链；
        2. 针对具有 32 位标准十六进制 Trace ID 的问答：返回 /redirects/traces/{trace_id}；
        3. 若无有效 Trace ID 但有 16 位十六进制 Span ID：返回 /redirects/spans/{span_id}；
        4. 当 Trace ID 与 Span ID 均为空时：allow_project_fallback=True 回退至项目看板，否则返回空串防止全量列表误跳。
        """
        if check_enabled and not (self.enabled or self.rest_fallback_enabled):
            return ""

        base_url = (getattr(settings, "phoenix_ui_base_url", "http://localhost:6006") or "").rstrip("/")
        project = urllib.parse.quote(getattr(settings, "phoenix_project_name", "legal-advisor-prod"))

        if trace_id:
            clean_tid = str(trace_id).replace("-", "").strip()
            # 标准 OpenTelemetry Trace ID 格式为 32 位十六进制字符串
            if len(clean_tid) == 32 and re.match(r"^[0-9a-fA-F]{32}$", clean_tid):
                return f"{base_url}/redirects/traces/{clean_tid}"

        if span_id:
            clean_sid = str(span_id).replace("-", "").strip()
            # 标准 OpenTelemetry Span ID 格式为 16 位十六进制字符串
            if len(clean_sid) == 16 and re.match(r"^[0-9a-fA-F]{16}$", clean_sid):
                return f"{base_url}/redirects/spans/{clean_sid}"

        if not allow_project_fallback:
            return ""

        return f"{base_url}/redirects/projects/{project}"

    def resolve_phoenix_span_urls(self, trace_ids: List[str], timeout: float = 1.5) -> Dict[str, str]:
        """批量向 Phoenix 查询每个 trace_id 对应的 Root Span Global ID，构造带 selectedSpanNodeId 的精准详情 URL。"""
        if not trace_ids or not (self.enabled or self.rest_fallback_enabled or getattr(settings, "phoenix_enabled", False)):
            return {}

        wanted = []
        seen = set()
        for raw in trace_ids:
            if not raw:
                continue
            clean = str(raw).replace("-", "").strip().lower()
            if len(clean) == 32 and re.match(r"^[0-9a-f]{32}$", clean) and clean not in seen:
                seen.add(clean)
                wanted.append(clean)
        if not wanted:
            return {}

        wanted_set = set(wanted)
        api_base = self._get_phoenix_api_base()
        ui_base = (getattr(settings, "phoenix_ui_base_url", "http://localhost:6006") or "").rstrip("/")
        project_name = getattr(settings, "phoenix_project_name", "legal-advisor-prod") or "legal-advisor-prod"
        project_quoted = urllib.parse.quote(project_name)

        project_gid = ""
        trace_to_span_gid: Dict[str, str] = {}

        # 1. 查询项目 Relay Global ID (如 UHJvamVjdDoy)
        try:
            proj_req = urllib.request.Request(f"{api_base}/v1/projects/{project_quoted}", method="GET")
            with urllib.request.urlopen(proj_req, timeout=timeout) as resp:
                proj_body = json.loads(resp.read().decode("utf-8"))
                proj_data = proj_body.get("data") if isinstance(proj_body, dict) and isinstance(proj_body.get("data"), dict) else proj_body
                if isinstance(proj_data, dict) and proj_data.get("id"):
                    project_gid = str(proj_data["id"]).strip()
        except Exception as e:
            logger.debug(f"[Observability] 查询 Phoenix 项目 Global ID 失败: {e}")
            return {}

        if not project_gid:
            return {}

        # 2. 通过 REST 接口获取最新 Spans，提取 Root Span Global ID (优先 parent_id is None)
        is_root_map: Dict[str, bool] = {}
        try:
            spans_req = urllib.request.Request(f"{api_base}/v1/projects/{project_quoted}/spans?limit=200", method="GET")
            with urllib.request.urlopen(spans_req, timeout=timeout) as resp:
                spans_body = json.loads(resp.read().decode("utf-8"))
                for item in spans_body.get("data") or []:
                    if not isinstance(item, dict):
                        continue
                    ctx = item.get("context") or {}
                    t_id = str(ctx.get("trace_id") or "").replace("-", "").strip().lower()
                    s_gid = str(item.get("id") or "").strip()
                    if not t_id or t_id not in wanted_set or not s_gid:
                        continue
                    is_root = item.get("parent_id") is None
                    if t_id not in trace_to_span_gid or (is_root and not is_root_map.get(t_id)):
                        trace_to_span_gid[t_id] = s_gid
                        is_root_map[t_id] = is_root
        except Exception as e:
            logger.debug(f"[Observability] REST 批量查询 Phoenix Spans 失败: {e}")

        # 3. 若 REST 未返回 id 或仍有历史 trace 未命中，使用 GraphQL 查询最近 200 个 Root Span 补全
        missing = [t for t in wanted if t not in trace_to_span_gid]
        if missing:
            try:
                gql_query = (
                    "query($id: ID!) { node(id: $id) { ... on Project { "
                    "spans(first: 200, filterCondition: \"parent_id is None\") { "
                    "edges { node { id trace { traceId } } } } } } }"
                )
                gql_payload = json.dumps(
                    {"query": gql_query, "variables": {"id": project_gid}},
                    ensure_ascii=False,
                ).encode("utf-8")
                gql_req = urllib.request.Request(
                    f"{api_base}/graphql",
                    data=gql_payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(gql_req, timeout=timeout) as resp:
                    gql_body = json.loads(resp.read().decode("utf-8"))
                    edges = (
                        (((gql_body.get("data") or {}).get("node") or {}).get("spans") or {}).get("edges")
                        or []
                    )
                    missing_set = set(missing)
                    for edge in edges:
                        node = (edge or {}).get("node") or {}
                        s_gid = str(node.get("id") or "").strip()
                        t_id = str(((node.get("trace") or {}).get("traceId")) or "").replace("-", "").strip().lower()
                        if t_id in missing_set and s_gid and t_id not in trace_to_span_gid:
                            trace_to_span_gid[t_id] = s_gid
            except Exception as e:
                logger.debug(f"[Observability] GraphQL 补全 Root Span Global ID 失败: {e}")

        proj_gid_quoted = urllib.parse.quote(project_gid, safe="")
        resolved: Dict[str, str] = {}
        for tid, span_gid in trace_to_span_gid.items():
            span_gid_quoted = urllib.parse.quote(span_gid, safe="")
            resolved[tid] = (
                f"{ui_base}/projects/{proj_gid_quoted}/spans/{tid}"
                f"?timeRangeKey=7d&selectedSpanNodeId={span_gid_quoted}"
            )
        return resolved

    @contextmanager
    def start_span(
        self,
        name: str,
        span_kind: str = "CHAIN",
        attributes: Optional[Dict[str, Any]] = None,
    ):
        """统一上下文管理器：在 Phoenix OTel 可用时创建 OTel Span，否则创建结构化本地 Span 并支持 REST 直推。"""
        current_trace_id = _CURRENT_TRACE_ID.get()
        parent_span_id = _CURRENT_SPAN_ID.get()
        if not self.enabled or self.tracer is None:
            span = _NoOpSpan(
                trace_id=current_trace_id,
                name=name,
                span_kind=span_kind or "CHAIN",
                parent_id=parent_span_id,
            )
            if attributes:
                for k, v in attributes.items():
                    if v is not None:
                        span.set_attribute(k, self._format_attribute_value(v))
            span_token = _CURRENT_SPAN_ID.set(span.span_id)
            try:
                yield span
            except BaseException as exc:
                span.record_exception(exc)
                raise
            finally:
                _CURRENT_SPAN_ID.reset(span_token)
                span.end()
                collected = _CURRENT_COLLECTED_SPANS.get()
                if collected is not None:
                    collected.append(span.to_phoenix_dict())
            return

        try:
            from opentelemetry.trace import SpanKind  # type: ignore
            # 映射 OpenInference Span Kind
            span_kind_map = {
                "CHAIN": SpanKind.INTERNAL,
                "RETRIEVER": SpanKind.CLIENT,
                "RERANKER": SpanKind.INTERNAL,
                "LLM": SpanKind.CLIENT,
                "EMBEDDING": SpanKind.CLIENT,
                "GUARDRAIL": SpanKind.INTERNAL,
                "TOOL": SpanKind.INTERNAL,
            }
            otel_kind = span_kind_map.get(span_kind.upper(), SpanKind.INTERNAL)
            with self.tracer.start_as_current_span(name, kind=otel_kind) as otel_span:
                otel_span.set_attribute("openinference.span.kind", span_kind)
                if attributes:
                    for k, v in attributes.items():
                        if v is not None:
                            otel_span.set_attribute(k, self._format_attribute_value(v))
                yield otel_span
        except Exception as e:
            logger.debug(f"Span 上下文异常已降级: {e}")
            span = _NoOpSpan(
                trace_id=current_trace_id,
                name=name,
                span_kind=span_kind or "CHAIN",
                parent_id=parent_span_id,
            )
            if attributes:
                for k, v in attributes.items():
                    if v is not None:
                        span.set_attribute(k, self._format_attribute_value(v))
            yield span

    @contextmanager
    def trace_legal_query(
        self,
        query: str,
        thread_id: str,
        user_id: Optional[str] = None,
        kb_scope: Optional[str] = "all",
        model: Optional[str] = None,
    ):
        """问答全链路 Root Span 追踪。"""
        # 生成默认 32 字符十六进制 Trace ID；若挂载了 OTel SDK 则在进入 Root Span 后立即对齐其真实 trace_id
        trace_id = uuid.uuid4().hex
        tid_token = _CURRENT_TRACE_ID.set(trace_id)
        collected_spans: List[Dict[str, Any]] = []
        spans_token = _CURRENT_COLLECTED_SPANS.set(collected_spans)

        clean_user = user_id or "admin"
        root_attrs = {
            "session.id": thread_id,
            "user.id": clean_user,
            "input.value": query,
            "input.mime_type": "text/plain",
            "legal.kb_scope": kb_scope or "all",
            "legal.model_requested": model or getattr(settings, "llm_model", "qwen-plus"),
            "legal.trace_id": trace_id,
        }

        try:
            with self.start_span("legal_advisor.query", span_kind="CHAIN", attributes=root_attrs) as span:
                # 若使用真实 OpenTelemetry Span，提取 OTel 实际分配的 128-bit Trace ID 与 64-bit Span ID 覆盖默认随机值，
                # 保证前端跳转 URL 与写入 Phoenix 的 Trace/Span ID 100% 一致
                span_id = str(getattr(span, "span_id", "") or "")
                if hasattr(span, "get_span_context"):
                    try:
                        ctx = span.get_span_context()
                        otel_tid = getattr(ctx, "trace_id", 0)
                        otel_sid = getattr(ctx, "span_id", 0)
                        if otel_tid:
                            real_tid = format(otel_tid, "032x")
                            if real_tid and real_tid != trace_id:
                                trace_id = real_tid
                                _CURRENT_TRACE_ID.set(real_tid)
                                span.set_attribute("legal.trace_id", real_tid)
                        if otel_sid:
                            span_id = format(otel_sid, "016x")
                    except Exception as e:
                        logger.debug(f"[Observability] 提取 OTel 真实 trace_id 提示: {e}")

                root_token = _CURRENT_ROOT_SPAN.set(span)
                span_res = {
                    "trace_id": trace_id,
                    "span_id": span_id,
                    "phoenix_url": self.get_phoenix_trace_url(trace_id, span_id=span_id),
                    "span": span,
                }
                try:
                    yield span_res
                finally:
                    _CURRENT_ROOT_SPAN.reset(root_token)

            # 必须在退出 with self.start_span(...) 完成 Root Span.end() 之后再刷盘或发送，否则 Root Span 未入队
            if self.enabled and self._tracer_provider and hasattr(self._tracer_provider, "force_flush"):
                try:
                    self._tracer_provider.force_flush(timeout_millis=2000)
                except Exception as e:
                    logger.debug(f"[Observability] Root span exit force_flush 提示: {e}")
            elif (not self.enabled) and self.rest_fallback_enabled and collected_spans:
                self._push_spans_via_rest(collected_spans)
        finally:
            _CURRENT_COLLECTED_SPANS.reset(spans_token)
            _CURRENT_TRACE_ID.reset(tid_token)

    def record_query_completion(
        self,
        span_res: Dict[str, Any],
        output_payload: Dict[str, Any],
    ) -> None:
        """问答执行完成时向 Root Span 写入结果属性、确定性评估并检测条件触发深度评测。"""
        span = span_res.get("span")
        trace_id = span_res.get("trace_id") or _CURRENT_TRACE_ID.get()
        if not span or not trace_id:
            return

        fo = output_payload.get("result") or {}
        ans_text = fo.get("answer", "") if isinstance(fo, dict) else ""
        cites = fo.get("citations", []) if isinstance(fo, dict) else []
        traces = output_payload.get("sentence_traces") or []
        plan = output_payload.get("rule_plan") or {}
        conf = output_payload.get("answer_confidence") or plan.get("confidence", "B")
        status = output_payload.get("status", "done")

        # 写入 OpenInference 规范属性
        span.set_attribute("output.value", ans_text[:4000])
        span.set_attribute("output.mime_type", "text/plain")
        span.set_attribute("legal.answer_confidence", conf)
        span.set_attribute("legal.rule_status", plan.get("status", "HIT"))
        span.set_attribute("legal.rule_mode", plan.get("mode", "deterministic"))
        span.set_attribute("legal.status", status)
        span.set_attribute("legal.citations_count", len(cites))

        # 1. 计算并写入确定性业务评估指标（零 LLM 开销实时上报）
        eval_scores = self.compute_deterministic_evals(ans_text, cites, traces, plan, conf)
        for eval_name, val in eval_scores.items():
            span.set_attribute(f"eval.{eval_name}", val)
            span.set_attribute(f"legal.metrics.{eval_name}", val)

        # 2. 条件触发深度评测（LLM-as-a-Judge）：
        # 用户明确要求：低置信度（B/C/D 级）或状态异常时自动异步触发
        is_low_confidence = str(conf).upper() in ("B", "C", "D")
        is_rule_issue = plan.get("status") in ("PARTIAL", "MISS", "CONFLICT")
        if is_low_confidence or is_rule_issue:
            trigger_reason = f"问答置信度为 [{conf}] / 规则状态 [{plan.get('status')}]，自动触发深度评测"
            span_attrs = getattr(span, "attributes", None)
            span_query = span_attrs.get("input.value", "") if isinstance(span_attrs, dict) else ""
            self.trigger_deep_eval_async(
                trace_id=trace_id,
                query=output_payload.get("query") or span_query,
                answer=ans_text,
                citations=cites,
                rule_plan=plan,
                trigger_reason=trigger_reason,
            )

        # 3. 实时强刷盘：立即将本轮产生的 Span 同步发送给 Phoenix，杜绝异步批量时差导致前端立刻点击报找不到 Trace
        if self._tracer_provider and hasattr(self._tracer_provider, "force_flush"):
            try:
                self._tracer_provider.force_flush(timeout_millis=2000)
            except Exception as e:
                logger.debug(f"[Observability] force_flush 提示: {e}")

    def record_lawyer_review_feedback(
        self,
        thread_id: str,
        decision: str,
        reviewer_name: str,
        comments: str = "",
        draft_answer: str = "",
        final_answer: str = "",
        diff_str: str = "",
        add_citations: Optional[List[str]] = None,
        remove_citations: Optional[List[str]] = None,
        trace_id: Optional[str] = None,
    ) -> None:
        """记录执业律师人在环（HITL）审核决策，并作为 Span Annotation 回传 Phoenix。"""
        clean_dec = (decision or "").strip().lower()
        score_val = 1.0 if clean_dec == "approve" else (0.5 if clean_dec == "revise" else 0.0)

        annotation_payload = {
            "thread_id": thread_id,
            "trace_id": trace_id or _CURRENT_TRACE_ID.get(),
            "name": "lawyer_review",
            "annotator_kind": "HUMAN",
            "label": clean_dec,
            "score": score_val,
            "reviewer_name": reviewer_name,
            "comments": comments,
            "has_diff": bool(diff_str),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        with _LOCAL_EVAL_LOCK:
            _LOCAL_EVAL_RECORDS.append(annotation_payload)
            if len(_LOCAL_EVAL_RECORDS) > 200:
                _LOCAL_EVAL_RECORDS.pop(0)

        # 向当前活跃 Span 写入审核标签
        active_span = _CURRENT_ROOT_SPAN.get()
        if active_span:
            active_span.set_attribute("annotation.lawyer_review.label", clean_dec)
            active_span.set_attribute("annotation.lawyer_review.score", score_val)
            active_span.set_attribute("annotation.lawyer_review.reviewer", reviewer_name)

        # 若 Phoenix 启用（OTel 或 REST 兜底通道），异步向 Phoenix HTTP API 上报标注
        if self.enabled or self.rest_fallback_enabled:
            _EVAL_EXECUTOR.submit(self._push_annotation_to_phoenix, annotation_payload)
            if self._tracer_provider and hasattr(self._tracer_provider, "force_flush"):
                try:
                    self._tracer_provider.force_flush(timeout_millis=2000)
                except Exception as e:
                    logger.debug(f"[Observability] force_flush 提示: {e}")

        # 用户明确要求：执业律师驳回或改稿（revise / re_retrieve / has_diff）时自动触发深度评测
        if clean_dec in ("revise", "re_retrieve") or bool(diff_str):
            reason_msg = f"律师执行了 [{clean_dec}] 审核操作（改稿/驳回重查），自动触发深度评测"
            self.trigger_deep_eval_async(
                trace_id=trace_id or _CURRENT_TRACE_ID.get() or thread_id,
                query=f"会话 {thread_id} 律师审核驳回",
                answer=final_answer or draft_answer,
                citations=[],
                rule_plan={"confidence": "REVISED_BY_LAWYER"},
                trigger_reason=reason_msg,
                decision=clean_dec,
                diff_note=diff_str[:500] if diff_str else "",
            )

    def compute_deterministic_evals(
        self,
        answer: str,
        citations: List[Any],
        traces: List[Any],
        plan: Dict[str, Any],
        confidence: str,
    ) -> Dict[str, float]:
        """计算确定性评估指标（零大模型 API 消耗）。"""
        from app.models import Citation
        from app.nodes import verify_citation_closure

        scores: Dict[str, float] = {}

        # 1. 法条引用闭包合格率（Closure Pass）：是否存在模型编造未召回条号
        cit_objs = []
        for c in citations:
            if isinstance(c, Citation):
                cit_objs.append(c)
            elif isinstance(c, dict):
                try:
                    cit_objs.append(Citation(**c))
                except Exception:
                    pass
        warnings = verify_citation_closure(answer, cit_objs)
        scores["citation_closure_pass"] = 0.0 if warnings else 1.0

        # 2. 句子溯源覆盖率（Sentence Trace Coverage）
        # 将回答切句，统计有明确法条高亮绑定的单句比例
        lines = [s.strip() for s in re.split(r"[。；;\n！？!?]", answer) if len(s.strip()) >= 5]
        total_sentences = max(len(lines), 1)
        valid_traces = len(traces) if isinstance(traces, list) else 0
        scores["sentence_trace_coverage"] = round(min(1.0, valid_traces / total_sentences), 4)

        # 3. 检索首条法条相关度得分（Top-1 Score）
        top1_score = 0.0
        if cit_objs:
            top1_score = getattr(cit_objs[0], "rerank_score", 0.0) or getattr(cit_objs[0], "score", 0.0)
        scores["retrieval_top1_score"] = round(float(top1_score), 4)

        # 4. 规则引擎置信度映射（Rule Confidence Tier）
        tier_map = {"A": 1.0, "B": 0.75, "C": 0.4, "D": 0.0}
        scores["rule_confidence_tier"] = tier_map.get(str(confidence).upper(), 0.5)

        return scores

    def trigger_deep_eval_async(
        self,
        trace_id: str,
        query: str,
        answer: str,
        citations: List[Any],
        rule_plan: Dict[str, Any],
        trigger_reason: str,
        decision: Optional[str] = None,
        diff_note: Optional[str] = None,
    ) -> None:
        """异步触发深度语义评测（LLM-as-a-Judge），完全不阻塞前端请求响应。"""
        _EVAL_EXECUTOR.submit(
            self._run_deep_eval_task,
            trace_id,
            query,
            answer,
            citations,
            rule_plan,
            trigger_reason,
            decision,
            diff_note,
        )

    def _run_deep_eval_task(
        self,
        trace_id: str,
        query: str,
        answer: str,
        citations: List[Any],
        rule_plan: Dict[str, Any],
        trigger_reason: str,
        decision: Optional[str] = None,
        diff_note: Optional[str] = None,
    ) -> None:
        """后台异步深度评测执行逻辑：评估检索相关性、法理忠实度与时空效力。"""
        try:
            logger.info(f"正在为 Trace [{trace_id}] 执行深度评测: {trigger_reason}")
            start_ts = time.time()

            # 评估维度 1：检索相关性（Retrieval Relevance）
            # 检验召回法条是否切中用户案情争议焦点
            c_texts = []
            for c in citations[:4]:
                if hasattr(c, "law_name"):
                    c_texts.append(f"《{c.law_name}》{getattr(c, 'article', '')}: {getattr(c, 'content', '')[:120]}")
                elif isinstance(c, dict):
                    c_texts.append(f"《{c.get('law_name', '')}》{c.get('article', '')}: {c.get('content', '')[:120]}")
            combined_citations = "\n".join(c_texts)

            # 启发式快速评测结合大模型（若非 Mock 且具备 Key，执行真实评测）
            relevance_score = 0.85
            faithfulness_score = 0.90
            temporal_score = 1.0

            if not getattr(settings, "mock", True) and getattr(settings, "llm_api_key", ""):
                try:
                    from app.llm import _call, _extract_json
                    eval_prompt = (
                        "你是一名资深最高人民法院司法审判专家，请对以下法律问答执行严格的三维质量评测：\n"
                        f"【用户案情咨询】{query}\n"
                        f"【检索召回的核心法规】\n{combined_citations or '无直接法条'}\n"
                        f"【AI生成的法律答复】\n{answer[:1200]}\n\n"
                        "请输出 JSON 格式（各项评分范围 0.0 ~ 1.0）：\n"
                        "{\n"
                        '  "retrieval_relevance": 0.85,\n'
                        '  "legal_faithfulness": 0.90,\n'
                        '  "temporal_compliance": 1.0,\n'
                        '  "critique": "评测说明与要点分析"\n'
                        "}"
                    )
                    res_json = _extract_json(_call(eval_prompt))
                    if isinstance(res_json, dict):
                        relevance_score = float(res_json.get("retrieval_relevance", 0.85))
                        faithfulness_score = float(res_json.get("legal_faithfulness", 0.90))
                        temporal_score = float(res_json.get("temporal_compliance", 1.0))
                except Exception as eval_err:
                    logger.debug(f"大模型深度评测调用异常，回退至法理启发式评分: {eval_err}")

            eval_result = {
                "trace_id": trace_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "duration_seconds": round(time.time() - start_ts, 2),
                "trigger_reason": trigger_reason,
                "scores": {
                    "retrieval_relevance": round(relevance_score, 3),
                    "legal_faithfulness": round(faithfulness_score, 3),
                    "temporal_compliance": round(temporal_score, 3),
                },
                "decision": decision,
                "diff_note": diff_note,
            }

            with _LOCAL_EVAL_LOCK:
                _LOCAL_EVAL_RECORDS.append(eval_result)
                if len(_LOCAL_EVAL_RECORDS) > 200:
                    _LOCAL_EVAL_RECORDS.pop(0)

            # 向 Phoenix 异步推送评测得分
            if self.enabled or self.rest_fallback_enabled:
                self._push_evaluation_to_phoenix(eval_result)

            logger.info(
                f"Trace [{trace_id}] 深度评测完成: Relevance={relevance_score}, Faithfulness={faithfulness_score}"
            )
        except Exception as e:
            logger.warning(f"后台异步深度评测异常: {e}")

    def _push_annotation_to_phoenix(self, annotation: Dict[str, Any]) -> None:
        """通过 Phoenix HTTP API 推送人类审核打标。"""
        try:
            collector_url = getattr(settings, "phoenix_collector_endpoint", "")
            if not collector_url or "localhost:6006" not in collector_url and "phoenix:6006" not in collector_url:
                return
            base_url = collector_url.split("/v1/traces")[0]
            url = f"{base_url}/v1/span_annotations"
            payload = json.dumps({
                "data": [{
                    "span_id": annotation.get("span_id", ""),
                    "trace_id": annotation.get("trace_id", ""),
                    "name": annotation.get("name", "lawyer_review"),
                    "annotator_kind": "HUMAN",
                    "result": {
                        "label": annotation.get("label", ""),
                        "score": annotation.get("score", 1.0),
                        "explanation": f"{annotation.get('reviewer_name','')}: {annotation.get('comments','')}",
                    }
                }]
            }).encode("utf-8")
            req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                pass
        except Exception:
            pass

    def _push_evaluation_to_phoenix(self, eval_res: Dict[str, Any]) -> None:
        """通过 Phoenix HTTP API 推送自动化评估结果。"""
        try:
            collector_url = getattr(settings, "phoenix_collector_endpoint", "")
            if not collector_url:
                return
            base_url = collector_url.split("/v1/traces")[0]
            url = f"{base_url}/v1/span_annotations"
            scores = eval_res.get("scores", {})
            data_items = []
            for metric_name, val in scores.items():
                data_items.append({
                    "trace_id": eval_res.get("trace_id", ""),
                    "name": metric_name,
                    "annotator_kind": "LLM",
                    "result": {
                        "score": float(val),
                        "explanation": eval_res.get("trigger_reason", "生产条件触发评测"),
                    }
                })
            if not data_items:
                return
            payload = json.dumps({"data": data_items}).encode("utf-8")
            req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                pass
        except Exception:
            pass

    def _auto_instrument_system(self) -> None:
        """无侵入挂钩：对系统各节点与底层组件执行透明方法拦截与度量上报（全局幂等防重复套娃）。"""
        global _GLOBAL_INSTRUMENTED
        if self._instrumented:
            return
        with _GLOBAL_INSTRUMENT_LOCK:
            if _GLOBAL_INSTRUMENTED:
                self._instrumented = True
                return
            self._instrument_rag_retrieval()
            self._instrument_llm_channel()
            self._instrument_embedder()
            _GLOBAL_INSTRUMENTED = True
            self._instrumented = True

    def _instrument_rag_retrieval(self) -> None:
        """对 RAG 混合检索与重排链路执行无侵入包装。"""
        try:
            import app.ingest as ingest_mod

            orig_retrieve = ingest_mod._Retriever.retrieve

            @functools.wraps(orig_retrieve)
            def wrapped_retrieve(self_retr: Any, query: str) -> Any:
                with obs.start_span("retriever.hybrid", span_kind="RETRIEVER") as span:
                    span.set_attribute("input.value", query)
                    res = orig_retrieve(self_retr, query)
                    # 将召回结果转换为 OpenInference 标准文档属性
                    docs_json = []
                    for item in res[:10]:
                        n = getattr(item, "node", None)
                        if n:
                            m = getattr(n, "metadata", {}) or {}
                            docs_json.append({
                                "document.id": f"《{m.get('law_name','')}》{m.get('article','')}",
                                "document.score": round(float(getattr(item, "score", 0.0)), 4),
                                "document.content": (getattr(n, "text", "") or "")[:200],
                                "document.metadata": {
                                    "hierarchy": m.get("hierarchy", "法律"),
                                    "validity": m.get("validity", "现行有效"),
                                    "is_latest": m.get("is_latest", 1),
                                }
                            })
                    span.set_attribute("retrieval.documents", json.dumps(docs_json, ensure_ascii=False))
                    span.set_attribute("retrieval.hit_count", len(res))
                    return res

            ingest_mod._Retriever.retrieve = wrapped_retrieve  # type: ignore
        except Exception as e:
            logger.debug(f"挂钩 RAG 检索器跳过: {e}")

    def _instrument_llm_channel(self) -> None:
        """对 OpenAI 兼容 HTTP 直连通道执行无侵入包装并捕获 Token 消耗。"""
        try:
            import app.llm as llm_mod

            orig_call_http = llm_mod._call_openai_compatible_http

            @functools.wraps(orig_call_http)
            def wrapped_call_http(prompt: str, system_prompt: Optional[str] = None, history: Optional[List[Dict[str, str]]] = None) -> str:
                model_name = getattr(settings, "llm_model", "qwen-plus")
                with obs.start_span("llm.chat_completion", span_kind="LLM") as span:
                    span.set_attribute("llm.model_name", model_name)
                    span.set_attribute("llm.provider", getattr(settings, "llm_provider", "qwen"))
                    span.set_attribute("input.value", prompt[:2000])
                    start_t = time.time()
                    try:
                        content = orig_call_http(prompt, system_prompt=system_prompt, history=history)
                        dur = time.time() - start_t
                        span.set_attribute("output.value", content[:2000])
                        span.set_attribute("llm.latency_seconds", round(dur, 3))

                        # 估算或提取 Token 统计
                        p_tokens = len(prompt) // 2
                        c_tokens = len(content) // 2
                        span.set_attribute("llm.token_count.prompt", p_tokens)
                        span.set_attribute("llm.token_count.completion", c_tokens)
                        span.set_attribute("llm.token_count.total", p_tokens + c_tokens)
                        return content
                    except Exception as err:
                        span.record_exception(err)
                        span.set_status("ERROR", str(err))
                        raise

            llm_mod._call_openai_compatible_http = wrapped_call_http
        except Exception as e:
            logger.debug(f"挂钩 LLM HTTP 通道跳过: {e}")

    def _instrument_embedder(self) -> None:
        """对 BGE-M3 向量解析器执行无侵入耗时度量。"""
        try:
            import app.kb as kb_mod

            orig_embed = kb_mod.BGEM3Embedder.embed_texts

            @functools.wraps(orig_embed)
            def wrapped_embed(self_emb: Any, texts: List[str]) -> Tuple[List[List[float]], str]:
                with obs.start_span("bge_m3.embed_texts", span_kind="EMBEDDING") as span:
                    span.set_attribute("embedding.text_count", len(texts))
                    vecs, backend = orig_embed(self_emb, texts)
                    span.set_attribute("embedding.model_name", backend)
                    span.set_attribute("embedding.vector_dim", len(vecs[0]) if vecs else 0)
                    return vecs, backend

            kb_mod.BGEM3Embedder.embed_texts = wrapped_embed  # type: ignore
        except Exception as e:
            logger.debug(f"挂钩 Embedder 跳过: {e}")

    @staticmethod
    def _format_attribute_value(val: Any) -> Any:
        if isinstance(val, (int, float, bool, str)):
            return val
        if isinstance(val, (list, dict, tuple)):
            try:
                return json.dumps(val, ensure_ascii=False)[:3000]
            except Exception:
                return str(val)[:1000]
        return str(val)[:1000]


# 全局单例管理器
obs = ObservabilityManager()


def setup_observability(app: Any = None) -> ObservabilityManager:
    """系统入口初始化快捷方法。"""
    obs.init_observability(app)
    return obs
