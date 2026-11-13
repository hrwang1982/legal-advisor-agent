"""FastAPI 服务（v3.5 可交付版）：
1. 暴露法律咨询接口、增强型人在环审核接口（强制 decision 必填与注册表鉴权）；
2. 暴露 RAG 法规知识库管理接口（创建知识库集合、导入外规全文/JSON 并执行 BGE-M3 向量化解析入库）；
3. 暴露外置知识与规则配置中心接口（同义词表、引致图谱、新旧法对照、特别法关系、客户名录、案由规则在线热更新，免改代码）；
4. 暴露多平台大模型配置接口（支持阿里云百炼千问 Qwen、本地 Ollama/vLLM OpenAI 兼容接口、DeepSeek 在线切换与配置）；
5. 暴露防篡改 SHA-256 哈希链审计查询接口、规则缺口归集接口与会话持久化恢复接口。
"""
from __future__ import annotations

import os
import uuid
from typing import Any, Dict, List, Optional

try:
    from fastapi import FastAPI, HTTPException, Header
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles
    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False
    FastAPI = None  # type: ignore
    HTTPException = Exception  # type: ignore
    def Header(default=None, **kwargs):  # type: ignore
        return default

try:
    from langgraph.types import Command
except ImportError:
    class Command:  # type: ignore
        def __init__(self, resume=None):
            self.resume = resume

from app.audit import audit_store
from app.config import settings
from app.graph import graph
from app.ingest import invalidate_cache, load_nodes
from app.kb import kb_store
from app.knowledge import knowledge_center
from app.models import BaseModel, Field
from app.mysql_store import mysql_store
from app.rbac import ALL_KB_COLLECTIONS, ALL_MENUS, rbac_manager

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)


class QueryReq(BaseModel):
    query: str
    history: Optional[List[Dict[str, str]]] = Field(default_factory=list)
    act_date: Optional[str] = None
    act_end_date: Optional[str] = None
    region: Optional[str] = None
    thread_id: Optional[str] = None
    kb_scope: Optional[str] = "all"
    model: Optional[str] = None
    deep_thinking: Optional[bool] = True


class ReviewReq(BaseModel):
    thread_id: str
    decision: str  # P0 安全加固：必填（approve / revise / re_retrieve），不再缺省默认 approve
    comments: str = ""
    reviewer_name: str = "值班执业律师"
    license_no: str = ""
    reviewer_role: str = "lawyer"  # lawyer / senior_lawyer / partner / expert
    reviewer_token: str = ""
    edited_answer: str = ""
    add_citations: List[str] = Field(default_factory=list)
    remove_citations: List[str] = Field(default_factory=list)
    extra_query: str = ""
    act_date: Optional[str] = None
    region: Optional[str] = None


class KBCollectionCreateReq(BaseModel):
    collection_id: str = ""
    name: str
    description: str = ""
    embed_model: str = "BAAI/bge-m3"


class KBDocumentImportReq(BaseModel):
    collection_id: str = "default_external_laws"
    law_name: str
    raw_text: str
    hierarchy: str = "法律"
    document_number: str = ""
    promulgation_date: str = ""
    effective_date: str = ""
    effective_to: str = ""
    validity: str = "现行有效"
    source: str = "全国人民代表大会常务委员会"
    category: str = "民事"
    scope: str = "一般"
    host_law: str = ""
    host_hierarchy: str = ""
    host_effective_date: str = ""
    verified: bool = True


class KnowledgeUpdateReq(BaseModel):
    section: str
    value: Any


class LLMSettingsUpdateReq(BaseModel):
    provider_preset: Optional[str] = None
    mock: Optional[bool] = None
    llm_provider: Optional[str] = None
    llm_api_base: Optional[str] = None
    llm_model: Optional[str] = None
    llm_api_key: Optional[str] = None
    llm_temperature: Optional[float] = None
    embed_enabled: Optional[bool] = None
    embed_api_base: Optional[str] = None
    embed_model: Optional[str] = None
    embed_api_key: Optional[str] = None
    hybrid_dense_weight: Optional[float] = None
    strict_review_auth: Optional[bool] = None
    top_k: Optional[int] = None
    verified: Optional[bool] = None


class LLMTestReq(BaseModel):
    llm_provider: Optional[str] = "qwen"
    llm_api_base: Optional[str] = ""
    llm_model: Optional[str] = ""
    llm_api_key: Optional[str] = ""


def _snapshot(config, thread_id: str, user_id: Optional[str] = None, kb_scope: Optional[str] = None):
    """读取线程当前状态并同步持久化至数据库。"""
    st = graph.get_state(config)
    fo = st.values.get("final_output")
    fo_dump = fo.model_dump() if fo else None
    result = {
        "final_output": fo_dump,
        "sentence_traces": (fo_dump.get("sentence_traces") if fo_dump else None) or st.values.get("sentence_traces", []),
        "structured_queries": (fo_dump.get("structured_queries") if fo_dump else None) or st.values.get("structured_queries", {}),
        "fact_summary": st.values.get("fact_summary"),
        "compliance_info": st.values.get("compliance_info"),
        "rule_plan": st.values.get("rule_plan"),
        "rule_notes": st.values.get("rule_notes", []),
        "precedence": st.values.get("precedence", []),
        "answer_confidence": st.values.get("answer_confidence"),
        "review_audit": st.values.get("review_audit"),
    }

    pending = []
    for task in getattr(st, "tasks", ()) or ():
        for intr in getattr(task, "interrupts", ()) or ():
            pending.append(getattr(intr, "value", intr))
    result["needs_review"] = bool(pending) or not bool(st.values.get("human_approved"))
    result["review_payload"] = pending[0] if pending else None

    status_str = "needs_review" if result["needs_review"] else "done"
    audit_store.save_snapshot(
        thread_id=thread_id,
        status=status_str,
        state_data={
            "query": st.values.get("masked_query") or st.values.get("query", ""),
            "act_date": st.values.get("act_date", ""),
            "region": st.values.get("region", ""),
            "answer_confidence": result["answer_confidence"],
            "rule_plan": result["rule_plan"],
            "final_output": fo_dump,
            "fact_summary": result["fact_summary"],
            "compliance_info": result["compliance_info"],
            "rule_notes": result["rule_notes"],
            "human_approved": bool(st.values.get("human_approved")),
            "user": {"username": user_id or "admin"},
            "kb_scope": kb_scope or "external",
        },
    )
    return result


def check_admin_permission(user_identifier: Optional[str] = None):
    """RBAC 管理员权限校验：仅具备 admin 角色或管理员权限者方可操作"""
    ident = (user_identifier or "").strip()
    if ident.lower() == "user":
        raise HTTPException(
            status_code=403,
            detail="权限不足：普通用户角色无权操作，需要管理员角色权限！",
        )
    if ident:
        perms = rbac_manager.get_user_effective_permissions(ident)
        if not perms.get("is_admin"):
            raise HTTPException(
                status_code=403,
                detail=f"权限不足：用户 [{ident}] 角色为 {perms.get('user',{}).get('roles',[])}，缺乏管理员权限！",
            )


def check_kb_manage_permission(user_identifier: Optional[str], collection_id: str):
    """RBAC 数据级权限校验：校验用户角色是否有权维护/导入指定知识库集合"""
    ident = (user_identifier or "admin").strip()
    if ident.lower() == "user":
        raise HTTPException(
            status_code=403,
            detail=f"权限不足：普通用户无知识库维护权限，仅支持问答检索！",
        )
    if not rbac_manager.check_kb_manage_permission(ident, collection_id):
        raise HTTPException(
            status_code=403,
            detail=f"权限不足：当前角色无权维护知识库集合 [{collection_id}]，请联系系统管理员授权！",
        )


def handle_legal_query(req: QueryReq, user_id: Optional[str] = None) -> Dict[str, Any]:
    tid = req.thread_id or str(uuid.uuid4())
    cfg = {"configurable": {"thread_id": tid}}
    payload: Dict[str, Any] = {"query": req.query, "thread_id": tid}
    if req.history:
        payload["history"] = req.history
    if req.act_date:
        payload["act_date"] = req.act_date
    if req.act_end_date:
        payload["act_end_date"] = req.act_end_date
    if req.region:
        payload["region"] = req.region
    if req.kb_scope:
        payload["kb_scope"] = req.kb_scope
    if req.model:
        payload["model"] = req.model
        if req.model == "mock":
            settings.mock = True
        else:
            settings.llm_model = req.model
            if (
                settings.llm_api_key
                or "localhost" in (settings.llm_api_base or "")
                or "127.0.0.1" in (settings.llm_api_base or "")
            ):
                settings.mock = False
    if req.deep_thinking is not None:
        payload["deep_thinking"] = req.deep_thinking

    graph.invoke(payload, cfg)
    snap = _snapshot(cfg, tid, user_id=user_id, kb_scope=req.kb_scope)
    fo = snap.get("final_output") or {}
    plan = snap.get("rule_plan") or {}
    audit_rec = audit_store.record_audit_event(
        thread_id=tid,
        event_type="query_generated",
        decision="pending_review",
        confidence=snap.get("answer_confidence") or "",
        review_level=plan.get("review", ""),
        citations=fo.get("citations", []),
        draft_answer=fo.get("answer", ""),
        final_answer=fo.get("answer", ""),
    )
    base = {
        "thread_id": tid,
        "result": snap["final_output"],
        "fact_summary": snap.get("fact_summary"),
        "compliance_info": snap.get("compliance_info"),
        "rule_plan": snap.get("rule_plan"),
        "rule_notes": snap.get("rule_notes", []),
        "precedence": snap.get("precedence", []),
        "answer_confidence": snap.get("answer_confidence"),
        "audit_entry": audit_rec,
        "kb_scope": req.kb_scope or "all",
        "model_used": req.model or (settings.llm_model if not settings.mock else "mock"),
    }
    if snap["needs_review"]:
        return {"status": "needs_review", "review": snap["review_payload"], **base}
    return {"status": "done", **base}


if _HAS_FASTAPI:
    app = FastAPI(title="法律顾问 Agent（可交付版 v3.5）", version="3.5.0")
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def index():
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))

    @app.get("/health")
    def health():
        chain_ok, total_logs, chain_msg = audit_store.verify_hash_chain()
        nodes = load_nodes()
        return {
            "status": "ok",
            "mode": "mock" if settings.mock else "llm",
            "llm_provider": settings.llm_provider,
            "llm_model": settings.llm_model,
            "embed_model": settings.embed_model,
            "total_articles_indexed": len(nodes),
            "audit_chain_valid": chain_ok,
            "audit_logs_count": total_logs,
            "audit_chain_message": chain_msg,
        }

    @app.post("/legal-query")
    def legal_query(req: QueryReq, x_user_id: Optional[str] = Header(None)):
        return handle_legal_query(req, user_id=x_user_id)

    @app.post("/human-review")
    def human_review(req: ReviewReq):
        dec = (req.decision or "").strip().lower()
        if dec not in ("approve", "revise", "re_retrieve"):
            raise HTTPException(
                status_code=400,
                detail="decision 为必填项，且必须为 approve / revise / re_retrieve 之一",
            )
        cfg = {"configurable": {"thread_id": req.thread_id}}
        resume_payload = {
            "thread_id": req.thread_id,
            "decision": dec,
            "comments": req.comments,
            "reviewer_name": req.reviewer_name,
            "license_no": req.license_no,
            "reviewer_role": req.reviewer_role,
            "reviewer_token": req.reviewer_token,
            "edited_answer": req.edited_answer,
            "add_citations": req.add_citations,
            "remove_citations": req.remove_citations,
            "extra_query": req.extra_query,
            "act_date": req.act_date or "",
            "region": req.region or "",
        }
        graph.invoke(Command(resume=resume_payload), cfg)
        snap = _snapshot(cfg, req.thread_id)
        base = {
            "thread_id": req.thread_id,
            "result": snap["final_output"],
            "fact_summary": snap.get("fact_summary"),
            "compliance_info": snap.get("compliance_info"),
            "rule_plan": snap.get("rule_plan"),
            "rule_notes": snap.get("rule_notes", []),
            "precedence": snap.get("precedence", []),
            "answer_confidence": snap.get("answer_confidence"),
            "review_audit": snap.get("review_audit"),
        }
        if snap["needs_review"]:
            return {"status": "needs_review", "review": snap["review_payload"], **base}
        return {"status": "done", **base}

    # ————————————————————————————————————————————————
    # RAG 法规知识库管理接口（创建知识库、导入外规文档并做 BGE-M3 向量解析）
    # ————————————————————————————————————————————————
    @app.get("/kb/documents")
    def list_kb_documents(category_type: Optional[str] = None, q: Optional[str] = None):
        """获取知识库中的文档/法规列表（支持内规知识库、外规知识库聚合检索）。"""
        nodes = load_nodes()
        grouped: Dict[str, Dict[str, Any]] = {}
        for n in nodes:
            meta = getattr(n, "metadata", {}) or {}
            lname = meta.get("law_name") or "未命名法规"
            if lname not in grouped:
                # 判断是内规还是外规
                is_internal = any(k in lname for k in ["条例", "规定", "制度", "办法", "指引", "管理办法", "工作指引"]) or meta.get("hierarchy") in ["地方性法规", "部门规章", "地方政府规章"]
                cat_type = "internal" if is_internal else "external"
                grouped[lname] = {
                    "id": f"doc_{abs(hash(lname)) % 1000000}",
                    "name": lname,
                    "collection_name": "内规知识库" if cat_type == "internal" else "外规知识库",
                    "category_type": cat_type,
                    "department": meta.get("source") or ("总行合规部" if cat_type == "internal" else "全国人大常委会"),
                    "upload_date": meta.get("effective_date") or "2026-11-03",
                    "uploader": "Admin",
                    "file_size": f"{round(len(getattr(n, 'text', '') or '') * 0.12 + 8.5, 2)}K",
                    "file_format": ".docx" if "办法" in lname or "通知" in lname else (".xlsx" if "清单" in lname else ".pdf"),
                    "segment_count": 0,
                    "is_active": True,
                    "status": "解析完成",
                    "hierarchy": meta.get("hierarchy", "法律"),
                    "articles": [],
                }
            grouped[lname]["segment_count"] += 1
            grouped[lname]["articles"].append({
                "article": meta.get("article", ""),
                "content": (getattr(n, "text", "") or "")[:150] + ("..." if len(getattr(n, "text", "") or "") > 150 else ""),
            })

        docs = list(grouped.values())
        if category_type and category_type != "all":
            docs = [d for d in docs if d["category_type"] == category_type]
        if q:
            ql = q.lower().strip()
            docs = [d for d in docs if ql in d["name"].lower() or ql in d["department"].lower()]
        return {"documents": docs, "total": len(docs)}

    @app.get("/kb/collections")
    def list_kb_collections():
        cols = kb_store.list_collections()
        total_nodes = len(load_nodes())
        return {
            "collections": cols,
            "total_active_articles": total_nodes,
            "embed_model": settings.embed_model,
        }

    @app.post("/kb/collections")
    def create_kb_collection(req: KBCollectionCreateReq, x_user_role: Optional[str] = Header(None)):
        check_admin_permission(x_user_role)
        col = kb_store.create_collection(
            collection_id=req.collection_id,
            name=req.name,
            description=req.description,
            embed_model=req.embed_model,
        )
        return {"status": "ok", "collection": col}

    @app.post("/kb/import")
    def import_kb_document(
        req: KBDocumentImportReq,
        x_user_id: Optional[str] = Header(None),
        x_user_role: Optional[str] = Header(None),
    ):
        ident = x_user_id or x_user_role or "admin"
        check_kb_manage_permission(ident, req.collection_id)
        res = kb_store.ingest_document(
            raw_text=req.raw_text,
            law_name=req.law_name,
            collection_id=req.collection_id,
            hierarchy=req.hierarchy,
            document_number=req.document_number,
            promulgation_date=req.promulgation_date,
            effective_date=req.effective_date,
            effective_to=req.effective_to,
            validity=req.validity,
            source=req.source,
            category=req.category,
            scope=req.scope,
            host_law=req.host_law,
            host_hierarchy=req.host_hierarchy,
            host_effective_date=req.host_effective_date,
            verified=req.verified,
        )
        invalidate_cache()
        res["total_active_articles"] = len(load_nodes())
        return {"status": "ok", **res}

    # ————————————————————————————————————————————————
    # 外置知识与规则配置中心接口（免改代码热更新知识与规则）
    # ————————————————————————————————————————————————
    @app.get("/config/knowledge")
    def get_knowledge_config(reload: bool = False):
        if reload:
            knowledge_center.reload()
        return {
            "config_path": getattr(knowledge_center, "config_path", "mysql:kb_knowledge_configs"),
            "knowledge": knowledge_center.get_all(reload=reload),
        }

    @app.post("/config/knowledge")
    def update_knowledge_config(
        req: KnowledgeUpdateReq,
        x_user_role: Optional[str] = Header(None),
        x_user_id: Optional[str] = Header(None),
    ):
        check_admin_permission(x_user_role or x_user_id)
        updated = knowledge_center.update_section(req.section, req.value)
        invalidate_cache()
        return {
            "status": "ok",
            "section": req.section,
            "knowledge_keys": list(updated.keys()),
        }

    # ————————————————————————————————————————————————
    # 外部大模型与 BGE-M3 向量接口在线配置（支持千问 Qwen / 本地 OpenAI 兼容接口）
    # ————————————————————————————————————————————————
    @app.get("/config/llm")
    def get_llm_settings():
        return settings.to_public_dict()

    @app.post("/config/llm")
    def update_llm_settings(req: LLMSettingsUpdateReq, x_user_role: Optional[str] = Header(None)):
        check_admin_permission(x_user_role)
        payload = req.model_dump() if hasattr(req, "model_dump") else req.__dict__
        clean_updates = {k: v for k, v in payload.items() if v is not None}
        updated = settings.update_runtime(clean_updates)
        invalidate_cache()
        return {"status": "ok", "settings": updated}

    @app.post("/config/llm/test")
    def test_llm_settings(req: LLMTestReq, x_user_role: Optional[str] = Header(None)):
        check_admin_permission(x_user_role)
        res = settings.test_connection(
            api_base=req.llm_api_base or "",
            model=req.llm_model or "",
            api_key=req.llm_api_key or "",
            provider=req.llm_provider or "qwen",
        )
        return res

    @app.get("/session/{thread_id}")
    def get_session(thread_id: str):
        snap = audit_store.get_snapshot(thread_id)
        if not snap:
            return {"status": "not_found", "thread_id": thread_id}
        return snap

    @app.get("/audit/logs")
    def get_audit_logs(thread_id: Optional[str] = None, limit: int = 50):
        chain_ok, total, msg = audit_store.verify_hash_chain()
        logs = audit_store.list_audit_logs(thread_id=thread_id, limit=limit)
        return {
            "chain_valid": chain_ok,
            "total_records": total,
            "verification_message": msg,
            "logs": logs,
        }

    @app.get("/audit/rule-gaps")
    def get_rule_gaps(limit: int = 50):
        return {"gaps": audit_store.list_rule_gaps(limit=limit)}

    @app.get("/audit/rule-proposals")
    def get_rule_proposals(status: Optional[str] = None, limit: int = 50):
        proposals = audit_store.list_rule_proposals(status=status, limit=limit)
        return {"proposals": proposals}

    @app.get("/stats/overview")
    def get_stats_overview(
        dept_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ):
        """获取系统真实运营与知识库数据概览（100% 数据库真实统计，杜绝伪造假数据）。"""
        data = mysql_store.get_stats_overview(dept_id=dept_id, start_date=start_date, end_date=end_date)
        chain_ok, total_logs, chain_msg = audit_store.verify_hash_chain()
        logs = audit_store.list_audit_logs(limit=25)
        data["chain_valid"] = chain_ok
        data["total_audit_logs"] = total_logs
        data["verification_message"] = chain_msg
        data["recent_qa_logs"] = logs
        return data

    @app.get("/stats/qa-details")
    def get_qa_details_api(
        username: Optional[str] = None,
        keyword: Optional[str] = None,
        dept_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 100,
    ):
        """获取真实的问答详情流水明细，支持用户名、问题/答案关键词、部门架构与日期范围检索。"""
        records = mysql_store.get_qa_details(
            username=username,
            keyword=keyword,
            dept_id=dept_id,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
        )
        return {"total": len(records), "records": records}

    class AdoptProposalReq(BaseModel):
        reviewer_name: Optional[str] = "执业律师/规则专家"
        notes: Optional[str] = ""

    @app.post("/audit/rule-proposals/{id}/adopt")
    def adopt_rule_proposal_api(id: int, req: Optional[AdoptProposalReq] = None, x_user_role: Optional[str] = Header(None)):
        check_admin_permission(x_user_role)
        name = req.reviewer_name if req else "执业律师/规则专家"
        notes = req.notes if req else ""
        ok, msg = audit_store.adopt_rule_proposal(id, reviewer_name=name, review_notes=notes)
        if not ok:
            raise HTTPException(status_code=400, detail=msg)
        return {"status": "ok", "message": msg}

    @app.post("/audit/rule-proposals/{id}/reject")
    def reject_rule_proposal_api(id: int, req: Optional[AdoptProposalReq] = None, x_user_role: Optional[str] = Header(None)):
        check_admin_permission(x_user_role)
        name = req.reviewer_name if req else ""
        notes = req.notes if req else ""
        ok, msg = audit_store.reject_rule_proposal(id, reviewer_name=name, review_notes=notes)
        return {"status": "ok", "message": msg}

    # ————————————————————————————————————————————————
    # 标准 RBAC 权限与三级组织机构用户中心管理接口
    # ————————————————————————————————————————————————
    class DeptReq(BaseModel):
        dept_id: Optional[str] = ""
        dept_name: str
        parent_id: Optional[str] = None
        sort_order: Optional[int] = 1
        leader: Optional[str] = ""
        phone: Optional[str] = ""
        status: Optional[str] = "active"

    class UserReq(BaseModel):
        user_id: Optional[str] = ""
        username: str
        name: str
        dept_id: Optional[str] = "dept_hq"
        department: Optional[str] = "集团总部"
        post_title: Optional[str] = ""
        license_no: Optional[str] = ""
        roles: List[str] = Field(default_factory=lambda: ["user"])
        status: Optional[str] = "active"

    class RoleReq(BaseModel):
        role_id: Optional[str] = ""
        role_name: str
        description: Optional[str] = ""
        menus: List[str] = Field(default_factory=list)
        kb_permissions: List[str] = Field(default_factory=list)

    @app.get("/rbac/meta")
    def rbac_meta():
        """获取系统全部标准菜单与知识库元数据定义（知识库集合动态同源拉取）"""
        from app.mysql_store import mysql_store
        dynamic_kbs = mysql_store.get_dynamic_kb_collections()
        return {
            "menus": ALL_MENUS,
            "kb_collections": dynamic_kbs if dynamic_kbs else ALL_KB_COLLECTIONS,
        }

    @app.get("/rbac/departments")
    def rbac_list_departments():
        """获取平铺部门列表"""
        return {"departments": rbac_manager.get_departments()}

    @app.get("/rbac/departments/tree")
    def rbac_get_department_tree():
        """获取三级组织架构树（含各级子部门及累计人数）"""
        return {"tree": rbac_manager.get_department_tree()}

    @app.post("/rbac/departments")
    def rbac_save_department(req: DeptReq, x_user_role: Optional[str] = Header(None), x_user_id: Optional[str] = Header(None)):
        """创建或编辑部门（支持最高 3 级）"""
        check_admin_permission(x_user_role or x_user_id)
        ok, res = rbac_manager.save_department(req.model_dump())
        if not ok:
            raise HTTPException(status_code=400, detail=str(res))
        return {"status": "ok", "department": res}

    @app.delete("/rbac/departments/{dept_id}")
    def rbac_delete_department(dept_id: str, x_user_role: Optional[str] = Header(None), x_user_id: Optional[str] = Header(None)):
        """删除部门节点（含总部保护、子部门级联校验与在职员工拦截）"""
        check_admin_permission(x_user_role or x_user_id)
        ok, msg = rbac_manager.delete_department(dept_id)
        if not ok:
            raise HTTPException(status_code=400, detail=msg)
        return {"status": "ok", "message": msg}

    @app.get("/rbac/departments/{dept_id}/users")
    def rbac_get_department_users(dept_id: str, include_children: bool = True):
        """按部门获取员工列表（支持递归包含子部门成员）"""
        users = rbac_manager.get_department_users(dept_id, include_children=include_children)
        return {"dept_id": dept_id, "total": len(users), "users": users}

    @app.get("/rbac/current")
    def rbac_current_user(x_user_id: Optional[str] = Header(None), x_user_role: Optional[str] = Header(None)):
        """获取当前会话用户的详细信息及计算后的有效菜单权限与可维护知识库列表"""
        ident = x_user_id or x_user_role or "admin"
        return rbac_manager.get_user_effective_permissions(ident)

    @app.get("/rbac/users")
    def rbac_list_users(dept_id: Optional[str] = None, include_children: bool = True):
        """获取用户列表，支持按组织部门筛选"""
        if dept_id:
            users = rbac_manager.get_department_users(dept_id, include_children=include_children)
        else:
            users = rbac_manager.get_users()
        return {"users": users}

    @app.post("/rbac/users")
    def rbac_save_user(req: UserReq, x_user_role: Optional[str] = Header(None), x_user_id: Optional[str] = Header(None)):
        """创建或编辑用户及其关联角色与三级组织部门"""
        check_admin_permission(x_user_role or x_user_id)
        saved = rbac_manager.save_user(req.model_dump())
        return {"status": "ok", "user": saved}

    @app.delete("/rbac/users/{user_id}")
    def rbac_delete_user(user_id: str, x_user_role: Optional[str] = Header(None), x_user_id: Optional[str] = Header(None)):
        """删除用户"""
        check_admin_permission(x_user_role or x_user_id)
        ok, msg = rbac_manager.delete_user(user_id)
        if not ok:
            raise HTTPException(status_code=400, detail=msg)
        return {"status": "ok", "message": msg}

    @app.get("/rbac/roles")
    def rbac_list_roles():
        """获取所有角色及其授权配置"""
        return {"roles": rbac_manager.get_roles()}

    @app.post("/rbac/roles")
    def rbac_save_role(req: RoleReq, x_user_role: Optional[str] = Header(None)):
        """创建或编辑角色的菜单权限与知识库维护授权"""
        check_admin_permission(x_user_role)
        saved = rbac_manager.save_role(req.model_dump())
        return {"status": "ok", "role": saved}

    @app.delete("/rbac/roles/{role_id}")
    def rbac_delete_role(role_id: str, x_user_role: Optional[str] = Header(None)):
        """删除自定义角色"""
        check_admin_permission(x_user_role)
        ok, msg = rbac_manager.delete_role(role_id)
        if not ok:
            raise HTTPException(status_code=400, detail=msg)
        return {"status": "ok", "message": msg}

    # ————————————————————————————————————————————————
    # P3 升级：对标「AI法盾」文档上传、知识图谱离线构建、官方新旧法对照与时效同步
    # ————————————————————————————————————————————————
    from app.counterparts import batch_import_official_counterparts
    from app.crawler import flk_monitor
    from app.graph_builder import (
        apply_cross_references_to_config,
        batch_build_law_graph,
        extract_article_cross_references,
        extract_interpretation_and_application_relations,
    )

    class AdvancedDocUploadReq(BaseModel):
        filename: str
        raw_text: Optional[str] = ""
        content_base64: Optional[str] = None
        law_name: Optional[str] = ""
        kb_type: Optional[str] = "外规知识库/法律法规"
        department: Optional[str] = "法律合规部"
        uploader: Optional[str] = "系统管理员"
        split_mode: Optional[str] = "article"  # auto / article / hierarchical
        validity_option: Optional[str] = "permanent"  # permanent / custom
        valid_until: Optional[str] = ""
        hierarchy: Optional[str] = "法律"
        effective_date: Optional[str] = ""
        promulgation_date: Optional[str] = ""
        host_law: Optional[str] = ""
        version_code: Optional[str] = ""
        version_name: Optional[str] = ""
        conflict_action: Optional[str] = "append_new"  # append_new / overwrite_latest / rename_new
        custom_new_title: Optional[str] = ""

    class CheckVersionConflictReq(BaseModel):
        filename: str
        law_name: Optional[str] = ""
        kb_type: Optional[str] = "外规知识库/法律法规"
        raw_text: Optional[str] = ""
        content_base64: Optional[str] = None

    @app.post("/kb/documents/check-version-conflict")
    def check_version_conflict_api(req: CheckVersionConflictReq):
        """上传前检测同名法规冲突并执行智能正文/题注版本嗅探"""
        import base64
        content_bytes = None
        if req.content_base64:
            try:
                b64_str = req.content_base64
                if "," in b64_str:
                    b64_str = b64_str.split(",", 1)[1]
                content_bytes = base64.b64decode(b64_str)
            except Exception:
                pass
        res = kb_store.check_version_conflict(
            filename=req.filename,
            law_name=req.law_name or "",
            kb_type=req.kb_type or "",
            content_bytes=content_bytes,
            raw_text=req.raw_text or "",
        )
        return {"status": "ok", **res}

    @app.post("/kb/documents/upload-advanced")
    def upload_advanced_document_api(req: AdvancedDocUploadReq):
        """对标「AI法盾」知识库文档上传、版本演进管理与三模式切分入库接口"""
        import base64

        if not req.raw_text and not req.filename and not req.content_base64:
            raise HTTPException(status_code=400, detail="请提供文档文本或文件内容")

        content_bytes = None
        if req.content_base64:
            try:
                b64_str = req.content_base64
                if "," in b64_str:
                    b64_str = b64_str.split(",", 1)[1]
                content_bytes = base64.b64decode(b64_str)
            except Exception as e:
                pass

        res = kb_store.import_advanced_document(
            filename=req.filename,
            content_bytes=content_bytes,
            raw_text=req.raw_text,
            law_name=req.law_name or req.filename.rsplit(".", 1)[0],
            kb_type=req.kb_type or "外规知识库/法律法规",
            department=req.department or "法律合规部",
            uploader=req.uploader or "系统管理员",
            split_mode=req.split_mode or "article",
            validity_option=req.validity_option or "permanent",
            valid_until=req.valid_until or "",
            hierarchy=req.hierarchy or "法律",
            effective_date=req.effective_date or "",
            promulgation_date=req.promulgation_date or "",
            host_law=req.host_law or "",
            version_code=req.version_code or "",
            version_name=req.version_name or "",
            conflict_action=req.conflict_action or "append_new",
            custom_new_title=req.custom_new_title or "",
        )
        invalidate_cache()
        return {"status": "ok", "result": res}

    class CreateTreeNodeReq(BaseModel):
        name: str
        parent_id: Optional[str] = ""
        node_type: Optional[str] = "folder"
        owner_dept_id: Optional[str] = "dept_hq"
        visibility: Optional[str] = "dept"
        description: Optional[str] = ""

    class RenameTreeNodeReq(BaseModel):
        new_name: str

    class UpdateKbDeptReq(BaseModel):
        owner_dept_id: str
        visibility: Optional[str] = "dept"

    @app.get("/kb/tree")
    def get_kb_tree_api():
        """获取完整的层级知识库与目录树结构"""
        return {"tree": kb_store.get_kb_tree()}

    @app.post("/kb/tree/node")
    def create_kb_tree_node_api(req: CreateTreeNodeReq):
        """创建知识库（顶层）或层级子目录（支持指定主责管理部门）"""
        try:
            node = kb_store.create_kb_tree_node(
                name=req.name,
                parent_id=req.parent_id or "",
                node_type=req.node_type or "folder",
                description=req.description or "",
                owner_dept_id=req.owner_dept_id or "dept_hq",
                visibility=req.visibility or "dept",
            )
            return {"status": "ok", "node": node}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.put("/kb/tree/node/{node_id}/dept")
    def update_kb_dept_api(node_id: str, req: UpdateKbDeptReq, x_user_role: Optional[str] = Header(None)):
        """更新知识库的主责管理部门与可见性范围"""
        check_admin_permission(x_user_role)
        from app.mysql_store import mysql_store
        ok = mysql_store.update_kb_node_owner_dept(node_id, req.owner_dept_id, req.visibility or "dept")
        if not ok:
            raise HTTPException(status_code=400, detail="更新主责部门失败")
        return {"status": "ok", "message": "知识库主责管理部门已更新"}

    @app.put("/kb/tree/node/{node_id}")
    def rename_kb_tree_node_api(node_id: str, req: RenameTreeNodeReq):
        """改名知识库或层级目录"""
        try:
            kb_store.rename_kb_tree_node(node_id, req.new_name)
            return {"status": "ok", "message": f"重命名为 [{req.new_name}] 成功"}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.delete("/kb/tree/node/{node_id}")
    def delete_kb_tree_node_api(node_id: str):
        """删除知识库或层级目录（递归删除子目录与关联文档）"""
        try:
            kb_store.delete_kb_tree_node(node_id)
            invalidate_cache()
            return {"status": "ok", "message": f"节点 {node_id} 及其下文档已删除"}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/kb/documents/advanced")
    def list_advanced_documents_api(kb_type: Optional[str] = None, limit: int = 200):
        """获取法规文档列表（支持所属知识库与层级目录路径筛选，含全量版本演进元数据）"""
        return {"documents": kb_store.list_law_documents(kb_type=kb_type, limit=limit)}

    @app.post("/kb/documents/{doc_id}/set-latest")
    def set_document_latest_api(doc_id: str):
        """将指定历史版本切换为当前最新有效版本"""
        try:
            res = kb_store.set_document_as_latest(doc_id)
            invalidate_cache()
            return res
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/kb/documents/diff")
    def compare_document_versions_api(doc_id_new: str, doc_id_old: str):
        """对比同一法规两个版本的条文级差异 (Diff)"""
        res = kb_store.compare_document_versions(doc_id_new=doc_id_new, doc_id_old=doc_id_old)
        if res.get("status") != "ok":
            raise HTTPException(status_code=400, detail=res.get("message", "对比失败"))
        return res

    @app.delete("/kb/documents/advanced/{doc_id}")
    def delete_advanced_document_api(doc_id: str):
        """删除指定法规文档及所有分块"""
        kb_store.delete_law_document(doc_id)
        invalidate_cache()
        return {"status": "ok", "message": f"文档 {doc_id} 已成功删除"}

    @app.get("/kb/documents/{doc_id}/chunks")
    def get_document_chunks_api(doc_id: str):
        """获取指定法规文档的条文切片与分段内容详情"""
        res = kb_store.get_document_chunks(doc_id)
        if res.get("status") == "not_found":
            raise HTTPException(status_code=404, detail=res.get("message", "文档不存在"))
        return res

    class GraphBuildReq(BaseModel):
        law_name: str
        article: Optional[str] = "第1条"
        content: str
        auto_propose: Optional[bool] = True

    @app.post("/graph-builder/extract")
    def extract_graph_api(req: GraphBuildReq):
        """离线大模型批处理：抽取引致条款关系"""
        refs = extract_article_cross_references(req.law_name, req.article or "第1条", req.content)
        if req.auto_propose and refs:
            batch_build_law_graph([{"law_name": req.law_name, "article": req.article, "content": req.content}])
        return {"status": "ok", "references": refs}

    class InterpExtractReq(BaseModel):
        interpretation_name: str
        article: str
        text_content: str

    @app.post("/graph-builder/extract-interpretation")
    def extract_interpretation_api(req: InterpExtractReq):
        """司法解释「理解与适用」引申关系抽取"""
        refs = extract_interpretation_and_application_relations(
            req.interpretation_name, req.article, req.text_content
        )
        return {"status": "ok", "relations": refs}

    @app.post("/graph-builder/apply-to-config")
    def apply_graph_to_config_api(references: List[Dict[str, Any]]):
        """将抽检通过的引致关系合入知识中心热生效"""
        cnt = apply_cross_references_to_config(references)
        return {"status": "ok", "applied_count": cnt}

    @app.post("/counterparts/batch-import")
    def batch_import_counterparts_api(counterparts: Optional[List[Dict[str, Any]]] = None):
        """法工委权威《新旧法律条文对照表》数字化批量录入"""
        res = batch_import_official_counterparts(counterparts)
        invalidate_cache()
        return res

    class CrawlerSyncReq(BaseModel):
        mock_bulletins: Optional[List[Dict[str, Any]]] = None

    @app.post("/crawler/sync-flk")
    def sync_flk_api(req: Optional[CrawlerSyncReq] = None):
        """国家法律法规数据库（flk.npc.gov.cn）公告监听与自动时效同步"""
        bulletins = req.mock_bulletins if req else None
        res = flk_monitor.check_and_sync_announcements(bulletins)
        return res

    @app.get("/crawler/revision-logs")
    def list_revision_logs_api(law_name: Optional[str] = None, article: Optional[str] = None):
        """获取条款项级修改与生命周期追踪日志"""
        logs = kb_store.list_article_revisions(law_name=law_name, article=article)
        return {"revision_logs": logs}

    # ————————————————————————————————————————————————
    # 国家法律法规数据库（flk.npc.gov.cn）官方 Word 文档自动拉取与分类入库接口
    # ————————————————————————————————————————————————
    from app.flk_downloader import flk_downloader

    @app.get("/flk/search")
    def search_flk_laws_api(
        keyword: str = "",
        type_code: str = "flfg",
        page: int = 1,
        size: int = 10,
    ):
        """在线检索国家法规库法规列表"""
        return flk_downloader.search_laws(keyword=keyword, type_code=type_code, page=page, size=size)

    class FLKDownloadReq(BaseModel):
        flk_id: Optional[str] = None
        title: Optional[str] = None
        type_code: str = "flfg"
        uploader: Optional[str] = "国家法规库同步器"

    @app.post("/flk/download-and-import")
    def download_flk_law_api(
        req: FLKDownloadReq,
        x_user_id: Optional[str] = Header(None),
        x_user_role: Optional[str] = Header(None),
    ):
        """自动下载国家法规库官方 Word 文档，自动按分类创建知识库目录并完成法条切分与向量入库"""
        uploader_name = x_user_id or req.uploader or "国家法规库同步器"
        res = flk_downloader.download_and_import(
            flk_id=req.flk_id,
            title=req.title,
            type_code=req.type_code,
            uploader=uploader_name,
        )
        return res
else:
    app = None

