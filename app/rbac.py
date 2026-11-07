"""基于角色的访问控制（RBAC）管理器：
1. 菜单访问权限：不同角色访问不同菜单，系统管理员 (admin) 拥有通配权限访问所有；
2. 知识库数据维护权限：不同角色对不同知识库（内规库、外规库等）配置独立的维护/导入权限；
3. 用户与角色解耦配置：支持为用户分配一个或多个角色；
4. 数据基于 JSON 文件持久化（app/data/rbac_config.json），开箱即用，支持热更新。
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

RBAC_CONFIG_PATH = os.getenv("RBAC_CONFIG_PATH", "app/data/rbac_config.json")

# 系统所有标准菜单定义
ALL_MENUS = [
    {"id": "menu:qa", "name": "智能问答", "icon": "💬", "desc": "法律咨询、三维检索、六段式意见书及句级溯源"},
    {"id": "menu:kb", "name": "知识库管理", "icon": "📁", "desc": "法规知识库列表、文档详情与外部法规导入"},
    {"id": "menu:llm", "name": "模型配置", "icon": "⚙️", "desc": "大模型与 BGE-M3 向量接口运行时参数配置"},
    {"id": "menu:rules", "name": "字典管理", "icon": "📖", "desc": "业务规则配置中心、同义词与提案采纳"},
    {"id": "menu:stats", "name": "数据统计", "icon": "📊", "desc": "12项核心指标数据概览与问答详情"},
    {"id": "menu:audit", "name": "审计与缺口", "icon": "🛡️", "desc": "SHA-256 防篡改审计哈希链与规则缺口清单"},
    {"id": "menu:eval", "name": "自动测评", "icon": "🧪", "desc": "法理规则基准断言与评测 (建设中)"},
    {"id": "menu:user", "name": "用户与权限中心", "icon": "👥", "desc": "RBAC 用户账号、角色菜单矩阵与知识库维护授权"},
]

# 系统标准知识库集合定义（支持数据级维护权限配置）
ALL_KB_COLLECTIONS = [
    {"id": "default_external_laws", "name": "外规知识库", "desc": "国家法律、行政法规、最高法司法解释"},
    {"id": "internal_rules", "name": "内规知识库", "desc": "总行规章制度、业务管理办法、合规指引"},
    {"id": "industry_standards", "name": "行业合规库", "desc": "金融行业规范、监管通报、自律准则"},
]

# 系统三级组织架构初始预设
DEFAULT_DEPARTMENTS: List[Dict[str, Any]] = [
    # 1. 顶级：总部
    {
        "dept_id": "dept_hq",
        "dept_name": "集团总部",
        "parent_id": None,
        "level": 1,
        "dept_path": "dept_hq",
        "sort_order": 1,
        "leader": "张合伙人",
        "phone": "010-88880001",
        "status": "active",
        "created_at": "2026-11-01T08:00:00Z",
        "updated_at": "2026-11-01T08:00:00Z",
    },
    # 2. 二级：各职能部室 / 业务条线
    {
        "dept_id": "dept_legal",
        "dept_name": "法务合规部",
        "parent_id": "dept_hq",
        "level": 2,
        "dept_path": "dept_hq/dept_legal",
        "sort_order": 1,
        "leader": "李合规",
        "phone": "010-88880002",
        "status": "active",
        "created_at": "2026-11-01T08:00:00Z",
        "updated_at": "2026-11-01T08:00:00Z",
    },
    {
        "dept_id": "dept_tech",
        "dept_name": "科技部",
        "parent_id": "dept_hq",
        "level": 2,
        "dept_path": "dept_hq/dept_tech",
        "sort_order": 2,
        "leader": "王技术",
        "phone": "010-88880003",
        "status": "active",
        "created_at": "2026-11-01T08:00:00Z",
        "updated_at": "2026-11-01T08:00:00Z",
    },
    {
        "dept_id": "dept_hr",
        "dept_name": "人力资源部",
        "parent_id": "dept_hq",
        "level": 2,
        "dept_path": "dept_hq/dept_hr",
        "sort_order": 3,
        "leader": "陈主管",
        "phone": "010-88880004",
        "status": "active",
        "created_at": "2026-11-01T08:00:00Z",
        "updated_at": "2026-11-01T08:00:00Z",
    },
    {
        "dept_id": "dept_biz",
        "dept_name": "金融商事业务部",
        "parent_id": "dept_hq",
        "level": 2,
        "dept_path": "dept_hq/dept_biz",
        "sort_order": 4,
        "leader": "赵律师",
        "phone": "010-88880005",
        "status": "active",
        "created_at": "2026-11-01T08:00:00Z",
        "updated_at": "2026-11-01T08:00:00Z",
    },
    # 3. 三级：科室 / 小组
    # 法务合规部下属
    {
        "dept_id": "dept_legal_risk",
        "dept_name": "诉讼风控科",
        "parent_id": "dept_legal",
        "level": 3,
        "dept_path": "dept_hq/dept_legal/dept_legal_risk",
        "sort_order": 1,
        "leader": "李合规",
        "phone": "010-88880021",
        "status": "active",
        "created_at": "2026-11-01T08:00:00Z",
        "updated_at": "2026-11-01T08:00:00Z",
    },
    {
        "dept_id": "dept_legal_contract",
        "dept_name": "合同审查科",
        "parent_id": "dept_legal",
        "level": 3,
        "dept_path": "dept_hq/dept_legal/dept_legal_contract",
        "sort_order": 2,
        "leader": "周审查",
        "phone": "010-88880022",
        "status": "active",
        "created_at": "2026-11-01T08:00:00Z",
        "updated_at": "2026-11-01T08:00:00Z",
    },
    # 科技部下属
    {
        "dept_id": "dept_tech_plat",
        "dept_name": "架构与平台研发组",
        "parent_id": "dept_tech",
        "level": 3,
        "dept_path": "dept_hq/dept_tech/dept_tech_plat",
        "sort_order": 1,
        "leader": "王技术",
        "phone": "010-88880031",
        "status": "active",
        "created_at": "2026-11-01T08:00:00Z",
        "updated_at": "2026-11-01T08:00:00Z",
    },
    {
        "dept_id": "dept_tech_ai",
        "dept_name": "算法与AI智能组",
        "parent_id": "dept_tech",
        "level": 3,
        "dept_path": "dept_hq/dept_tech/dept_tech_ai",
        "sort_order": 2,
        "leader": "王知识",
        "phone": "010-88880032",
        "status": "active",
        "created_at": "2026-11-01T08:00:00Z",
        "updated_at": "2026-11-01T08:00:00Z",
    },
    # 人力资源部下属
    {
        "dept_id": "dept_hr_perf",
        "dept_name": "薪酬绩效科",
        "parent_id": "dept_hr",
        "level": 3,
        "dept_path": "dept_hq/dept_hr/dept_hr_perf",
        "sort_order": 1,
        "leader": "陈员工",
        "phone": "010-88880041",
        "status": "active",
        "created_at": "2026-11-01T08:00:00Z",
        "updated_at": "2026-11-01T08:00:00Z",
    },
    {
        "dept_id": "dept_hr_rel",
        "dept_name": "招聘与员工关系科",
        "parent_id": "dept_hr",
        "level": 3,
        "dept_path": "dept_hq/dept_hr/dept_hr_rel",
        "sort_order": 2,
        "leader": "孙HR",
        "phone": "010-88880042",
        "status": "active",
        "created_at": "2026-11-01T08:00:00Z",
        "updated_at": "2026-11-01T08:00:00Z",
    },
    # 金融商事部下属
    {
        "dept_id": "dept_biz_bank",
        "dept_name": "银行金融法律组",
        "parent_id": "dept_biz",
        "level": 3,
        "dept_path": "dept_hq/dept_biz/dept_biz_bank",
        "sort_order": 1,
        "leader": "赵律师",
        "phone": "010-88880051",
        "status": "active",
        "created_at": "2026-11-01T08:00:00Z",
        "updated_at": "2026-11-01T08:00:00Z",
    },
    {
        "dept_id": "dept_biz_corp",
        "dept_name": "企业商事法律组",
        "parent_id": "dept_biz",
        "level": 3,
        "dept_path": "dept_hq/dept_biz/dept_biz_corp",
        "sort_order": 2,
        "leader": "钱律师",
        "phone": "010-88880052",
        "status": "active",
        "created_at": "2026-11-01T08:00:00Z",
        "updated_at": "2026-11-01T08:00:00Z",
    },
]

# 初始开箱即用的默认配置
DEFAULT_RBAC_DATA: Dict[str, Any] = {
    "roles": [
        {
            "role_id": "admin",
            "role_name": "系统管理员",
            "description": "拥有全局最高权限，可访问所有菜单并维护所有知识库",
            "is_system": True,
            "menus": ["*"],
            "kb_permissions": ["*"],
        },
        {
            "role_id": "compliance_officer",
            "role_name": "合规法务专家",
            "description": "负责合规审核、业务规则维护及内外部规章制度维护",
            "is_system": True,
            "menus": ["menu:qa", "menu:kb", "menu:rules", "menu:stats", "menu:audit"],
            "kb_permissions": ["default_external_laws", "internal_rules"],
        },
        {
            "role_id": "kb_editor",
            "role_name": "知识库运维专员",
            "description": "专注于外部法规文档的导入、向量化切片与解析校验",
            "is_system": True,
            "menus": ["menu:qa", "menu:kb"],
            "kb_permissions": ["default_external_laws"],
        },
        {
            "role_id": "lawyer",
            "role_name": "执业审核律师",
            "description": "负责人在环咨询意见书审核、执业证注册表校验与审计追溯",
            "is_system": True,
            "menus": ["menu:qa", "menu:stats", "menu:audit"],
            "kb_permissions": [],
        },
        {
            "role_id": "user",
            "role_name": "普通业务人员",
            "description": "普通员工，仅具有智能问答与检索查阅权限，无管理维护权",
            "is_system": True,
            "menus": ["menu:qa"],
            "kb_permissions": [],
        },
    ],
    "users": [
        {
            "user_id": "u1001",
            "username": "admin",
            "name": "张合伙人",
            "dept_id": "dept_hq",
            "department": "集团总部",
            "post_title": "高级合伙人 / 律所主任",
            "license_no": "11101201010888888",
            "roles": ["admin"],
            "status": "active",
            "created_at": "2026-11-01T08:00:00Z",
        },
        {
            "user_id": "u1002",
            "username": "li_compliance",
            "name": "李合规",
            "dept_id": "dept_legal_risk",
            "department": "法务合规部 / 诉讼风控科",
            "post_title": "风控法务主管",
            "license_no": "13101201810012345",
            "roles": ["compliance_officer"],
            "status": "active",
            "created_at": "2026-11-02T09:30:00Z",
        },
        {
            "user_id": "u1003",
            "username": "wang_editor",
            "name": "王知识",
            "dept_id": "dept_tech_ai",
            "department": "科技部 / 算法与AI智能组",
            "post_title": "法律知识图谱专家",
            "license_no": "EMP-205",
            "roles": ["kb_editor"],
            "status": "active",
            "created_at": "2026-11-02T10:15:00Z",
        },
        {
            "user_id": "u1004",
            "username": "zhang_lawyer",
            "name": "赵律师",
            "dept_id": "dept_biz_bank",
            "department": "金融商事业务部 / 银行金融法律组",
            "post_title": "资深执业律师",
            "license_no": "13101201910056789",
            "roles": ["lawyer"],
            "status": "active",
            "created_at": "2026-11-03T14:20:00Z",
        },
        {
            "user_id": "u1005",
            "username": "chen_user",
            "name": "陈员工",
            "dept_id": "dept_hr_perf",
            "department": "人力资源部 / 薪酬绩效科",
            "post_title": "人事绩效专员",
            "license_no": "EMP-908",
            "roles": ["user"],
            "status": "active",
            "created_at": "2026-11-04T08:00:00Z",
        },
    ],
}


class RBACManager:
    """RBAC 核心权限、组织机构与数据管理类"""

    def __init__(self, config_path: str = RBAC_CONFIG_PATH):
        self.config_path = config_path
        self._ensure_config()

    def _ensure_config(self) -> None:
        """确保 MySQL 数据已初始化，不存在则写入初始默认配置与三级组织机构数据"""
        try:
            from app.mysql_store import mysql_store
            existing_depts = mysql_store.get_departments()
            if not existing_depts:
                for d in DEFAULT_DEPARTMENTS:
                    mysql_store.save_department(d)
        except Exception as e:
            logger.error(f"[RBAC] 持久化角色与用户数据失败: {e}")
            import traceback
            traceback.print_exc()

        data = self._load_raw()
        if not data.get("roles") or not data.get("users"):
            self._save_raw(DEFAULT_RBAC_DATA)

    def _load_raw(self) -> Dict[str, Any]:
        """从 MySQL 表 sys_rbac_roles 与 sys_rbac_users 加载配置（兼容 dept_id 与 post_title）"""
        try:
            from app.mysql_store import mysql_store
            roles_rows = mysql_store.query("SELECT role_id, role_name, description, is_system, menus_json, kb_permissions_json FROM sys_rbac_roles")
            users_rows = mysql_store.query("SELECT user_id, username, name, dept_id, department, post_title, license_no, roles_json, status, created_at FROM sys_rbac_users")

            roles = []
            for r in roles_rows:
                r = dict(r)
                roles.append({
                    "role_id": r["role_id"],
                    "role_name": r["role_name"],
                    "description": r.get("description") or "",
                    "is_system": bool(r.get("is_system")),
                    "menus": json.loads(r["menus_json"]) if r.get("menus_json") else [],
                    "kb_permissions": json.loads(r["kb_permissions_json"]) if r.get("kb_permissions_json") else [],
                })

            users = []
            for u in users_rows:
                u = dict(u)
                users.append({
                    "user_id": u["user_id"],
                    "username": u["username"],
                    "name": u["name"],
                    "dept_id": u.get("dept_id") or "dept_hq",
                    "department": u.get("department") or "",
                    "post_title": u.get("post_title") or "",
                    "license_no": u.get("license_no") or "",
                    "roles": json.loads(u["roles_json"]) if u.get("roles_json") else [],
                    "status": u.get("status") or "active",
                    "created_at": u.get("created_at") or "",
                })

            if roles and users:
                return {"roles": roles, "users": users}
        except Exception as e:
            logger.error(f"[RBAC] 从数据库加载角色与用户异常: {e}")

        return dict(DEFAULT_RBAC_DATA)

    def _save_raw(self, data: Dict[str, Any]) -> None:
        """持久化至 MySQL 数据库专用表 sys_rbac_roles 与 sys_rbac_users"""
        try:
            from app.mysql_store import mysql_store
            # 同步更新子表 sys_rbac_roles 与 sys_rbac_users
            for r in data.get("roles", []):
                mysql_store.execute(
                    """
                    REPLACE INTO sys_rbac_roles (role_id, role_name, description, is_system, menus_json, kb_permissions_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        r.get("role_id"),
                        r.get("role_name"),
                        r.get("description", ""),
                        1 if r.get("is_system") else 0,
                        json.dumps(r.get("menus", []), ensure_ascii=False),
                        json.dumps(r.get("kb_permissions", []), ensure_ascii=False),
                    ),
                )
            for u in data.get("users", []):
                mysql_store.execute(
                    """
                    REPLACE INTO sys_rbac_users (user_id, username, name, dept_id, department, post_title, license_no, roles_json, status, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        u.get("user_id"),
                        u.get("username"),
                        u.get("name"),
                        u.get("dept_id") or "dept_hq",
                        u.get("department", ""),
                        u.get("post_title", ""),
                        u.get("license_no", ""),
                        json.dumps(u.get("roles", []), ensure_ascii=False),
                        u.get("status", "active"),
                        u.get("created_at", ""),
                    ),
                )
        except Exception as e:
            logger.error(f"[RBAC] 持久化角色与用户数据失败: {e}")
            import traceback
            traceback.print_exc()

    # —————————————————— 角色管理 ——————————————————
    def get_roles(self) -> List[Dict[str, Any]]:
        return self._load_raw().get("roles", [])

    def get_role_by_id(self, role_id: str) -> Optional[Dict[str, Any]]:
        for r in self.get_roles():
            if r.get("role_id") == role_id:
                return r
        return None

    def save_role(self, role_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建或修改角色（含允许访问菜单与知识库维护授权）"""
        data = self._load_raw()
        roles: List[Dict[str, Any]] = data.get("roles", [])
        rid = (role_data.get("role_id") or "").strip()
        if not rid:
            rid = f"role_{uuid.uuid4().hex[:6]}"

        menus = role_data.get("menus") or ["menu:qa"]
        kb_perms = role_data.get("kb_permissions") or []

        existing = next((r for r in roles if r.get("role_id") == rid), None)
        if existing:
            existing["role_name"] = role_data.get("role_name", existing.get("role_name"))
            existing["description"] = role_data.get("description", existing.get("description"))
            existing["menus"] = menus
            existing["kb_permissions"] = kb_perms
            ret = existing
        else:
            new_role = {
                "role_id": rid,
                "role_name": role_data.get("role_name", rid),
                "description": role_data.get("description", ""),
                "is_system": False,
                "menus": menus,
                "kb_permissions": kb_perms,
            }
            roles.append(new_role)
            ret = new_role

        data["roles"] = roles
        self._save_raw(data)
        return ret

    def delete_role(self, role_id: str) -> Tuple[bool, str]:
        """删除角色（系统内置角色不可删）"""
        data = self._load_raw()
        roles = data.get("roles", [])
        target = next((r for r in roles if r.get("role_id") == role_id), None)
        if not target:
            return False, "未找到该角色"
        if target.get("is_system"):
            return False, "系统内置基础角色不可删除"

        # 检查是否有关联用户
        users = data.get("users", [])
        in_use = any(role_id in (u.get("roles") or []) for u in users)
        if in_use:
            return False, "仍有用户关联该角色，请先调整用户角色后再删除"

        data["roles"] = [r for r in roles if r.get("role_id") != role_id]
        self._save_raw(data)
        try:
            from app.mysql_store import mysql_store
            mysql_store.execute("DELETE FROM sys_rbac_roles WHERE role_id = ?", (role_id,))
        except Exception as e:
            logger.error(f"[RBAC] 持久化角色与用户数据失败: {e}")
            import traceback
            traceback.print_exc()
        return True, "删除成功"

    # —————————————————— 组织机构三级部门管理 ——————————————————
    def get_departments(self) -> List[Dict[str, Any]]:
        """获取所有平铺的组织架构部门列表"""
        try:
            from app.mysql_store import mysql_store
            depts = mysql_store.get_departments()
            if depts:
                return depts
            # 若为空自动初始化
            for d in DEFAULT_DEPARTMENTS:
                mysql_store.save_department(d)
            return mysql_store.get_departments()
        except Exception:
            return list(DEFAULT_DEPARTMENTS)

    def get_department_by_id(self, dept_id: str) -> Optional[Dict[str, Any]]:
        for d in self.get_departments():
            if d.get("dept_id") == dept_id:
                return d
        return None

    def get_department_tree(self) -> List[Dict[str, Any]]:
        """构建三级组织架构树（含直属与累计下属员工数）"""
        depts = self.get_departments()
        users = self.get_users()

        # 统计每个部门直属人员数
        direct_counts: Dict[str, int] = {}
        for u in users:
            d_id = u.get("dept_id") or "dept_hq"
            direct_counts[d_id] = direct_counts.get(d_id, 0) + 1

        # 构建节点字典
        node_map: Dict[str, Dict[str, Any]] = {}
        for d in depts:
            node = dict(d)
            node["children"] = []
            node["direct_user_count"] = direct_counts.get(d["dept_id"], 0)
            node["user_count"] = node["direct_user_count"]
            node_map[d["dept_id"]] = node

        # 组装树形结构
        root_nodes: List[Dict[str, Any]] = []
        for d in depts:
            dept_id = d["dept_id"]
            parent_id = d.get("parent_id")
            if parent_id and parent_id in node_map:
                node_map[parent_id]["children"].append(node_map[dept_id])
            else:
                root_nodes.append(node_map[dept_id])

        # 后序遍历汇总下属员工数
        def _calc_total(node: Dict[str, Any]) -> int:
            total = node["direct_user_count"]
            for ch in node.get("children", []):
                total += _calc_total(ch)
            node["user_count"] = total
            return total

        for r in root_nodes:
            _calc_total(r)

        return root_nodes

    def save_department(self, dept_data: Dict[str, Any]) -> Tuple[bool, Any]:
        """创建或修改部门（严格控制三级层级）"""
        did = (dept_data.get("dept_id") or "").strip()
        dname = (dept_data.get("dept_name") or "").strip()
        if not dname:
            return False, "部门名称不能为空"

        parent_id = dept_data.get("parent_id")
        if parent_id:
            parent_id = str(parent_id).strip()

        # 计算 level 与 dept_path
        if not parent_id:
            level = 1
            if not did:
                did = f"dept_{uuid.uuid4().hex[:6]}"
            dept_path = did
        else:
            p_node = self.get_department_by_id(parent_id)
            if not p_node:
                return False, f"上级部门 [{parent_id}] 不存在"
            p_level = int(p_node.get("level") or 1)
            if p_level >= 3:
                return False, "组织架构层级最高支持 3 级（总部 -> 二级部室 -> 三级科室），禁止再创建下级部门"
            level = p_level + 1
            if not did:
                did = f"dept_{uuid.uuid4().hex[:6]}"
            dept_path = f"{p_node.get('dept_path', parent_id)}/{did}"

        saved_item = {
            "dept_id": did,
            "dept_name": dname,
            "parent_id": parent_id or None,
            "level": level,
            "dept_path": dept_path,
            "sort_order": int(dept_data.get("sort_order") or 1),
            "leader": dept_data.get("leader") or "",
            "phone": dept_data.get("phone") or "",
            "status": dept_data.get("status") or "active",
        }

        try:
            from app.mysql_store import mysql_store
            ok = mysql_store.save_department(saved_item)
            if ok:
                return True, saved_item
            return False, "底层保存失败"
        except Exception as e:
            return False, f"保存异常: {e}"

    def delete_department(self, dept_id: str) -> Tuple[bool, str]:
        """删除部门（含根节点保护、子部门拦截与在职员工拦截）"""
        if dept_id == "dept_hq":
            return False, "根组织【集团总部】不可删除！"

        depts = self.get_departments()
        target = next((d for d in depts if d["dept_id"] == dept_id), None)
        if not target:
            return False, "未找到该部门"

        # 防御 1: 检查是否存在下属子部门
        has_children = any(d.get("parent_id") == dept_id for d in depts)
        if has_children:
            return False, "该部门下仍有子部门，请先调整或删除下属部门后再删除该部门！"

        # 防御 2: 检查是否有在职员工归属该部门
        users = self.get_users()
        has_users = any(u.get("dept_id") == dept_id for u in users)
        if has_users:
            return False, "该部门下仍有在职员工，请先在员工列表中调整员工所属部门后再删除！"

        try:
            from app.mysql_store import mysql_store
            ok = mysql_store.delete_department(dept_id)
            if ok:
                return True, "删除部门成功"
            return False, "底层删除失败"
        except Exception as e:
            return False, f"删除异常: {e}"

    def get_department_users(self, dept_id: str, include_children: bool = True) -> List[Dict[str, Any]]:
        """按部门获取员工列表（支持包含下属科室成员）"""
        users = self.get_users()
        if not dept_id or dept_id == "all":
            return users

        if not include_children:
            return [u for u in users if (u.get("dept_id") or "dept_hq") == dept_id]

        target_dept = self.get_department_by_id(dept_id)
        if not target_dept:
            return [u for u in users if (u.get("dept_id") or "dept_hq") == dept_id]

        # 递归找到该部门及其所有子孙部门 ID
        target_path = target_dept.get("dept_path") or dept_id
        depts = self.get_departments()
        matched_dept_ids = {
            d["dept_id"] for d in depts
            if d.get("dept_path", "").startswith(target_path) or d["dept_id"] == dept_id
        }
        return [u for u in users if (u.get("dept_id") or "dept_hq") in matched_dept_ids]

    # —————————————————— 用户管理 ——————————————————
    def get_users(self) -> List[Dict[str, Any]]:
        return self._load_raw().get("users", [])

    def get_user_by_username_or_id(self, identifier: str) -> Optional[Dict[str, Any]]:
        users = self.get_users()
        for u in users:
            if u.get("user_id") == identifier or u.get("username") == identifier:
                return u
        return None

    def save_user(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建或修改用户及角色绑定（支持绑定三级组织部门）"""
        data = self._load_raw()
        users: List[Dict[str, Any]] = data.get("users", [])
        uid = (user_data.get("user_id") or "").strip()
        uname = (user_data.get("username") or "").strip()

        # 部门关联处理
        dept_id = user_data.get("dept_id") or "dept_hq"
        dept_info = self.get_department_by_id(dept_id)
        if dept_info:
            # 自动生成如: 法务合规部 / 诉讼风控科
            if dept_info.get("parent_id") and dept_info["parent_id"] != "dept_hq":
                p_dept = self.get_department_by_id(dept_info["parent_id"])
                p_name = p_dept.get("dept_name") if p_dept else ""
                full_dept_name = f"{p_name} / {dept_info['dept_name']}" if p_name else dept_info["dept_name"]
            else:
                full_dept_name = dept_info["dept_name"]
        else:
            full_dept_name = user_data.get("department") or "集团总部"

        existing = next((u for u in users if u.get("user_id") == uid or (uname and u.get("username") == uname)), None)
        if existing:
            existing["name"] = user_data.get("name", existing.get("name"))
            existing["dept_id"] = dept_id
            existing["department"] = user_data.get("department") or full_dept_name
            existing["post_title"] = user_data.get("post_title", existing.get("post_title", ""))
            existing["license_no"] = user_data.get("license_no", existing.get("license_no"))
            existing["roles"] = user_data.get("roles") or existing.get("roles") or ["user"]
            existing["status"] = user_data.get("status", existing.get("status", "active"))
            ret = existing
        else:
            new_uid = uid or f"u{uuid.uuid4().hex[:6]}"
            new_user = {
                "user_id": new_uid,
                "username": uname or new_uid,
                "name": user_data.get("name") or uname or "未命名用户",
                "dept_id": dept_id,
                "department": user_data.get("department") or full_dept_name,
                "post_title": user_data.get("post_title") or "",
                "license_no": user_data.get("license_no") or "",
                "roles": user_data.get("roles") or ["user"],
                "status": user_data.get("status") or "active",
                "created_at": "2026-11-04T10:00:00Z",
            }
            users.append(new_user)
            ret = new_user

        data["users"] = users
        self._save_raw(data)
        return ret

    def delete_user(self, user_id: str) -> Tuple[bool, str]:
        data = self._load_raw()
        users = data.get("users", [])
        target = next((u for u in users if u.get("user_id") == user_id), None)
        if not target:
            return False, "未找到该用户"
        if target.get("username") == "admin":
            return False, "初始超级管理员 admin 账号不可删除"

        data["users"] = [u for u in users if u.get("user_id") != user_id]
        self._save_raw(data)
        try:
            from app.mysql_store import mysql_store
            mysql_store.execute("DELETE FROM sys_rbac_users WHERE user_id = ?", (user_id,))
        except Exception as e:
            logger.error(f"[RBAC] 持久化角色与用户数据失败: {e}")
            import traceback
            traceback.print_exc()
        return True, "删除成功"

    # —————————————————— 权限计算与校验 ——————————————————
    def get_user_effective_permissions(self, identifier: Optional[str] = None) -> Dict[str, Any]:
        """计算指定用户的最终有效权限（合并多角色）"""
        user = self.get_user_by_username_or_id(identifier or "admin")
        if not user:
            # 回退为普通用户
            return {
                "user": {"user_id": "guest", "username": "guest", "name": "访客", "roles": ["user"]},
                "is_admin": False,
                "allowed_menus": ["menu:qa"],
                "manageable_kbs": [],
            }

        roles_list = self.get_roles()
        role_map = {r["role_id"]: r for r in roles_list}

        user_role_ids = user.get("roles") or ["user"]
        is_admin = "admin" in user_role_ids

        all_menus: set[str] = set()
        manageable_kbs: set[str] = set()

        if is_admin:
            all_menus = {m["id"] for m in ALL_MENUS}
            manageable_kbs = {"*"}
        else:
            for rid in user_role_ids:
                r = role_map.get(rid)
                if not r:
                    continue
                if "*" in (r.get("menus") or []):
                    all_menus = {m["id"] for m in ALL_MENUS}
                else:
                    all_menus.update(r.get("menus") or [])

                if "*" in (r.get("kb_permissions") or []):
                    manageable_kbs = {"*"}
                else:
                    manageable_kbs.update(r.get("kb_permissions") or [])

        dept_id = user.get("dept_id") or "dept_hq"
        dept_info = self.get_department_by_id(dept_id)

        return {
            "user": user,
            "department": dept_info or {"dept_id": "dept_hq", "dept_name": user.get("department") or "集团总部", "level": 1},
            "is_admin": is_admin,
            "allowed_menus": sorted(list(all_menus)),
            "manageable_kbs": sorted(list(manageable_kbs)),
        }

    def check_menu_permission(self, identifier: Optional[str], menu_id: str) -> bool:
        """检查用户是否有权访问指定菜单"""
        perms = self.get_user_effective_permissions(identifier)
        if perms.get("is_admin"):
            return True
        return menu_id in perms.get("allowed_menus", [])

    def check_kb_manage_permission(self, identifier: Optional[str], collection_id: str) -> bool:
        """检查用户是否有权维护（导入/删除）指定知识库"""
        perms = self.get_user_effective_permissions(identifier)
        if perms.get("is_admin"):
            return True
        kbs = perms.get("manageable_kbs", [])
        if "*" in kbs:
            return True
        return collection_id in kbs


rbac_manager = RBACManager()

