"""基于角色的访问控制（RBAC）管理器：
1. 菜单访问权限：不同角色访问不同菜单，系统管理员 (admin) 拥有通配权限访问所有；
2. 知识库数据维护权限：不同角色对不同知识库（内规库、外规库等）配置独立的维护/导入权限；
3. 用户与角色解耦配置：支持为用户分配一个或多个角色；
4. 数据基于 JSON 文件持久化（app/data/rbac_config.json），开箱即用，支持热更新。
"""
from __future__ import annotations

import json
import os
import uuid
from typing import Any, Dict, List, Optional, Tuple

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
            "department": "管理委员会",
            "license_no": "11101201010888888",
            "roles": ["admin"],
            "status": "active",
            "created_at": "2026-11-01T08:00:00Z",
        },
        {
            "user_id": "u1002",
            "username": "li_compliance",
            "name": "李合规",
            "department": "法务合规部",
            "license_no": "13101201810012345",
            "roles": ["compliance_officer"],
            "status": "active",
            "created_at": "2026-11-02T09:30:00Z",
        },
        {
            "user_id": "u1003",
            "username": "wang_editor",
            "name": "王知识",
            "department": "知识资产管理中心",
            "license_no": "EMP-205",
            "roles": ["kb_editor"],
            "status": "active",
            "created_at": "2026-11-02T10:15:00Z",
        },
        {
            "user_id": "u1004",
            "username": "zhang_lawyer",
            "name": "赵律师",
            "department": "金融商事业务部",
            "license_no": "13101201910056789",
            "roles": ["lawyer"],
            "status": "active",
            "created_at": "2026-11-03T14:20:00Z",
        },
        {
            "user_id": "u1005",
            "username": "chen_user",
            "name": "陈员工",
            "department": "零售金融部",
            "license_no": "EMP-908",
            "roles": ["user"],
            "status": "active",
            "created_at": "2026-11-04T08:00:00Z",
        },
    ],
}


class RBACManager:
    """RBAC 核心权限与数据管理类"""

    def __init__(self, config_path: str = RBAC_CONFIG_PATH):
        self.config_path = config_path
        self._ensure_config()

    def _ensure_config(self) -> None:
        """确保 MySQL 数据已初始化，不存在则写入初始默认配置"""
        data = self._load_raw()
        if not data.get("roles"):
            self._save_raw(DEFAULT_RBAC_DATA)

    def _load_raw(self) -> Dict[str, Any]:
        """从 MySQL 表 sys_rbac_roles 与 sys_rbac_users 加载配置"""
        try:
            from app.mysql_store import mysql_store
            roles_rows = mysql_store.query("SELECT role_id, role_name, description, is_system, menus_json, kb_permissions_json FROM sys_rbac_roles")
            users_rows = mysql_store.query("SELECT user_id, username, name, department, license_no, roles_json, status, created_at FROM sys_rbac_users")

            roles = []
            for r in roles_rows:
                roles.append({
                    "role_id": r["role_id"],
                    "role_name": r["role_name"],
                    "description": r["description"] or "",
                    "is_system": bool(r["is_system"]),
                    "menus": json.loads(r["menus_json"]) if r["menus_json"] else [],
                    "kb_permissions": json.loads(r["kb_permissions_json"]) if r["kb_permissions_json"] else [],
                })

            users = []
            for u in users_rows:
                users.append({
                    "user_id": u["user_id"],
                    "username": u["username"],
                    "name": u["name"],
                    "department": u["department"] or "",
                    "license_no": u["license_no"] or "",
                    "roles": json.loads(u["roles_json"]) if u["roles_json"] else [],
                    "status": u["status"] or "active",
                    "created_at": u["created_at"] or "",
                })

            if roles and users:
                return {"roles": roles, "users": users}

            cfg = mysql_store.get_config("rbac_config")
            if cfg and isinstance(cfg, dict) and "roles" in cfg and "users" in cfg:
                return cfg
        except Exception:
            pass

        return dict(DEFAULT_RBAC_DATA)

    def _save_raw(self, data: Dict[str, Any]) -> None:
        """持久化至 MySQL 数据库专用表 sys_rbac_roles 与 sys_rbac_users"""
        try:
            from app.mysql_store import mysql_store
            mysql_store.save_config("rbac_config", data)
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
                    REPLACE INTO sys_rbac_users (user_id, username, name, department, license_no, roles_json, status, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        u.get("user_id"),
                        u.get("username"),
                        u.get("name"),
                        u.get("department", ""),
                        u.get("license_no", ""),
                        json.dumps(u.get("roles", []), ensure_ascii=False),
                        u.get("status", "active"),
                        u.get("created_at", ""),
                    ),
                )
        except Exception:
            pass

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
        except Exception:
            pass
        return True, "删除成功"

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
        """创建或修改用户及角色绑定"""
        data = self._load_raw()
        users: List[Dict[str, Any]] = data.get("users", [])
        uid = (user_data.get("user_id") or "").strip()
        uname = (user_data.get("username") or "").strip()

        existing = next((u for u in users if u.get("user_id") == uid or (uname and u.get("username") == uname)), None)
        if existing:
            existing["name"] = user_data.get("name", existing.get("name"))
            existing["department"] = user_data.get("department", existing.get("department"))
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
                "department": user_data.get("department") or "业务部",
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
        except Exception:
            pass
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

        return {
            "user": user,
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

