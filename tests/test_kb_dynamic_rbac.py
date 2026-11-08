"""
知识库动态同源同步与多部门管理管控自动化测试套件
全面覆盖：
1. 知识库与授权矩阵动态同源（创建即纳管、改名即同步）；
2. 知识库主责管理部门 (owner_dept_id) 绑定与自动归属；
3. 基于部门管辖权 (ABAC) 与角色权限 (RBAC) 融合的权限裁决；
4. 知识库删除时权限级联清理闭环（杜绝失效脏权限残留）。
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.kb import kb_store
from app.mysql_store import mysql_store
from app.rbac import rbac_manager


class TestKbDynamicRBAC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mysql_store.init_schema()
        rbac_manager._ensure_config()

    def test_01_dynamic_kb_collections_synchronization(self):
        """测试 1：知识库管理与权限中心动态同源同步（创建即纳入管控）"""
        # 1. 在知识库管理中创建一个全新业务知识库，绑定主责部门为金融商事业务部 (dept_biz)
        test_kb_name = "信贷风控专项制度库"
        created = kb_store.create_kb_tree_node(
            name=test_kb_name,
            node_type="kb",
            description="面向企业信贷与普惠金融风控规章",
            owner_dept_id="dept_biz",
            visibility="dept"
        )
        new_kb_id = created["node_id"]
        self.assertTrue(new_kb_id.startswith("kb_"))
        self.assertEqual(created["owner_dept_id"], "dept_biz")

        # 2. 验证 mysql_store 动态获取的知识库集合立即包含该知识库
        dynamic_kbs = mysql_store.get_dynamic_kb_collections()
        kb_ids = [k["id"] for k in dynamic_kbs]
        self.assertIn(new_kb_id, kb_ids)

        found_kb = next(k for k in dynamic_kbs if k["id"] == new_kb_id)
        self.assertEqual(found_kb["name"], test_kb_name)
        self.assertEqual(found_kb["owner_dept_id"], "dept_biz")
        self.assertIn("金融商事业务部", found_kb["owner_dept_name"])

        # 3. 清理测试知识库
        kb_store.delete_kb_tree_node(new_kb_id)

    def test_02_department_ownership_permission_decision(self):
        """测试 2：基于部门管辖权与角色的知识库维护权限裁决"""
        # 1. 创建属于科技部 (dept_tech) 的知识库
        tech_kb = kb_store.create_kb_tree_node(
            name="数据安全与算法内规库",
            node_type="kb",
            owner_dept_id="dept_tech",
            visibility="dept"
        )
        tech_kb_id = tech_kb["node_id"]

        # 2. 检查管理员：超级管理员拥有全权
        admin_perms = rbac_manager.get_user_effective_permissions("admin")
        self.assertTrue(admin_perms["is_admin"])
        self.assertTrue(rbac_manager.check_kb_manage_permission("admin", tech_kb_id))

        # 3. 检查科技部知识库专员 (王知识 wang_editor): 所属 dept_tech，拥有 kb_editor 角色
        wang_perms = rbac_manager.get_user_effective_permissions("wang_editor")
        self.assertIn(tech_kb_id, wang_perms["manageable_kbs"])
        self.assertTrue(rbac_manager.check_kb_manage_permission("wang_editor", tech_kb_id))

        # 4. 检查非主责部门人员且未授权角色 (陈员工 chen_user): 所属人力资源部，角色 user
        self.assertFalse(rbac_manager.check_kb_manage_permission("chen_user", tech_kb_id))

        # 清理
        kb_store.delete_kb_tree_node(tech_kb_id)

    def test_03_role_matrix_explicit_grant_and_update(self):
        """测试 3：跨部门显式角色授权与权限更新"""
        # 创建一个跨部门通用库
        corp_kb = kb_store.create_kb_tree_node(
            name="集团合规通报库",
            node_type="kb",
            owner_dept_id="dept_legal",
            visibility="public"
        )
        corp_kb_id = corp_kb["node_id"]

        # 获取知识库运维专员角色 kb_editor
        role_editor = rbac_manager.get_role_by_id("kb_editor")
        self.assertIsNotNone(role_editor)

        # 显式授权该角色可维护 corp_kb_id
        orig_perms = list(role_editor.get("kb_permissions") or [])
        updated_perms = list(set(orig_perms + [corp_kb_id]))
        role_editor["kb_permissions"] = updated_perms
        rbac_manager.save_role(role_editor)

        # 验证专员具备该权限
        self.assertTrue(rbac_manager.check_kb_manage_permission("wang_editor", corp_kb_id))

        # 恢复角色配置
        role_editor["kb_permissions"] = orig_perms
        rbac_manager.save_role(role_editor)

        # 清理知识库
        kb_store.delete_kb_tree_node(corp_kb_id)

    def test_04_cascade_cleanup_when_kb_deleted(self):
        """测试 4：知识库删除时自动级联清理角色表中的失效残留权限"""
        # 1. 创建知识库
        temp_kb = kb_store.create_kb_tree_node(
            name="待销毁临时规章库",
            node_type="kb",
            owner_dept_id="dept_hq"
        )
        temp_kb_id = temp_kb["node_id"]

        # 2. 为 compliance_officer 角色授予该库权限
        comp_role = rbac_manager.get_role_by_id("compliance_officer")
        orig_kbs = list(comp_role.get("kb_permissions") or [])
        comp_role["kb_permissions"] = orig_kbs + [temp_kb_id]
        rbac_manager.save_role(comp_role)

        # 确认已添加
        self.assertIn(temp_kb_id, rbac_manager.get_role_by_id("compliance_officer")["kb_permissions"])

        # 3. 删除该知识库
        kb_store.delete_kb_tree_node(temp_kb_id)

        # 4. 验证权限中心角色记录中已自动清除了该失效 node_id，无脏数据
        reloaded_role = rbac_manager.get_role_by_id("compliance_officer")
        self.assertNotIn(temp_kb_id, reloaded_role.get("kb_permissions") or [])

        # 5. 验证动态集合中已不再包含该知识库
        dyn_kbs = mysql_store.get_dynamic_kb_collections()
        dyn_ids = [k["id"] for k in dyn_kbs]
        self.assertNotIn(temp_kb_id, dyn_ids)


if __name__ == "__main__":
    unittest.main()

