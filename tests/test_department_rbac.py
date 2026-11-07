"""
三级组织架构与 RBAC 权限联动自动化测试套件
全面覆盖：
1. 三级树状结构构建与递归层级检验（L1总部 -> L2部室 -> L3科室/小组）；
2. 直属与下属累计在职员工数 (user_count) 自底向上后序递归统计；
3. 严格三级上限防御拦截（禁止创建第 4 级部门）；
4. 部门删除三大安全保护（根部门保护、子部门存在拦截、在职员工存在拦截）；
5. 用户绑定部门 (dept_id) 与专业职务 (post_title)，自动映射计算全路径 (department)；
6. 部门员工直属查询 vs 递归穿透包含所有下级科室查询；
7. 角色与菜单权限矩阵不受任何结构变更干扰。
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rbac import rbac_manager


class TestDepartmentRBAC(unittest.TestCase):
    def setUp(self):
        rbac_manager._ensure_config()

    def test_01_department_tree_levels_and_counting(self):
        """测试 1：三级组织架构树结构完整性与员工人数统计自底向上汇总"""
        tree = rbac_manager.get_department_tree()
        self.assertIsInstance(tree, list)
        self.assertGreaterEqual(len(tree), 1)

        # 根节点必须为 L1 总部
        root = tree[0]
        self.assertEqual(root["level"], 1)
        self.assertEqual(root["dept_id"], "dept_hq")
        self.assertIn("children", root)
        self.assertGreater(len(root["children"]), 0)

        # 验证二级部室与三级科室
        dept_ids_l2 = [sub["dept_id"] for sub in root["children"]]
        self.assertIn("dept_legal", dept_ids_l2)
        self.assertIn("dept_tech", dept_ids_l2)

        # 验证二级部门的 level 为 2，且三级科室 level 为 3
        legal_dept = next(d for d in root["children"] if d["dept_id"] == "dept_legal")
        self.assertEqual(legal_dept["level"], 2)
        self.assertGreater(len(legal_dept["children"]), 0)

        litigation_team = next(c for c in legal_dept["children"] if c["dept_id"] == "dept_legal_risk")
        self.assertEqual(litigation_team["level"], 3)
        self.assertEqual(len(litigation_team.get("children", [])), 0)

        # 验证总人数自底向上累计：根节点总人数 >= 各二级部门总人数之和
        sum_l2_users = sum(sub["user_count"] for sub in root["children"])
        self.assertGreaterEqual(root["user_count"], sum_l2_users)

    def test_02_flat_departments_and_paths(self):
        """测试 2：获取平铺组织部门列表与全路径映射"""
        depts = rbac_manager.get_departments()
        self.assertIsInstance(depts, list)
        self.assertGreaterEqual(len(depts), 10)

        dept_map = {d["dept_id"]: d for d in depts}
        self.assertIn("dept_legal_risk", dept_map)
        lit = dept_map["dept_legal_risk"]
        self.assertEqual(lit["level"], 3)
        self.assertEqual(lit["parent_id"], "dept_legal")
        self.assertIn("dept_hq", lit["dept_path"])
        self.assertIn("dept_legal", lit["dept_path"])
        self.assertIn("dept_legal_risk", lit["dept_path"])

    def test_03_create_department_and_level_limit_defense(self):
        """测试 3：部门创建与严格三级上限防御（禁止创建第 4 级）"""
        # 1. 在二级科技部(dept_tech)下创建合法三级科室
        new_l3 = {
            "dept_id": "dept_tech_sec_unit",
            "dept_name": "网络数据安全科",
            "parent_id": "dept_tech",
            "leader": "周工程师",
            "phone": "010-88001122",
            "sort_order": 90
        }
        ok, saved = rbac_manager.save_department(new_l3)
        self.assertTrue(ok)
        self.assertEqual(saved["level"], 3)
        self.assertEqual(saved["parent_id"], "dept_tech")

        # 2. 尝试在新建的第 3 级科室下创建第 4 级部门 -> 必须触发层级越界防御
        invalid_l4 = {
            "dept_id": "dept_invalid_l4",
            "dept_name": "非法四级越界工作组",
            "parent_id": "dept_tech_sec_unit",
            "leader": "越界人员"
        }
        ok4, err4 = rbac_manager.save_department(invalid_l4)
        self.assertFalse(ok4)
        self.assertIn("最高支持 3 级", err4)

        # 清理刚才测试创建的合法三级科室
        del_ok, _ = rbac_manager.delete_department("dept_tech_sec_unit")
        self.assertTrue(del_ok)

    def test_04_delete_protection_rules(self):
        """测试 4：组织部门删除三大安全防御规则"""
        # 规则 1：禁止删除总部根机构
        ok1, err1 = rbac_manager.delete_department("dept_hq")
        self.assertFalse(ok1)
        self.assertIn("不可删除", err1)

        # 规则 2：禁止删除存在下属子部门的部门（如二级法务合规部下设科室）
        ok2, err2 = rbac_manager.delete_department("dept_legal")
        self.assertFalse(ok2)
        self.assertIn("子部门", err2)

        # 规则 3：禁止删除存在在职员工的部门（如诉讼风控科有李合规）
        ok3, err3 = rbac_manager.delete_department("dept_legal_risk")
        self.assertFalse(ok3)
        self.assertIn("在职员工", err3)

    def test_05_user_dept_binding_and_filtering(self):
        """测试 5：用户部门绑定、职务赋予及直属与递归穿透查询"""
        # 1. 注册一名新测试员工，挂载到三级科室“算法与AI智能组”(dept_tech_ai)
        test_user = {
            "user_id": "u_test_ai_001",
            "username": "ai_researcher_test",
            "name": "高算法",
            "dept_id": "dept_tech_ai",
            "post_title": "大模型高级专家",
            "roles": ["user"]
        }
        saved_u = rbac_manager.save_user(test_user)
        self.assertEqual(saved_u["dept_id"], "dept_tech_ai")
        self.assertEqual(saved_u["post_title"], "大模型高级专家")
        self.assertIn("科技部", saved_u["department"])
        self.assertIn("算法与AI智能组", saved_u["department"])

        # 2. 直属三级科室查询：包含该测试人员
        l3_users = rbac_manager.get_department_users("dept_tech_ai", include_children=False)
        l3_usernames = [u["username"] for u in l3_users]
        self.assertIn("ai_researcher_test", l3_usernames)

        # 3. 上级二级部门 (dept_tech) 递归穿透查询 (include_children=True)：必须包含挂在该三级科室的人员
        rec_users = rbac_manager.get_department_users("dept_tech", include_children=True)
        rec_usernames = [u["username"] for u in rec_users]
        self.assertIn("ai_researcher_test", rec_usernames)

        # 4. 上级二级部门不递归查询 (include_children=False)：不得包含仅挂在三级科室的人员
        norec_users = rbac_manager.get_department_users("dept_tech", include_children=False)
        norec_usernames = [u["username"] for u in norec_users]
        self.assertNotIn("ai_researcher_test", norec_usernames)

        # 5. 总部 (dept_hq) 递归穿透查询：包含所有下辖员工
        all_rec_users = rbac_manager.get_department_users("dept_hq", include_children=True)
        all_rec_names = [u["username"] for u in all_rec_users]
        self.assertIn("ai_researcher_test", all_rec_names)
        self.assertIn("admin", all_rec_names)

        # 清理测试用户
        rbac_manager.delete_user("u_test_ai_001")

    def test_06_rbac_roles_and_kb_permissions_integrity(self):
        """测试 6：组织架构引入后，原有 RBAC 角色与知识库矩阵完好性"""
        roles = rbac_manager.get_roles()
        self.assertGreaterEqual(len(roles), 4)

        admin_role = next(r for r in roles if r["role_id"] == "admin")
        self.assertIn("*", admin_role["menus"])
        self.assertIn("*", admin_role["kb_permissions"])

        compliance_role = next(r for r in roles if r["role_id"] == "compliance_officer")
        self.assertIn("menu:rules", compliance_role["menus"])

        # 校验权限计算：超级管理员拥有通配全权 *
        admin_perm = rbac_manager.get_user_effective_permissions("admin")
        self.assertTrue(admin_perm["is_admin"])
        self.assertIn("*", admin_perm["manageable_kbs"])


if __name__ == "__main__":
    unittest.main()
