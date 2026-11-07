import unittest
from app.knowledge import knowledge_center

class TestKnowledgeConfigSync(unittest.TestCase):
    def setUp(self):
        self.initial_all = knowledge_center.get_all()

    def test_knowledge_config_not_empty(self):
        """测试外置知识规则中心具备各项非空基础业务规则。"""
        all_data = knowledge_center.get_all()
        # 1. 验证 config_path 属性存在且正常访问
        self.assertTrue(hasattr(knowledge_center, "config_path"))
        self.assertTrue(bool(knowledge_center.config_path))

        # 2. 验证前端下拉框对应的 7 个核心业务规则模块均存在且包含有效数据
        required_sections = [
            "legal_synonym_map",
            "cross_reference_graph",
            "article_succession_list",
            "special_general_relations",
            "client_registry",
            "case_type_rules",
            "reviewers_registry",
        ]
        for sec in required_sections:
            self.assertIn(sec, all_data, f"业务知识规则 {sec} 必须存在")
            val = all_data[sec]
            self.assertTrue(bool(val), f"业务知识规则 {sec} 数据绝不能显示为空")

        # 验证同义词表包含常用法律转换
        synonyms = all_data["legal_synonym_map"]
        self.assertIn("炒鱿鱼", synonyms)
        self.assertIn("违法解除", synonyms["炒鱿鱼"])

    def test_update_and_persist_section(self):
        """测试按 section 局部更新业务规则并持久化回写。"""
        # 针对同义词表添加一条测试用例并持久化
        test_key = "测试口语词_临时"
        test_val = "测试标准法言法语_生效"
        synonyms = dict(self.initial_all.get("legal_synonym_map") or {})
        synonyms[test_key] = test_val

        updated = knowledge_center.update_section("legal_synonym_map", synonyms)
        self.assertIn("legal_synonym_map", updated)
        self.assertEqual(updated["legal_synonym_map"].get(test_key), test_val)

        # 重新从底层读取验证持久化
        reloaded = knowledge_center.reload()
        self.assertEqual(reloaded["legal_synonym_map"].get(test_key), test_val)

        # 还原现场
        synonyms.pop(test_key, None)
        knowledge_center.update_section("legal_synonym_map", synonyms)
        final_data = knowledge_center.reload()
        self.assertNotIn(test_key, final_data["legal_synonym_map"])

if __name__ == "__main__":
    unittest.main()
