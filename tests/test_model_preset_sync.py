import unittest
from app.config import settings, LLM_PROVIDER_PRESETS

class TestModelPresetSync(unittest.TestCase):
    def setUp(self):
        settings.load_from_disk()

    def test_available_models_only_includes_verified(self):
        pub = settings.to_public_dict()
        avail_ids = [m["id"] for m in pub["available_models"]]
        # mock 必然可用
        self.assertIn("mock", avail_ids)

        # 检查未配置或未 verified 的平台模型，绝不能泄漏进入可用列表
        for p_key, p_cred in settings.platform_credentials.items():
            preset = LLM_PROVIDER_PRESETS.get(p_key, {})
            p_models = [m["id"] for m in preset.get("models", [])]
            if not p_cred.get("verified"):
                for mid in p_models:
                    self.assertNotIn(mid, avail_ids, f"未验证平台 {p_key} 的模型 {mid} 不应出现在可用列表中")

    def test_switch_and_verify_provider(self):
        # 模拟管理员测试并保存 deepseek
        settings.update_runtime({
            "mock": False,
            "llm_provider": "deepseek",
            "llm_api_base": "https://api.deepseek.com/v1",
            "llm_model": "deepseek-chat",
            "llm_api_key": "sk-fake-key-test-12345678",
            "verified": True
        })

        # 再次获取配置，deepseek 的模型必须出现在 available_models 中
        pub = settings.to_public_dict()
        avail_ids = [m["id"] for m in pub["available_models"]]
        self.assertIn("deepseek-chat", avail_ids)
        self.assertIn("deepseek-reasoner", avail_ids)

        # 切换到未验证的 siliconflow，其模型绝不应该出现在可用列表中
        settings.update_runtime({
            "mock": False,
            "llm_provider": "siliconflow",
            "llm_api_base": "https://api.siliconflow.cn/v1",
            "llm_model": "Qwen/Qwen2.5-72B-Instruct",
            "llm_api_key": "",
            "verified": False
        })
        pub_sf = settings.to_public_dict()
        avail_ids_sf = [m["id"] for m in pub_sf["available_models"]]
        self.assertNotIn("Qwen/Qwen2.5-72B-Instruct", avail_ids_sf)

        # 恢复默认 qwen
        settings.update_runtime({
            "mock": False,
            "llm_provider": "qwen",
            "llm_api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "llm_model": "qwen-turbo",
            "llm_api_key": "sk-56cfaaa8be4b45c1aa047a74a6762dc0",
            "verified": True
        })

    def test_llm_test_connection_no_key(self):
        # 测试在非 mock 模式下没有 api_key 测试失败
        res = settings.test_connection(
            api_base="https://api.deepseek.com/v1",
            model="deepseek-chat",
            api_key="",
            provider="deepseek"
        )
        self.assertFalse(res["ok"])
        self.assertIn("API Key", res["message"])

if __name__ == "__main__":
    unittest.main()
