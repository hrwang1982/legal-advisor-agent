import unittest
from app.config import settings, LLM_PROVIDER_PRESETS

class TestModelPresetSync(unittest.TestCase):
    def setUp(self):
        # 记录初始状态以便 tearDown 严格还原
        settings.load_from_disk()
        self.initial_qwen_key = settings.platform_credentials.get("qwen", {}).get("api_key", "")

    def tearDown(self):
        # 还原干净现场：恢复默认 qwen 激活，清空其他平台测试产生的临时凭据
        settings.platform_credentials["deepseek"]["api_key"] = ""
        settings.platform_credentials["deepseek"]["verified"] = False
        settings.platform_credentials["siliconflow"]["api_key"] = ""
        settings.platform_credentials["siliconflow"]["verified"] = False
        settings.platform_credentials["local_openai"]["api_key"] = ""
        settings.platform_credentials["local_openai"]["verified"] = False

        settings.update_runtime({
            "mock": False,
            "llm_provider": "qwen",
            "llm_api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "llm_model": "qwen-turbo",
            "llm_api_key": self.initial_qwen_key,
            "verified": True,
        })

    def test_available_models_only_includes_verified(self):
        """测试白名单机制：未测通的模型绝不展示在智能问答列表中。"""
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
                    self.assertNotIn(mid, avail_ids, f"未验证平台 {p_key} 的模型 {mid} 绝不应出现在可用列表中")

    def test_credentials_physical_isolation_on_switch(self):
        """测试各平台凭据物理隔离：从通义千问切换到硅基流动时，输入/配置 Key 坚决置空，绝不串用千问 Key。"""
        # 确认当前在 qwen 平台且千问拥有有效 Key
        self.assertEqual(settings.llm_provider, "qwen")
        self.assertTrue(bool(settings.llm_api_key))
        qwen_key = settings.llm_api_key

        # 1. 切换平台预设至 siliconflow (此时未配置专属 Key)
        pub_sf = settings.update_runtime({"provider_preset": "siliconflow"})

        # 核心断言：siliconflow 的当前生效 Key 和独立凭据 Key 必须坚决置空！
        self.assertEqual(pub_sf["llm_api_key"], "")
        self.assertEqual(pub_sf["platform_credentials"]["siliconflow"]["api_key"], "")
        self.assertFalse(pub_sf["platform_credentials"]["siliconflow"]["verified"])

        # 核心断言：千问的专属 Key 依然安然无恙，未受任何污染或清空
        self.assertEqual(pub_sf["platform_credentials"]["qwen"]["api_key"], qwen_key)

        # 核心断言：由于 siliconflow 尚未填 Key 且未测通，其模型绝不在 available_models 中
        avail_ids_sf = [m["id"] for m in pub_sf["available_models"]]
        self.assertNotIn("Qwen/Qwen2.5-72B-Instruct", avail_ids_sf)

        # 2. 模拟用户为 siliconflow 填入专属 Key 并测试保存
        sf_custom_key = "sk-siliconflow-exclusive-test-key-9999"
        pub_sf_configured = settings.update_runtime({
            "llm_provider": "siliconflow",
            "llm_api_key": sf_custom_key,
            "verified": True,
        })
        self.assertEqual(pub_sf_configured["llm_api_key"], sf_custom_key)
        self.assertEqual(pub_sf_configured["platform_credentials"]["siliconflow"]["api_key"], sf_custom_key)
        self.assertTrue(pub_sf_configured["platform_credentials"]["siliconflow"]["verified"])

        # 验证此时 siliconflow 的模型进入白名单
        avail_ids_after = [m["id"] for m in pub_sf_configured["available_models"]]
        self.assertIn("Qwen/Qwen2.5-72B-Instruct", avail_ids_after)

        # 3. 切回 qwen 预设，必须回显千问专属 Key，绝不串入 siliconflow 的 Key
        pub_qwen_again = settings.update_runtime({"provider_preset": "qwen"})
        self.assertEqual(pub_qwen_again["llm_api_key"], qwen_key)
        self.assertEqual(pub_qwen_again["platform_credentials"]["qwen"]["api_key"], qwen_key)

        # 4. 再次切回 siliconflow，必须回显刚刚保存的 siliconflow 专属 Key，绝不串入千问的 Key
        pub_sf_again = settings.update_runtime({"provider_preset": "siliconflow"})
        self.assertEqual(pub_sf_again["llm_api_key"], sf_custom_key)
        self.assertEqual(pub_sf_again["platform_credentials"]["siliconflow"]["api_key"], sf_custom_key)

    def test_provider_field_in_available_models(self):
        """测试白名单模型必须携带 provider 字段，便于问答列表精准路由。"""
        pub = settings.to_public_dict()
        for m in pub["available_models"]:
            self.assertIn("provider", m)
            self.assertTrue(bool(m["provider"]))

    def test_llm_test_connection_validation(self):
        """测试连通性探测接口的校验规则。"""
        # 测试在非 mock 模式下没有 api_key 时测试失败
        res = settings.test_connection(
            api_base="https://api.deepseek.com/v1",
            model="deepseek-chat",
            api_key="",
            provider="deepseek"
        )
        self.assertFalse(res["ok"])
        self.assertIn("API Key", res["message"])

        # 本地服务在未配置 key 时不强制拦截 key，但若服务未启动应友好提示连接失败
        res_local = settings.test_connection(
            api_base="http://127.0.0.1:59999/v1",
            model="qwen2.5:14b",
            api_key="",
            provider="local_openai"
        )
        self.assertFalse(res_local["ok"])
        self.assertTrue("拒绝" in res_local["message"] or "异常" in res_local["message"])

if __name__ == "__main__":
    unittest.main()
