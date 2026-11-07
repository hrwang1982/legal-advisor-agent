"""运行时配置与多平台大模型预设（支持 OpenAI 兼容接口：DeepSeek / 阿里百炼千问 Qwen / 本地 Ollama / vLLM / Xinference）。
支持运行时在线热更新 LLM 与 Embedding（默认推荐 BAAI/bge-m3）配置。
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict


def _bool(v: str | None, default: bool = False) -> bool:
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


# 常用大模型与向量模型平台预设（供前端一键切换或配置参考，提供官方真实可用模型列表）
LLM_PROVIDER_PRESETS: Dict[str, Dict[str, Any]] = {
    "qwen": {
        "name": "阿里云百炼 / 通义千问 (DashScope OpenAI兼容)",
        "llm_api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "llm_model": "qwen-plus",
        "embed_api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "embed_model": "text-embedding-v3",
        "models": [
            {"id": "qwen-plus", "name": "qwen-plus (均衡旗舰 · 推荐法律问答)"},
            {"id": "qwen-max", "name": "qwen-max (复杂推理 · 深度法理分析)"},
            {"id": "qwen-long", "name": "qwen-long (超长案卷 · 百万字法规解析)"},
            {"id": "qwen3.6-plus", "name": "qwen3.6-plus (新一代旗舰 · 复杂问答推荐)"},
            {"id": "qwen3.8-flash", "name": "qwen3.8-flash (新一代极速 · 毫秒级流式响应)"},
        ],
    },
    "local_openai": {
        "name": "本地 OpenAI 兼容服务 (vLLM / Ollama / Xinference + bge-m3)",
        "llm_api_base": "http://localhost:11434/v1",
        "llm_model": "qwen2.5:14b",
        "embed_api_base": "http://localhost:11434/v1",
        "embed_model": "BAAI/bge-m3",
        "models": [
            {"id": "qwen2.5:14b", "name": "qwen2.5:14b (本地 Ollama 推荐)"},
            {"id": "qwen2.5:7b", "name": "qwen2.5:7b (轻量本地测试)"},
            {"id": "qwen2.5:32b", "name": "qwen2.5:32b (本地高阶推理)"},
        ],
    },
    "deepseek": {
        "name": "DeepSeek 官方 API + BGE-M3 向量",
        "llm_api_base": "https://api.deepseek.com/v1",
        "llm_model": "deepseek-chat",
        "embed_api_base": "http://localhost:11434/v1",
        "embed_model": "BAAI/bge-m3",
        "models": [
            {"id": "deepseek-chat", "name": "deepseek-chat (DeepSeek-V3 官方通用)"},
            {"id": "deepseek-reasoner", "name": "deepseek-reasoner (DeepSeek-R1 深度思考推理)"},
        ],
    },
    "siliconflow": {
        "name": "SiliconFlow (Qwen + BAAI/bge-m3 在线兼容接口)",
        "llm_api_base": "https://api.siliconflow.cn/v1",
        "llm_model": "Qwen/Qwen2.5-72B-Instruct",
        "embed_api_base": "https://api.siliconflow.cn/v1",
        "embed_model": "BAAI/bge-m3",
        "models": [
            {"id": "Qwen/Qwen2.5-72B-Instruct", "name": "Qwen/Qwen2.5-72B-Instruct (硅基流动千问)"},
            {"id": "deepseek-ai/DeepSeek-V3", "name": "deepseek-ai/DeepSeek-V3 (硅基流动 V3)"},
            {"id": "deepseek-ai/DeepSeek-R1", "name": "deepseek-ai/DeepSeek-R1 (硅基流动 R1)"},
        ],
    },
}


class Settings:
    # —— 大模型平台标识：qwen / local_openai / deepseek / siliconflow / custom ——
    llm_provider: str = os.getenv("LLM_PROVIDER", "qwen")

    # —— LLM（OpenAI 兼容接口：支持千问 DashScope / 本地 vLLM/Ollama / DeepSeek）——
    llm_api_key: str = os.getenv("LLM_API_KEY", os.getenv("DASHSCOPE_API_KEY", ""))
    llm_api_base: str = os.getenv(
        "LLM_API_BASE",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    llm_model: str = os.getenv("LLM_MODEL", "qwen-plus")
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    llm_timeout: float = float(os.getenv("LLM_TIMEOUT", "30.0"))
    llm_max_retries: int = int(os.getenv("LLM_MAX_RETRIES", "2"))

    # —— 嵌入向量模型（推荐 BAAI/bge-m3，支持 OpenAI /v1/embeddings 兼容接口与本地 bge-m3 特征向量）——
    embed_enabled: bool = _bool(os.getenv("EMBED_ENABLED", "true"), True)
    embed_api_key: str = os.getenv("EMBED_API_KEY", os.getenv("LLM_API_KEY", ""))
    embed_api_base: str = os.getenv("EMBED_API_BASE", "")
    embed_model: str = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
    embed_dim: int = int(os.getenv("EMBED_DIM", "1024"))
    hybrid_dense_weight: float = float(os.getenv("HYBRID_DENSE_WEIGHT", "0.45"))

    # —— 运行与安全门禁模式 ——
    mock: bool = _bool(os.getenv("MOCK", "true"), True)  # 默认 MOCK，免密钥即可跑通演示
    strict_review_auth: bool = _bool(os.getenv("STRICT_REVIEW_AUTH", "false"), False)
    top_k: int = int(os.getenv("TOP_K", "5"))

    # —— 数据 / 知识配置 / 向量库路径 ——
    data_path: str = os.getenv("DATA_PATH", "app/data/sample_laws.json")
    knowledge_config_path: str = os.getenv(
        "KNOWLEDGE_CONFIG_PATH",
        "app/data/knowledge_config.json",
    )
    llm_config_path: str = os.getenv("LLM_CONFIG_PATH", "app/data/llm_config.json")
    kb_db_path: str = os.getenv("KB_DB_PATH", "app/data/kb_vectors.db")

    # —— 存储引擎扩展（MySQL + Elasticsearch 8.x 选配，缺省自动使用内置 SQLite + 内存 BM25+BGE-M3）——
    storage_backend: str = os.getenv("STORAGE_BACKEND", "sqlite")  # mysql / sqlite
    mysql_host: str = os.getenv("MYSQL_HOST", "localhost")
    mysql_port: int = int(os.getenv("MYSQL_PORT", "3306"))
    mysql_user: str = os.getenv("MYSQL_USER", "root")
    mysql_password: str = os.getenv("MYSQL_PASSWORD", "")
    mysql_database: str = os.getenv("MYSQL_DATABASE", "legal_advisor")

    # —— Elasticsearch 8.x 混合检索配置 ——
    es_enabled: bool = _bool(os.getenv("ES_ENABLED", "false"), False)
    es_hosts: str = os.getenv("ES_HOSTS", "http://localhost:9200")
    es_index: str = os.getenv("ES_INDEX", "legal_chunks")
    es_api_key: str = os.getenv("ES_API_KEY", "")

    # —— 服务 ——
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))

    # 各平台独立凭据与测试验证状态（核心：未填 key 或未测试通过的模版不可用）
    platform_credentials: Dict[str, Dict[str, Any]] = {
        "qwen": {
            "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "api_key": os.getenv("LLM_API_KEY", os.getenv("DASHSCOPE_API_KEY", "sk-56cfaaa8be4b45c1aa047a74a6762dc0")),
            "model": "qwen-plus",
            "verified": bool(os.getenv("LLM_API_KEY", os.getenv("DASHSCOPE_API_KEY", "sk-56cfaaa8be4b45c1aa047a74a6762dc0"))),
        },
        "deepseek": {
            "api_base": "https://api.deepseek.com/v1",
            "api_key": "",
            "model": "deepseek-chat",
            "verified": False,
        },
        "siliconflow": {
            "api_base": "https://api.siliconflow.cn/v1",
            "api_key": "",
            "model": "Qwen/Qwen2.5-72B-Instruct",
            "verified": False,
        },
        "local_openai": {
            "api_base": "http://localhost:11434/v1",
            "api_key": "",
            "model": "qwen2.5:14b",
            "verified": False,
        },
    }

    def test_connection(
        self,
        api_base: str,
        model: str,
        api_key: str,
        provider: str = "qwen",
    ) -> Dict[str, Any]:
        """真实探测大模型连通性与密钥有效性。"""
        api_base = (api_base or "").strip().rstrip("/")
        model = (model or "").strip()
        api_key = (api_key or "").strip()

        if model.lower() == "mock":
            return {"ok": True, "message": "MOCK 模式检测通过（零外部依赖）"}

        if not api_base:
            return {"ok": False, "message": "测试失败：未配置 API Base 地址"}
        if not model:
            return {"ok": False, "message": "测试失败：未指定 LLM 模型名称"}

        is_local = "localhost" in api_base or "127.0.0.1" in api_base
        if not is_local and not api_key:
            return {"ok": False, "message": "测试失败：当前平台必须填入对应的有效 API Key"}

        url = f"{api_base}/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 10,
        }
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        import time
        import urllib.request
        import urllib.error
        import json

        t0 = time.time()
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=15.0) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            latency_ms = int((time.time() - t0) * 1000)
            choices = body.get("choices") or []
            if choices:
                return {
                    "ok": True,
                    "message": f"连通测试通过！模型 [{model}] 响应正常（耗时 {latency_ms}ms）",
                    "latency_ms": latency_ms,
                }
            return {
                "ok": False,
                "message": f"测试异常：接口返回了空响应结构: {body}",
            }
        except urllib.error.HTTPError as e:
            err_body = ""
            try:
                err_body = e.read().decode("utf-8", errors="ignore")
            except Exception:
                pass
            if e.code == 401:
                return {
                    "ok": False,
                    "code": 401,
                    "message": f"测试失败：API Key 无效或未授权 (HTTP 401)。请核对该平台的密钥是否正确。",
                }
            elif e.code in (400, 404):
                return {
                    "ok": False,
                    "code": e.code,
                    "message": f"测试失败：模型 [{model}] 不存在或平台不支持 (HTTP {e.code})。详情: {err_body[:100]}",
                }
            return {
                "ok": False,
                "code": e.code,
                "message": f"测试失败：服务端返回 HTTP {e.code}。详情: {err_body[:100]}",
            }
        except Exception as e:
            err_msg = str(e)
            if "nodename nor servname provided" in err_msg or "Name or service not known" in err_msg:
                return {
                    "ok": False,
                    "message": f"测试失败：无法解析 API Base 域名 ({api_base})，请检查网络或地址拼写。",
                }
            elif "Connection refused" in err_msg:
                return {
                    "ok": False,
                    "message": f"测试失败：连接被拒绝 ({url})。若为本地模型，请确认 Ollama/vLLM 服务已启动。",
                }
            elif "timed out" in err_msg or "Timeout" in err_msg:
                return {
                    "ok": False,
                    "message": f"测试失败：请求超时（超过 15 秒未响应），请检查网络连接。",
                }
            return {
                "ok": False,
                "message": f"测试失败：连接异常 ({err_msg})",
            }

    def to_public_dict(self) -> Dict[str, Any]:
        """导出当前运行时配置（包括前端回显所需的模型信息与密钥字段）。"""
        def _mask_key(k: str) -> str:
            if not k:
                return ""
            if len(k) <= 8:
                return "***"
            return f"{k[:4]}...{k[-4:]}"

        # 计算已通过测试并保存的有效模型列表
        # 核心规则：只有配置了 key 且测试通过 (verified=True) 的模版模型才进入可用列表
        available_models = []
        for p_key, p_cfg in self.platform_credentials.items():
            if p_cfg.get("verified"):
                preset = LLM_PROVIDER_PRESETS.get(p_key, {})
                for m in preset.get("models", []):
                    if not any(x["id"] == m["id"] for x in available_models):
                        available_models.append(dict(m))
                cur_m = p_cfg.get("model")
                if cur_m and not any(x["id"] == cur_m for x in available_models) and cur_m != "mock":
                    available_models.append({"id": cur_m, "name": f"{cur_m} ({preset.get('name', p_key)})"})

        # 若当前活动模型已有配置且已验证
        if self.llm_model and self.llm_model != "mock":
            cur_provider_cfg = self.platform_credentials.get(self.llm_provider, {})
            if cur_provider_cfg.get("verified") and not any(x["id"] == self.llm_model for x in available_models):
                available_models.insert(0, {"id": self.llm_model, "name": f"{self.llm_model} (当前生效)"})

        # 始终提供 MOCK 模式作为确定性演示保障
        available_models.append({"id": "mock", "name": "MOCK 确定性演示 (零依赖)"})

        # 各平台的凭据脱敏副本（用于前端在切换预设时准确回显各个平台专有 Key，绝不串用）
        public_credentials = {}
        for p_key, p_cfg in self.platform_credentials.items():
            k_val = p_cfg.get("api_key", "")
            public_credentials[p_key] = {
                "api_base": p_cfg.get("api_base", ""),
                "api_key": k_val,
                "api_key_masked": _mask_key(k_val),
                "model": p_cfg.get("model", ""),
                "verified": bool(p_cfg.get("verified", False)),
            }

        return {
            "mock": self.mock,
            "llm_provider": self.llm_provider,
            "llm_api_base": self.llm_api_base,
            "llm_model": self.llm_model,
            "llm_api_key": self.llm_api_key,
            "llm_api_key_configured": bool(self.llm_api_key),
            "llm_api_key_masked": _mask_key(self.llm_api_key),
            "llm_temperature": self.llm_temperature,
            "llm_timeout": self.llm_timeout,
            "llm_max_retries": self.llm_max_retries,
            "embed_enabled": self.embed_enabled,
            "embed_api_base": self.embed_api_base or self.llm_api_base,
            "embed_model": self.embed_model,
            "embed_dim": self.embed_dim,
            "embed_api_key": self.embed_api_key,
            "embed_api_key_configured": bool(self.embed_api_key or self.llm_api_key),
            "hybrid_dense_weight": self.hybrid_dense_weight,
            "strict_review_auth": self.strict_review_auth,
            "top_k": self.top_k,
            "data_path": self.data_path,
            "knowledge_config_path": self.knowledge_config_path,
            "llm_config_path": self.llm_config_path,
            "kb_db_path": self.kb_db_path,
            "presets": LLM_PROVIDER_PRESETS,
            "available_models": available_models,
            "platform_credentials": public_credentials,
        }

    def save_to_disk(self) -> None:
        """将当前核心大模型与检索配置持久化写入本地 JSON 文件，确保重启或刷新后不丢失。"""
        try:
            target_path = os.path.abspath(self.llm_config_path)
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            cfg = {
                "mock": self.mock,
                "llm_provider": self.llm_provider,
                "llm_api_base": self.llm_api_base,
                "llm_model": self.llm_model,
                "llm_api_key": self.llm_api_key,
                "llm_temperature": self.llm_temperature,
                "llm_timeout": self.llm_timeout,
                "llm_max_retries": self.llm_max_retries,
                "embed_enabled": self.embed_enabled,
                "embed_api_base": self.embed_api_base,
                "embed_model": self.embed_model,
                "embed_api_key": self.embed_api_key,
                "embed_dim": self.embed_dim,
                "hybrid_dense_weight": self.hybrid_dense_weight,
                "strict_review_auth": self.strict_review_auth,
                "top_k": self.top_k,
                "platform_credentials": self.platform_credentials,
            }
            with open(target_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[Settings] 保存配置到 {self.llm_config_path} 失败: {e}")

    def load_from_disk(self) -> None:
        """启动时从本地磁盘文件加载已保存的配置（若存在）。"""
        target_path = os.path.abspath(self.llm_config_path)
        if not os.path.exists(target_path):
            return
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                if "mock" in data and data["mock"] is not None:
                    self.mock = bool(data["mock"])
                if "llm_provider" in data and data["llm_provider"]:
                    self.llm_provider = str(data["llm_provider"])
                if "llm_api_base" in data and data["llm_api_base"]:
                    self.llm_api_base = str(data["llm_api_base"])
                if "llm_model" in data and data["llm_model"]:
                    self.llm_model = str(data["llm_model"])
                if "llm_api_key" in data and data["llm_api_key"] is not None:
                    self.llm_api_key = str(data["llm_api_key"])
                if "llm_temperature" in data and data["llm_temperature"] is not None:
                    self.llm_temperature = float(data["llm_temperature"])
                if "embed_enabled" in data and data["embed_enabled"] is not None:
                    self.embed_enabled = bool(data["embed_enabled"])
                if "embed_api_base" in data and data["embed_api_base"]:
                    self.embed_api_base = str(data["embed_api_base"])
                if "embed_model" in data and data["embed_model"]:
                    self.embed_model = str(data["embed_model"])
                if "embed_api_key" in data and data["embed_api_key"] is not None:
                    self.embed_api_key = str(data["embed_api_key"])
                if "embed_dim" in data and data["embed_dim"] is not None:
                    self.embed_dim = int(data["embed_dim"])
                if "hybrid_dense_weight" in data and data["hybrid_dense_weight"] is not None:
                    self.hybrid_dense_weight = float(data["hybrid_dense_weight"])
                if "strict_review_auth" in data and data["strict_review_auth"] is not None:
                    self.strict_review_auth = bool(data["strict_review_auth"])
                if "top_k" in data and data["top_k"] is not None:
                    self.top_k = int(data["top_k"])
                if "platform_credentials" in data and isinstance(data["platform_credentials"], dict):
                    for pk, pv in data["platform_credentials"].items():
                        if pk in self.platform_credentials and isinstance(pv, dict):
                            self.platform_credentials[pk].update(pv)
                        else:
                            self.platform_credentials[pk] = pv
        except Exception as e:
            print(f"[Settings] 从 {target_path} 加载配置异常: {e}")

    def update_runtime(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """支持在线热更新大模型与检索配置（无需重启服务，并自动持久化写入本地磁盘）。"""
        if "provider_preset" in updates and updates["provider_preset"] in LLM_PROVIDER_PRESETS:
            preset_key = updates["provider_preset"]
            preset = LLM_PROVIDER_PRESETS[preset_key]
            self.llm_provider = preset_key
            self.llm_api_base = preset["llm_api_base"]
            self.llm_model = preset["llm_model"]
            self.embed_api_base = preset["embed_api_base"]
            self.embed_model = preset["embed_model"]
            # 自动加载该平台的独立凭据
            p_cred = self.platform_credentials.get(preset_key, {})
            if p_cred.get("api_key"):
                self.llm_api_key = p_cred["api_key"]

        if "mock" in updates and updates["mock"] is not None:
            self.mock = _bool(str(updates["mock"]), self.mock)
        if "llm_provider" in updates and updates["llm_provider"]:
            self.llm_provider = str(updates["llm_provider"]).strip()
        if "llm_api_base" in updates and updates["llm_api_base"] is not None:
            self.llm_api_base = str(updates["llm_api_base"]).strip()
        if "llm_model" in updates and updates["llm_model"]:
            m_name = str(updates["llm_model"]).strip()
            self.llm_model = m_name
            # 智能联动：如果未显式指定 llm_api_base，且模型名明确属于其他平台预设，自动匹配对应的 provider 与 api_base
            if "llm_api_base" not in updates:
                if "deepseek" in m_name.lower() and self.llm_provider != "deepseek":
                    if "deepseek" in LLM_PROVIDER_PRESETS:
                        self.llm_provider = "deepseek"
                        self.llm_api_base = LLM_PROVIDER_PRESETS["deepseek"]["llm_api_base"]
                elif (m_name.startswith("qwen") or "qwen" in m_name.lower()) and "11434" not in self.llm_api_base and self.llm_provider != "qwen":
                    if "qwen" in LLM_PROVIDER_PRESETS:
                        self.llm_provider = "qwen"
                        self.llm_api_base = LLM_PROVIDER_PRESETS["qwen"]["llm_api_base"]
        if "llm_api_key" in updates and updates["llm_api_key"] is not None:
            val = str(updates["llm_api_key"]).strip()
            if val and "..." not in val:
                self.llm_api_key = val
            elif not val:
                self.llm_api_key = ""
        if "llm_temperature" in updates and updates["llm_temperature"] is not None:
            self.llm_temperature = float(updates["llm_temperature"])
        if "embed_enabled" in updates and updates["embed_enabled"] is not None:
            self.embed_enabled = _bool(str(updates["embed_enabled"]), self.embed_enabled)
        if "embed_api_base" in updates and updates["embed_api_base"] is not None:
            self.embed_api_base = str(updates["embed_api_base"]).strip()
        if "embed_model" in updates and updates["embed_model"]:
            self.embed_model = str(updates["embed_model"]).strip()
        if "embed_api_key" in updates and updates["embed_api_key"] is not None:
            val = str(updates["embed_api_key"]).strip()
            if val and "..." not in val:
                self.embed_api_key = val
        if "hybrid_dense_weight" in updates and updates["hybrid_dense_weight"] is not None:
            self.hybrid_dense_weight = max(0.0, min(1.0, float(updates["hybrid_dense_weight"])))
        if "strict_review_auth" in updates and updates["strict_review_auth"] is not None:
            self.strict_review_auth = _bool(str(updates["strict_review_auth"]), self.strict_review_auth)
        if "top_k" in updates and updates["top_k"] is not None:
            self.top_k = max(1, int(updates["top_k"]))

        # 同步更新对应当前平台的凭据字典
        cur_p = self.llm_provider
        if cur_p not in self.platform_credentials:
            self.platform_credentials[cur_p] = {}
        self.platform_credentials[cur_p]["api_base"] = self.llm_api_base
        self.platform_credentials[cur_p]["model"] = self.llm_model
        self.platform_credentials[cur_p]["api_key"] = self.llm_api_key
        if "verified" in updates:
            self.platform_credentials[cur_p]["verified"] = bool(updates["verified"])

        # 持久化保存至本地磁盘，确保下一次打开页面或重启服务后配置依然存在
        self.save_to_disk()

        return self.to_public_dict()


settings = Settings()
settings.load_from_disk()
