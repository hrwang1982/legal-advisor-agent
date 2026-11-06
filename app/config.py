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


# 常用大模型与向量模型平台预设（供前端一键切换或配置参考）
LLM_PROVIDER_PRESETS: Dict[str, Dict[str, Any]] = {
    "qwen": {
        "name": "阿里云百炼 / 通义千问 (DashScope OpenAI兼容)",
        "llm_api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "llm_model": "qwen-plus",
        "embed_api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "embed_model": "text-embedding-v3",
    },
    "local_openai": {
        "name": "本地 OpenAI 兼容服务 (vLLM / Ollama / Xinference + bge-m3)",
        "llm_api_base": "http://localhost:11434/v1",
        "llm_model": "qwen2.5:14b",
        "embed_api_base": "http://localhost:11434/v1",
        "embed_model": "BAAI/bge-m3",
    },
    "deepseek": {
        "name": "DeepSeek 官方 API + BGE-M3 向量",
        "llm_api_base": "https://api.deepseek.com/v1",
        "llm_model": "deepseek-chat",
        "embed_api_base": "http://localhost:11434/v1",
        "embed_model": "BAAI/bge-m3",
    },
    "siliconflow": {
        "name": "SiliconFlow (Qwen + BAAI/bge-m3 在线兼容接口)",
        "llm_api_base": "https://api.siliconflow.cn/v1",
        "llm_model": "Qwen/Qwen2.5-72B-Instruct",
        "embed_api_base": "https://api.siliconflow.cn/v1",
        "embed_model": "BAAI/bge-m3",
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

    def to_public_dict(self) -> Dict[str, Any]:
        """导出当前运行时配置（包括前端回显所需的模型信息与密钥字段）。"""
        def _mask_key(k: str) -> str:
            if not k:
                return ""
            if len(k) <= 8:
                return "***"
            return f"{k[:4]}...{k[-4:]}"

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

        if "mock" in updates and updates["mock"] is not None:
            self.mock = _bool(str(updates["mock"]), self.mock)
        if "llm_provider" in updates and updates["llm_provider"]:
            self.llm_provider = str(updates["llm_provider"]).strip()
        if "llm_api_base" in updates and updates["llm_api_base"] is not None:
            self.llm_api_base = str(updates["llm_api_base"]).strip()
        if "llm_model" in updates and updates["llm_model"]:
            self.llm_model = str(updates["llm_model"]).strip()
        if "llm_api_key" in updates and updates["llm_api_key"] is not None:
            val = str(updates["llm_api_key"]).strip()
            if val and "..." not in val:
                self.llm_api_key = val
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

        # 持久化保存至本地磁盘，确保下一次打开页面或重启服务后配置依然存在
        self.save_to_disk()

        return self.to_public_dict()


settings = Settings()
settings.load_from_disk()
