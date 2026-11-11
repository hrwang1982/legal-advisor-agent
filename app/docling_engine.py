"""独立 Docling 深度文档解析微服务客户端引擎 (Docling Engine & Microservice Client)
架构职责：
1. 【微服务优先】：优先通过内部网络调用独立的 Docling 微服务容器 (http://docling:8002/parse)；
2. 【全模态高保真】：解析复杂排版法典、裁判文书、批复函件，准确还原阅读顺序与 Markdown 表格；
3. 【透明自愈与平滑降级】：当 Docling 微服务离线或网络异常时，平滑降级至现有本地解析器（pypdf / python-docx / rapidocr）。
"""
from __future__ import annotations

import base64
import json
import logging
import os
import urllib.error
import urllib.request
from typing import Optional

logger = logging.getLogger(__name__)

_LAST_DOCLING_ERROR: Optional[str] = None


def is_docling_available(timeout: float = 1.0) -> bool:
    """快速探测独立 Docling 容器 (legal-advisor-docling) 是否在线且引擎已就绪。"""
    service_url = os.getenv("DOCLING_SERVICE_URL", "http://docling:8002/parse")
    health_url = service_url.rsplit("/", 1)[0] + "/health"

    candidate_urls = [health_url]
    for alt in ("http://legal-advisor-docling:8002/health", "http://localhost:8002/health"):
        if alt not in candidate_urls:
            candidate_urls.append(alt)

    for h_url in candidate_urls:
        try:
            req = urllib.request.Request(
                h_url,
                headers={"User-Agent": "LegalAdvisorAgent/1.0"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    # 确保引擎就绪，或者状态为 ok
                    return bool(data.get("engine_ready") or data.get("status") == "ok")
        except Exception:
            continue
    return False


def parse_document_via_docling(
    filename: str,
    file_bytes: bytes,
    timeout: float = 120.0,
) -> Optional[str]:
    """通过微服务网络调用独立 Docling 容器解析文档，返回高保真 Markdown 文本。
    遇到网络中断、容器离线或解析异常时返回 None，触发主流程平滑降级。
    """
    global _LAST_DOCLING_ERROR
    _LAST_DOCLING_ERROR = None

    service_url = os.getenv("DOCLING_SERVICE_URL", "") or "http://docling:8002/parse"
    candidate_urls = [service_url]
    for alt in ("http://legal-advisor-docling:8002/parse", "http://localhost:8002/parse"):
        if alt not in candidate_urls:
            candidate_urls.append(alt)

    b64_str = base64.b64encode(file_bytes).decode("ascii")
    payload = json.dumps({
        "filename": filename,
        "file_base64": b64_str,
    }).encode("utf-8")

    for target_url in candidate_urls:
        try:
            req = urllib.request.Request(
                target_url,
                data=payload,
                headers={"Content-Type": "application/json", "User-Agent": "LegalAdvisorAgent/1.0"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    if data.get("status") == "ok":
                        md_text = str(data.get("markdown") or "").strip()
                        if md_text:
                            logger.info(
                                f"✅ Docling 容器深度解析成功: {filename}, "
                                f"耗时 {data.get('elapse_ms')}ms, "
                                f"页数 {data.get('pages_count')}, "
                                f"表格数 {data.get('tables_count')}"
                            )
                            return md_text
                    else:
                        _LAST_DOCLING_ERROR = data.get("message") or "Docling 解析服务返回失败状态"
                        logger.warning(f"Docling 服务解析失败: {_LAST_DOCLING_ERROR}")
                        return None
        except urllib.error.URLError as e:
            _LAST_DOCLING_ERROR = f"连接 Docling 微服务异常 ({target_url}): {e}"
            logger.debug(_LAST_DOCLING_ERROR)
            continue
        except Exception as e:
            _LAST_DOCLING_ERROR = f"Docling 解析请求异常: {e}"
            logger.warning(_LAST_DOCLING_ERROR)
            return None

    return None

