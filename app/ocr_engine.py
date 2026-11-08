"""轻量级本地 OCR 识别与微服务客户端引擎 (OCR Engine & Microservice Client)
架构层次：
1. 【微服务优先】：优先通过内部网络调用独立的 OCR 微服务容器 (http://ocr:8001/ocr)；
2. 【视觉大模型增强】：当微服务离线或遇到复杂艺术字/公章时，支持调用 Vision 多模态大模型抽取；
3. 【本地单例兜底】：若环境安装了 RapidOCR，在进程内单例执行；
4. 【证书/合同印章特化】：对法律场景常见的证书、奖状、签约盖章文件进行结构化实体语义整理。
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_RAPID_OCR_INSTANCE = None
_RAPID_OCR_INIT_ATTEMPTED = False


def get_rapid_ocr():
    """获取 RapidOCR 单例实例。仅在初次调用时延迟初始化，避免冷启动性能损耗。"""
    global _RAPID_OCR_INSTANCE, _RAPID_OCR_INIT_ATTEMPTED
    if not _RAPID_OCR_INIT_ATTEMPTED:
        _RAPID_OCR_INIT_ATTEMPTED = True
        try:
            from rapidocr_onnxruntime import RapidOCR

            _RAPID_OCR_INSTANCE = RapidOCR()
            logger.info("✅ 本地 RapidOCR 引擎已成功就绪并初始化")
        except Exception as e:
            logger.debug(f"本地 RapidOCR 引擎未安装或初始化跳过: {e}")
            _RAPID_OCR_INSTANCE = None
    return _RAPID_OCR_INSTANCE


def is_remote_ocr_online(timeout: float = 1.0) -> bool:
    """快速探测独立 OCR 容器 (legal-advisor-ocr) 是否在线。"""
    ocr_url = os.getenv("OCR_SERVICE_URL", "http://ocr:8001/ocr")
    health_url = ocr_url.rsplit("/", 1)[0] + "/health"
    try:
        req = urllib.request.Request(health_url, headers={"User-Agent": "LegalAdvisorAgent/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        # 同时尝试检查 localhost:8001 (宿主机调试场景)
        if "ocr:8001" in health_url:
            try:
                local_h = health_url.replace("ocr:8001", "localhost:8001")
                req = urllib.request.Request(local_h, headers={"User-Agent": "LegalAdvisorAgent/1.0"})
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return resp.status == 200
            except Exception:
                pass
    return False


def call_remote_ocr_service(img_bytes: bytes, timeout: float = 8.0) -> Optional[str]:
    """通过微服务网络调用独立 OCR 容器 (http://ocr:8001/ocr)。"""
    ocr_urls = [os.getenv("OCR_SERVICE_URL", "http://ocr:8001/ocr")]
    if "ocr:8001" in ocr_urls[0]:
        ocr_urls.append("http://localhost:8001/ocr")

    b64_str = base64.b64encode(img_bytes).decode("ascii")
    payload = json.dumps({"image_base64": b64_str, "min_confidence": 0.4}).encode("utf-8")

    for target_url in ocr_urls:
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
                    if data.get("status") == "ok" and data.get("text"):
                        logger.info(f"✅ 独立 OCR 容器识别成功: 耗时 {data.get('elapse_ms')}ms")
                        return str(data["text"]).strip()
        except Exception as e:
            logger.debug(f"调用 OCR 微服务 {target_url} 异常: {e}")
    return None


def call_vision_llm_ocr(img_bytes: bytes, timeout: float = 15.0) -> Optional[str]:
    """调用多模态视觉大模型（如 DashScope Qwen-VL / OpenAI 兼容接口）对复杂图像进行图文与印章识别。"""
    from app.config import settings

    if not settings.llm_api_key or settings.mock:
        return None

    api_url = f"{settings.llm_api_base.rstrip('/')}/chat/completions"
    model = "qwen-vl-plus" if "dashscope" in settings.llm_api_base.lower() else (settings.llm_model or "gpt-4o")
    b64_str = base64.b64encode(img_bytes).decode("ascii")

    prompt = (
        "你是一个高精度法律文档与证据票据 OCR 专家。请完整识别并提取该图片中的所有文字，"
        "包括标题、颁发对象姓名、正文、落款单位、盖章公章内容、编号及日期。"
        "请直接输出识别到的排版纯文本，不要任何客套前言。"
    )

    body = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_str}"}},
                ],
            }
        ],
        "temperature": 0.01,
        "max_tokens": 1500,
    }

    try:
        req = urllib.request.Request(
            api_url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {settings.llm_api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                resp_data = json.loads(resp.read().decode("utf-8"))
                content = resp_data["choices"][0]["message"]["content"]
                if content and content.strip():
                    logger.info("✅ 视觉多模态大模型 OCR 识别成功")
                    return content.strip()
    except Exception as e:
        logger.debug(f"Vision LLM OCR 调用跳过: {e}")
    return None


def is_ocr_available() -> bool:
    """查询当前系统是否具备 OCR 识别能力。"""
    if is_remote_ocr_online():
        return True
    from app.config import settings
    if settings.llm_api_key and not settings.mock:
        return True
    return get_rapid_ocr() is not None


def ocr_image_bytes(img_bytes: bytes, min_confidence: float = 0.5) -> str:
    """多级协同 OCR 文本提取核心流程：
    1. 优先调用独立 OCR 容器 (legal-advisor-ocr:8001)；
    2. 备选调用 Vision 多模态大模型（适合带底纹、复杂艺术字及红色公章）；
    3. 备选调用进程内 RapidOCR；
    4. 兜底输出微服务部署指引提示。
    """
    if not img_bytes or len(img_bytes) < 16:
        return ""

    # 通路 1：独立 OCR 微服务容器调用
    remote_res = call_remote_ocr_service(img_bytes)
    if remote_res and remote_res.strip():
        return remote_res.strip()

    # 通路 2：多模态大模型视觉识别增强
    vision_res = call_vision_llm_ocr(img_bytes)
    if vision_res and vision_res.strip():
        return vision_res.strip()

    # 通路 3：本地 RapidOCR 单例
    engine = get_rapid_ocr()
    if engine:
        try:
            result, _ = engine(img_bytes)
            if result:
                valid_lines = []
                for item in result:
                    if len(item) >= 3:
                        txt = str(item[1]).strip()
                        try:
                            score = float(item[2])
                        except (ValueError, TypeError):
                            score = 1.0
                        if txt and score >= min_confidence:
                            valid_lines.append(txt)
                if valid_lines:
                    return "\n".join(valid_lines)
        except Exception as e:
            logger.warning(f"Local RapidOCR processing error: {e}")

    # 通路 4：未识别或未就绪时的引导
    return (
        "【系统提示：当前独立 OCR 容器服务尚未启动或仍在加载。】\n"
        "推荐部署方式：在终端执行 `docker compose up -d ocr` 启动专用的 legal-advisor-ocr 容器，"
        "或在【模型配置】中配置通义千问 / OpenAI 兼容多模态 API Key，即可实现证书印章、扫描件 PDF 与图片的秒级高精自动识别。"
    )


def ocr_pdf_pages(pdf_bytes: bytes, max_pages: int = 30) -> str:
    """对纯扫描版/图片版 PDF 提取内嵌位图并执行多级 OCR 识别。"""
    try:
        import pypdf

        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        total_pages = min(len(reader.pages), max_pages)
        page_texts = []

        for p_idx in range(total_pages):
            page = reader.pages[p_idx]
            page_extracted_lines = []
            try:
                for img_obj in getattr(page, "images", []):
                    img_data = getattr(img_obj, "data", None)
                    if img_data:
                        ocr_txt = ocr_image_bytes(img_data)
                        if ocr_txt.strip() and "系统提示" not in ocr_txt:
                            page_extracted_lines.append(ocr_txt.strip())
            except Exception as pe:
                logger.debug(f"Error extracting images from page {p_idx}: {pe}")

            if page_extracted_lines:
                page_texts.append(
                    f"【第 {p_idx + 1} 页（扫描识别）】\n" + "\n".join(page_extracted_lines)
                )

        if page_texts:
            return "\n\n".join(page_texts)
    except Exception as e:
        logger.warning(f"PDF OCR processing warning: {e}")
    return ""


def ocr_docx_embedded_images(docx_bytes: bytes) -> List[Tuple[str, str]]:
    """从 docx 二进制流中解压 word/media/ 内嵌图片并识别。"""
    extracted: List[Tuple[str, str]] = []
    try:
        import zipfile

        with zipfile.ZipFile(io.BytesIO(docx_bytes)) as zf:
            img_files = sorted(
                [
                    f
                    for f in zf.namelist()
                    if f.startswith("word/media/")
                    and f.lower().endswith(
                        (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
                    )
                ]
            )
            for img_name in img_files:
                try:
                    img_data = zf.read(img_name)
                    txt = ocr_image_bytes(img_data)
                    clean_txt = txt.strip()
                    if clean_txt and len(clean_txt) >= 4 and "系统提示" not in clean_txt:
                        base_name = img_name.rsplit("/", 1)[-1]
                        extracted.append((base_name, clean_txt))
                except Exception as ie:
                    logger.debug(f"Failed to OCR docx image {img_name}: {ie}")
    except Exception as e:
        logger.warning(f"DOCX embedded images OCR warning: {e}")
    return extracted
