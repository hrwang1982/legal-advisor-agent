"""轻量级本地 OCR 识别引擎（纯私有化无外网依赖）
核心能力：
1. 单张图片（PNG / JPG / BMP / WEBP）的高精度文字提取；
2. 扫描件 PDF（纯图片 PDF）逐页图片提取与 OCR 自动拼接；
3. Word (.docx) 文档内嵌插图（word/media/）的自动提取与图文混排识别；
4. 毫秒级单例生命周期管理与平滑降级（无安装时不阻塞、不抛错）。
"""
from __future__ import annotations

import io
import logging
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
            logger.info("✅ RapidOCR 引擎已成功就绪并初始化")
        except Exception as e:
            logger.info(f"RapidOCR 引擎未安装或初始化跳过: {e}")
            _RAPID_OCR_INSTANCE = None
    return _RAPID_OCR_INSTANCE


def is_ocr_available() -> bool:
    """查询当前环境是否具备本地 OCR 引擎能力。"""
    return get_rapid_ocr() is not None


def ocr_image_bytes(img_bytes: bytes, min_confidence: float = 0.5) -> str:
    """对单张图片的原始二进制字节流执行 OCR 识别，返回按阅读顺序排列的纯文本。"""
    if not img_bytes:
        return ""
    engine = get_rapid_ocr()
    if not engine:
        return ""

    try:
        # RapidOCR 接收 bytes, ndarray 或 path
        result, _ = engine(img_bytes)
        if not result:
            return ""

        # result 格式: [ [[box_points], text, score], ... ]
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
        return "\n".join(valid_lines)
    except Exception as e:
        logger.warning(f"OCR image processing warning: {e}")
        return ""


def ocr_pdf_pages(pdf_bytes: bytes, max_pages: int = 30) -> str:
    """对纯扫描版/图片版 PDF 提取内嵌位图并执行 OCR 识别。
    按页码聚合文本，形成结构化文档内容供切分器进一步分段。
    """
    if not is_ocr_available():
        return ""

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
                        if ocr_txt.strip():
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
    """从 docx 二进制流中解压 word/media/ 内嵌图片并识别。
    返回列表: [(图片文件名, 识别出的文本), ...]
    """
    if not is_ocr_available():
        return []

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
                    # 仅保留具有实际语义的图文文本（过滤纯单字符或极短干扰符）
                    if clean_txt and len(clean_txt) >= 4:
                        base_name = img_name.rsplit("/", 1)[-1]
                        extracted.append((base_name, clean_txt))
                except Exception as ie:
                    logger.debug(f"Failed to OCR docx image {img_name}: {ie}")
    except Exception as e:
        logger.warning(f"DOCX embedded images OCR warning: {e}")
    return extracted

