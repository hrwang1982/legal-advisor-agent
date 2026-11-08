"""独立 OCR 识别微服务 (Legal Advisor OCR Service)
服务端口: 8001
核心职责:
1. 接收图片二进制/Base64 数据，完成图像去噪、对比度拉伸及文本行检测；
2. 提取证书、合同印章、扫描件 PDF 页面中的中英文印刷体及艺术体文字；
3. 输出纯文本与分词行，供法律顾问主业务服务切分和向量化入库。
"""
from __future__ import annotations

import base64
import io
import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("legal_ocr_service")

app = FastAPI(title="Legal Advisor OCR Microservice", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_OCR_ENGINE = None


def get_engine():
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            _OCR_ENGINE = RapidOCR()
            logger.info("RapidOCR 引擎初始化成功")
        except Exception as e:
            logger.error(f"RapidOCR 引擎初始化失败: {e}")
            _OCR_ENGINE = None
    return _OCR_ENGINE


class OCRRequest(BaseModel):
    image_base64: Optional[str] = None
    filename: Optional[str] = None
    min_confidence: Optional[float] = 0.5


class OCRResponse(BaseModel):
    status: str
    text: str
    lines_count: int
    elapse_ms: float
    message: Optional[str] = None


@app.get("/health")
def health_check():
    engine_ready = get_engine() is not None
    return {
        "status": "ok",
        "service": "legal-advisor-ocr",
        "engine_ready": engine_ready,
        "engine_type": "rapidocr-onnxruntime",
    }


def _process_image_bytes(img_bytes: bytes, min_confidence: float = 0.5) -> List[str]:
    """执行 OCR 文本行检测与识别。"""
    engine = get_engine()
    if not engine:
        raise RuntimeError("OCR 引擎尚未就绪")

    result, _ = engine(img_bytes)
    if not result:
        return []

    lines: List[str] = []
    for item in result:
        if len(item) >= 3:
            txt = str(item[1]).strip()
            try:
                score = float(item[2])
            except (ValueError, TypeError):
                score = 1.0
            if txt and score >= min_confidence:
                lines.append(txt)
    return lines


@app.post("/ocr", response_model=OCRResponse)
def ocr_endpoint(req: OCRRequest):
    start_t = time.perf_counter()
    if not req.image_base64:
        raise HTTPException(status_code=400, detail="缺少 image_base64 字段")

    raw_b64 = req.image_base64
    if "," in raw_b64:
        raw_b64 = raw_b64.split(",", 1)[1]

    try:
        img_bytes = base64.b64decode(raw_b64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Base64 解码失败: {e}")

    try:
        lines = _process_image_bytes(img_bytes, min_confidence=req.min_confidence or 0.5)
        text = "\n".join(lines)
        elapse = (time.perf_counter() - start_t) * 1000
        return OCRResponse(
            status="ok",
            text=text,
            lines_count=len(lines),
            elapse_ms=round(elapse, 2),
            message="识别成功" if lines else "未识别到清晰文字",
        )
    except Exception as e:
        logger.error(f"OCR 处理异常: {e}")
        return OCRResponse(
            status="error",
            text="",
            lines_count=0,
            elapse_ms=round((time.perf_counter() - start_t) * 1000, 2),
            message=str(e),
        )


@app.post("/ocr/upload", response_model=OCRResponse)
async def ocr_upload_file(file: UploadFile = File(...)):
    start_t = time.perf_counter()
    img_bytes = await file.read()
    try:
        lines = _process_image_bytes(img_bytes, min_confidence=0.5)
        text = "\n".join(lines)
        elapse = (time.perf_counter() - start_t) * 1000
        return OCRResponse(
            status="ok",
            text=text,
            lines_count=len(lines),
            elapse_ms=round(elapse, 2),
            message="识别成功" if lines else "未识别到清晰文字",
        )
    except Exception as e:
        return OCRResponse(
            status="error",
            text="",
            lines_count=0,
            elapse_ms=round((time.perf_counter() - start_t) * 1000, 2),
            message=str(e),
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

