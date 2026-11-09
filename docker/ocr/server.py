"""独立 OCR 识别微服务 (Legal Advisor OCR Service)
服务端口: 8001
核心职责:
1. 接收图片二进制/Base64 数据，完成图像预处理（格式统一、超大图缩放、小图放大）；
2. 提取证书、承诺书、合同印章、扫描件中的中英文印刷体及手写体文字；
3. 按阅读顺序（自上而下、自左而右）重排文本行，同一行的多个文本块合并输出；
4. 健康检查以引擎真实就绪为准（engine_ready=false 时返回 503）。
"""
from __future__ import annotations

import base64
import io
import logging
import threading
import time
from typing import Any, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("legal_ocr_service")

app = FastAPI(title="Legal Advisor OCR Microservice", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_OCR_ENGINE = None
_ENGINE_ERROR: Optional[str] = None
_ENGINE_LOCK = threading.Lock()
_INFER_LOCK = threading.Lock()


def get_engine():
    """线程安全的单例初始化；失败时记录错误原因（不在每次健康检查时反复刷日志）。"""
    global _OCR_ENGINE, _ENGINE_ERROR
    if _OCR_ENGINE is not None:
        return _OCR_ENGINE
    with _ENGINE_LOCK:
        if _OCR_ENGINE is None:
            try:
                from rapidocr_onnxruntime import RapidOCR

                _OCR_ENGINE = RapidOCR()
                _ENGINE_ERROR = None
                logger.info("✅ RapidOCR 引擎初始化成功")
            except Exception as e:
                if _ENGINE_ERROR != str(e):
                    logger.error(f"RapidOCR 引擎初始化失败: {e}")
                _ENGINE_ERROR = str(e)
                _OCR_ENGINE = None
    return _OCR_ENGINE


@app.on_event("startup")
def _warmup():
    get_engine()


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
    body = {
        "status": "ok" if engine_ready else "degraded",
        "service": "legal-advisor-ocr",
        "engine_ready": engine_ready,
        "engine_type": "rapidocr-onnxruntime",
        "engine_error": _ENGINE_ERROR,
    }
    return JSONResponse(content=body, status_code=200 if engine_ready else 503)


def _preprocess(img_bytes: bytes) -> Any:
    """图像预处理：统一转 RGB（处理 PNG 透明通道/调色板/CMYK），超大图缩放，小图放大。"""
    try:
        import numpy as np
        from PIL import Image

        img = Image.open(io.BytesIO(img_bytes))
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGBA")
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[-1])
            img = bg
        elif img.mode != "RGB":
            img = img.convert("RGB")

        w, h = img.size
        max_side = max(w, h)
        if max_side > 3000:
            scale = 3000 / max_side
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        elif max_side < 800:
            scale = 800 / max_side
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        # RapidOCR / OpenCV 采用 BGR 通道顺序
        arr = np.array(img)[:, :, ::-1].copy()
        return arr
    except Exception as e:
        logger.warning(f"图像预处理失败，回退原始字节: {e}")
        return img_bytes


def _reading_order(result: List[Any], min_confidence: float) -> List[str]:
    """按阅读顺序重排：先按行（y 中心聚类），行内按 x 排序并以空格合并。"""
    boxes = []
    for item in result:
        if len(item) < 3:
            continue
        txt = str(item[1]).strip()
        try:
            score = float(item[2])
        except (ValueError, TypeError):
            score = 1.0
        if not txt or score < min_confidence:
            continue
        pts = item[0]
        try:
            ys = [float(p[1]) for p in pts]
            xs = [float(p[0]) for p in pts]
            yc, h, x0 = sum(ys) / len(ys), max(ys) - min(ys), min(xs)
        except Exception:
            yc, h, x0 = 0.0, 0.0, 0.0
        boxes.append((yc, h, x0, txt))

    boxes.sort(key=lambda b: (b[0], b[2]))
    lines: List[List[tuple]] = []
    for b in boxes:
        if lines:
            last = lines[-1]
            ref_y = sum(x[0] for x in last) / len(last)
            ref_h = max(max(x[1] for x in last), b[1], 1.0)
            if abs(b[0] - ref_y) <= ref_h * 0.5:
                last.append(b)
                continue
        lines.append([b])
    return ["  ".join(x[3] for x in sorted(ln, key=lambda x: x[2])) for ln in lines]


def _process_image_bytes(img_bytes: bytes, min_confidence: float = 0.5) -> List[str]:
    engine = get_engine()
    if not engine:
        raise RuntimeError(f"OCR 引擎尚未就绪: {_ENGINE_ERROR}")

    img_input = _preprocess(img_bytes)
    with _INFER_LOCK:
        result, _ = engine(img_input)
    if not result:
        return []
    return _reading_order(result, min_confidence)


def _build_response(img_bytes: bytes, min_conf: float, start_t: float) -> OCRResponse:
    try:
        lines = _process_image_bytes(img_bytes, min_confidence=min_conf)
        return OCRResponse(
            status="ok",
            text="\n".join(lines),
            lines_count=len(lines),
            elapse_ms=round((time.perf_counter() - start_t) * 1000, 2),
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
    min_conf = req.min_confidence if req.min_confidence is not None else 0.5
    return _build_response(img_bytes, min_conf, start_t)


@app.post("/ocr/upload", response_model=OCRResponse)
async def ocr_upload_file(file: UploadFile = File(...)):
    start_t = time.perf_counter()
    img_bytes = await file.read()
    return _build_response(img_bytes, 0.5, start_t)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
