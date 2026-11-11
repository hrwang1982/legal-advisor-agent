"""独立 Docling 深度文档解析微服务 (Legal Advisor Docling Service)
服务端口: 8002
核心职责:
1. 接收各类法务合规文档（PDF、Word/DOCX、PPTX、XLSX、图片等）；
2. 基于 Docling 深度学习布局分析（DocLayNet）与表格结构还原（TableFormer）进行全模态文档结构化解析；
3. 准确还原法条双栏排版顺序、保留表格 Markdown 语法结构，输出高保真 Markdown 文本；
4. 提供健康检查、耗时度量与线程安全防护。
"""
from __future__ import annotations

import base64
import io
import logging
import os
import tempfile
import threading
import time
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("legal_docling_service")

app = FastAPI(title="Legal Advisor Docling Microservice", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_CONVERTER_INSTANCE = None
_CONVERTER_ERROR: Optional[str] = None
_CONVERTER_LOCK = threading.Lock()
_INFER_LOCK = threading.Lock()


def get_converter():
    """线程安全的 Docling 转换器单例懒加载初始化。"""
    global _CONVERTER_INSTANCE, _CONVERTER_ERROR
    if _CONVERTER_INSTANCE is not None:
        return _CONVERTER_INSTANCE
    with _CONVERTER_LOCK:
        if _CONVERTER_INSTANCE is None:
            try:
                from docling.document_converter import DocumentConverter

                logger.info("⏳ 正在初始化 Docling DocumentConverter 引擎...")
                _CONVERTER_INSTANCE = DocumentConverter()
                _CONVERTER_ERROR = None
                logger.info("✅ Docling 引擎初始化成功，已就绪")
            except Exception as e:
                _CONVERTER_ERROR = str(e)
                logger.error(f"❌ Docling 引擎初始化异常: {e}")
                _CONVERTER_INSTANCE = None
    return _CONVERTER_INSTANCE


@app.on_event("startup")
def _warmup():
    threading.Thread(target=get_converter, daemon=True).start()


@app.get("/health")
def health_check():
    """健康检查探针：返回微服务运行状态与 Docling 引擎加载情况"""
    ready = _CONVERTER_INSTANCE is not None
    resp = {
        "status": "ok" if ready else "initializing",
        "service": "legal-advisor-docling",
        "engine_ready": ready,
    }
    if _CONVERTER_ERROR:
        resp["error"] = _CONVERTER_ERROR
    return JSONResponse(status_code=200 if ready else 200, content=resp)


class ParseRequest(BaseModel):
    filename: Optional[str] = "document.pdf"
    file_base64: Optional[str] = None


class ParseResponse(BaseModel):
    status: str
    markdown: str
    pages_count: Optional[int] = 0
    tables_count: Optional[int] = 0
    elapse_ms: float
    message: Optional[str] = None


def _run_docling_convert(filename: str, file_bytes: bytes) -> Dict[str, Any]:
    """核心转换执行逻辑：落地安全临时文件，调用 Docling 进行解析并导出为 Markdown。"""
    converter = get_converter()
    if converter is None:
        raise RuntimeError(f"Docling 引擎尚未就绪或初始化失败: {_CONVERTER_ERROR or '加载中'}")

    ext = (filename.rsplit(".", 1)[-1] if "." in filename else "bin").lower()
    suffix = f".{ext}" if ext else ".tmp"

    start_t = time.time()
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        with _INFER_LOCK:
            logger.info(f"📄 开始执行 Docling 深度文档解析: {filename} (大小: {len(file_bytes) / 1024:.1f} KB)")
            conv_result = converter.convert(tmp_path)
            md_text = conv_result.document.export_to_markdown()

            pages_count = 0
            tables_count = 0
            try:
                if hasattr(conv_result.document, "pages"):
                    pages_count = len(conv_result.document.pages)
                if hasattr(conv_result.document, "tables"):
                    tables_count = len(conv_result.document.tables)
            except Exception:
                pass

        elapse = round((time.time() - start_t) * 1000, 2)
        logger.info(f"✅ Docling 解析完成: {filename}, 耗时 {elapse}ms, 页数 {pages_count}, 表格数 {tables_count}")
        return {
            "status": "ok",
            "markdown": md_text,
            "pages_count": pages_count,
            "tables_count": tables_count,
            "elapse_ms": elapse,
        }
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


@app.post("/parse", response_model=ParseResponse)
def parse_document_json(req: ParseRequest):
    """JSON Base64 解析接口"""
    if not req.file_base64:
        raise HTTPException(status_code=400, detail="缺少 file_base64 数据")

    try:
        raw_b64 = req.file_base64
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]
        file_bytes = base64.b64decode(raw_b64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Base64 解码失败: {e}")

    try:
        res = _run_docling_convert(req.filename or "document.pdf", file_bytes)
        return ParseResponse(**res)
    except Exception as err:
        logger.error(f"Docling 解析异常: {err}")
        return ParseResponse(
            status="error",
            markdown="",
            pages_count=0,
            tables_count=0,
            elapse_ms=0,
            message=str(err),
        )


@app.post("/parse-file")
async def parse_document_file(file: UploadFile = File(...)):
    """Multipart 表单文件上传解析接口"""
    try:
        file_bytes = await file.read()
        res = _run_docling_convert(file.filename or "document.pdf", file_bytes)
        return res
    except Exception as err:
        logger.error(f"Docling 解析文件异常: {err}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "markdown": "",
                "pages_count": 0,
                "tables_count": 0,
                "elapse_ms": 0,
                "message": str(err),
            },
        )

