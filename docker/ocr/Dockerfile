FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 系统共享库：
# - libgomp1：onnxruntime 多线程推理
# - libglib2.0-0 / libxcb1 / libgl1 / libsm6 / libxext6 / libxrender1：
#   opencv 运行时依赖（缺失 libxcb.so.1 会导致 RapidOCR 初始化失败 → engine_ready=false）
RUN sed -i 's|deb.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list.d/debian.sources 2>/dev/null || true \
    && apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    libglib2.0-0 \
    libxcb1 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt \
    # 用无 GUI 依赖的 headless 版本替换 opencv-python，彻底规避 X11/libxcb 依赖
    && (pip uninstall -y opencv-python || true) \
    && pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python-headless

COPY server.py .

# 构建期预热：确保模型可加载，失败则构建失败（避免带病上线）
RUN python -c "from rapidocr_onnxruntime import RapidOCR; RapidOCR(); print('RapidOCR warmup OK')"

EXPOSE 8001

# 健康检查以引擎真实就绪为准（engine_ready=false 时返回 503）
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=5 CMD curl -sf http://localhost:8001/health || exit 1

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001"]
