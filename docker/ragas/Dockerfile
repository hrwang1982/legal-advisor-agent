FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN sed -i 's|deb.debian.org|mirrors.aliyun.com|g' /etc/apt/sources.list.d/debian.sources 2>/dev/null || true \
    && apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

COPY server.py .

EXPOSE 8003

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 CMD curl -sf http://localhost:8003/health || exit 1

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8003"]
