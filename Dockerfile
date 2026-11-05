FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 先装依赖，利用层缓存
COPY requirements.txt .
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# 再拷代码
COPY app ./app

EXPOSE 8000

# 默认 MOCK 模式，免密钥即可演示；接真实 LLM 时在 docker-compose 覆盖环境变量
ENV MOCK=true \
    HOST=0.0.0.0 \
    PORT=8000

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
