# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including ffmpeg, vulkan runtime, and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    build-essential \
    libvulkan1 \
    vulkan-tools \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
COPY wheels/ /tmp/wheels/

# استخدام الكاش الدائم + التثبيت من /tmp/wheels أولاً دون إعادة التحميل
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --default-timeout=1000 --retries 10 \
    --find-links /tmp/wheels \
    -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu124 \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 \
    && pip install --default-timeout=1000 --retries 10 \
    --find-links /tmp/wheels \
    nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 \
    && rm -rf /tmp/wheels

COPY . /app

ENV PYTHONPATH=/app
ENV MEETING_HOST=0.0.0.0
ENV MEETING_PORT=3011
ENV MEETING_MODELS_DIR=/app/models
ENV MEETING_UPLOAD_DIR=/app/uploads
ENV MEETING_DB_PATH=/app/data/meetings.db

# ربط ملفات تشغيل كرت الشاشة بالنظام ليجدها LLaMA فوراً
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH

EXPOSE 3011

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "3011", "--timeout-keep-alive", "3600"]