FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including ffmpeg and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

ENV PYTHONPATH=/app
ENV MEETING_HOST=0.0.0.0
ENV MEETING_PORT=8000
ENV MEETING_MODELS_DIR=/app/models
ENV MEETING_UPLOAD_DIR=/app/uploads
ENV MEETING_DB_PATH=/app/data/meetings.db

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
