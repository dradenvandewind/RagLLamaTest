
# ──────────────────────────────────────────────
# Stage 1: builder — install dependencies
# ──────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# Minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies into an isolated virtualenv
COPY requirements.txt .
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# ──────────────────────────────────────────────
# Stage 2: runtime — final lightweight image
# ──────────────────────────────────────────────
FROM python:3.12-slim AS runtime

LABEL maintainer="erwanleblond@gmail.com"
LABEL description="RAG LlamaIndex async — transcription vidéo YouTube"

RUN apt-get update && apt-get install -y ffmpeg firefox-esr

WORKDIR /app

# Copy only the virtualenv
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
#RUN apt-get update && apt-get install -y nodejs && \
#    ln -sf $(which node) /usr/local/bin/nodejs
RUN apt-get update && apt-get install -y nodejs npm && \
    npm install -g @yt-dlp/sandbox

# Copy application code
COPY app/ ./app/

# Persistent directories (mounted via Docker volumes)
RUN mkdir -p /app/chroma_db /app/data



# Default environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CHROMA_PATH=/app/chroma_db \
    OPENAI_MODEL=gpt-4o-mini \
    EMBED_MODEL=text-embedding-3-small \
    LLM_TEMPERATURE=0.1 \
    CHUNK_SIZE=512 \
    CHUNK_OVERLAP=64 \
    PORT=8000 \
    WORKERS=4
COPY cookies.txt /app/cookies.txt
RUN chmod 644 /app/cookies.txt

# Non-root user for security
RUN useradd -m -u 1001 raguser && chown -R raguser:raguser /app
USER raguser

EXPOSE $PORT

# Native Docker healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

CMD uvicorn app.main:app \
    --host 0.0.0.0 \
    --port ${PORT} \
    --workers ${WORKERS} \
    --log-level info
