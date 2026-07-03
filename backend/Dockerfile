# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Qwen3-1.7B is downloaded on first boot into this dir; mount it as a volume
    # so the ~7GB (float32) download is fetched once and reused across restarts.
    HF_HOME=/cache/huggingface

WORKDIR /app

# Install the CPU-only torch wheel first (the default PyPI wheel pulls heavy
# CUDA deps we don't need — this app runs Qwen3-1.7B on CPU in Docker).
# The pin is quoted: unquoted, the shell would parse `>=2.5.0` as a redirect.
RUN pip install --no-cache-dir "torch>=2.5.0" \
    --index-url https://download.pytorch.org/whl/cpu

# Remaining deps (torch already satisfied, so it's skipped here).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8000

# --ws websockets-sansio + a longer ping timeout avoid a keepalive-ping
# AssertionError in the legacy WS implementation when a sustained CPU
# generation run starves the event loop (see README "Run it locally").
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--ws", "websockets-sansio", "--ws-ping-timeout", "30"]
