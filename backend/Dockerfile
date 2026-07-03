# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # GPT-2 is downloaded on first boot into this dir; mount it as a volume so
    # the ~500MB download is fetched once and reused across restarts.
    HF_HOME=/cache/huggingface

WORKDIR /app

# Install the CPU-only torch wheel first (the default PyPI wheel pulls heavy
# CUDA deps we don't need — this app runs GPT-2 small on CPU by design).
RUN pip install --no-cache-dir torch>=2.5.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Remaining deps (torch already satisfied, so it's skipped here).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
