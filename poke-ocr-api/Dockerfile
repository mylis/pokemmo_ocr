# syntax=docker/dockerfile:1.6
FROM python:3.11-slim

# Prevent Python from writing pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps:
# - libgl1/libglib2.0-0: needed by opencv
# - libgomp1: OpenMP runtime (helps numpy/pytorch performance)
# - tesseract-ocr: optional; EasyOCR may use it in some setups
# - build-essential: some wheels may need compiling
# - procps/gawk: debugging tools (ps/awk)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    tesseract-ocr \
    build-essential \
    procps \
    gawk \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# -------------------------
# Python deps (cached)
# -------------------------
COPY requirements.txt /app/requirements.txt

# Use BuildKit cache to speed up rebuilds (pip downloads/wheels)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip \
 && pip install -r /app/requirements.txt \
 && pip install gunicorn uvicorn

# Copy the rest of the app
COPY . /app

EXPOSE 8000

# EasyOCR settings
# Store models somewhere NOT in /root and pre-download them at build time.
ENV EASYOCR_GPU=1 \
    EASYOCR_MODULE_PATH=/opt/easyocr

# Concurrency defaults (tweak at runtime)
ENV OCR_WORKERS=2 \
    OCR_INFLIGHT=2 \
    MAX_UPLOAD_BYTES=250000000 \
    MAX_IMAGE_BYTES=15000000 \
    MAX_VIDEO_BYTES=200000000 \
    WEB_CONCURRENCY=2 \
    GUNICORN_MAX_REQUESTS=800 \
    GUNICORN_MAX_REQUESTS_JITTER=200 \
    GUNICORN_TIMEOUT=120 \
    GUNICORN_KEEPALIVE=5

# -------------------------
# Pre-download EasyOCR models (prevents multi-worker race on boot)
# NOTE: gpu=False here is fine; it downloads the same models.
# -------------------------
RUN python - <<'PY'
import os
import easyocr

path = os.environ.get("EASYOCR_MODULE_PATH", "/opt/easyocr")
os.makedirs(path, exist_ok=True)

# This will download + unzip models into model_storage_directory
easyocr.Reader(["en"], gpu=False, model_storage_directory=path)
print("EasyOCR models ready at:", path)
PY

# Run API (Gunicorn master + Uvicorn workers)
CMD ["bash", "-lc", "gunicorn app:app -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --workers ${WEB_CONCURRENCY} --access-logfile - --error-logfile - --log-level info --max-requests ${GUNICORN_MAX_REQUESTS} --max-requests-jitter ${GUNICORN_MAX_REQUESTS_JITTER} --timeout ${GUNICORN_TIMEOUT} --keep-alive ${GUNICORN_KEEPALIVE} "]