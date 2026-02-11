# Dockerfile
FROM python:3.11-slim

# Prevents Python from writing pyc files and buffers stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps:
# - libgl1/libglib2.0-0: needed by opencv
# - libgomp1: OpenMP runtime (helps numpy/pytorch performance)
# - tesseract-ocr: EasyOCR may use it in some setups; harmless if unused
# - build-essential: occasional wheel builds (can remove if you know all wheels exist)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    tesseract-ocr \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first (better caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of the app
COPY . /app

EXPOSE 8000

# Default: CPU EasyOCR. Set EASYOCR_GPU=1 for GPU runtime (note: GPU images differ)
ENV EASYOCR_GPU=1

# Concurrency defaults (tweak at runtime)
ENV OCR_WORKERS=2 \
    OCR_INFLIGHT=2 \
    MAX_UPLOAD_BYTES=250000000 \
    MAX_IMAGE_BYTES=15000000 \
    MAX_VIDEO_BYTES=200000000

# Recommended for multiple concurrent users:
# - use multiple worker processes (adjust based on CPU cores)
# - keep OCR_INFLIGHT modest per worker
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
