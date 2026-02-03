# Dockerfile
FROM python:3.11-slim

# Prevents Python from writing pyc files and buffers
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps:
# - libgl1/libglib2.0-0: needed by opencv
# - build-essential: helps with some wheels if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first (better caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of the app
COPY . /app

# If your JSON files live in /app/data, keep it as-is.
# Expose port
EXPOSE 8000

# Default: CPU EasyOCR. Set EASYOCR_GPU=1 for GPU runtime (see notes below).
ENV EASYOCR_GPU=0

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
