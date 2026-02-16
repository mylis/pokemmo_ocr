@echo off
docker build -t poke-ocr-api .
docker rm -f pokemmo-ocr-api >nul 2>&1
docker run -d ^
  --name pokemmo-ocr-api ^
  --restart unless-stopped ^
  --gpus all ^
  --shm-size=1g ^
  -e EASYOCR_GPU=1 ^
  -e WEB_CONCURRENCY=2 ^
  -e GUNICORN_MAX_REQUESTS=800 ^
  -e GUNICORN_MAX_REQUESTS_JITTER=200 ^
  -p 8000:8000 ^
  poke-ocr-api
