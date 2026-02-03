docker build -t poke-ocr-api . && \
docker stop pokemmo-ocr-api && \
docker rm pokemmo-ocr-api && \
docker run -d --name pokemmo-ocr-api --gpus all -e EASYOCR_GPU=1 -p 8000:8000 poke-ocr-api
