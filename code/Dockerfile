FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg gcc && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip
RUN pip install ultralytics freetype-py requests fastapi[standard] uvicorn pydantic tqdm lap cython
RUN pip install cython_bbox
EXPOSE 8001
