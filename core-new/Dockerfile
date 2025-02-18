FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-runtime

RUN apt-get update && apt-get install -y ffmpeg && apt-get clean && rm -rf /var/lib/apt/lists/*
ENV PATH="/opt/conda/bin:$PATH"
RUN pip install --upgrade pip
RUN pip install ultralytics freetype-py requests fastapi[standard] uvicorn pydantic
EXPOSE 8001
