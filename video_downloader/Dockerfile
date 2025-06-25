FROM python:3.10-slim

WORKDIR /app
COPY ./video_downloader/requirements.txt /app/requirements.txt
RUN apt-get update && apt-get install -y libpq-dev gcc libgl1 ffmpeg && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN pip install -r requirements.txt
COPY ./video_downloader /app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]
