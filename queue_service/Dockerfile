FROM python:3.10-slim

WORKDIR /app
COPY ./queue_service/requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt
COPY ./queue_service /app
CMD ["./entrypoint.sh"]
