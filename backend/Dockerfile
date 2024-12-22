FROM python:3.10-slim

WORKDIR /app
COPY ./backend/requirements.txt /app/requirements.txt
RUN apt-get update && apt-get install -y libpq-dev gcc && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN pip install -r requirements.txt
COPY ./backend /app
CMD ["python", "app.py"]
