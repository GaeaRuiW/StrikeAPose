import os

listen_port = 8000

video_dir = "/data/videos"

# Postgres
postgres_user = "postgres"
postgres_password = os.getenv("POSTGRES_PASSWORD", "postgres")
postgres_db = "pose"
postgres_host = os.getenv("POSTGRES_HOST", "localhost")
postgres_port = 5432
postgres_uri = f"postgresql://{postgres_user}:{postgres_password}@{postgres_host}:{postgres_port}/{postgres_db}"

# Redis
redis_host = os.getenv("REDIS_HOST", "localhost")
redis_port = 6379
redis_db = 0

# Dashboard
length_to_show = 20
