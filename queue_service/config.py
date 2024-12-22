import os

redis_host = os.getenv("REDIS_HOST", "localhost")
redis_port = 6379
redis_db = 0

scan_interval = 5

core_service_host = os.getenv("CORE_SERVICE_HOST", "localhost")
backend_host = os.getenv("BACKEND_HOST", "localhost")