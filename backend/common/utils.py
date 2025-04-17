import os
import statistics

import bcrypt
import redis
from config import length_to_show, redis_db, redis_host, redis_port


def get_redis_connection():
    return redis.Redis(host=redis_host, port=redis_port, db=redis_db)


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def check_password(password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed_password.encode())


def get_length_to_show():
    return os.getenv("LENGTH_TO_SHOW", length_to_show)


def calculate_stats(data: list[float | int | None]) -> tuple[float, float]:
    filtered_data = [x for x in data if x is not None]

    n = len(filtered_data)
    if n == 0:
        return 0.0, 0.0

    average = sum(filtered_data) / n

    if n < 2:
        std_dev = 0.0
    else:
        std_dev = statistics.stdev(filtered_data)

    return round(average, 2), round(std_dev, 2)
