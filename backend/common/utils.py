import os

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
