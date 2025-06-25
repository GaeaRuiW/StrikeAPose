import os
import asyncio
import statistics

import bcrypt
import cv2
import redis
import aiofiles
from config import length_to_show, redis_db, redis_host, redis_port


def get_redis_connection():
    """获取同步Redis连接（向后兼容）"""
    return redis.Redis(host=redis_host, port=redis_port, db=redis_db)


async def get_async_redis_connection():
    """获取异步Redis连接"""
    import aioredis
    return await aioredis.from_url(f"redis://{redis_host}:{redis_port}/{redis_db}")


async def async_redis_operation(operation_func, *args, **kwargs):
    """
    异步执行Redis操作的包装器
    对于简单的Redis操作，在线程池中执行同步操作
    """
    def _redis_operation():
        redis_client = get_redis_connection()
        return operation_func(redis_client, *args, **kwargs)

    return await asyncio.get_event_loop().run_in_executor(None, _redis_operation)


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

    std_dev = 0.0 if n < 2 else statistics.stdev(filtered_data)
    return round(average, 4), round(std_dev, 4)


async def generate_thumbnail(video_path: str, thumbnail_path: str, time: int = 1):
    def _generate_thumbnail():
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, time * 1000)
        success, image = cap.read()
        if success:
            cv2.imwrite(thumbnail_path, image)
        cap.release()

    await asyncio.get_event_loop().run_in_executor(None, _generate_thumbnail)

async def convert_to_mp4(input_path, output_path, ffmpeg_executable="ffmpeg"):
    # 异步检查输入文件是否存在
    def _check_file_exists(path):
        return os.path.exists(path)

    if not await asyncio.get_event_loop().run_in_executor(None, _check_file_exists, input_path):
        print(f"错误：输入文件不存在: {input_path}")
        return False

    output_dir = os.path.dirname(output_path)
    if output_dir:
        def _check_and_create_dir(dir_path):
            if not os.path.exists(dir_path):
                try:
                    os.makedirs(dir_path)
                    return True, f"已创建输出目录: {dir_path}"
                except OSError as e:
                    return False, f"错误：无法创建输出目录 {dir_path}: {e}"
            return True, None

        success, message = await asyncio.get_event_loop().run_in_executor(None, _check_and_create_dir, output_dir)
        if not success:
            print(message)
            return False
        elif message:
            print(message)

    command = [
        ffmpeg_executable,
        "-i", input_path,
        "-c:v", "libx264",
        "-c:a", "aac",
        "-pix_fmt", "yuv420p",
        "-y",
        output_path
    ]

    print(f"\n正在转换: {input_path} -> {output_path}")
    print(f"执行命令: {' '.join(command)}")

    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()

        if process.returncode == 0:
            print("转换成功!")
            return True
        else:
            print(f"错误：FFmpeg 转换失败 (返回码: {process.returncode})")
            print("FFmpeg 标准输出:\n", stdout.decode())
            print("FFmpeg 标准错误:\n", stderr.decode())
            # 如果转换失败，尝试删除可能已创建的不完整输出文件
            def _remove_file_if_exists(file_path):
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        return True, f"已删除不完整的输出文件: {file_path}"
                    except OSError as remove_err:
                        return False, f"警告：无法删除不完整的输出文件 {file_path}: {remove_err}"
                return True, None

            success, message = await asyncio.get_event_loop().run_in_executor(None, _remove_file_if_exists, output_path)
            if message:
                print(message)
            return False
    except FileNotFoundError:
        print(f"错误：找不到 FFmpeg 可执行文件 '{ffmpeg_executable}'。请检查路径或 PATH 设置。")
        return False
    except Exception as e:
        print(f"发生未知错误: {e}")
        return False


async def async_file_exists(file_path: str) -> bool:
    """异步检查文件是否存在"""
    return await asyncio.get_event_loop().run_in_executor(None, os.path.exists, file_path)


async def async_remove_file(file_path: str) -> bool:
    """异步删除文件"""
    def _remove_file():
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
        except OSError:
            return False
        return True

    return await asyncio.get_event_loop().run_in_executor(None, _remove_file)


async def async_makedirs(dir_path: str) -> bool:
    """异步创建目录"""
    def _makedirs():
        try:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            return True
        except OSError:
            return False

    return await asyncio.get_event_loop().run_in_executor(None, _makedirs)


async def async_read_file(file_path: str) -> str:
    """异步读取文件内容"""
    async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
        return await f.read()


async def async_write_file(file_path: str, content: str) -> bool:
    """异步写入文件内容"""
    try:
        async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
            await f.write(content)
        return True
    except Exception:
        return False


# 异步Redis操作便利函数
async def async_redis_rpush(queue_name: str, value: str) -> int:
    """异步向Redis队列右侧推入值"""
    def _rpush(redis_client, queue, val):
        return redis_client.rpush(queue, val)

    return await async_redis_operation(_rpush, queue_name, value)


async def async_redis_lrem(queue_name: str, count: int, value: str) -> int:
    """异步从Redis队列中移除值"""
    def _lrem(redis_client, queue, cnt, val):
        return redis_client.lrem(queue, cnt, val)

    return await async_redis_operation(_lrem, queue_name, count, value)


async def async_redis_lpop(queue_name: str) -> str:
    """异步从Redis队列左侧弹出值"""
    def _lpop(redis_client, queue):
        result = redis_client.lpop(queue)
        return result.decode() if result else None

    return await async_redis_operation(_lpop, queue_name)


async def async_redis_llen(queue_name: str) -> int:
    """异步获取Redis队列长度"""
    def _llen(redis_client, queue):
        return redis_client.llen(queue)

    return await async_redis_operation(_llen, queue_name)