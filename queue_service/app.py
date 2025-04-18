import time
from datetime import datetime, timedelta
import redis
import requests
from config import (
    backend_host, 
    core_service_host, 
    redis_db, 
    redis_host,
    redis_port, 
    scan_interval,
    action_timeout
)

redis_conn = redis.Redis(host=redis_host, port=redis_port, db=redis_db)

def check_running_actions():
    # 获取所有运行中的actions
    running_actions = redis_conn.lrange("running_actions", 0, -1)
    for action in running_actions:
        action_name = action.decode()
        # 获取action的开始时间
        start_time_str = redis_conn.get(f"action_start_time:{action_name}")
        if start_time_str:
            start_time = datetime.strptime(start_time_str.decode(), "%Y-%m-%d %H:%M:%S")
            # 检查是否超时
            if datetime.now() - start_time > timedelta(seconds=action_timeout):
                # 解析action信息
                patient_id, action_id, video_id = action_name.split("-")
                # 将action标记为超时
                requests.post(
                    f"http://{backend_host}:8000/api/v1/actions/update_action_status",
                    json={
                        "action_id": action_id,
                        "status": "timeout",
                        "action": action_name
                    }
                )
                # 从运行队列中移除
                redis_conn.lrem("running_actions", 0, action)
                # 删除开始时间记录
                redis_conn.delete(f"action_start_time:{action_name}")

while True:
    # 检查运行中的actions是否超时
    check_running_actions()
    
    # 如果有正在运行的action，等待下一次检查
    if redis_conn.llen("running_actions") > 0:
        time.sleep(scan_interval)
        continue
        
    # 如果没有等待的action，等待下一次检查
    if redis_conn.llen("waiting_actions") == 0:
        time.sleep(scan_interval)
        continue
        
    # 处理等待中的action
    action = redis_conn.lpop("waiting_actions")
    action_name = action.decode()
    action = action_name.split("-")
    patient_id = action[0]
    action_id = action[1]
    video_id = action[2]
    
    video = requests.get(f"http://{backend_host}:8000/api/v1/videos/get_video_by_id/{video_id}").json()
    video_path = video["video_path"]

    action = requests.post(
        f"http://{core_service_host}:8001/inference/", 
        json={
            "video_path": video_path, 
            "action_id": action_id, 
            "action": action_name
        }
    )
    
    if action.status_code == 200:
        # 将action添加到运行队列并记录开始时间
        redis_conn.rpush("running_actions", action_name)
        redis_conn.set(
            f"action_start_time:{action_name}", 
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        requests.post(
            f"http://{backend_host}:8000/api/v1/actions/update_action_status",
            json={
                "status": "running", 
                "action_id": action_id, 
                "action": action_name
            }
        )
        requests.post(
            f"http://{backend_host}:8000/api/v1/actions/update_action_progress",
            json={
                "action_id": action_id, 
                "progress": "running"
            }
        )
    else:
        redis_conn.rpush("error_actions", action_name)
        requests.post(
            f"http://{backend_host}:8000/api/v1/actions/update_action_status",
            json={
                "status": f"error: {action.text}", 
                "action_id": action_id, 
                "action": action_name
            }
        )
        requests.post(
            f"http://{backend_host}:8000/api/v1/actions/update_action_progress",
            json={
                "action_id": action_id, 
                "progress": "error"
            }
        )
    time.sleep(scan_interval)
