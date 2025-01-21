import time

import redis
import requests
from config import (backend_host, core_service_host, redis_db, redis_host,
                    redis_port, scan_interval)

redis_conn = redis.Redis(host=redis_host, port=redis_port, db=redis_db)

while True:
    # print("Scanning for new actions")
    if redis_conn.llen("running_actions") > 0:
        time.sleep(scan_interval)
        continue
    if redis_conn.llen("waiting_actions") == 0:
        time.sleep(scan_interval)
        continue
    action = redis_conn.lpop("waiting_actions")
    # f"{action.user_id}-{data.action_id}-{action.video_id}"
    action_name = action.decode()
    action = action_name.split("-")
    user_id = action[0]
    action_id = action[1]
    video_id = action[2]
    video = requests.get(f"http://{backend_host}:8000/api/v1/videos/get_video_by_id/{video_id}").json()
    video_path = video["video_path"]

    action = requests.post(f"http://{core_service_host}:8001/inference/", json={
                           "video_path": video_path, "action_id": action_id, "action": action_name})
    if action.status_code == 200:
        redis_conn.rpush("running_actions",
                         f"{user_id}-{action_id}-{video_id}")
        requests.post(f"http://{backend_host}:8000/api/v1/actions/update_action_status",
                      json={"status": "running", "action_id": action_id, "action": action_name})
    else:
        redis_conn.rpush("error_actions", f"{user_id}-{action_id}-{video_id}")
        requests.post(f"http://{backend_host}:8000/api/v1/actions/update_action_status",
                      json={"status": f"error: {action.text}", "action_id": action_id, "action": action_name})
    time.sleep(scan_interval)
