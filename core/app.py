import os
import warnings
from threading import Thread

import cv2
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from rtmpose_trt_inference import *
from steps_analysis import *

warnings.filterwarnings("ignore")
backend_host = os.getenv("BACKEND_HOST", "127.0.0.1")
update_action_url = f"http://{backend_host}:8000/api/v1/actions/update_action"
update_action_status_url = f"http://{backend_host}:8000/api/v1/actions/update_action_status"
insert_inference_video_url = f"http://{backend_host}:8000/api/v1/videos/insert_inference_video"

def flip_video(video_path):
    print("flipping video")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    file_extension = os.path.splitext(video_path)[1].lower()
    codec_map = {
        '.mp4': 'mp4v',  # MP4 文件
        '.avi': 'XVID',  # AVI 文件
        '.mov': 'avc1',  # MOV 文件（H.264）
        '.mkv': 'X264',  # MKV 文件（H.264）
        '.flv': 'FLV1',  # FLV 文件
        '.wmv': 'WMV2',  # WMV 文件
        '.webm': 'VP80',  # WebM 文件（VP8 编码）
        '.mpeg': 'MPEG',  # MPEG 文件
    }
    
    fourcc_code = codec_map.get(file_extension, 'mp4v')
    fourcc = cv2.VideoWriter_fourcc(*fourcc_code)
    out = cv2.VideoWriter(video_path.replace('original', 'flipped'), fourcc, fps, (int(cap.get(3)), int(cap.get(4))))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        out.write(frame)
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("flipping video done")
    return video_path.replace('original', 'flipped')

class InferenceRequest(BaseModel):
    action_id: int
    video_path: str
    action: str
    diff: int = 3
    num_circle: int = 3
    smooth_sigma: int = 15
    vis: bool = True


app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def jwt_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": str(exc.detail)},
    )


@app.post("/inference/")
async def inference_api(inference_request: InferenceRequest):
    action_id = inference_request.action_id
    video_path = inference_request.video_path
    output_path = video_path.replace('original', 'inference')
    output_path = os.path.dirname(output_path)
    action = inference_request.action
    differece_frames = int(inference_request.diff)
    num_circle = int(inference_request.num_circle)
    smooth_sigma = int(inference_request.smooth_sigma)
    vis = bool(inference_request.vis)

    def inference_thread():
        try:
            flipped_video_path = flip_video(video_path)
            result = inference(flipped_video_path, output_path,
                               differece_frames, smooth_sigma, num_circle, vis)
            if result is not None:
                data = {
                    "action_id": action_id,
                    "data": result
                }
                print(result)
                requests.put(update_action_url, json=data)
                requests.post(f"{insert_inference_video_url}/{action_id}")
                requests.post(update_action_status_url, json={"action_id": action_id, "status": "success", "action": action})
            else:
                print('No result! please check the csv and log file.')
                requests.post(update_action_status_url, json={"action_id": action_id, "status": "failed: no result", "action": action})
        except Exception as e:
            print(e)
            requests.post(update_action_status_url, json={"action_id": action_id, "status": f"failed: {str(e)}", "action": action})

    thread = Thread(target=inference_thread)
    thread.start()
    return {"message": "Inference started!"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)
