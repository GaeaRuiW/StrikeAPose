import os
import warnings
from threading import Thread

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
    vis = True if inference_request.vis else False

    def inference_thread():
        try:
            result = inference(video_path, output_path,
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
