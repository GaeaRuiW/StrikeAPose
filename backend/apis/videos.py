import os
import uuid
from datetime import datetime

from config import video_dir
from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import StreamingResponse
from models import SessionDep, Users, VideoPath

router = APIRouter(tags=["videos"], prefix="/videos")


@router.post("/upload/{user_id}")
async def upload_video(user_id: int, video: UploadFile = File(...), session: SessionDep = SessionDep):
    user = session.query(Users).filter(
        Users.id == user_id and Users.is_deleted == False).first()
    if not user:
        return {"message": "User not found"}
    if not video.content_type.startswith("video/"):
        return {"message": "Invalid file type"}
    if not video.filename.endswith(".mp4"):
        return {"message": "Only mp4 files are allowed"}

    file_size = 0
    chunk_size = 1024 * 1024
    gen_uuid = uuid.uuid4().hex[:8]
    video_path = f"{video_dir}/original/{user_id}-{video.filename}-{gen_uuid}.mp4"
    with open(video_path, "wb") as f:
        while True:
            chunk = await video.read(chunk_size)
            if not chunk:
                break
            file_size += len(chunk)
            f.write(chunk)
    new_video = VideoPath(video_path=video_path, user_id=user_id, original_video=True, inference_video=False, is_deleted=False,
                          create_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), update_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    session.add(new_video)
    session.commit()
    return {"message": "Video uploaded successfully", "video_id": new_video.id}


@router.get("/stream/{video_type}/{user_id}/{video_id}")
async def stream_video(video_type: str, user_id: int, video_id: int, session: SessionDep = SessionDep, request: Request = None):
    if video_type not in ["original", "inference"]:
        return {"message": "Invalid video type"}
    video = session.query(VideoPath).filter(
        VideoPath.id == video_id,
        VideoPath.user_id == user_id,
        VideoPath.original_video == (video_type == "original"),
        VideoPath.inference_video == (video_type == "inference"),
        VideoPath.is_deleted == False
    ).first()

    if not video:
        return {"message": "Video not found"}

    video_path = video.video_path
    file_size = os.path.getsize(video_path)

    if range_header := request.headers.get("range", None):
        start, end = range_header.replace("bytes=", "").split("-")
        start = int(start)
        end = int(end) if end else file_size - 1
        chunk_size = end - start + 1

        def iter_file():
            with open(video_path, "rb") as f:
                f.seek(start)
                while chunk := f.read(chunk_size):
                    yield chunk

        headers = {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Type": "video/mp4",
        }

        return StreamingResponse(iter_file(), status_code=206, headers=headers)

    return StreamingResponse(open(video_path, "rb"), media_type="video/mp4")


@router.get("/get_videos/{user_id}")
async def get_videos(user_id: int, session: SessionDep = SessionDep):
    videos = session.query(VideoPath).filter(
        VideoPath.user_id == user_id and VideoPath.is_deleted == False).all()
    return {"videos": [video.to_dict() for video in videos]}

@router.get("/get_inference_video_by_original_id/{original_video_id}")
async def get_video_by_original(original_video_id: int, session: SessionDep = SessionDep):
    video = session.query(VideoPath).filter(
        VideoPath.id == original_video_id and VideoPath.is_deleted == False).first()
    video_path = video.video_path
    video_path = video_path.replace("original", "inference")
    reference_video = session.query(VideoPath).filter(
        VideoPath.video_path == video_path,
        VideoPath.original_video == False,
        VideoPath.inference_video == True,
        VideoPath.is_deleted == False).first()
    return reference_video.to_dict() if reference_video else {"message": "Reference video not found"}

@router.get("/get_video_by_id/{video_id}")
async def get_video_by_id(video_id: int, session: SessionDep = SessionDep):
    video = session.query(VideoPath).filter(
        VideoPath.id == video_id and VideoPath.is_deleted == False).first()
    return video.to_dict() if video else {"message": "Video not found"}


@router.post("/insert_inference_video/{action_id}")
async def insert_inference_video(action_id: int, session: SessionDep = SessionDep):
    video = session.query(VideoPath).filter(
        VideoPath.action_id == action_id and VideoPath.is_deleted == False).first()
    if not video:
        return {"message": "Video not found"}
    new_video_path = video.video_path.replace("original", "inference")
    new_video = VideoPath(video_path=new_video_path, user_id=video.user_id, original_video=False, inference_video=True, is_deleted=False, action_id=action_id,
                          create_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), update_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    session.add(new_video)
    session.commit()
    return {"message": "Inference video inserted successfully", "video_id": new_video.id}
