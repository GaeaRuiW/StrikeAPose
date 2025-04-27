import os
import uuid
from datetime import datetime

from config import video_dir
from fastapi import APIRouter, File, Request, UploadFile, Body
from fastapi.responses import StreamingResponse
from models import SessionDep, Patients, VideoPath, Doctors, Action, Stage, StepsInfo
from pydantic import BaseModel

router = APIRouter(tags=["videos"], prefix="/videos")


class DeleteVideo(BaseModel):
    video_id: int
    doctor_id: int
    patient_id: int


@router.delete("/delete_video")
async def delete_video(video: DeleteVideo = Body(...), session: SessionDep = SessionDep):
    doctor = session.query(Doctors).filter(
        Doctors.id == video.doctor_id, Doctors.is_deleted == False).first()
    if not doctor:
        return {"message": "Doctor not found"}
    if doctor.role_id != 1:
        video_ = session.query(VideoPath).filter(VideoPath.id == video.video_id,
                                                 VideoPath.is_deleted == False).first()
        if not video_:
            return {"message": "Video not found or this doctor does not have permission to delete this video"}
    else:
        video_ = session.query(VideoPath).filter(VideoPath.id == video.video_id, VideoPath.is_deleted == False, VideoPath.patient_id == video.patient_id).first()
    if not video_:
        return {"message": "Video not found"}
    action_id = video_.action_id
    if not action_id:
        session.delete(video_)
        session.commit()
        return {"message": "Video deleted successfully"}
    all_videos = session.query(VideoPath).filter(
        VideoPath.action_id == action_id, VideoPath.is_deleted == False).all()
    for video_ in all_videos:
        video_path = video_.video_path
        if os.path.exists(video_path):
            print(f"Deleting video: {video_path}")
            os.remove(video_path)
        if os.path.exists(video_path.replace("mp4", "json")):
            print(f"Deleting json: {video_path.replace('mp4', 'json')}")
            os.remove(video_path.replace("mp4", "json"))
        session.delete(video_)

    all_actions = session.query(Action).filter(
        Action.id == action_id, Action.is_deleted == False).all()
    all_parent_actions = session.query(Action).filter(
        Action.parent_id == action_id, Action.is_deleted == False).all()
    all_actions.extend(all_parent_actions)
    if not all_actions:
        session.commit()
        return {"message": "Video deleted successfully"}
    for action_ in all_actions:
        all_stages = session.query(Stage).filter(
        Stage.action_id == action_id, Stage.is_deleted == False).all()
        if not all_stages:
            session.commit()
            return {"message": "Video deleted successfully"}
        for stage_ in all_stages:
            steps = session.query(StepsInfo).filter(
                StepsInfo.stage_id == stage_.id, StepsInfo.is_deleted == False).all()
            if not steps:
                pass
            for step in steps:
                session.delete(step)
            session.delete(stage_)
        session.delete(action_)
    session.commit()

    return {"message": "Video deleted successfully"}

@router.post("/upload/{patient_id}")
async def upload_video(patient_id: int, video: UploadFile = File(...), session: SessionDep = SessionDep):
    patient = session.query(Patients).filter(
        Patients.id == patient_id, Patients.is_deleted == False).first()
    if not patient:
        return {"message": "Patient not found"}
    if not video.content_type.startswith("video/"):
        return {"message": "Invalid file type"}
    if not video.filename.endswith(".mp4"):
        return {"message": "Only mp4 files are allowed"}

    file_size = 0
    chunk_size = 1024 * 1024
    gen_uuid = uuid.uuid4().hex[:8]
    video_path = f"{video_dir}/original/{patient_id}-{video.filename}-{gen_uuid}.mp4"
    with open(video_path, "wb") as f:
        while True:
            chunk = await video.read(chunk_size)
            if not chunk:
                break
            file_size += len(chunk)
            f.write(chunk)
    new_video = VideoPath(video_path=video_path, patient_id=patient_id, original_video=True, inference_video=False, is_deleted=False,
                          create_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), update_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    session.add(new_video)
    session.commit()
    return {"message": "Video uploaded successfully", "video_id": new_video.id}


@router.get("/stream/{video_type}/{patient_id}/{video_id}")
async def stream_video(video_type: str, patient_id: int, video_id: int, session: SessionDep = SessionDep, request: Request = None):
    if video_type not in ["original", "inference"]:
        return {"message": "Invalid video type"}
    video = session.query(VideoPath).filter(
        VideoPath.id == video_id,
        VideoPath.patient_id == patient_id,
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


@router.get("/get_videos/{patient_id}")
async def get_videos(patient_id: int, session: SessionDep = SessionDep):
    videos = session.query(VideoPath).filter(
        VideoPath.patient_id == patient_id, VideoPath.is_deleted == False).all()
    if not videos:
        return {"message": "No videos found"}
    videos = sorted(videos, key=lambda x: x.create_time)
    return {"videos": [video.to_dict() for video in videos]}

@router.get("/get_inference_video_by_original_id/{original_video_id}")
async def get_video_by_original(original_video_id: int, session: SessionDep = SessionDep):
    video = session.query(VideoPath).filter(
        VideoPath.id == original_video_id, VideoPath.is_deleted == False).first()
    if not video:
        return {"message": "Video not found"}
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
        VideoPath.id == video_id, VideoPath.is_deleted == False).first()
    return video.to_dict() if video else {"message": "Video not found"}


@router.post("/insert_inference_video/{action_id}")
async def insert_inference_video(action_id: int, session: SessionDep = SessionDep):
    video = session.query(VideoPath).filter(
        VideoPath.action_id == action_id, VideoPath.is_deleted == False).first()
    if not video:
        return {"message": "Video not found"}
    new_video_path = video.video_path.replace("original", "inference")
    new_video = VideoPath(video_path=new_video_path, patient_id=video.patient_id, original_video=False, inference_video=True, is_deleted=False, action_id=action_id,
                          create_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), update_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    session.add(new_video)
    session.commit()
    return {"message": "Inference video inserted successfully", "video_id": new_video.id}
