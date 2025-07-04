import io
import os
import shutil
import tempfile
import uuid
from datetime import datetime

import httpx
from common.utils import convert_to_mp4, generate_thumbnail
from config import video_dir
from fastapi import APIRouter, Body, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from models import (Action, Doctors, Patients, SessionDep, Stage, StepsInfo,
                    VideoPath)
from pydantic import BaseModel
from sqlalchemy import select
from starlette.concurrency import run_in_threadpool

router = APIRouter(tags=["videos"], prefix="/videos")


class DeleteVideo(BaseModel):
    video_id: int
    doctor_id: int
    patient_id: int


class UpdateNotes(BaseModel):
    video_id: int
    doctor_id: int
    patient_id: int
    notes: str

class DownloadVideoFromUrl(BaseModel):
    patient_id: int
    url: str


@router.delete("/delete_video")
async def delete_video(video: DeleteVideo = Body(...), session: SessionDep = SessionDep):
    result = await session.execute(select(Doctors).where(
        Doctors.id == video.doctor_id, Doctors.is_deleted == False))
    doctor = result.scalar_one_or_none()
    if not doctor:
        return {"message": "Doctor not found"}
    if doctor.role_id != 1:
        result = await session.execute(select(VideoPath).where(
            VideoPath.id == video.video_id, VideoPath.is_deleted == False))
        video_ = result.scalar_one_or_none()
        if not video_:
            return {"message": "Video not found or this doctor does not have permission to delete this video"}
    else:
        result = await session.execute(select(VideoPath).where(
            VideoPath.id == video.video_id, VideoPath.is_deleted == False,
            VideoPath.patient_id == video.patient_id))
        video_ = result.scalar_one_or_none()
    if not video_:
        return {"message": "Video not found"}
    action_id = video_.action_id
    if not action_id:
        video_.is_deleted = True
        await session.commit()
        return {"message": "Video deleted successfully"}
    result = await session.execute(select(VideoPath).where(
        VideoPath.action_id == action_id, VideoPath.is_deleted == False))
    all_videos = result.scalars().all()
    for video_ in all_videos:
        video_.is_deleted = True

    result = await session.execute(select(Action).where(
        Action.original_video_id == video_.id, Action.is_deleted == False))
    all_actions = result.scalars().all()

    result = await session.execute(select(Action).where(
        Action.parent_id == action_id, Action.is_deleted == False))
    all_parent_actions = result.scalars().all()
    all_actions.extend(all_parent_actions)

    for action_ in all_actions:
        result = await session.execute(select(Stage).where(
            Stage.action_id == action_id, Stage.is_deleted == False))
        all_stages = result.scalars().all()
        for stage_ in all_stages:
            result = await session.execute(select(StepsInfo).where(
                StepsInfo.stage_id == stage_.id, StepsInfo.is_deleted == False))
            steps = result.scalars().all()
            for step in steps:
                step.is_deleted = True
            stage_.is_deleted = True
        action_.is_deleted = True

    await session.commit()

    return {"message": "Video deleted successfully"}

@router.post("/download_from_url")
async def download_from_url(params: DownloadVideoFromUrl = Body(...), session: SessionDep = SessionDep):
    result = await session.execute(select(Patients).where(
        Patients.id == params.patient_id, Patients.is_deleted == False))
    patient = result.scalar_one_or_none()
    if not patient:
        return {"message": "Patient not found"}
    gen_uuid = uuid.uuid4().hex[:8]
    output_path = os.path.join(
        video_dir, "original", f"{patient.id}-{gen_uuid}.mp4")
    async with httpx.AsyncClient() as client:
        response =  await client.post(
            "http://downloader:8080/api/download", json={"url": params.url, "output_path": output_path, "quality": "best", "info_only": False}, timeout=300
        )
        if response.status_code != 200:
            return response.json(), response.status_code
    new_video = VideoPath(
        video_path=output_path,
        patient_id=patient.id,
        original_video=True,
        inference_video=False,
        is_deleted=False,
        notes="下载视频",
        create_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        update_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    session.add(new_video)
    await session.commit()
    return response.json(), response.status_code


@router.post("/upload/{patient_id}")
async def upload_video(patient_id: int, video: UploadFile = File(...), session: SessionDep = SessionDep):
    result = await session.execute(select(Patients).where(
        Patients.id == patient_id, Patients.is_deleted == False))
    patient = result.scalar_one_or_none()
    if not patient:
        # Use HTTPException
        raise HTTPException(status_code=404, detail="Patient not found")

    if not video.content_type or not video.content_type.startswith("video/"):
        raise HTTPException(
            status_code=400, detail="Invalid file content type")

    # More robust format check (still relies on filename)
    supported_formats = ('.avi', '.mov', '.wmv', '.mkv', '.flv', '.mp4v', '.m4v', '.rmvb',
                         '.webm', '.mpeg', '.mpg', '.ts', '.vob', '.mp4')
    original_filename = video.filename or "unknown_video"
    file_ext = os.path.splitext(original_filename)[1].lower()

    if not file_ext:
        raise HTTPException(status_code=400, detail="File has no extension")

    if file_ext not in supported_formats:
        raise HTTPException(
            status_code=400, detail=f"Unsupported file format: {file_ext}")

    gen_uuid = uuid.uuid4().hex[:8]
    # Sanitize filename a bit
    safe_base_filename = os.path.splitext(original_filename)[
        0].replace(" ", "_").replace("/", "_")
    final_filename_base = f"{patient_id}-{safe_base_filename}-{gen_uuid}"
    # Define final path assuming MP4 output
    final_video_path = os.path.join(
        video_dir, "original", f"{final_filename_base}.mp4")
    final_video_dir = os.path.dirname(final_video_path)

    # Ensure output directory exists
    os.makedirs(final_video_dir, exist_ok=True)

    temp_file_path = None
    try:
        # --- Step 1: Save uploaded file to a temporary location ---
        # Use NamedTemporaryFile for automatic cleanup on error (usually)
        # Keep the original extension for the temporary file
        # Use a temporary directory known to be on the same filesystem as final_video_dir
        # if possible, otherwise shutil.move might perform a copy. /tmp is common.
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext, dir="/tmp") as temp_f:
            temp_file_path = temp_f.name
            size = 0
            chunk_size = 1024 * 1024  # 1MB chunks
            while True:
                # Use await to read the async stream
                chunk = await video.read(chunk_size)
                if not chunk:
                    break
                size += len(chunk)
                temp_f.write(chunk)
            # Logging
            print(f"Temporary file saved: {temp_file_path}, Size: {size}")

        # --- Step 2: Convert if necessary OR move if already MP4 ---
        if file_ext != ".mp4":
            print(f"Converting {temp_file_path} to {final_video_path}...")
            # Run the synchronous convert_to_mp4 (using FFmpeg) in a threadpool
            # to avoid blocking the async event loop.
            conversion_success = await run_in_threadpool(convert_to_mp4, temp_file_path, final_video_path)

            if not conversion_success:
                # convert_to_mp4 should ideally log its own errors
                print(f"Conversion failed for {temp_file_path}")
                raise HTTPException(
                    status_code=500, detail="Video conversion failed")
            else:
                print(f"Conversion successful: {final_video_path}")
                # Conversion done, original temp file can be removed now
                os.remove(temp_file_path)
                temp_file_path = None  # Mark as removed

        else:
            # It's already MP4, just move the temporary file to the final location
            print(f"Moving {temp_file_path} to {final_video_path}...")
            shutil.move(temp_file_path, final_video_path)
            temp_file_path = None  # Mark as moved

        # --- Step 3: Add record to database ---
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_video = VideoPath(
            video_path=final_video_path,
            patient_id=patient_id,
            original_video=True,
            inference_video=False,
            is_deleted=False,
            notes="视频",
            create_time=current_time,
            update_time=current_time
        )
        session.add(new_video)
        await session.commit()
        await session.refresh(new_video)  # Get the generated ID

        return {"message": "Video uploaded successfully", "video_id": new_video.id}

    except HTTPException as http_exc:
        # If an HTTPException was raised earlier, re-raise it
        raise http_exc
    except Exception as e:
        print(f"ERROR during video upload processing: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"An internal error occurred: {e}"
        ) from e
    finally:
        # --- Cleanup: Ensure temporary file is deleted if something went wrong ---
        if temp_file_path and os.path.exists(temp_file_path):
            print(f"Cleaning up leftover temporary file: {temp_file_path}")
            try:
                os.remove(temp_file_path)
            except OSError as rm_err:
                print(
                    f"Error removing temporary file {temp_file_path}: {rm_err}")
        # Ensure the underlying file stream is closed
        await video.close()


@router.get("/video/{video_type}/{patient_id}/{video_id}")
async def get_video(video_type: str, patient_id: int, video_id: int, session: SessionDep = SessionDep):
    if video_type not in ["original", "inference"]:
        return {"message": "Invalid video type"}
    result = await session.execute(select(VideoPath).where(
        VideoPath.id == video_id,
        VideoPath.patient_id == patient_id,
        VideoPath.original_video == (video_type == "original"),
        VideoPath.inference_video == (video_type == "inference"),
        VideoPath.is_deleted == False
    ))
    video = result.scalar_one_or_none()

    if not video:
        return {"message": "Video not found"}

    video_path = video.video_path
    if not os.path.isfile(video_path):
        print(
            f"Error: Database record found for video ID {video_id}, but file not found at {video_path}")
        raise HTTPException(
            status_code=404, detail="Video file not found on server.")

    return FileResponse(
        path=video_path,
        media_type="video/mp4",
        filename=os.path.basename(video_path)
    )


@router.get("/stream/{video_type}/{patient_id}/{video_id}")
async def stream_video(video_type: str, patient_id: int, video_id: int, session: SessionDep = SessionDep):
    if video_type not in ["original", "inference"]:
        raise HTTPException(status_code=400, detail="Invalid video type")

    result = await session.execute(select(VideoPath).where(
        VideoPath.id == video_id,
        VideoPath.patient_id == patient_id,
        VideoPath.original_video == (video_type == "original"),
        VideoPath.inference_video == (video_type == "inference"),
        VideoPath.is_deleted == False
    ))
    video = result.scalar_one_or_none()

    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    video_path = video.video_path
    if not os.path.isfile(video_path):
        raise HTTPException(
            status_code=404, detail="Video file not found on server.")

    return FileResponse(
        path=video_path,
        media_type="video/mp4",
        filename=os.path.basename(video_path)
    )


@router.get("/thumbnail_image/{video_type}/{patient_id}/{video_id}")
async def get_thumbnail_image(video_type: str, patient_id: int, video_id: int, session: SessionDep = SessionDep):
    if video_type not in ["original", "inference"]:
        return {"message": "Invalid video type"}
    result = await session.execute(select(VideoPath).where(
        VideoPath.id == video_id,
        VideoPath.patient_id == patient_id,
        VideoPath.original_video == (video_type == "original"),
        VideoPath.inference_video == (video_type == "inference"),
        VideoPath.is_deleted == False
    ))
    video = result.scalar_one_or_none()

    if not video:
        return {"message": "Video not found"}

    image_path = video.video_path.replace("mp4", "jpg")
    if not os.path.exists(image_path):
        await generate_thumbnail(video.video_path, image_path, time=1)

    import aiofiles
    async with aiofiles.open(image_path, "rb") as f:
        content = await f.read()

    return StreamingResponse(io.BytesIO(content), media_type="image/jpeg")


@router.get("/get_videos/{patient_id}")
async def get_videos(patient_id: int, session: SessionDep = SessionDep):
    result = await session.execute(select(VideoPath).where(
        VideoPath.patient_id == patient_id, VideoPath.is_deleted == False))
    videos = result.scalars().all()
    if not videos:
        return {"message": "No videos found"}
    videos = sorted(videos, key=lambda x: x.create_time, reverse=True)
    return {"videos": [video.to_dict() for video in videos]}


# @router.get("/get_inference_video_by_original_id/{original_video_id}")
# def get_video_by_original(original_video_id: int, session: SessionDep = SessionDep):
#     video = session.query(VideoPath).filter(
#         VideoPath.id == original_video_id, VideoPath.is_deleted == False).first()
#     if not video:
#         return {"message": "Video not found"}
#     video_path = video.video_path
#     video_path = video_path.replace("original", "inference")
#     reference_video = session.query(VideoPath).filter(
#         VideoPath.video_path == video_path,
#         VideoPath.original_video == False,
#         VideoPath.inference_video == True,
#         VideoPath.is_deleted == False).first()
#     return reference_video.to_dict() if reference_video else {"message": "Reference video not found"}


@router.get("/get_video_by_id/{video_id}")
async def get_video_by_id(video_id: int, session: SessionDep = SessionDep):
    result = await session.execute(select(VideoPath).where(
        VideoPath.id == video_id, VideoPath.is_deleted == False))
    video = result.scalar_one_or_none()
    return video.to_dict() if video else {"message": "Video not found"}


@router.post("/insert_inference_video/{action_id}")
async def insert_inference_video(action_id: int, session: SessionDep = SessionDep):
    result = await session.execute(select(VideoPath).where(
        VideoPath.action_id == action_id, VideoPath.is_deleted == False))
    video = result.scalar_one_or_none()
    if not video:
        return {"message": "Video not found"}
    new_video_path = video.video_path.replace("original", "inference")
    new_video_path = f"{os.path.dirname(new_video_path)}/{action_id}-{os.path.basename(new_video_path)}"
    new_video = VideoPath(video_path=new_video_path, patient_id=video.patient_id, original_video=False, inference_video=True, is_deleted=False, action_id=action_id, notes=None,
                          create_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), update_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    session.add(new_video)
    await session.commit()
    # Refresh to get the ID and avoid detached instance
    await session.refresh(new_video)
    return {"message": "Inference video inserted successfully", "video_id": new_video.id}


@router.patch("/notes")
async def update_video_notes(params: UpdateNotes = Body(...), session: SessionDep = SessionDep):
    result = await session.execute(select(Doctors).where(
        Doctors.id == params.doctor_id, Doctors.is_deleted == False))
    doctor = result.scalar_one_or_none()
    if not doctor:
        return {"message": "Doctor not found"}
    if doctor.role_id != 1:
        result = await session.execute(select(VideoPath).where(
            VideoPath.id == params.video_id, VideoPath.is_deleted == False))
        video_ = result.scalar_one_or_none()
        if not video_:
            return {"message": "Video not found or this doctor does not have permission to update this video"}
    else:
        result = await session.execute(select(VideoPath).where(
            VideoPath.id == params.video_id, VideoPath.is_deleted == False,
            VideoPath.patient_id == params.patient_id))
        video_ = result.scalar_one_or_none()
    if not video_:
        return {"message": "Video not found"}
    video_.notes = params.notes
    await session.commit()
    return {"message": "update notes successfully"}
