from datetime import datetime
from typing import List, Optional

from common.utils import get_redis_connection
from fastapi import APIRouter, Body
from models import Action, SessionDep, Stage, StepsInfo, VideoPath
from pydantic import BaseModel


class CreateAction(BaseModel):
    user_id: int
    video_id: int


class StepsInfoData(BaseModel):
    start_frame: int
    end_frame: int
    step_length: float
    step_speed: float
    front_leg: str
    hip_min_degree: float
    hip_max_degree: float
    first_step: bool
    steps_diff: float
    stride_length: float


class UpdateActionData(BaseModel):
    stage_n: int
    start_frame: int
    end_frame: int
    steps_info: Optional[List[StepsInfoData]]


class UpdateAction(BaseModel):
    action_id: int
    data: Optional[List[UpdateActionData]]


router = APIRouter(tags=["actions"], prefix="/actions")


@router.post("/")
async def create_action(action: CreateAction = Body(...), session: SessionDep = SessionDep):
    video = session.query(VideoPath).filter(VideoPath.id == action.video_id,
                                            VideoPath.user_id == action.user_id, VideoPath.original_video == True, VideoPath.is_deleted == False).first()
    if not video:
        return {"message": "Video not found"}
    new_action = Action(user_id=action.user_id,
                        video_id=action.video_id, status="waiting", is_deleted=False, create_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), update_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    session.add(new_action)
    session.commit()
    action_id = new_action.id
    redis_client = get_redis_connection()
    redis_client.rpush("waiting_actions",
                       f"{action.user_id}-{action_id}-{action.video_id}")
    video.action_id = action_id
    video.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    session.add(video)
    session.commit()
    return {"message": "Action created successfully", "action_id": action_id}


@router.get("/get_actions/{user_id}")
async def get_actions(user_id: int, session: SessionDep = SessionDep):
    actions = session.query(Action).filter(
        Action.user_id == user_id and Action.is_deleted == False).all()
    return {"actions": [action.to_dict() for action in actions]}


@router.delete("/delete_action/{action_id}")
async def delete_action(action_id: int, session: SessionDep = SessionDep):
    action = session.query(Action).filter(
        Action.id == action_id, Action.is_deleted == False).first()
    if not action:
        return {"message": "Action not found"}
    action.is_deleted = True
    action.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    session.add(action)
    videos = session.query(VideoPath).filter(
        VideoPath.action_id == action_id, VideoPath.is_deleted == False).all()
    for video in videos:
        video.is_deleted = True
        video.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        session.add(video)
    stages = session.query(Stage).filter(
        Stage.action_id == action_id, Stage.is_deleted == False).all()
    for stage in stages:
        stage.is_deleted = True
        stage.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        session.add(stage)
        steps_info = session.query(StepsInfo).filter(
            StepsInfo.stage_id == stage.id, StepsInfo.is_deleted == False).all()
        for step_info in steps_info:
            step_info.is_deleted = True
            step_info.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            session.add(step_info)
    session.commit()
    return {"message": "Action deleted successfully"}


@router.put("/update_action")
async def update_action(data: UpdateAction = Body(...), session: SessionDep = SessionDep):
    action = session.query(Action).filter(
        Action.id == data.action_id, Action.is_deleted == False).first()
    if not action:
        return {"message": "Action not found"}
    for stage_data in sorted(data.data, key=lambda x: x.stage_n):
        stage = Stage(action_id=data.action_id, stage_n=stage_data.stage_n,
                      start_frame=stage_data.start_frame, end_frame=stage_data.end_frame, is_deleted=False, create_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), update_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        session.add(stage)
        session.commit()
        for n, step_info in enumerate(stage_data.steps_info):
            step_info_db = StepsInfo(stage_id=stage.id, step_id=n + 1,
                                     start_frame=step_info.start_frame, end_frame=step_info.end_frame,
                                     step_length=step_info.step_length, step_speed=step_info.step_speed,
                                     front_leg=step_info.front_leg, hip_min_degree=step_info.hip_min_degree,
                                     hip_max_degree=step_info.hip_max_degree, first_step=step_info.first_step,
                                     steps_diff=step_info.steps_diff, stride_length=step_info.stride_length,
                                     is_deleted=False, create_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), update_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            session.add(step_info_db)
    session.commit()
    return {"message": "Action updated successfully"}


@router.post("/update_action_status")
async def update_action_status(action_id: int, status: str, session: SessionDep = SessionDep):
    action = session.query(Action).filter(
        Action.id == action_id, Action.is_deleted == False).first()
    if not action:
        return {"message": "Action not found"}
    action.status = status
    action.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    session.update(action)
    session.commit()
    return {"message": "Action status updated successfully"}
