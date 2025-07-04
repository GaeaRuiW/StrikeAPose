from datetime import datetime
from typing import List, Optional
from sqlalchemy import select
from common.utils import async_redis_rpush, async_redis_lrem
from fastapi import APIRouter, Body
from models import Action, SessionDep, Stage, StepsInfo, VideoPath
from pydantic import BaseModel


class CreateAction(BaseModel):
    patient_id: int
    video_id: int
    parent_id: Optional[int] = None


class StepsInfoData(BaseModel):
    start_frame: int
    end_frame: int
    step_length: float
    step_speed: float
    front_leg: str
    support_time: float
    liftoff_height: float
    hip_min_degree: float
    hip_max_degree: float
    first_step: bool
    steps_diff: float
    stride_length: float
    step_width: float


class UpdateActionData(BaseModel):
    stage_n: int
    start_frame: int
    end_frame: int
    steps_info: Optional[List[StepsInfoData]]


class UpdateAction(BaseModel):
    action_id: int
    inference_video_id: Optional[int] = None
    data: Optional[List[UpdateActionData]]


class UpdateActionStatus(BaseModel):
    action_id: int
    status: str
    action: str


class UpdateActionProgress(BaseModel):
    action_id: int
    progress: str


router = APIRouter(tags=["actions"], prefix="/actions")
# redis_conn = get_redis_connection()


@router.post("/")
async def create_action(action: CreateAction = Body(...), session: SessionDep = SessionDep):
    result = await session.execute(select(VideoPath).where(VideoPath.id == action.video_id,
                                            VideoPath.patient_id == action.patient_id, VideoPath.original_video == True, VideoPath.is_deleted == False))
    video = result.scalar_one_or_none()
    if not video:
        return {"message": "Video not found"}

    if action.parent_id:
        result = await session.execute(select(Action).where(
            Action.id == action.parent_id, Action.is_deleted == False))
        parent_id_exists = result.scalar_one_or_none()
        if not parent_id_exists:
            return {"message": "Parent action not found"}

    new_action = Action(patient_id=action.patient_id,
                        original_video_id=action.video_id, status="waiting", progress="waiting for processing", is_deleted=False, create_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), update_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        parent_id=action.parent_id or None)
    session.add(new_action)
    await session.commit()
    await session.refresh(new_action)  # Refresh to get the ID and avoid detached instance
    action_id = new_action.id

    result = await session.execute(select(Action).where(
        Action.id == action_id, Action.is_deleted == False))
    action_ = result.scalar_one_or_none()
    if not action_:
        return {"message": "Action not found"}
    action_.parent_id = action.parent_id or action_id
    await session.commit()

    await async_redis_rpush("waiting_actions",
                            f"{action.patient_id}-{action_id}-{action.video_id}")
    video.action_id = action_id
    video.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    session.add(video)
    await session.commit()
    return {"message": "Action created successfully", "action_id": action_id}


@router.get("/get_actions/{patient_id}")
async def get_actions(patient_id: int, session: SessionDep = SessionDep):
    result = await session.execute(select(Action).where(
        Action.patient_id == patient_id, Action.is_deleted == False))
    actions = result.scalars().all()
    if not actions:
        return {"message": "No actions found"}
    actions = sorted(actions, key=lambda x: x.create_time, reverse=True)
    return {"actions": [action.to_dict() for action in actions]}


@router.get("/get_action_by_id/{action_id}")
async def get_action_by_id(action_id: int, session: SessionDep = SessionDep):
    result = await session.execute(select(Action).where(
        Action.id == action_id, Action.is_deleted == False))
    action = result.scalar_one_or_none()

    if action:
        return {"action": action.to_dict()}
    else:
        return {"message": "Action not found"}


@router.get("/get_action_by_parent_id/{parent_id}")
async def get_action_by_parent_id(parent_id: int, session: SessionDep = SessionDep):
    result = await session.execute(select(Action).where(
        Action.parent_id == parent_id, Action.is_deleted == False))
    actions = result.scalars().all()
    if not actions:
        return {"message": "No actions found"}
    actions = sorted(actions, key=lambda x: x.create_time, reverse=True)
    return {"action": [a.to_dict() for a in actions]}


@router.delete("/delete_action/{action_id}")
async def delete_action(action_id: int, session: SessionDep = SessionDep):
    result = await session.execute(select(Action).where(
        Action.id == action_id, Action.is_deleted == False))
    action = result.scalar_one_or_none()
    if not action:
        return {"message": "Action not found"}
    action.is_deleted = True
    action.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Get all actions with this ID (should be just one)
    result = await session.execute(select(Action).where(
        Action.id == action_id, Action.is_deleted == False))
    actions = result.scalars().all()
    for action_ in actions:
        action_.is_deleted = True
        action_.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Get stages for this action
        result = await session.execute(select(Stage).where(
            Stage.action_id == action_.id, Stage.is_deleted == False))
        stages = result.scalars().all()
        for stage in stages:
            stage.is_deleted = True
            stage.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Get steps info for this stage
            result = await session.execute(select(StepsInfo).where(
                StepsInfo.stage_id == stage.id, StepsInfo.is_deleted == False))
            steps_info = result.scalars().all()
            for step_info in steps_info:
                step_info.is_deleted = True
                step_info.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # videos = session.query(VideoPath).filter(
        #     VideoPath.action_id == action_id, VideoPath.is_deleted == False).all()
        # for video in videos:
        #     video.is_deleted = True
        #     video.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # stages = session.query(Stage).filter(
    #     Stage.action_id == action_id, Stage.is_deleted == False).all()
    # for stage in stages:
    #     stage.is_deleted = True
    #     stage.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #     steps_info = session.query(StepsInfo).filter(
    #         StepsInfo.stage_id == stage.id, StepsInfo.is_deleted == False).all()
    #     for step_info in steps_info:
    #         step_info.is_deleted = True
    #         step_info.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    await session.commit()
    await session.refresh(action)
    await async_redis_lrem("waiting_actions", 0,
                           f"{action.patient_id}-{action_id}-{action.original_video_id}")
    await async_redis_lrem("running_actions", 0,
                           f"{action.patient_id}-{action_id}-{action.original_video_id}")
    return {"message": "Action deleted successfully"}


@router.put("/update_action")
async def update_action(data: UpdateAction = Body(...), session: SessionDep = SessionDep):
    result = await session.execute(select(Action).where(
        Action.id == data.action_id, Action.is_deleted == False))
    action = result.scalar_one_or_none()
    if not action:
        return {"message": "Action not found"}
    if data.inference_video_id:
        action.inference_video_id = data.inference_video_id
        action.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        await session.commit()
    for stage_data in sorted(data.data, key=lambda x: x.stage_n):
        stage = Stage(action_id=data.action_id, stage_n=stage_data.stage_n,
                      start_frame=stage_data.start_frame, end_frame=stage_data.end_frame, is_deleted=False, create_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), update_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        session.add(stage)
        await session.commit()
        await session.refresh(stage)  # Refresh to get the ID and avoid detached instance
        for n, step_info in enumerate(stage_data.steps_info):
            step_info_db = StepsInfo(stage_id=stage.id, step_id=n + 1,
                                     start_frame=step_info.start_frame, end_frame=step_info.end_frame,
                                     step_length=step_info.step_length, step_speed=step_info.step_speed,
                                     front_leg=step_info.front_leg, hip_min_degree=step_info.hip_min_degree,
                                     hip_max_degree=step_info.hip_max_degree, first_step=step_info.first_step,
                                     steps_diff=step_info.steps_diff, stride_length=step_info.stride_length,
                                     step_width=step_info.step_width,
                                     support_time=step_info.support_time, liftoff_height=step_info.liftoff_height,
                                     is_deleted=False, create_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), update_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            session.add(step_info_db)
    await session.commit()
    return {"message": "Action updated successfully"}


@router.post("/update_action_status")
async def update_action_status(action_status: UpdateActionStatus, session: SessionDep = SessionDep):
    action_id = action_status.action_id
    status = action_status.status
    action_name = action_status.action
    if status != "running":
        await async_redis_lrem("running_actions", 0, action_name)
    result = await session.execute(select(Action).where(
        Action.id == action_id, Action.is_deleted == False))
    action = result.scalar_one_or_none()
    if not action:
        return {"message": "Action not found"}
    action.status = status
    action.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    await session.commit()
    return {"message": "Action status updated successfully"}


@router.post("/update_action_progress")
async def update_action_progress(action_progress: UpdateActionProgress, session: SessionDep = SessionDep):
    action_id = action_progress.action_id
    progress = action_progress.progress
    result = await session.execute(select(Action).where(
        Action.id == action_id, Action.is_deleted == False))
    action = result.scalar_one_or_none()
    if not action:
        return {"message": "Action not found"}
    action.progress = progress
    action.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    await session.commit()
    return {"message": "Action progress updated successfully"}
