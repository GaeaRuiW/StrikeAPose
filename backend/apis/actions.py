from datetime import datetime
from typing import List, Optional

from common.utils import get_redis_connection
from fastapi import APIRouter, Body
from models import (Action, SessionDep, StepHipDegree, StepLength, StepSpeed,
                    StepStride, VideoPath)
from pydantic import BaseModel


class CreateAction(BaseModel):
    user_id: int
    video_id: int


class StepLengthData(BaseModel):
    step_id: int
    size: float
    left_feet: Optional[bool]
    right_feet: Optional[bool]


class StepSpeedData(BaseModel):
    step_id: int
    size: float
    left_feet: Optional[bool]
    right_feet: Optional[bool]


class StepStrideData(BaseModel):
    stride_id: int
    size: float


class StepHipDegreeData(BaseModel):
    step_id: int
    low: float
    high: float
    left_feet: Optional[bool]
    right_feet: Optional[bool]


class UpdateActionData(BaseModel):
    step_length: Optional[List[StepLengthData]]
    step_speed: Optional[List[StepSpeedData]]
    step_stride: Optional[List[StepStrideData]]
    step_hip_degree: Optional[List[StepHipDegreeData]]


class UpdateAction(BaseModel):
    action_id: int
    data: UpdateActionData

# data = {
#     "step_length": [
#         {
#             "step_id": 1,
#             "size": 10,
#             "left_feet": True,
#             "right_feet": False
#         },
#         {
#             "step_id": 2,
#             "size": 20,
#             "left_feet": False,
#             "right_feet": True
#         }
#     ],
#     "speed": [
#         {
#             "step_id": 1,
#             "speed": 10.0,
#             "left_feet": True,
#             "right_feet": False
#         },
#         {
#             "step_id": 2,
#             "speed": 20.0,
#             "left_feet": False,
#             "right_feet": True
#         }
#     ],
#     "step_stride": [
#         {
#             "stride_id": 1,
#             "stride": 10.0,
#         },
#         {
#             "stride_id": 2,
#             "stride": 20.0,
#         }
#     ],
#     "step_hip_degree": [
#         {
#             "step_id": 1,
#             "low": 10.0,
#             "high": 20.0,
#             "left_feet": True,
#             "right_feet": False
#         },
#         {
#             "step_id": 2,
#             "low": 20.0,
#             "high": 30.0,
#             "left_feet": False,
#             "right_feet": True
#         }
#     ]
# }


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
    step_hip_degrees = session.query(StepHipDegree).filter(StepHipDegree.action_id == action_id,
                                                           StepHipDegree.is_deleted == False).all()
    for step_hip_degree in step_hip_degrees:
        step_hip_degree.is_deleted = True
        step_hip_degree.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        session.add(step_hip_degree)
    step_lengths = session.query(StepLength).filter(
        StepLength.action_id == action_id, StepLength.is_deleted == False).all()
    for step_length in step_lengths:
        step_length.is_deleted = True
        step_length.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        session.add(step_length)
    speeds = session.query(StepSpeed).filter(
        StepSpeed.action_id == action_id, StepSpeed.is_deleted == False).all()
    for speed in speeds:
        speed.is_deleted = True
        speed.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        session.add(speed)
    step_strides = session.query(StepStride).filter(
        StepStride.action_id == action_id, StepStride.is_deleted == False).all()
    for step_stride in step_strides:
        step_stride.is_deleted = True
        step_stride.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        session.add(step_stride)
    session.commit()
    return {"message": "Action deleted successfully"}


@router.put("/update_action")
async def update_action(data: UpdateAction = Body(...), session: SessionDep = SessionDep):
    action = session.query(Action).filter(
        Action.id == data.action_id, Action.is_deleted == False).first()
    if not action:
        return {"message": "Action not found"}
    step_length_data = sorted(data.data.step_length, key=lambda x: x.step_id)
    speed_data = sorted(data.data.step_speed, key=lambda x: x.step_id)
    step_stride_data = sorted(data.data.step_stride, key=lambda x: x.stride_id)
    step_hip_degree_data = sorted(data.data.step_hip_degree, key=lambda x: x.step_id)
    for step_length in step_length_data:
        step_length_db = StepLength(action_id=data.action_id, step_id=step_length.step_id,
                                    size=step_length.size, left_feet=step_length.left_feet,
                                    right_feet=step_length.right_feet, is_deleted=False, create_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), update_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        session.add(step_length_db)
    for speed in speed_data:
        speed_db = StepSpeed(action_id=data.action_id, step_id=speed.step_id,
                                  size=speed.size, left_feet=speed.left_feet,
                                  right_feet=speed.right_feet, is_deleted=False, create_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), update_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        session.add(speed_db)
    for step_stride in step_stride_data:
        step_stride_db = StepStride(action_id=data.action_id, stride_id=step_stride.stride_id,
                                    size=step_stride.size, is_deleted=False, create_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), update_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        session.add(step_stride_db)
    for step_hip_degree in step_hip_degree_data:
        step_hip_degree_db = StepHipDegree(action_id=data.action_id, step_id=step_hip_degree.step_id,
                                           low=step_hip_degree.low, high=step_hip_degree.high,
                                           left_feet=step_hip_degree.left_feet, right_feet=step_hip_degree.right_feet,
                                           is_deleted=False, create_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), update_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        session.add(step_hip_degree_db)
    session.commit()
    redis_con = get_redis_connection()
    # redis_con.rpush("running_actions", f"{action.user_id}-{data.action_id}-{action.video_id}")
    # remove action from running list
    redis_con.lrem("running_actions", 0, f"{action.user_id}-{data.action_id}-{action.video_id}")
    action.status = "completed"
    action.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
