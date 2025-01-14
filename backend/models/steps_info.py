from datetime import datetime

from sqlmodel import Field, SQLModel


class StepsInfo(SQLModel, table=True):
    id: int = Field(primary_key=True)
    stage_id: int
    step_id: int
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
    create_time: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    update_time: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    is_deleted: bool

    def __init__(self, stage_id: int, step_id: int, start_frame: int, end_frame: int, step_length: float, step_speed: float, front_leg: str, hip_min_degree: float, hip_max_degree: float, first_step: bool, steps_diff: float, stride_length: float, create_time: str, update_time: str, is_deleted: bool):
        self.stage_id = stage_id
        self.step_id = step_id
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.step_length = step_length
        self.step_speed = step_speed
        self.front_leg = front_leg
        self.hip_min_degree = hip_min_degree
        self.hip_max_degree = hip_max_degree
        self.first_step = first_step
        self.steps_diff = steps_diff
        self.stride_length = stride_length
        self.create_time = create_time
        self.update_time = update_time
        self.is_deleted = is_deleted

    def to_dict(self):
        return {
            "id": self.id,
            "stage_id": self.stage_id,
            "step_id": self.step_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "step_length": self.step_length,
            "step_speed": self.step_speed,
            "front_leg": self.front_leg,
            "hip_min_degree": self.hip_min_degree,
            "hip_max_degree": self.hip_max_degree,
            "first_step": self.first_step,
            "steps_diff": self.steps_diff,
            "stride_length": self.stride_length,
            "create_time": self.create_time,
            "update_time": self.update_time,
            "is_deleted": self.is_deleted
        }