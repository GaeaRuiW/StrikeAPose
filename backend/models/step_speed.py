from datetime import datetime

from sqlmodel import Field, SQLModel


class StepSpeed(SQLModel, table=True):
    id: int = Field(primary_key=True)
    action_id: int
    step_id: int
    size: float
    left_feet: bool
    right_feet: bool
    create_time: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    update_time: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    is_deleted: bool

    def __init__(self, action_id: int, step_id: int, size: float, left_feet: bool, right_feet: bool, create_time: str, update_time: str, is_deleted: bool):
        self.action_id = action_id
        self.step_id = step_id
        self.size = size
        self.left_feet = left_feet
        self.right_feet = right_feet
        self.create_time = create_time
        self.update_time = update_time
        self.is_deleted = is_deleted

    def to_dict(self):
        return {
            "id": self.id,
            "action_id": self.action_id,
            "step_id": self.step_id,
            "size": self.size,
            "left_feet": self.left_feet,
            "right_feet": self.right_feet,
            "create_time": self.create_time,
            "update_time": self.update_time,
            "is_deleted": self.is_deleted
        }
