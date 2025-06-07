from datetime import datetime

from sqlmodel import Field, SQLModel


class Stage(SQLModel, table=True):
    id: int = Field(primary_key=True)
    object_id: int
    stage_n: int
    start_frame: int
    end_frame: int
    create_time: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    update_time: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    is_deleted: bool

    def __init__(self, object_id: int, stage_n: int, start_frame: int, end_frame: int, create_time: str, update_time: str, is_deleted: bool):
        self.object_id = object_id
        self.stage_n = stage_n
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.create_time = create_time
        self.update_time = update_time
        self.is_deleted = is_deleted

    def to_dict(self):
        return {
            "id": self.id,
            "object_id": self.object_id,
            "stage_n": self.stage_n,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "create_time": self.create_time,
            "update_time": self.update_time,
            "is_deleted": self.is_deleted
        }
