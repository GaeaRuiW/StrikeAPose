from datetime import datetime

from sqlmodel import Field, SQLModel


class Action(SQLModel, table=True):
    id: int = Field(primary_key=True)
    video_id: int
    patient_id: int
    status: str
    progress: str
    create_time: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    update_time: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    is_deleted: bool

    def __init__(self, video_id: int, patient_id: int, status: str, progress: str, create_time: str, update_time: str, is_deleted: bool):
        self.video_id = video_id
        self.patient_id = patient_id
        self.status = status
        self.progress = progress
        self.create_time = create_time
        self.update_time = update_time
        self.is_deleted = is_deleted

    def to_dict(self):
        return {
            "id": self.id,
            "video_id": self.video_id,
            "patient_id": self.patient_id,
            "status": self.status,
            "progress": self.progress,
            "create_time": self.create_time,
            "update_time": self.update_time,
            "is_deleted": self.is_deleted
        }
