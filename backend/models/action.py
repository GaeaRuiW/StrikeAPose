from datetime import datetime

from sqlmodel import Field, SQLModel


class Action(SQLModel, table=True):
    id: int = Field(primary_key=True)
    parent_id: int = Field(default=None, foreign_key="action.id", nullable=True)
    original_video_id: int
    inference_video_id: int = Field(default=None, nullable=True)
    patient_id: int
    status: str
    progress: str
    create_time: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    update_time: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    is_deleted: bool

    def __init__(self, patient_id: int, status: str, progress: str, create_time: str, update_time: str, is_deleted: bool, parent_id: int = None, original_video_id: int = None, inference_video_id: int = None):
        self.patient_id = patient_id
        self.status = status
        self.progress = progress
        self.create_time = create_time
        self.update_time = update_time
        self.is_deleted = is_deleted
        self.parent_id = parent_id
        self.original_video_id = original_video_id
        self.inference_video_id = inference_video_id

    def to_dict(self):
        return {
            "id": self.id,
            "original_video_id": self.original_video_id,
            "inference_video_id": self.inference_video_id,
            "patient_id": self.patient_id,
            "status": self.status,
            "progress": self.progress,
            "create_time": self.create_time,
            "update_time": self.update_time,
            "is_deleted": self.is_deleted,
            "parent_id": self.parent_id
        }
