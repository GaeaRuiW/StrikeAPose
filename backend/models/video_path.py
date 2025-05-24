from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel


class VideoPath(SQLModel, table=True):
    id: int = Field(primary_key=True)
    patient_id: int
    action_id: Optional[int] = Field(default=None, nullable=True)
    original_video: bool
    inference_video: bool
    video_path: str
    notes: str = Field(default=None, nullable=True)
    create_time: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    update_time: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    is_deleted: bool

    def __init__(self, patient_id: int, original_video: bool, inference_video: bool, video_path: str, create_time: str, update_time: str, is_deleted: bool, notes: str, action_id: int = None):
        self.patient_id = patient_id
        self.action_id = action_id
        self.original_video = original_video
        self.inference_video = inference_video
        self.video_path = video_path
        self.create_time = create_time
        self.update_time = update_time
        self.is_deleted = is_deleted
        self.notes = notes

    def to_dict(self):
        return {
            "id": self.id,
            "patient_id": self.patient_id,
            "action_id": self.action_id,
            "original_video": self.original_video,
            "inference_video": self.inference_video,
            "video_path": self.video_path,
            "create_time": self.create_time,
            "update_time": self.update_time,
            "notes": self.notes,
            "is_deleted": self.is_deleted
        }
