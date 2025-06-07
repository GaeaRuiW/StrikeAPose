from datetime import datetime
from typing import Optional
from sqlmodel import Field, SQLModel


class Patients(SQLModel, table=True):
    id: int = Field(primary_key=True)
    username: str = Field(index=True)
    age: Optional[int] = Field(default=None, nullable=True)
    gender: Optional[str] = Field(default=None, nullable=True)
    case_id: str = Field(index=True, nullable=False, unique=True)
    doctor_id: Optional[int] = Field(default=None, nullable=True)
    notes: Optional[str] = Field(default=None, nullable=True)
    create_time: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    update_time: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    is_deleted: bool

    def __init__(self, username: str, age: int, gender: str, case_id: str, doctor_id: int, create_time: str, update_time: str, is_deleted: bool, notes: Optional[str] = None):
        self.username = username
        self.age = age
        self.gender = gender
        self.case_id = case_id
        self.doctor_id = doctor_id
        self.notes = notes
        self.create_time = create_time
        self.update_time = update_time
        self.is_deleted = is_deleted

    def to_dict(self):
        return {
            "id": self.id,
            "username": self.username,
            "age": self.age,
            "gender": self.gender,
            "case_id": self.case_id,
            "doctor_id": self.doctor_id,
            "notes": self.notes,
            "create_time": self.create_time,
            "update_time": self.update_time,
            # is_deleted is often not needed in frontend GET responses unless specifically required
        }

