
from datetime import datetime
from typing import Optional
from sqlmodel import Field, SQLModel


class Doctors(SQLModel, table=True):
    id: int = Field(primary_key=True)
    username: str = Field(index=True)
    password: str
    email: str = Field(unique=True, index=True, nullable=False)
    phone: Optional[str] = Field(default=None, nullable=True)
    department: Optional[str] = Field(default="康复科", nullable=True)
    role_id: Optional[int] = Field(default=2, nullable=True) # Default to 2 (Doctor role) instead of 1 (Admin)
    notes: Optional[str] = Field(default=None, nullable=True)
    create_time: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    update_time: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    is_deleted: bool

    def __init__(self, username: str, password: str, email: str, phone: str, create_time: str, update_time: str, is_deleted: bool, department: str = "康复科", role_id: int = 2, notes: Optional[str] = None):
        self.username = username
        self.password = password
        self.email = email
        self.phone = phone
        self.role_id = role_id
        self.notes = notes
        self.create_time = create_time
        self.update_time = update_time
        self.is_deleted = is_deleted
        self.department = department

    def to_dict(self):
        return {
            "id": self.id,
            "username": self.username,
            # Password should not be sent to frontend
            "email": self.email,
            "phone": self.phone,
            "role_id": self.role_id,
            "notes": self.notes,
            "create_time": self.create_time,
            "update_time": self.update_time,
            "is_deleted": self.is_deleted,
            "department": self.department
            # patientCount will be added by the API endpoint if needed
        }
