from datetime import datetime

from sqlmodel import Field, SQLModel


class Objects(SQLModel, table=True):
    id: int = Field(primary_key=True)
    name: str
    action_id: int
    create_time: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    update_time: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    is_deleted: bool

    def __init__(self, name: str, action_id: int, create_time: str, update_time: str, is_deleted: bool):
        self.name = name
        self.action_id = action_id
        self.create_time = create_time
        self.update_time = update_time
        self.is_deleted = is_deleted

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "action_id": self.action_id,
            "create_time": self.create_time,
            "update_time": self.update_time,
            "is_deleted": self.is_deleted
        }
