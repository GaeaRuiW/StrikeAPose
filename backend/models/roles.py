from sqlmodel import Field, SQLModel


class Roles(SQLModel, table=True):
    id: int = Field(primary_key=True)
    role_name: str
    role_desc: str

    def __init__(self, role_name: str, role_desc: str):
        self.role_name = role_name
        self.role_desc = role_desc

    def to_dict(self):
        return {
            "id": self.id,
            "role_name": self.role_name,
            "role_desc": self.role_desc
        }
