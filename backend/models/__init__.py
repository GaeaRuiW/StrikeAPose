from config import postgres_uri
from fastapi import Depends
from models.action import Action
from models.roles import Roles
from models.stage import Stage
from models.steps_info import StepsInfo
from models.patients import Patients
from models.doctors import Doctors
from models.video_path import VideoPath
from models.objects import Objects
from sqlmodel import Session, SQLModel, create_engine
from typing_extensions import Annotated

engine = create_engine(postgres_uri)
print("engine", engine)


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]
