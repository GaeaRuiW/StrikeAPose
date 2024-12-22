from config import postgres_uri
from fastapi import Depends
from models.action import Action
from models.roles import Roles
from models.step_hip_degree import StepHipDegree
from models.step_length import StepLength
from models.step_speed import StepSpeed
from models.step_stride import StepStride
from models.users import Users
from models.video_path import VideoPath
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
