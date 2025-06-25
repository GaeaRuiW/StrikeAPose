from config import postgres_uri
from fastapi import Depends
from models.action import Action
from models.roles import Roles
from models.stage import Stage
from models.steps_info import StepsInfo
from models.patients import Patients
from models.doctors import Doctors
from models.video_path import VideoPath
from sqlmodel import SQLModel
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from typing_extensions import Annotated

# Convert PostgreSQL URI to async format
async_postgres_uri = postgres_uri.replace("postgresql://", "postgresql+asyncpg://")
engine = create_async_engine(async_postgres_uri)
print("async engine", engine)


async def create_db_and_tables():
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)


async def get_session():
    async with AsyncSession(engine) as session:
        yield session


SessionDep = Annotated[AsyncSession, Depends(get_session)]
