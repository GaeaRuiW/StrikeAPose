import os

from apis.actions import router as action_router
from apis.dashboard import router as dashboard_router
from apis.doctors import router as doctor_router
from apis.management import router as management_router
from apis.patients import router as patient_router
from apis.table import router as table_router
from apis.videos import router as video_router
from config import listen_port, postgres_uri, video_dir
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from models import Roles, create_db_and_tables

app = FastAPI(doc_url=None)

app.include_router(action_router, prefix="/api/v1", tags=["actions"])
app.include_router(dashboard_router, prefix="/api/v1", tags=["dashboard"])
app.include_router(doctor_router, prefix="/api/v1", tags=["doctors"])
app.include_router(patient_router, prefix="/api/v1", tags=["patients"])
app.include_router(video_router, prefix="/api/v1", tags=["videos"])
app.include_router(table_router, prefix="/api/v1", tags=["tables"])
app.include_router(management_router, prefix="/api/v1", tags=["management"])

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def jwt_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": str(exc.detail)},
    )


@app.on_event("startup")
async def startup_event():
    if not os.path.exists(f"{video_dir}/original"):
        os.makedirs(f"{video_dir}/original")
    if not os.path.exists(f"{video_dir}/flipped"):
        os.makedirs(f"{video_dir}/flipped")
    if not os.path.exists(f"{video_dir}/inference"):
        os.makedirs(f"{video_dir}/inference")
    await create_db_and_tables()

    from models import engine
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy import select
    async with AsyncSession(engine) as session:
        result = await session.execute(select(Roles).where(Roles.id == 1))
        admin_role = result.scalar_one_or_none()
        if not admin_role:
            admin_role = Roles(
                role_name="admin",
                role_desc="Administrator role with full access"
            )
            session.add(admin_role)
            await session.commit()
            print("Admin role created successfully")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=listen_port, reload=True)
