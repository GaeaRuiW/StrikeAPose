from apis.actions import router as action_router
from apis.dashboard import router as dashboard_router
from apis.users import router as user_router
from apis.videos import router as video_router
from config import listen_port
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from models import create_db_and_tables

app = FastAPI()

app.include_router(action_router, prefix="/api/v1", tags=["actions"])
app.include_router(dashboard_router, prefix="/api/v1", tags=["dashboard"])
app.include_router(user_router, prefix="/api/v1", tags=["users"])
app.include_router(video_router, prefix="/api/v1", tags=["videos"])

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
    create_db_and_tables()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=listen_port, reload=True)
