from datetime import datetime

from common.utils import hash_password, check_password
from fastapi import APIRouter, Body
from models import SessionDep, Users
from pydantic import BaseModel

router = APIRouter(tags=["users"], prefix="/users")


class CreateUserModel(BaseModel):
    username: str
    password: str
    email: str
    phone: str


class UpdateUserModel(BaseModel):
    email: str
    phone: str
    password: str


class LoginModel(BaseModel):
    username: str
    password: str

class DeleteUserModel(BaseModel):
    password: str

@router.post("/register")
def register_user(user: CreateUserModel = Body(..., embed=True), session: SessionDep = SessionDep):
    user = Users(username=user.username, password=hash_password(
        user.password), email=user.email, phone=user.phone, role_id=1, create_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), update_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), is_deleted=False)
    session.add(user)
    session.commit()
    return user.to_dict()


@router.get("/get_all_users")
def get_all_users(session: SessionDep = SessionDep):
    users = session.query(Users).filter(Users.is_deleted == False).all()
    return [user.to_dict() for user in users]


@router.get("/get_user_by_id/{user_id}")
def get_user_by_id(user_id: int, session: SessionDep = SessionDep):
    user = session.query(Users).filter(
        Users.id == user_id and Users.is_deleted == False).first()
    return user.to_dict() if user else {"message": "User not found"}


@router.put("/update_user_by_id/{user_id}")
def update_user_by_id(user_id: int, user: UpdateUserModel = Body(..., embed=True), session: SessionDep = SessionDep):
    user_db = session.query(Users).filter(
        Users.id == user_id and Users.is_deleted == False).first()
    if not user:
        return {"message": "User not found"}
    user_db.email = user.email
    user_db.phone = user.phone
    user_db.password = hash_password(user.password)
    user_db.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    session.commit()
    return user_db.to_dict()


@router.delete("/delete_user_by_id/{user_id}")
def delete_user_by_id(user_id: int, user_model: DeleteUserModel, session: SessionDep = SessionDep):
    user = session.query(Users).filter(
        Users.id == user_id and Users.is_deleted == False).first()
    if not user:
        return {"message": "User not found"}
    if not check_password(user_model.password, user.password):
        return {"message": "Invalid password"}
    user.is_deleted = True
    user.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    session.commit()
    return {"message": "User deleted successfully"}


@router.get("/get_user_by_username/{username}")
def get_user_by_username(username: str, session: SessionDep = SessionDep):
    user = session.query(Users).filter(
        Users.username == username and Users.is_deleted == False).first()
    return user.to_dict() if user else {"message": "User not found"}


@router.post("/login")
def login_user(user_model: LoginModel = Body(..., embed=True), session: SessionDep = SessionDep):
    if (
        user := session.query(Users)
        .filter(Users.username == user_model.username and Users.is_deleted == False)
        .first()
    ):
        return (
            {"message": "Login successful", "user": user.to_dict()} if check_password(user_model.password, user.password) else {"message": "Invalid password"}
        )
    else:
        return {"message": "User not found"}
