from datetime import datetime

from common.utils import hash_password, check_password
from fastapi import APIRouter, Body
from sqlalchemy import select
from models import SessionDep, Doctors, Patients
from pydantic import BaseModel

router = APIRouter(tags=["doctors"], prefix="/doctors")


class CreateDoctorModel(BaseModel):
    username: str
    password: str
    email: str
    phone: str
    department: str = "康复科"


class UpdateDoctorModel(BaseModel):
    doctor_id: int
    email: str
    phone: str
    password: str
    department: str = "康复科"


class LoginModel(BaseModel):
    username: str
    password: str


class DeleteDoctorModel(BaseModel):
    password: str


@router.post("/register")
async def register_doctor(doctor: CreateDoctorModel = Body(..., embed=True), session: SessionDep = SessionDep):
    result = await session.execute(select(Doctors).where(
        Doctors.email == doctor.email, Doctors.is_deleted == False))
    if existing_doctor := result.scalar_one_or_none():
        return {"message": "Email already exists"}
    doctor_obj = Doctors(username=doctor.username, password=hash_password(
        doctor.password), email=doctor.email, phone=doctor.phone, create_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), update_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), is_deleted=False, department=doctor.department)
    session.add(doctor_obj)
    await session.commit()
    # Refresh to get the ID and avoid detached instance
    await session.refresh(doctor_obj)
    return doctor_obj.to_dict()


@router.get("/get_all_doctors")
async def get_all_doctors(session: SessionDep = SessionDep):
    result = await session.execute(select(Doctors).where(Doctors.is_deleted == False))
    doctors = result.scalars().all()
    return [doctor.to_dict() for doctor in doctors]


@router.get("/get_doctor_by_id/{doctor_id}")
async def get_doctor_by_id(doctor_id: int, session: SessionDep = SessionDep):
    result = await session.execute(select(Doctors).where(
        Doctors.id == doctor_id, Doctors.is_deleted == False))
    doctor = result.scalar_one_or_none()
    return doctor.to_dict() if doctor else {"message": "Doctor not found"}


@router.put("/update_doctor_by_id")
async def update_doctor_by_id(doctor: UpdateDoctorModel = Body(..., embed=True), session: SessionDep = SessionDep):
    result = await session.execute(select(Doctors).where(
        Doctors.id == doctor.doctor_id, Doctors.is_deleted == False))
    doctor_db = result.scalar_one_or_none()
    if not doctor_db:
        return {"message": "Doctor not found"}
    doctor_db.email = doctor.email
    doctor_db.phone = doctor.phone
    doctor_db.password = hash_password(doctor.password)
    doctor_db.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    doctor_db.department = doctor.department
    await session.commit()
    await session.refresh(doctor_db)  # Refresh to avoid detached instance
    return doctor_db.to_dict()


@router.delete("/delete_doctor_by_id/{doctor_id}")
async def delete_doctor_by_id(doctor_id: int, doctor_model: DeleteDoctorModel, session: SessionDep = SessionDep):
    result = await session.execute(select(Doctors).where(
        Doctors.id == doctor_id, Doctors.is_deleted == False))
    doctor = result.scalar_one_or_none()
    if not doctor:
        return {"message": "Doctor not found"}
    if not check_password(doctor_model.password, doctor.password):
        return {"message": "Invalid password"}
    doctor.is_deleted = True
    doctor.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    await session.commit()
    return {"message": "Doctor deleted successfully"}


@router.get("/get_doctor_by_name/{name}")
async def get_doctor_by_name(name: str, session: SessionDep = SessionDep):
    result = await session.execute(select(Doctors).where(
        Doctors.username == name, Doctors.is_deleted == False))
    doctor = result.scalar_one_or_none()
    return doctor.to_dict() if doctor else {"message": "Doctor not found"}


@router.post("/login")
async def login_doctor(doctor_model: LoginModel = Body(..., embed=True), session: SessionDep = SessionDep):
    from sqlalchemy import select
    result = await session.execute(select(Doctors).where(
        Doctors.username == doctor_model.username, Doctors.is_deleted == False))
    if doctor := result.scalar_one_or_none():
        return (
            {"message": "Login successful", "doctor": doctor.to_dict()} if check_password(
                doctor_model.password, doctor.password) else {"message": "Invalid password"}
        )
    else:
        return {"message": "Doctor not found"}
