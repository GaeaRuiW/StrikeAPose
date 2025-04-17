from datetime import datetime

from common.utils import hash_password, check_password
from fastapi import APIRouter, Body
from models import SessionDep, Doctors, Patients
from pydantic import BaseModel

router = APIRouter(tags=["doctors"], prefix="/doctors")


class CreateDoctorModel(BaseModel):
    username: str
    password: str
    email: str
    phone: str


class UpdateDoctorModel(BaseModel):
    doctor_id: int
    email: str
    phone: str
    password: str


class LoginModel(BaseModel):
    username: str
    password: str

class DeleteDoctorModel(BaseModel):
    password: str

@router.post("/register")
def register_doctor(doctor: CreateDoctorModel = Body(..., embed=True), session: SessionDep = SessionDep):
    doctor = Doctors(username=doctor.username, password=hash_password(
        doctor.password), email=doctor.email, phone=doctor.phone, create_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), update_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), is_deleted=False)
    session.add(doctor)
    session.commit()
    return doctor.to_dict()


@router.get("/get_all_doctors")
def get_all_doctors(session: SessionDep = SessionDep):
    doctors = session.query(Doctors).filter(Doctors.is_deleted == False).all()
    return [doctor.to_dict() for doctor in doctors]


@router.get("/get_doctor_by_id/{doctor_id}")
def get_doctor_by_id(doctor_id: int, session: SessionDep = SessionDep):
    doctor = session.query(Doctors).filter(
        Doctors.id == doctor_id and Doctors.is_deleted == False).first()
    return doctor.to_dict() if doctor else {"message": "Doctor not found"}


@router.put("/update_doctor_by_id")
def update_doctor_by_id(doctor: UpdateDoctorModel = Body(..., embed=True), session: SessionDep = SessionDep):
    doctor_db = session.query(Doctors).filter(
        Doctors.id == doctor.doctor_id and Doctors.is_deleted == False).first()
    if not doctor_db:
        return {"message": "Doctor not found"}
    doctor_db.email = doctor.email
    doctor_db.phone = doctor.phone
    doctor_db.password = hash_password(doctor.password)
    doctor_db.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    session.commit()
    return doctor_db.to_dict()


@router.delete("/delete_doctor_by_id/{doctor_id}")
def delete_doctor_by_id(doctor_id: int, doctor_model: DeleteDoctorModel, session: SessionDep = SessionDep):
    doctor = session.query(Doctors).filter(
        Doctors.id == doctor_id and Doctors.is_deleted == False).first()
    if not doctor:
        return {"message": "Doctor not found"}
    if not check_password(doctor_model.password, doctor.password):
        return {"message": "Invalid password"}
    doctor.is_deleted = True
    doctor.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    session.commit()
    return {"message": "Doctor deleted successfully"}


@router.get("/get_doctor_by_name/{name}")
def get_doctor_by_name(name: str, session: SessionDep = SessionDep):
    doctor = session.query(Doctors).filter(
        Doctors.username == name and Doctors.is_deleted == False).first()
    return doctor.to_dict() if doctor else {"message": "Doctor not found"}


@router.post("/login")
def login_doctor(doctor_model: LoginModel = Body(..., embed=True), session: SessionDep = SessionDep):
    if (
        doctor := session.query(Doctors)
        .filter(Doctors.username == doctor_model.username and Doctors.is_deleted == False)
        .first()
    ):
        return (
            {"message": "Login successful", "doctor": doctor.to_dict()} if check_password(doctor_model.password, doctor.password) else {"message": "Invalid password"}
        )
    else:
        return {"message": "Doctor not found"}
