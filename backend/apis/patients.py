from datetime import datetime
from typing import Optional
from sqlalchemy import desc, asc

from fastapi import APIRouter, Body, Query
from models import Doctors, Patients, SessionDep
from pydantic import BaseModel

router = APIRouter(tags=["patients"], prefix="/patients")


class CreatePatientModel(BaseModel):
    username: str
    age: int
    gender: str
    case_id: str
    doctor_id: int


class UpdatePatientModel(BaseModel):
    age: int
    gender: str
    case_id: str
    doctor_id: int
    patient_id: int
    username: str

class PatientLoginModel(BaseModel):
    case_id: str
    verify_case_id: str


@router.post("/patient_login")
def patient_login(patient: PatientLoginModel = Body(..., embed=True), session: SessionDep = SessionDep):
    patient_ = session.query(Patients).filter(
        Patients.case_id == patient.case_id, Patients.is_deleted == False).first()
    if not patient_:
        return {"message": "Patient not found"}
    if patient_.case_id != patient.verify_case_id:
        return {"message": "Case ID verification failed"}
    return {"message": "Login successful", "patient": patient_.to_dict()}


@router.put("/insert_patient")
def insert_patient(patient: CreatePatientModel = Body(..., embed=True), session: SessionDep = SessionDep):
    patient = Patients(username=patient.username, age=patient.age, gender=patient.gender, case_id=patient.case_id, doctor_id=patient.doctor_id,
                       create_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), update_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), is_deleted=False)
    session.add(patient)
    session.commit()
    return patient.to_dict()


@router.get("/get_all_patient_by_doctor_id/{doctor_id}")
def get_all_patient_by_doctor_id(doctor_id: int, session: SessionDep = SessionDep):
    doctor = session.query(Doctors).filter(Doctors.id == doctor_id).first()
    if not doctor:
        return {"message": "Doctor not found"}
    if doctor.role_id != 1:
        patients = session.query(Patients).filter(
            Patients.doctor_id == doctor_id, Patients.is_deleted == False).all()
    else:
        patients = session.query(Patients).filter(
            Patients.is_deleted == False).all()
    return [patient.to_dict() for patient in patients]


@router.put("/update_patient_by_id")
def update_patient_by_id(patient: UpdatePatientModel = Body(..., embed=True), session: SessionDep = SessionDep):
    doctor = session.query(Doctors).filter(
        Doctors.id == patient.doctor_id, Doctors.is_deleted == False).first()
    if not doctor:
        return {"message": "Doctor not found"}
    if doctor.role_id != 1:
        patient_ = session.query(Patients).filter(Patients.id == patient.patient_id,
                                                 Patients.doctor_id == patient.doctor_id, Patients.is_deleted == False).first()
        if not patient_:
            return {"message": "Patient not found or this doctor does not have permission to update this patient"}
    else:
        patient_ = session.query(Patients).filter(Patients.id == patient.patient_id, Patients.is_deleted == False).first()
        if not patient_:
            return {"message": "Patient not found or this doctor does not have permission to update this patient"}
    patient_.age = patient.age
    patient_.gender = patient.gender
    patient_.case_id = patient.case_id
    patient_.doctor_id = patient.doctor_id
    patient_.username = patient.username
    patient_.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    session.commit()
    return patient_.to_dict()


@router.delete("/delete_patient_by_id/{patient_id}/{doctor_id}")
def delete_patient_by_id(patient_id: int, doctor_id: int, session: SessionDep = SessionDep):
    doctor = session.query(Doctors).filter(
        Doctors.id == doctor_id, Doctors.is_deleted == False).first()
    if not doctor:
        return {"message": "Doctor not found"}
    if doctor.role_id != 1:
        patient_ = session.query(Patients).filter(
            Patients.id == patient_id, Patients.doctor_id == doctor_id, Patients.is_deleted == False).first()
        if not patient_:
            return {"message": "Patient not found or this doctor does not have permission to delete this patient"}

    else:
        patient_ = session.query(Patients).filter(
            Patients.id == patient_id, Patients.is_deleted == False).first()
        if not patient_:
            return {"message": "Patient not found"}
    patient_.is_deleted = True
    patient_.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    session.commit()
    return {"message": "Patient deleted successfully"}


@router.get("/get_patients_with_page")
def get_patient_with_page(
    page: int = Query(default=1, ge=1, description="页码"),
    page_size: int = Query(default=10, ge=1, le=100, description="每页数量"),
    sort_by: Optional[str] = Query(
        default="id", description="排序字段(id, username, email, create_time)"),
    sort_order: Optional[str] = Query(
        default="asc", description="排序方向(asc, desc)"),
    doctor_id: int = Query(None, ge=1),
    session: SessionDep = SessionDep
):
    patients = session.query(Patients).filter(Patients.is_deleted == False)
    if doctor_id:
        patients = patients.filter(Patients.doctor_id == doctor_id)
    if sort_by:
        if sort_by == "id":
            patients = patients.order_by(Patients.id)
        elif sort_by == "username":
            patients = patients.order_by(Patients.username)
        elif sort_by == "email":
            patients = patients.order_by(Patients.email)
        elif sort_by == "create_time":
            patients = patients.order_by(Patients.create_time)
    total = patients.count()
    if sort_order == "desc":
        patients = patients.order_by(desc(Patients.id))
    else:
        patients = patients.order_by(asc(Patients.id))
    patients = patients.offset((page - 1) * page_size).limit(page_size).all()
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "patients": [patient.to_dict() for patient in patients]
    }
