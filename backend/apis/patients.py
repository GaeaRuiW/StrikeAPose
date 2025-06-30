from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Body, Query
from models import Doctors, Patients, SessionDep, VideoPath, Action
from pydantic import BaseModel
from sqlalchemy import asc, desc, func, select, update

router = APIRouter(tags=["patients"], prefix="/patients")


class CreatePatientModel(BaseModel):
    username: str
    age: int
    gender: str
    case_id: str
    patient_id: int
    doctor_id: int


class UpdatePatientModel(BaseModel):
    age: int
    gender: str
    case_id: str
    doctor_id: int
    patient_id: int  # 当前患者ID，用于查找患者
    new_patient_id: int  # 新的患者ID（病历号）
    username: str


class PatientLoginModel(BaseModel):
    case_id: str
    verify_case_id: str


@router.post("/patient_login")
async def patient_login(patient: PatientLoginModel = Body(..., embed=True), session: SessionDep = SessionDep):
    result = await session.execute(select(Patients).where(
        Patients.case_id == patient.case_id, Patients.is_deleted == False))
    if patient_ := result.scalar_one_or_none():
        return (
            {"message": "Case ID verification failed"}
            if patient_.case_id != patient.verify_case_id
            else {"message": "Login successful", "patient": patient_.to_dict()}
        )
    else:
        return {"message": "Patient not found"}


@router.put("/insert_patient")
async def insert_patient(patient: CreatePatientModel = Body(..., embed=True), session: SessionDep = SessionDep):
    result = await session.execute(select(Doctors).where(
        Doctors.id == patient.doctor_id, Doctors.is_deleted == False))
    doctor = result.scalar_one_or_none()
    if not doctor:
        return {"message": "Doctor not found"}

    # 检查case_id是否已存在
    result = await session.execute(select(Patients).where(
        Patients.case_id == patient.case_id, Patients.is_deleted == False))
    if patient_ := result.scalar_one_or_none():
        return {"message": "Case ID already exists"}

    # 检查patient_id是否已存在
    result = await session.execute(select(Patients).where(
        Patients.id == patient.patient_id, Patients.is_deleted == False))
    if patient_ := result.scalar_one_or_none():
        return {"message": "Patient ID already exists"}

    patient_obj = Patients(username=patient.username, age=patient.age, gender=patient.gender, case_id=patient.case_id, doctor_id=patient.doctor_id,
                           create_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), update_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), is_deleted=False)
    # 手动设置ID
    patient_obj.id = patient.patient_id
    session.add(patient_obj)
    await session.commit()
    # Refresh to get the ID and avoid detached instance
    await session.refresh(patient_obj)
    return patient_obj.to_dict()


@router.get("/get_all_patient_by_doctor_id/{doctor_id}")
async def get_all_patient_by_doctor_id(doctor_id: int, session: SessionDep = SessionDep):
    result = await session.execute(select(Doctors).where(Doctors.id == doctor_id))
    doctor = result.scalar_one_or_none()
    if not doctor:
        return {"message": "Doctor not found"}
    if doctor.role_id != 1:
        result = await session.execute(select(Patients).where(
            Patients.doctor_id == doctor_id, Patients.is_deleted == False))
    else:
        result = await session.execute(select(Patients).where(
            Patients.is_deleted == False))
    patients = result.scalars().all()
    return [patient.to_dict() for patient in patients]


@router.get("/get_last_upload_video_patient/{doctor_id}")
async def get_last_upload_video_patient(doctor_id: int, session: SessionDep = SessionDep):
    result = await session.execute(select(Doctors).where(Doctors.id == doctor_id))
    doctor = result.scalar_one_or_none()
    if not doctor:
        return {"message": "Doctor not found"}

    # Build the query using SQLAlchemy 2.0 syntax
    query = select(VideoPath).join(
        Patients, VideoPath.patient_id == Patients.id)

    if doctor.role_id != 1:  # Not an admin
        query = query.where(Patients.doctor_id == doctor_id)

    query = query.where(
        Patients.is_deleted == False,
        VideoPath.is_deleted == False,
        # REMOVED: VideoPath.original_video == True,
    )

    result = await session.execute(query)
    all_videos = result.scalars().all()

    if not all_videos:
        return {"message": "No video found", "patient_id": None, "video_id": None}

    # Step 2: Sort the results reliably in Python.
    latest_video = sorted(
        all_videos, key=lambda v: v.create_time, reverse=True)[0]

    return {
        "patient_id": latest_video.patient_id,
        "video_id": latest_video.id,
        "message": "success",
    }


@router.get("/check_patient_id_exists/{patient_id}")
async def check_patient_id_exists(patient_id: int, session: SessionDep = SessionDep):
    """检查病历号是否已存在"""
    result = await session.execute(select(Patients).where(
        Patients.id == patient_id, Patients.is_deleted == False))
    patient = result.scalar_one_or_none()
    return {"exists": patient is not None}


@router.put("/update_patient_by_id")
async def update_patient_by_id(patient: UpdatePatientModel = Body(..., embed=True), session: SessionDep = SessionDep):
    result = await session.execute(select(Doctors).where(
        Doctors.id == patient.doctor_id, Doctors.is_deleted == False))
    doctor = result.scalar_one_or_none()
    if not doctor:
        return {"message": "Doctor not found"}

    # 查找要更新的患者
    if doctor.role_id != 1:
        result = await session.execute(select(Patients).where(
            Patients.id == patient.patient_id,
            Patients.doctor_id == patient.doctor_id,
            Patients.is_deleted == False))
    else:
        result = await session.execute(select(Patients).where(
            Patients.id == patient.patient_id, Patients.is_deleted == False))
    patient_ = result.scalar_one_or_none()
    if not patient_:
        return {"message": "Patient not found or this doctor does not have permission to update this patient"}

    # 如果要修改病历号，需要特殊处理
    if patient.new_patient_id != patient.patient_id:
        # 检查新病历号是否已存在
        result = await session.execute(select(Patients).where(
            Patients.id == patient.new_patient_id, Patients.is_deleted == False))
        if result.scalar_one_or_none():
            return {"message": "New patient ID already exists"}

        # 创建新的患者记录
        new_patient = Patients(
            username=patient.username,
            age=patient.age,
            gender=patient.gender,
            case_id=patient.case_id,
            doctor_id=patient.doctor_id,
            create_time=patient_.create_time,  # 保持原创建时间
            update_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            is_deleted=False
        )
        new_patient.id = patient.new_patient_id
        session.add(new_patient)

        # 更新相关表的外键引用

        # 更新VideoPath表
        await session.execute(
            update(VideoPath).where(
                VideoPath.patient_id == patient.patient_id
            ).values(patient_id=patient.new_patient_id)
        )

        # 更新Action表
        await session.execute(
            update(Action).where(
                Action.patient_id == patient.patient_id
            ).values(patient_id=patient.new_patient_id)
        )

        # 删除旧的患者记录
        patient_.is_deleted = True

        await session.commit()
        await session.refresh(new_patient)
        return new_patient.to_dict()
    else:
        # 如果不修改病历号，正常更新
        patient_.age = patient.age
        patient_.gender = patient.gender
        patient_.case_id = patient.case_id
        patient_.doctor_id = patient.doctor_id
        patient_.username = patient.username
        patient_.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        await session.commit()
        await session.refresh(patient_)
        return patient_.to_dict()


@router.delete("/delete_patient_by_id/{patient_id}/{doctor_id}")
async def delete_patient_by_id(patient_id: int, doctor_id: int, session: SessionDep = SessionDep):
    result = await session.execute(select(Doctors).where(
        Doctors.id == doctor_id, Doctors.is_deleted == False))
    doctor = result.scalar_one_or_none()
    if not doctor:
        return {"message": "Doctor not found"}
    if doctor.role_id != 1:
        result = await session.execute(select(Patients).where(
            Patients.id == patient_id, Patients.doctor_id == doctor_id, Patients.is_deleted == False))
        patient_ = result.scalar_one_or_none()
        if not patient_:
            return {"message": "Patient not found or this doctor does not have permission to delete this patient"}
    else:
        result = await session.execute(select(Patients).where(
            Patients.id == patient_id, Patients.is_deleted == False))
        patient_ = result.scalar_one_or_none()
        if not patient_:
            return {"message": "Patient not found"}
    patient_.is_deleted = True
    patient_.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    await session.commit()
    return {"message": "Patient deleted successfully"}


@router.get("/get_patients_with_page")
async def get_patient_with_page(
    page: int = Query(default=1, ge=1, description="页码"),
    page_size: int = Query(default=10, ge=1, le=100, description="每页数量"),
    sort_by: Optional[str] = Query(
        default="id", description="排序字段(id, username, email, create_time)"),
    sort_order: Optional[str] = Query(
        default="asc", description="排序方向(asc, desc)"),
    doctor_id: int = Query(None, ge=1),
    session: SessionDep = SessionDep
):

    # Build base query
    query = select(Patients).where(Patients.is_deleted == False)
    if doctor_id:
        query = query.where(Patients.doctor_id == doctor_id)

    # Apply sorting
    if sort_by:
        if sort_by == "id":
            query = query.order_by(Patients.id)
        elif sort_by == "username":
            query = query.order_by(Patients.username)
        elif sort_by == "create_time":
            query = query.order_by(Patients.create_time)

    # Get total count
    count_query = select(func.count(Patients.id)).where(
        Patients.is_deleted == False)
    if doctor_id:
        count_query = count_query.where(Patients.doctor_id == doctor_id)
    result = await session.execute(count_query)
    total = result.scalar()

    # Apply final sorting and pagination
    if sort_order == "desc":
        query = query.order_by(desc(Patients.id))
    else:
        query = query.order_by(asc(Patients.id))

    query = query.offset((page - 1) * page_size).limit(page_size)
    result = await session.execute(query)
    patients = result.scalars().all()

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "patients": [patient.to_dict() for patient in patients]
    }
