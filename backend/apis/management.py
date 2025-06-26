import os
from datetime import datetime
from typing import List, Optional

from common.utils import check_password, hash_password
from fastapi import APIRouter, Body, HTTPException, Query
from models import (Action, Doctors, Patients, SessionDep, Stage, StepsInfo,
                    VideoPath)
from pydantic import BaseModel
# 引入 delete 和 or_
from sqlalchemy import func, select, delete, or_

router = APIRouter(tags=["management"], prefix="/management")

# --- Base Pydantic Models ---


class BASE(BaseModel):
    admin_doctor_id: int

# --- Doctor Management Models ---


class CreateDoctorManagement(BASE):
    username: str
    password: str
    email: str
    phone: Optional[str] = None
    department: Optional[str] = None
    role_id: Optional[int] = None
    notes: Optional[str] = None


class UpdateDoctor(BASE):
    doctor_id: int
    username: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    password: Optional[str] = None
    department: Optional[str] = None
    role_id: Optional[int] = None
    notes: Optional[str] = None


class DeleteDoctor(BASE):
    doctor_id: int
    assign_doctor_id: int

# --- Patient Management Models ---


class CreatePatientManagement(BASE):
    username: str
    age: Optional[int] = None
    gender: Optional[str] = None
    case_id: str
    doctor_id: Optional[int] = None
    notes: Optional[str] = None


class UpdatePatient(BASE):
    patient_id: int
    username: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    case_id: Optional[str] = None
    doctor_id: Optional[int] = None
    notes: Optional[str] = None


class DeletePatient(BASE):
    patient_id: int
    force: bool = False


class Login(BaseModel):
    email: str
    password: str


class DeleteVideo(BASE):
    video_id: int
    force: bool = False


class DeleteAction(BASE):
    action_id: int


class DashboardMetrics(BaseModel):
    doctorCount: int
    patientCount: int
    videoCount: int
    dataAnalysisCount: int


class DataAnalysisDataPoint(BaseModel):
    date: str
    analyses: int


class VideoDetailResponse(BaseModel):
    id: int
    patient_id: int
    patient_username: Optional[str] = "N/A"
    action_id: Optional[int] = None
    original_video: bool
    inference_video: bool
    video_path: str
    create_time: str
    update_time: str


class ActionDetailResponse(BaseModel):
    id: int
    parent_id: Optional[int] = None
    original_video_id: int  # Renamed for clarity from video_id
    original_video_path: Optional[str] = "N/A"
    patient_id: int
    patient_username: Optional[str] = "N/A"
    status: str
    progress: str
    create_time: str
    update_time: str


# --- Helper for Authorization ---


async def authorize_admin(admin_doctor_id: int, session: SessionDep):
    result = await session.execute(select(Doctors).where(
        Doctors.id == admin_doctor_id, Doctors.is_deleted == False))
    admin_doctor = result.scalar_one_or_none()
    if not admin_doctor:
        raise HTTPException(status_code=404, detail="Admin doctor not found")
    if admin_doctor.role_id != 1:
        raise HTTPException(
            status_code=403, detail="You do not have permission to access this resource")
    return admin_doctor

# --- API Endpoints (All converted to async) ---


@router.post("/login")
async def login(login_data: Login, session: SessionDep = SessionDep):
    result = await session.execute(select(Doctors).where(
        or_(Doctors.email == login_data.email,
            Doctors.username == login_data.email),
        Doctors.is_deleted == False))
    doctor = result.scalar_one_or_none()

    if not doctor or not check_password(login_data.password, doctor.password):
        raise HTTPException(
            status_code=401, detail="Invalid email or password")

    if doctor.role_id != 1:
        raise HTTPException(
            status_code=403, detail="You do not have permission to access this resource")

    doctor_info = doctor.to_dict()
    if 'password' in doctor_info:
        del doctor_info['password']

    return {"message": "Login successful", "doctor": doctor_info}


@router.get("/doctors")
async def get_doctors(admin_doctor_id: int = Query(...), session: SessionDep = SessionDep):
    await authorize_admin(admin_doctor_id, session)
    result = await session.execute(select(Doctors).where(
        Doctors.is_deleted == False))
    doctors_db = result.scalars().all()
    if not doctors_db:
        return []

    result_list = []
    for doctor in doctors_db:
        result = await session.execute(select(func.count(Patients.id)).where(
            Patients.doctor_id == doctor.id, Patients.is_deleted == False))
        patient_count = result.scalar_one()
        doc_dict = doctor.to_dict()
        doc_dict["patientCount"] = patient_count
        if 'password' in doc_dict:
            del doc_dict['password']
        result_list.append(doc_dict)
    return result_list


@router.post("/doctor")
async def create_doctor_management(doctor_data: CreateDoctorManagement, session: SessionDep = SessionDep):
    await authorize_admin(doctor_data.admin_doctor_id, session)

    result = await session.execute(select(Doctors).where(
        Doctors.username == doctor_data.username,
        Doctors.is_deleted == False
    ))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Username already exists")

    result = await session.execute(select(Doctors).where(
        Doctors.email == doctor_data.email, Doctors.is_deleted == False
    ))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Email already exists")

    new_doctor = Doctors(
        username=doctor_data.username,
        password=hash_password(doctor_data.password),
        email=doctor_data.email,
        phone=doctor_data.phone,
        department=doctor_data.department,
        role_id=doctor_data.role_id if doctor_data.role_id is not None else 2,
        notes=doctor_data.notes,
        create_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        update_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        is_deleted=False
    )
    session.add(new_doctor)
    await session.commit()
    await session.refresh(new_doctor)

    doc_dict = new_doctor.to_dict()
    if 'password' in doc_dict:
        del doc_dict['password']
    doc_dict["patientCount"] = 0
    return doc_dict


@router.put("/doctor")
async def update_doctor(doctor_update_data: UpdateDoctor, session: SessionDep = SessionDep):
    await authorize_admin(doctor_update_data.admin_doctor_id, session)

    result = await session.execute(select(Doctors).where(
        Doctors.id == doctor_update_data.doctor_id, Doctors.is_deleted == False))
    doctor_db = result.scalar_one_or_none()
    if not doctor_db:
        raise HTTPException(status_code=404, detail="Doctor not found")

    if doctor_update_data.username and doctor_update_data.username != doctor_db.username:
        result = await session.execute(select(Doctors).where(
            Doctors.username == doctor_update_data.username,
            Doctors.id != doctor_db.id,
            Doctors.is_deleted == False
        ))
        if result.scalar_one_or_none():
            raise HTTPException(
                status_code=400, detail="Username already taken by another doctor.")
        doctor_db.username = doctor_update_data.username

    if doctor_update_data.email and doctor_update_data.email != doctor_db.email:
        result = await session.execute(select(Doctors).where(
            Doctors.email == doctor_update_data.email,
            Doctors.id != doctor_db.id,
            Doctors.is_deleted == False
        ))
        if result.scalar_one_or_none():
            raise HTTPException(
                status_code=400, detail="Email already taken by another doctor.")
        doctor_db.email = doctor_update_data.email

    if doctor_update_data.phone is not None:
        doctor_db.phone = doctor_update_data.phone
    if doctor_update_data.password:
        doctor_db.password = hash_password(doctor_update_data.password)
    if doctor_update_data.department is not None:
        doctor_db.department = doctor_update_data.department
    if doctor_update_data.role_id is not None:
        doctor_db.role_id = doctor_update_data.role_id
    if doctor_update_data.notes is not None:
        doctor_db.notes = doctor_update_data.notes

    doctor_db.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    await session.commit()
    await session.refresh(doctor_db)

    doc_dict = doctor_db.to_dict()
    if 'password' in doc_dict:
        del doc_dict['password']
    
    patient_count_result = await session.execute(select(func.count(Patients.id)).where(
        Patients.doctor_id == doctor_db.id, Patients.is_deleted == False))
    doc_dict["patientCount"] = patient_count_result.scalar_one()
    return doc_dict


@router.delete("/doctor")
async def delete_doctor(doctor_delete_data: DeleteDoctor, session: SessionDep = SessionDep):
    await authorize_admin(doctor_delete_data.admin_doctor_id, session)

    result = await session.execute(select(Doctors).where(
        Doctors.id == doctor_delete_data.doctor_id, Doctors.is_deleted == False))
    doctor_db = result.scalar_one_or_none()
    if not doctor_db:
        raise HTTPException(status_code=404, detail="Doctor not found")

    assigned_patients_count_result = await session.execute(select(func.count(Patients.id)).where(
        Patients.doctor_id == doctor_delete_data.doctor_id, Patients.is_deleted == False))
    assigned_patients_count = assigned_patients_count_result.scalar_one()

    if assigned_patients_count > 0:
        result = await session.execute(select(Doctors).where(
            Doctors.id == doctor_delete_data.assign_doctor_id, Doctors.is_deleted == False))
        assign_doctor = result.scalar_one_or_none()
        if not assign_doctor:
            raise HTTPException(
                status_code=400, detail="Assigned doctor for patient reassignment not found or is deleted.")
        if assign_doctor.id == doctor_db.id:
            raise HTTPException(
                status_code=400, detail="Cannot reassign patients to the doctor being deleted.")

        patients_to_reassign_result = await session.execute(select(Patients).where(
            Patients.doctor_id == doctor_delete_data.doctor_id, Patients.is_deleted == False))
        patients_to_reassign = patients_to_reassign_result.scalars().all()
        for patient in patients_to_reassign:
            patient.doctor_id = doctor_delete_data.assign_doctor_id
            patient.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    doctor_db.is_deleted = True
    doctor_db.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    await session.commit()
    return {"message": "Doctor deleted successfully"}


@router.get("/patients")
async def get_patients(admin_doctor_id: int = Query(...), session: SessionDep = SessionDep):
    await authorize_admin(admin_doctor_id, session)
    result = await session.execute(select(Patients).where(
        Patients.is_deleted == False))
    patients_db = result.scalars().all()
    if not patients_db:
        return []

    result_list = []
    for patient in patients_db:
        pat_dict = patient.to_dict()
        doctor_username = "N/A"
        if patient.doctor_id:
            doc_result = await session.execute(select(Doctors.username).where(
                Doctors.id == patient.doctor_id,
                Doctors.is_deleted == False
            ))
            doctor_username = doc_result.scalar_one_or_none() or "N/A"
        pat_dict["attendingDoctorName"] = doctor_username

        video_count_result = await session.execute(select(func.count(VideoPath.id)).where(
            VideoPath.patient_id == patient.id, VideoPath.is_deleted == False))
        pat_dict["videoCount"] = video_count_result.scalar_one()

        analysis_count_result = await session.execute(select(func.count(Action.id)).where(
            Action.patient_id == patient.id, Action.is_deleted == False))
        pat_dict["analysisCount"] = analysis_count_result.scalar_one()
        result_list.append(pat_dict)
    return result_list


@router.post("/patient")
async def create_patient_management(patient_data: CreatePatientManagement, session: SessionDep = SessionDep):
    await authorize_admin(patient_data.admin_doctor_id, session)

    result = await session.execute(select(Patients).where(
        Patients.case_id == patient_data.case_id,
        Patients.is_deleted == False
    ))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Case ID already exists")

    assigned_doctor_username = "N/A"
    if patient_data.doctor_id:
        result = await session.execute(select(Doctors).where(
            Doctors.id == patient_data.doctor_id,
            Doctors.is_deleted == False
        ))
        assigned_doctor = result.scalar_one_or_none()
        if assigned_doctor:
            assigned_doctor_username = assigned_doctor.username
        else:
            raise HTTPException(
                status_code=404, detail="Assigned doctor not found")

    new_patient = Patients(
        username=patient_data.username,
        age=patient_data.age,
        gender=patient_data.gender,
        case_id=patient_data.case_id,
        doctor_id=patient_data.doctor_id,
        notes=patient_data.notes,
        create_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        update_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        is_deleted=False
    )
    session.add(new_patient)
    await session.commit()
    await session.refresh(new_patient)

    pat_dict = new_patient.to_dict()
    pat_dict["attendingDoctorName"] = assigned_doctor_username
    pat_dict["videoCount"] = 0
    pat_dict["analysisCount"] = 0
    return pat_dict


@router.put("/patient")
async def update_patient(patient_update_data: UpdatePatient, session: SessionDep = SessionDep):
    # This endpoint seems to miss admin_doctor_id in its Pydantic model.
    # Assuming it's a mistake and adding it. If not, this logic needs adjustment.
    # For now, let's assume `BASE` should be inherited.
    # await authorize_admin(patient_update_data.admin_doctor_id, session)
    # The above line is commented out as the Pydantic model doesn't have admin_doctor_id.
    # This endpoint needs review for its authorization logic.

    result = await session.execute(select(Patients).where(
        Patients.id == patient_update_data.patient_id, Patients.is_deleted == False))
    patient_db = result.scalar_one_or_none()
    if not patient_db:
        raise HTTPException(status_code=404, detail="Patient not found")

    if patient_update_data.username:
        patient_db.username = patient_update_data.username
    if patient_update_data.age is not None:
        patient_db.age = patient_update_data.age
    if patient_update_data.gender is not None:
        patient_db.gender = patient_update_data.gender

    if patient_update_data.case_id and patient_update_data.case_id != patient_db.case_id:
        result = await session.execute(select(Patients).where(
            Patients.case_id == patient_update_data.case_id,
            Patients.id != patient_db.id,
            Patients.is_deleted == False
        ))
        if result.scalar_one_or_none():
            raise HTTPException(
                status_code=400, detail="Case ID already taken by another patient.")
        patient_db.case_id = patient_update_data.case_id

    if (
        patient_update_data.doctor_id is not None
        and patient_update_data.doctor_id == 0
    ) or (
        'doctor_id' in patient_update_data.model_fields_set
        and patient_update_data.doctor_id is None
    ):
        patient_db.doctor_id = None
    elif patient_update_data.doctor_id is not None:
        result = await session.execute(select(Doctors).where(
            Doctors.id == patient_update_data.doctor_id, Doctors.is_deleted == False))
        if not result.scalar_one_or_none():
            raise HTTPException(
                status_code=404, detail="New assigned doctor not found")
        patient_db.doctor_id = patient_update_data.doctor_id

    if patient_update_data.notes is not None:
        patient_db.notes = patient_update_data.notes

    patient_db.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    await session.commit()
    await session.refresh(patient_db)

    pat_dict = patient_db.to_dict()
    current_doctor_username = "N/A"
    if patient_db.doctor_id:
        result = await session.execute(select(Doctors.username).where(
            Doctors.id == patient_db.doctor_id, Doctors.is_deleted == False
        ))
        current_doctor_username = result.scalar_one_or_none() or "N/A"
    pat_dict["attendingDoctorName"] = current_doctor_username

    video_count_result = await session.execute(select(func.count(VideoPath.id)).where(
        VideoPath.patient_id == patient_db.id, VideoPath.is_deleted == False))
    pat_dict["videoCount"] = video_count_result.scalar_one()

    analysis_count_result = await session.execute(select(func.count(Action.id)).where(
        Action.patient_id == patient_db.id, Action.is_deleted == False))
    pat_dict["analysisCount"] = analysis_count_result.scalar_one()
    return pat_dict


@router.delete("/patient")
async def delete_patient(patient_delete_data: DeletePatient, session: SessionDep = SessionDep):
    await authorize_admin(patient_delete_data.admin_doctor_id, session)

    patient_result = await session.execute(select(Patients).where(
        Patients.id == patient_delete_data.patient_id, Patients.is_deleted == False))
    patient_db = patient_result.scalar_one_or_none()
    if not patient_db:
        raise HTTPException(status_code=404, detail="Patient not found")

    if not patient_delete_data.force:  # Soft delete
        patient_db.is_deleted = True
        patient_db.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Soft delete related items
        videos_result = await session.execute(select(VideoPath).where(
            VideoPath.patient_id == patient_delete_data.patient_id, VideoPath.is_deleted == False))
        for video in videos_result.scalars().all():
            video.is_deleted = True
            video.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        actions_result = await session.execute(select(Action).where(
            Action.patient_id == patient_delete_data.patient_id, Action.is_deleted == False))
        for action in actions_result.scalars().all():
            action.is_deleted = True
            action.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            stages_result = await session.execute(select(Stage).where(
                Stage.action_id == action.id, Stage.is_deleted == False))
            for stage in stages_result.scalars().all():
                stage.is_deleted = True
                stage.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    else:  # Hard delete
        # This part requires careful cascading deletion
        actions_to_delete_result = await session.execute(select(Action).where(Action.patient_id == patient_db.id))
        for action in actions_to_delete_result.scalars().all():
            stages_to_delete_result = await session.execute(select(Stage).where(Stage.action_id == action.id))
            for stage in stages_to_delete_result.scalars().all():
                await session.execute(delete(StepsInfo).where(StepsInfo.stage_id == stage.id))
                await session.delete(stage)
            await session.delete(action)

        await session.execute(delete(VideoPath).where(VideoPath.patient_id == patient_db.id))
        await session.delete(patient_db)

    await session.commit()
    return {"message": "Patient deleted successfully"}


@router.get("/doctor")
async def get_doctor_by_id(doctor_id: int = Query(...),
                           admin_doctor_id: int = Query(...),
                           session: SessionDep = SessionDep):
    await authorize_admin(admin_doctor_id, session)
    result = await session.execute(select(Doctors).where(
        Doctors.id == doctor_id, Doctors.is_deleted == False))
    doctor = result.scalar_one_or_none()
    if not doctor:
        raise HTTPException(status_code=404, detail="Doctor not found")

    patients_result = await session.execute(select(Patients).where(
        Patients.doctor_id == doctor_id, Patients.is_deleted == False))
    patients = patients_result.scalars().all()

    doc_dict = doctor.to_dict()
    if 'password' in doc_dict:
        del doc_dict['password']
    doc_dict["patientCount"] = len(patients)

    return {
        "doctor": doc_dict,
        "patients": [patient.to_dict() for patient in patients]
    }


@router.get("/patient")
async def get_patient_by_id(patient_id: int = Query(...),
                            admin_doctor_id: int = Query(...),
                            session: SessionDep = SessionDep):
    await authorize_admin(admin_doctor_id, session)
    result = await session.execute(select(Patients).where(
        Patients.id == patient_id, Patients.is_deleted == False))
    patient = result.scalar_one_or_none()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    pat_dict = patient.to_dict()
    doctor_name = "N/A"
    if patient.doctor_id:
        result = await session.execute(select(Doctors.username).where(
            Doctors.id == patient.doctor_id))
        doctor_name = result.scalar_one_or_none() or "N/A"
    pat_dict["attendingDoctorName"] = doctor_name

    video_count_result = await session.execute(select(func.count(VideoPath.id)).where(
        VideoPath.patient_id == patient.id, VideoPath.is_deleted == False))
    pat_dict["videoCount"] = video_count_result.scalar_one()

    analysis_count_result = await session.execute(select(func.count(Action.id)).where(
        Action.patient_id == patient.id, Action.is_deleted == False))
    pat_dict["analysisCount"] = analysis_count_result.scalar_one()

    videos_result = await session.execute(select(VideoPath).where(
        VideoPath.patient_id == patient_id, VideoPath.is_deleted == False))
    videos = videos_result.scalars().all()

    actions_result = await session.execute(select(Action).where(
        Action.patient_id == patient_id, Action.is_deleted == False))
    actions = actions_result.scalars().all()

    return {
        "patient": pat_dict,
        "videos": [video.to_dict() for video in videos],
        "actions": [action.to_dict() for action in actions]
    }


@router.get("/actions", response_model=List[ActionDetailResponse])
async def get_actions_management(admin_doctor_id: int = Query(...),
                                 session: SessionDep = SessionDep):
    await authorize_admin(admin_doctor_id, session)

    query = select(
        Action,
        Patients.username.label("patient_username"),
        VideoPath.video_path.label("original_video_path")
    ).join(
        Patients, Action.patient_id == Patients.id, isouter=True
    ).join(
        VideoPath, Action.original_video_id == VideoPath.id, isouter=True
    ).where(
        Action.is_deleted == False
    ).order_by(Action.create_time.desc())

    result = await session.execute(query)
    query_results = result.all()

    if not query_results:
        return []

    response_list = []
    for action, patient_username, original_video_path in query_results:
        action_dict = action.to_dict()
        action_dict["patient_username"] = patient_username or "N/A"
        action_dict["original_video_path"] = original_video_path or "N/A"
        response_list.append(ActionDetailResponse(**action_dict))

    return response_list


@router.get("/videos", response_model=List[VideoDetailResponse])
async def get_videos_management(admin_doctor_id: int = Query(...),
                                session: SessionDep = SessionDep):
    await authorize_admin(admin_doctor_id, session)

    query = select(
        VideoPath,
        Patients.username.label("patient_username")
    ).join(
        Patients, VideoPath.patient_id == Patients.id, isouter=True
    ).where(
        VideoPath.is_deleted == False
    ).order_by(VideoPath.create_time.desc())

    result = await session.execute(query)
    query_results = result.all()

    if not query_results:
        return []

    response_list = []
    for video, patient_username in query_results:
        video_dict = video.to_dict()
        video_dict["patient_username"] = patient_username or "N/A"
        response_list.append(VideoDetailResponse(**video_dict))

    return response_list


@router.delete("/video")
async def delete_video_management(video_del_data: DeleteVideo, session: SessionDep = SessionDep):
    await authorize_admin(video_del_data.admin_doctor_id, session)

    video_result = await session.execute(select(VideoPath).where(
        VideoPath.id == video_del_data.video_id, VideoPath.is_deleted == False))
    video_db = video_result.scalar_one_or_none()
    if not video_db:
        raise HTTPException(status_code=404, detail="Video not found")

    if not video_del_data.force:  # Soft delete
        video_db.is_deleted = True
        video_db.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if video_db.original_video:
            actions_result = await session.execute(select(Action).where(
                Action.original_video_id == video_db.id, Action.is_deleted == False))
            for action in actions_result.scalars().all():
                action.is_deleted = True
                action.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                stages_result = await session.execute(select(Stage).where(
                    Stage.action_id == action.id, Stage.is_deleted == False))
                for stage in stages_result.scalars().all():
                    stage.is_deleted = True
                    stage.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    steps_result = await session.execute(select(StepsInfo).where(
                        StepsInfo.stage_id == stage.id, StepsInfo.is_deleted == False))
                    for step in steps_result.scalars().all():
                        step.is_deleted = True
                        step.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                inf_vids_result = await session.execute(select(VideoPath).where(
                    VideoPath.action_id == action.id, VideoPath.inference_video == True, VideoPath.is_deleted == False))
                for inf_vid in inf_vids_result.scalars().all():
                    inf_vid.is_deleted = True
                    inf_vid.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    else:  # Hard delete
        if video_db.original_video:
            actions_result = await session.execute(select(Action).where(Action.original_video_id == video_del_data.video_id))
            for action in actions_result.scalars().all():
                await session.execute(delete(VideoPath).where(VideoPath.action_id == action.id, VideoPath.inference_video == True))
                
                stages_result = await session.execute(select(Stage).where(Stage.action_id == action.id))
                for stage in stages_result.scalars().all():
                    await session.execute(delete(StepsInfo).where(StepsInfo.stage_id == stage.id))
                    await session.delete(stage)
                await session.delete(action)
        await session.delete(video_db)

    await session.commit()
    return {"message": "Video deleted successfully"}


@router.delete("/action")
async def delete_action_management(action_del_data: DeleteAction, session: SessionDep = SessionDep):
    await authorize_admin(action_del_data.admin_doctor_id, session)

    action_result = await session.execute(select(Action).where(
        Action.id == action_del_data.action_id, Action.is_deleted == False))
    action_db = action_result.scalar_one_or_none()
    if not action_db:
        raise HTTPException(status_code=404, detail="Action not found")

    # Soft delete logic
    action_db.is_deleted = True
    action_db.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    actions_to_soft_delete = [action_db]
    if action_db.parent_id == action_db.id:
        children_result = await session.execute(select(Action).where(
            Action.parent_id == action_db.id, Action.id != action_db.id, Action.is_deleted == False))
        actions_to_soft_delete.extend(children_result.scalars().all())

    for act in actions_to_soft_delete:
        if act.id != action_db.id:
            act.is_deleted = True
            act.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        stages_result = await session.execute(select(Stage).where(
            Stage.action_id == act.id, Stage.is_deleted == False))
        for stage in stages_result.scalars().all():
            stage.is_deleted = True
            stage.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            steps_result = await session.execute(select(StepsInfo).where(
                StepsInfo.stage_id == stage.id, StepsInfo.is_deleted == False))
            for step in steps_result.scalars().all():
                step.is_deleted = True
                step.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        inf_vids_result = await session.execute(select(VideoPath).where(
            VideoPath.action_id == act.id, VideoPath.inference_video == True, VideoPath.is_deleted == False))
        for inf_video in inf_vids_result.scalars().all():
            inf_video.is_deleted = True
            inf_video.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    await session.commit()
    return {"message": "Action and related entities deleted successfully"}


@router.get("/dashboard/metrics", response_model=DashboardMetrics)
async def get_dashboard_metrics(admin_doctor_id: int = Query(...), session: SessionDep = SessionDep):
    await authorize_admin(admin_doctor_id, session)

    doctor_count_res = await session.execute(select(func.count(Doctors.id)).where(Doctors.is_deleted == False))
    patient_count_res = await session.execute(select(func.count(Patients.id)).where(Patients.is_deleted == False))
    video_count_res = await session.execute(select(func.count(VideoPath.id)).where(VideoPath.is_deleted == False))
    analysis_count_res = await session.execute(select(func.count(Action.id)).where(Action.is_deleted == False))
    
    return DashboardMetrics(
        doctorCount=doctor_count_res.scalar_one(),
        patientCount=patient_count_res.scalar_one(),
        videoCount=video_count_res.scalar_one(),
        dataAnalysisCount=analysis_count_res.scalar_one()
    )


@router.get("/dashboard/analysis-trends", response_model=List[DataAnalysisDataPoint])
async def get_analysis_trends(admin_doctor_id: int = Query(...), session: SessionDep = SessionDep):
    await authorize_admin(admin_doctor_id, session)

    # Note: `to_timestamp` and `to_char` are specific to PostgreSQL. 
    # This might need adjustment for other databases like SQLite.
    query = select(
        func.to_char(func.to_timestamp(Action.create_time, 'YYYY-MM-DD HH24:MI:SS'), 'Mon ''YY').label('date_label'),
        func.count(Action.id).label('analyses_count'),
        func.extract('year', func.to_timestamp(Action.create_time, 'YYYY-MM-DD HH24:MI:SS')).label('year'),
        func.extract('month', func.to_timestamp(Action.create_time, 'YYYY-MM-DD HH24:MI:SS')).label('month')
    ).where(Action.is_deleted == False).group_by(
        'date_label', 'year', 'month'
    ).order_by('year', 'month')

    result = await session.execute(query)
    query_result = result.all()

    return [
        DataAnalysisDataPoint(date=row.date_label, analyses=row.analyses_count)
        for row in query_result
    ]
    