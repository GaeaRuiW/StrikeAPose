import os
from datetime import datetime

from common.utils import hash_password
from fastapi import APIRouter, Body, Query
from models import (Action, Doctors, Patients, SessionDep, Stage, StepsInfo,
                    VideoPath)
from pydantic import BaseModel

router = APIRouter(tags=["management"], prefix="/management")


class Login(BaseModel):
    username: str
    password: str


class BaseQuery(BaseModel):
    page: int = 1
    size: int = 10
    doctor_id: int = 0


class BASE(BaseModel):
    admin_doctor_id: int


class UpdateDoctor(BASE):
    doctor_id: int
    email: str
    phone: str
    password: str
    department: str = "康复科"
    role_id: int = 1


class DeleteDoctor(BASE):
    doctor_id: int
    assign_doctor_id: int


class UpdatePatient(BASE):
    username: str
    age: int
    gender: str
    case_id: str
    doctor_id: int


class DeletePatient(BASE):
    patient_id: int
    force: bool = False


class DeleteVideo(BASE):
    video_id: int
    force: bool = False


class DeleteAction(BASE):
    action_id: int


@router.post("/login")
def login(login: Login = Body(..., embed=True), session: SessionDep = SessionDep):
    doctor = session.query(Doctors).filter(
        Doctors.username == login.username, Doctors.password == login.password, Doctors.is_deleted == False).first()
    if not doctor:
        return {"message": "Invalid username or password"}
    if doctor.role_id != 1:
        return {"message": "You do not have permission to access this resource"}
    return {"message": "Login successful", "doctor": doctor.to_dict()}


@router.get("/doctors")
def get_doctor(admin_doctor_id: int = Query(...), session: SessionDep = SessionDep):
    doctor = session.query(Doctors).filter(
        Doctors.id == admin_doctor_id, Doctors.is_deleted == False).first()
    if not doctor:
        return {"message": "Doctor not found"}
    if doctor.role_id != 1:
        return {"message": "You do not have permission to access this resource"}
    doctors = session.query(Doctors).filter(
        Doctors.is_deleted == False).all()
    if not doctors:
        return {"message": "No doctors found"}
    return [doctor.to_dict() for doctor in doctors]


@router.get("/patients")
def get_patients(admin_doctor_id: int = Query(...), session: SessionDep = SessionDep):
    doctor = session.query(Doctors).filter(
        Doctors.id == admin_doctor_id, Doctors.is_deleted == False).first()
    if not doctor:
        return {"message": "Doctor not found"}
    if doctor.role_id != 1:
        return {"message": "You do not have permission to access this resource"}
    patients = session.query(Patients).filter(
        Patients.is_deleted == False).all()
    if not patients:
        return {"message": "No patients found"}
    return [patient.to_dict() for patient in patients]


router.get("/doctor")


def get_doctor_by_id(doctor_id: int = Query(...),
                     admin_doctor_id: int = Query(...),
                     session: SessionDep = SessionDep):
    admin_doctor = session.query(Doctors).filter(
        Doctors.id == admin_doctor_id, Doctors.is_deleted == False).first()
    if not admin_doctor or admin_doctor.role_id != 1:
        return {"message": "You do not have permission to access this resource"}
    doctor = session.query(Doctors).filter(
        Doctors.id == doctor_id, Doctors.is_deleted == False).first()
    if not doctor:
        return {"message": "Doctor not found"}
    patients = session.query(Patients).filter(
        Patients.doctor_id == doctor_id, Patients.is_deleted == False).all()
    return {
        "doctor": doctor.to_dict(),
        "patients": [patient.to_dict() for patient in patients]
    }


@router.get("/patient")
def get_patient_by_id(patient_id: int = Query(...),
                      admin_doctor_id: int = Query(...),
                      session: SessionDep = SessionDep):
    admin_doctor = session.query(Doctors).filter(
        Doctors.id == admin_doctor_id, Doctors.is_deleted == False).first()
    if not admin_doctor or admin_doctor.role_id != 1:
        return {"message": "You do not have permission to access this resource"}
    patient = session.query(Patients).filter(
        Patients.id == patient_id, Patients.is_deleted == False).first()
    if not patient:
        return {"message": "Patient not found"}
    videos = session.query(VideoPath).filter(
        VideoPath.patient_id == patient_id, VideoPath.is_deleted == False).all()
    actions = session.query(Action).filter(
        Action.patient_id == patient_id, Action.is_deleted == False).all()
    return {
        "patient": patient.to_dict(),
        "videos": [video.to_dict() for video in videos],
        "actions": [action.to_dict() for action in actions]
    }


@router.get("/actions")
def get_actions(admin_doctor_id: int = Query(...),
                session: SessionDep = SessionDep):
    admin_doctor = session.query(Doctors).filter(
        Doctors.id == admin_doctor_id, Doctors.is_deleted == False).first()
    if not admin_doctor or admin_doctor.role_id != 1:
        return {"message": "You do not have permission to access this resource"}
    actions = session.query(Action).filter(
        Action.is_deleted == False).all()
    if not actions:
        return {"message": "No actions found"}
    actions = [action.to_dict() for action in actions]
    for action in actions:
        stages = session.query(Stage).filter(
            Stage.action_id == action.id, Stage.is_deleted == False).all()
        action["stages"] = [stage.to_dict() for stage in stages]
        for stage in stages:
            steps = session.query(StepsInfo).filter(
                StepsInfo.stage_id == stage.id, StepsInfo.is_deleted == False).all()
            action["stages"][stage.stage_n]["steps"] = [step.to_dict()
                                                        for step in steps]
    return actions


@router.get("/videos")
def get_videos(admin_doctor_id: int = Query(...),
               session: SessionDep = SessionDep):
    admin_doctor = session.query(Doctors).filter(
        Doctors.id == admin_doctor_id, Doctors.is_deleted == False).first()
    if not admin_doctor or admin_doctor.role_id != 1:
        return {"message": "You do not have permission to access this resource"}
    videos = session.query(VideoPath).filter(
        VideoPath.is_deleted == False).all()
    if not videos:
        return {"message": "No videos found"}
    return [video.to_dict() for video in videos]


@router.get("/video")
def get_video_by_id(video_id: int = Query(...),
                    admin_doctor_id: int = Query(...),
                    session: SessionDep = SessionDep):
    admin_doctor = session.query(Doctors).filter(
        Doctors.id == admin_doctor_id, Doctors.is_deleted == False).first()
    if not admin_doctor or admin_doctor.role_id != 1:
        return {"message": "You do not have permission to access this resource"}
    video = session.query(VideoPath).filter(
        VideoPath.id == video_id, VideoPath.is_deleted == False).first()
    if not video:
        return {"message": "Video not found"}
    actions = session.query(Action).filter(
        Action.video_id == video_id, Action.is_deleted == False).all()
    return {
        "video": video.to_dict(),
        "actions": [action.to_dict() for action in actions]
    }


@router.get("/action")
def get_action_by_id(action_id: int = Query(...),
                     admin_doctor_id: int = Query(...),
                     session: SessionDep = SessionDep):
    admin_doctor = session.query(Doctors).filter(
        Doctors.id == admin_doctor_id, Doctors.is_deleted == False).first()
    if not admin_doctor or admin_doctor.role_id != 1:
        return {"message": "You do not have permission to access this resource"}
    action = session.query(Action).filter(
        Action.id == action_id, Action.is_deleted == False).first()
    if not action:
        return {"message": "Action not found"}
    son_actions = session.query(Action).filter(
        Action.parent_id == action_id, Action.is_deleted == False).all()
    return {
        "action": action.to_dict(),
        "son_actions": [son_action.to_dict() for son_action in son_actions]
    }


@router.put("/doctor")
def update_doctor(doctor: UpdateDoctor = Body(..., embed=True), session: SessionDep = SessionDep):
    admin_doctor = session.query(Doctors).filter(
        Doctors.id == doctor.admin_doctor_id, Doctors.is_deleted == False).first()
    if not admin_doctor or admin_doctor.role_id != 1:
        return {"message": "You do not have permission to access this resource"}
    doctor_db = session.query(Doctors).filter(
        Doctors.id == doctor.doctor_id, Doctors.is_deleted == False).first()
    if not doctor_db:
        return {"message": "Doctor not found"}
    doctor_db.email = doctor.email
    doctor_db.phone = doctor.phone
    doctor_db.password = hash_password(doctor.password)
    doctor_db.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    doctor_db.department = doctor.department
    doctor_db.role_id = doctor.role_id
    session.commit()
    return doctor_db.to_dict()


@router.put("/patient")
def update_patient(patient: UpdatePatient = Body(..., embed=True), session: SessionDep = SessionDep):
    admin_doctor = session.query(Doctors).filter(
        Doctors.id == patient.admin_doctor_id, Doctors.is_deleted == False).first()
    if not admin_doctor or admin_doctor.role_id != 1:
        return {"message": "You do not have permission to access this resource"}
    patient_db = session.query(Patients).filter(
        Patients.id == patient.patient_id, Patients.is_deleted == False).first()
    if not patient_db:
        return {"message": "Patient not found"}
    patient_db.username = patient.username
    patient_db.age = patient.age
    patient_db.gender = patient.gender
    patient_db.case_id = patient.case_id
    patient_db.doctor_id = patient.doctor_id
    patient_db.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    session.commit()
    return patient_db.to_dict()


@router.delete("/doctor")
def delete_doctor(doctor: DeleteDoctor = Body(..., embed=True), session: SessionDep = SessionDep):
    admin_doctor = session.query(Doctors).filter(
        Doctors.id == doctor.admin_doctor_id, Doctors.is_deleted == False).first()
    if not admin_doctor or admin_doctor.role_id != 1:
        return {"message": "You do not have permission to access this resource"}
    doctor_db = session.query(Doctors).filter(
        Doctors.id == doctor.doctor_id, Doctors.is_deleted == False).first()
    if not doctor_db:
        return {"message": "Doctor not found"}
    doctor_db.is_deleted = True
    doctor_db.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    assigned_patients = session.query(Patients).filter(
        Patients.doctor_id == doctor.doctor_id, Patients.is_deleted == False).all()
    assign_doctor = session.query(Doctors).filter(
        Doctors.id == doctor.assign_doctor_id, Doctors.is_deleted == False).first()
    if not assign_doctor:
        return {"message": "Assigned doctor not found"}
    if assigned_patients:
        for patient in assigned_patients:
            patient.doctor_id = doctor.assign_doctor_id
            patient.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    session.commit()
    return {"message": "Doctor deleted successfully"}


@router.delete("/patient")
def delete_patient(patient: DeletePatient = Body(..., embed=True), session: SessionDep = SessionDep):
    admin_doctor = session.query(Doctors).filter(
        Doctors.id == patient.admin_doctor_id, Doctors.is_deleted == False).first()
    if not admin_doctor or admin_doctor.role_id != 1:
        return {"message": "You do not have permission to access this resource"}
    patient_db = session.query(Patients).filter(
        Patients.id == patient.patient_id, Patients.is_deleted == False).first()
    if not patient_db:
        return {"message": "Patient not found"}
    if not patient.force:
        patient_db.is_deleted = True
        patient_db.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        videos = session.query(VideoPath).filter(
            VideoPath.patient_id == patient.patient_id, VideoPath.is_deleted == False).all()
        for video in videos:
            video.is_deleted = True
            video.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        actions = session.query(Action).filter(
            Action.patient_id == patient.patient_id, Action.is_deleted == False).all()
        for action in actions:
            action.is_deleted = True
            action.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            stages = session.query(Stage).filter(
                Stage.action_id == action.id, Stage.is_deleted == False).all()
            for stage in stages:
                stage.is_deleted = True
                stage.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                steps = session.query(StepsInfo).filter(
                    StepsInfo.stage_id == stage.id, StepsInfo.is_deleted == False).all()
                for step in steps:
                    step.is_deleted = True
                    step.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    else:
        videos = session.query(VideoPath).filter(
            VideoPath.patient_id == patient.patient_id, VideoPath.is_deleted == False).all()
        for video in videos:
            video_path = video.video_path
            if os.path.exists(video_path):
                print(f"Deleting video: {video_path}")
                os.remove(video_path)
            if os.path.exists(video_path.replace("mp4", "json")):
                print(f"Deleting json: {video_path.replace('mp4', 'json')}")
                os.remove(video_path.replace("mp4", "json"))
            session.delete(video)
        actions = session.query(Action).filter(
            Action.patient_id == patient.patient_id, Action.is_deleted == False).all()
        for action in actions:
            stages = session.query(Stage).filter(
                Stage.action_id == action.id, Stage.is_deleted == False).all()
            for stage in stages:
                steps = session.query(StepsInfo).filter(
                    StepsInfo.stage_id == stage.id, StepsInfo.is_deleted == False).all()
                for step in steps:
                    session.delete(step)
                session.delete(stage)
            session.delete(action)
        session.delete(patient_db)
    session.commit()
    return {"message": "Patient deleted successfully"}


@router.delete("/video")
def delete_video(video: DeleteVideo = Body(..., embed=True), session: SessionDep = SessionDep):
    admin_doctor = session.query(Doctors).filter(
        Doctors.id == video.admin_doctor_id, Doctors.is_deleted == False).first()
    if not admin_doctor or admin_doctor.role_id != 1:
        return {"message": "You do not have permission to access this resource"}
    video_db = session.query(VideoPath).filter(
        VideoPath.id == video.video_id, VideoPath.is_deleted == False).first()
    if not video_db:
        return {"message": "Video not found"}
    if not video.force:
        video_db.is_deleted = True
        video_db.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    actions = session.query(Action).filter(
        Action.video_id == video.video_id, Action.is_deleted == False).all()
    for action in actions:
        action.is_deleted = True
        action.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        stages = session.query(Stage).filter(
            Stage.action_id == action.id, Stage.is_deleted == False).all()
        for stage in stages:
            stage.is_deleted = True
            stage.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            steps = session.query(StepsInfo).filter(
                StepsInfo.stage_id == stage.id, StepsInfo.is_deleted == False).all()
            for step in steps:
                step.is_deleted = True
                step.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    else:
        actions = session.query(Action).filter(
            Action.video_id == video.video_id, Action.is_deleted == False).all()
        for action in actions:
            stages = session.query(Stage).filter(
                Stage.action_id == action.id, Stage.is_deleted == False).all()
            for stage in stages:
                steps = session.query(StepsInfo).filter(
                    StepsInfo.stage_id == stage.id, StepsInfo.is_deleted == False).all()
                for step in steps:
                    session.delete(step)
                session.delete(stage)
            session.delete(action)
        session.delete(video_db)
    session.commit()
    return {"message": "Video deleted successfully"}


@router.delete("/action")
def delete_action(action: DeleteAction = Body(..., embed=True), session: SessionDep = SessionDep):
    admin_doctor = session.query(Doctors).filter(
        Doctors.id == action.admin_doctor_id, Doctors.is_deleted == False).first()
    if not admin_doctor or admin_doctor.role_id != 1:
        return {"message": "You do not have permission to access this resource"}
    action_db = session.query(Action).filter(
        Action.id == action.action_id, Action.is_deleted == False).first()
    if not action_db:
        return {"message": "Action not found"}
    action_db.is_deleted = True
    action_db.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    stages = session.query(Stage).filter(
        Stage.action_id == action.action_id, Stage.is_deleted == False).all()
    for stage in stages:
        stage.is_deleted = True
        stage.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        steps = session.query(StepsInfo).filter(
            StepsInfo.stage_id == stage.id, StepsInfo.is_deleted == False).all()
        for step in steps:
            step.is_deleted = True
            step.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    son_actions = session.query(Action).filter(
        Action.parent_id == action.action_id, Action.is_deleted == False).all()
    for son_action in son_actions:
        son_action.is_deleted = True
        son_action.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        stages = session.query(Stage).filter(
            Stage.action_id == son_action.id, Stage.is_deleted == False).all()
        for stage in stages:
            stage.is_deleted = True
            stage.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            steps = session.query(StepsInfo).filter(
                StepsInfo.stage_id == stage.id, StepsInfo.is_deleted == False).all()
            for step in steps:
                step.is_deleted = True
                step.update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    session.commit()
    return {"message": "Action deleted successfully"}
