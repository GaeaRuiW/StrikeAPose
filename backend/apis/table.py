from common.utils import calculate_stats
from fastapi import APIRouter
from models import SessionDep, Stage, StepsInfo

router = APIRouter(tags=["table"], prefix="/table")


@router.get("/step_hip_degree/{action_id}")
def get_average_step_hip_degree(action_id: int, session: Session = SessionDep):
    step_hip_degree_low = []
    step_hip_degree_high = []

    # Optimized Query: Get relevant stage IDs first
    stages_ids_query = session.query(Stage.id).filter(
        Stage.action_id == action_id, Stage.is_deleted == False
    )
    stage_ids = [id_tuple[0] for id_tuple in stages_ids_query.all()]

    if not stage_ids:
        # Return early if no relevant stages found
        return {"low_average": 0, "high_average": 0, "average": 0,
                "low_standard_deviation": 0, "high_standard_deviation": 0,
                "standard_deviation": 0, "chart_url": f"/dashboard/step_hip_degree/{action_id}"}

    # Single query for all steps info
    steps_info_query = session.query(StepsInfo).filter(
        StepsInfo.stage_id.in_(stage_ids),
        StepsInfo.is_deleted == False
    )

    for step_info in steps_info_query.all():
        # Check for None before appending
        if step_info.hip_min_degree is not None:
            step_hip_degree_low.append(step_info.hip_min_degree)
        if step_info.hip_max_degree is not None:
            step_hip_degree_high.append(step_info.hip_max_degree)

    low_average, low_standard_deviation = calculate_stats(step_hip_degree_low)
    high_average, high_standard_deviation = calculate_stats(
        step_hip_degree_high)

    # Calculate overall stats using combined data
    all_degrees = step_hip_degree_low + step_hip_degree_high
    average, standard_deviation = calculate_stats(all_degrees)

    return {
        "low_average": low_average,
        "high_average": high_average,
        "average": average,
        "low_standard_deviation": low_standard_deviation,
        "high_standard_deviation": high_standard_deviation,
        "standard_deviation": standard_deviation,  # Corrected: uses combined data
        "chart_url": f"/dashboard/step_hip_degree/{action_id}"
    }


@router.get("/step_length/{action_id}")
def get_average_step_length(action_id: int, session: Session = SessionDep):
    left_step_length = []
    right_step_length = []

    stages_ids_query = session.query(Stage.id).filter(
        Stage.action_id == action_id, Stage.is_deleted == False
    )
    stage_ids = [id_tuple[0] for id_tuple in stages_ids_query.all()]

    if not stage_ids:
        return {"left_average": 0, "right_average": 0, "average": 0,
                "left_standard_deviation": 0, "right_standard_deviation": 0,
                "standard_deviation": 0, "chart_url": f"/dashboard/step_length/{action_id}"}

    steps_info_query = session.query(StepsInfo).filter(
        StepsInfo.stage_id.in_(stage_ids),
        StepsInfo.is_deleted == False
    )

    for step_info in steps_info_query.all():
        if step_info.step_length is not None:  # Check for None
            if step_info.front_leg == "left":
                left_step_length.append(step_info.step_length)
            elif step_info.front_leg == "right":  # Use elif for clarity
                right_step_length.append(step_info.step_length)
            # Consider else case if front_leg can be something else or None

    left_average, left_standard_deviation = calculate_stats(left_step_length)
    right_average, right_standard_deviation = calculate_stats(
        right_step_length)

    all_lengths = left_step_length + right_step_length
    average, standard_deviation = calculate_stats(all_lengths)

    return {
        "left_average": left_average,
        "right_average": right_average,
        "average": average,
        "left_standard_deviation": left_standard_deviation,
        "right_standard_deviation": right_standard_deviation,
        "standard_deviation": standard_deviation,  # Corrected: uses combined data
        "chart_url": f"/dashboard/step_length/{action_id}"
    }


@router.get("/step_speed/{action_id}")
def get_average_step_speed(action_id: int, session: Session = SessionDep):
    left_step_speed = []
    right_step_speed = []

    stages_ids_query = session.query(Stage.id).filter(
        Stage.action_id == action_id, Stage.is_deleted == False
    )
    stage_ids = [id_tuple[0] for id_tuple in stages_ids_query.all()]

    if not stage_ids:
        return {"left_average": 0, "right_average": 0, "average": 0,
                "left_standard_deviation": 0, "right_standard_deviation": 0,
                "standard_deviation": 0, "chart_url": f"/dashboard/step_speed/{action_id}"}

    steps_info_query = session.query(StepsInfo).filter(
        StepsInfo.stage_id.in_(stage_ids),
        StepsInfo.is_deleted == False
    )

    for step_info in steps_info_query.all():
        if step_info.step_speed is not None:  # Check for None
            if step_info.front_leg == "left":
                left_step_speed.append(step_info.step_speed)
            elif step_info.front_leg == "right":
                right_step_speed.append(step_info.step_speed)

    left_average, left_standard_deviation = calculate_stats(left_step_speed)
    right_average, right_standard_deviation = calculate_stats(right_step_speed)

    all_speeds = left_step_speed + right_step_speed
    average, standard_deviation = calculate_stats(all_speeds)

    return {
        "left_average": left_average,
        "right_average": right_average,
        "average": average,
        "left_standard_deviation": left_standard_deviation,
        "right_standard_deviation": right_standard_deviation,
        "standard_deviation": standard_deviation,  # Corrected: uses combined data
        "chart_url": f"/dashboard/step_speed/{action_id}"
    }


@router.get("/step_stride/{action_id}")
def get_average_step_stride(action_id: int, session: Session = SessionDep):
    step_stride = []

    stages_ids_query = session.query(Stage.id).filter(
        Stage.action_id == action_id, Stage.is_deleted == False
    )
    stage_ids = [id_tuple[0] for id_tuple in stages_ids_query.all()]

    if not stage_ids:
        return {"average": 0, "standard_deviation": 0,
                "chart_url": f"/dashboard/step_stride/{action_id}"}

    steps_info_query = session.query(StepsInfo).filter(
        StepsInfo.stage_id.in_(stage_ids),
        StepsInfo.is_deleted == False,
        StepsInfo.first_step == False  # Keep this filter if intended
    )

    for step_info in steps_info_query.all():
        if step_info.stride_length is not None:  # Check for None
            step_stride.append(step_info.stride_length)

    average, standard_deviation = calculate_stats(step_stride)

    return {
        "average": average,
        "standard_deviation": standard_deviation,
        "chart_url": f"/dashboard/step_stride/{action_id}"
    }


@router.get("/step_difference/{action_id}")
def get_average_step_difference(action_id: int, session: Session = SessionDep):
    step_difference = []

    stages_ids_query = session.query(Stage.id).filter(
        Stage.action_id == action_id, Stage.is_deleted == False
    )
    stage_ids = [id_tuple[0] for id_tuple in stages_ids_query.all()]

    if not stage_ids:
        return {"average": 0, "standard_deviation": 0,
                "chart_url": f"/dashboard/step_difference/{action_id}"}

    steps_info_query = session.query(StepsInfo).filter(
        StepsInfo.stage_id.in_(stage_ids),
        StepsInfo.is_deleted == False,
        StepsInfo.first_step == False
    )

    for step_info in steps_info_query.all():
        if step_info.steps_diff is not None:
            step_difference.append(step_info.steps_diff)

    average, standard_deviation = calculate_stats(step_difference)

    return {
        "average": average,
        "standard_deviation": standard_deviation,
        "chart_url": f"/dashboard/step_difference/{action_id}"
    }


@router.get("/support_time/{action_id}")
def get_average_support_time(action_id: int, session: Session = SessionDep):
    left_support_time = []
    right_support_time = []

    stages_ids_query = session.query(Stage.id).filter(
        Stage.action_id == action_id, Stage.is_deleted == False
    )
    stage_ids = [id_tuple[0] for id_tuple in stages_ids_query.all()]

    if not stage_ids:
        return {"left_average": 0, "right_average": 0, "average": 0,
                "left_standard_deviation": 0, "right_standard_deviation": 0,
                "standard_deviation": 0, "chart_url": f"/dashboard/support_time/{action_id}"}

    steps_info_query = session.query(StepsInfo).filter(
        StepsInfo.stage_id.in_(stage_ids),
        StepsInfo.is_deleted == False
    )

    for step_info in steps_info_query.all():
        if step_info.support_time is not None:  # Check for None
            if step_info.front_leg == "left":
                left_support_time.append(step_info.support_time)
            elif step_info.front_leg == "right":
                right_support_time.append(step_info.support_time)

    left_average, left_standard_deviation = calculate_stats(left_support_time)
    right_average, right_standard_deviation = calculate_stats(
        right_support_time)

    all_times = left_support_time + right_support_time
    average, standard_deviation = calculate_stats(all_times)

    return {
        "left_average": left_average,
        "right_average": right_average,
        "average": average,
        "left_standard_deviation": left_standard_deviation,
        "right_standard_deviation": right_standard_deviation,
        "standard_deviation": standard_deviation,
        "chart_url": f"/dashboard/support_time/{action_id}"
    }


@router.get("/liftoff_height/{action_id}")
def get_average_liftoff_height(action_id: int, session: Session = SessionDep):
    left_liftoff_height = []
    right_liftoff_height = []

    stages_ids_query = session.query(Stage.id).filter(
        Stage.action_id == action_id, Stage.is_deleted == False
    )
    stage_ids = [id_tuple[0] for id_tuple in stages_ids_query.all()]

    if not stage_ids:
        return {"left_average": 0, "right_average": 0, "average": 0,
                "left_standard_deviation": 0, "right_standard_deviation": 0,
                "standard_deviation": 0, "chart_url": f"/dashboard/liftoff_height/{action_id}"}

    steps_info_query = session.query(StepsInfo).filter(
        StepsInfo.stage_id.in_(stage_ids),
        StepsInfo.is_deleted == False
    )

    for step_info in steps_info_query.all():
        if step_info.liftoff_height is not None:
            if step_info.front_leg == "left":
                left_liftoff_height.append(step_info.liftoff_height)
            elif step_info.front_leg == "right":
                right_liftoff_height.append(step_info.liftoff_height)

    left_average, left_standard_deviation = calculate_stats(
        left_liftoff_height)
    right_average, right_standard_deviation = calculate_stats(
        right_liftoff_height)

    all_heights = left_liftoff_height + right_liftoff_height
    average, standard_deviation = calculate_stats(all_heights)

    return {
        "left_average": left_average,
        "right_average": right_average,
        "average": average,
        "left_standard_deviation": left_standard_deviation,
        "right_standard_deviation": right_standard_deviation,
        "standard_deviation": standard_deviation,
        "chart_url": f"/dashboard/liftoff_height/{action_id}"
    }
