from common.utils import calculate_stats
from fastapi import APIRouter
from models import SessionDep, Stage, StepsInfo

router = APIRouter(tags=["table"], prefix="/table")


@router.get("/step_hip_degree/{action_id}")
def get_average_step_hip_degree(action_id: int, session: SessionDep = SessionDep):
    left_step_hip_degree_low = []
    left_step_hip_degree_high = []
    right_step_hip_degree_high = []
    right_step_hip_degree_low = []
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
            # step_hip_degree_low.append(step_info.hip_min_degree)
            if step_info.front_leg == "left":
                left_step_hip_degree_low.append(step_info.hip_min_degree)
            elif step_info.front_leg == "right":
                right_step_hip_degree_low.append(step_info.hip_min_degree)
            step_hip_degree_low.append(step_info.hip_min_degree)
        if step_info.hip_max_degree is not None:
            # step_hip_degree_high.append(step_info.hip_max_degree)
            if step_info.front_leg == "left":
                left_step_hip_degree_high.append(step_info.hip_max_degree)
            elif step_info.front_leg == "right":
                right_step_hip_degree_high.append(step_info.hip_max_degree)
            step_hip_degree_high.append(step_info.hip_max_degree)

    low_average, low_standard_deviation = calculate_stats(step_hip_degree_low)
    high_average, high_standard_deviation = calculate_stats(
        step_hip_degree_high)
    left_low_average, left_low_standard_deviation = calculate_stats(
        left_step_hip_degree_low)
    left_high_average, left_high_standard_deviation = calculate_stats(
        left_step_hip_degree_high)
    right_low_average, right_low_standard_deviation = calculate_stats(
        right_step_hip_degree_low)
    right_high_average, right_high_standard_deviation = calculate_stats(
        right_step_hip_degree_high)

    # Calculate overall stats using combined data
    all_degrees = step_hip_degree_low + step_hip_degree_high
    average, standard_deviation = calculate_stats(all_degrees)
    all_left_degrees = left_step_hip_degree_low + left_step_hip_degree_high
    all_right_degrees = right_step_hip_degree_low + right_step_hip_degree_high
    left_average, left_standard_deviation = calculate_stats(all_left_degrees)
    right_average, right_standard_deviation = calculate_stats(
        all_right_degrees)
    min_value = min(all_degrees) if all_degrees else 0
    max_value = max(all_degrees) if all_degrees else 0
    left_min_value = min(all_left_degrees) if all_left_degrees else 0
    left_max_value = max(all_left_degrees) if all_left_degrees else 0
    right_min_value = min(all_right_degrees) if all_right_degrees else 0
    right_max_value = max(all_right_degrees) if all_right_degrees else 0

    return {
        "low_average": low_average,
        "high_average": high_average,
        "left_low_average": left_low_average,
        "left_high_average": left_high_average,
        "right_low_average": right_low_average,
        "right_high_average": right_high_average,
        "left_average": left_average,
        "right_average": right_average,
        "average": average,
        "min_value": min_value,
        "max_value": max_value,
        "left_min_value": left_min_value,
        "left_max_value": left_max_value,
        "right_min_value": right_min_value,
        "right_max_value": right_max_value,
        "left_standard_deviation": left_standard_deviation,
        "right_standard_deviation": right_standard_deviation,
        "left_low_standard_deviation": left_low_standard_deviation,
        "left_high_standard_deviation": left_high_standard_deviation,
        "right_low_standard_deviation": right_low_standard_deviation,
        "right_high_standard_deviation": right_high_standard_deviation,
        "low_standard_deviation": low_standard_deviation,
        "high_standard_deviation": high_standard_deviation,
        "standard_deviation": standard_deviation,  # Corrected: uses combined data
        "chart_url": f"/dashboard/step_hip_degree/{action_id}"
    }


@router.get("/step_length/{action_id}")
def get_average_step_length(action_id: int, session: SessionDep = SessionDep):
    left_step_length = []
    right_step_length = []
    step_length = []

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
            step_length.append(step_info.step_length)

    left_average, left_standard_deviation = calculate_stats(left_step_length)
    right_average, right_standard_deviation = calculate_stats(
        right_step_length)
    average, standard_deviation = calculate_stats(step_length)
    left_min_value = min(left_step_length) if left_step_length else 0
    left_max_value = max(left_step_length) if left_step_length else 0
    right_min_value = min(right_step_length) if right_step_length else 0
    right_max_value = max(right_step_length) if right_step_length else 0
    min_value = min(step_length) if step_length else 0
    max_value = max(step_length) if step_length else 0


    return {
        "left_average": left_average,
        "right_average": right_average,
        "average": average,
        "left_standard_deviation": left_standard_deviation,
        "right_standard_deviation": right_standard_deviation,
        "standard_deviation": standard_deviation,
        "left_min_value": left_min_value,
        "left_max_value": left_max_value,
        "right_min_value": right_min_value,
        "right_max_value": right_max_value,
        "min_value": min_value,
        "max_value": max_value,
        "chart_url": f"/dashboard/step_length/{action_id}"
    }

@router.get("/step_width/{action_id}")
def get_average_step_width(action_id: int, session: SessionDep = SessionDep):
    step_width = []
    left_step_width = []
    right_step_width = []

    stages_ids_query = session.query(Stage.id).filter(
        Stage.action_id == action_id, Stage.is_deleted == False
    )
    stage_ids = [id_tuple[0] for id_tuple in stages_ids_query.all()]

    if not stage_ids:
        return {"average": 0, "standard_deviation": 0,
                "chart_url": f"/dashboard/step_width/{action_id}"}

    steps_info_query = session.query(StepsInfo).filter(
        StepsInfo.stage_id.in_(stage_ids),
        StepsInfo.is_deleted == False
    )

    for step_info in steps_info_query.all():
        if step_info.step_width is not None:
            if step_info.front_leg == "left":
                left_step_width.append(step_info.step_width)
            elif step_info.front_leg == "right":
                right_step_width.append(step_info.step_width)
            step_width.append(step_info.step_width)

    average, standard_deviation = calculate_stats(step_width)
    left_average, left_standard_deviation = calculate_stats(left_step_width)
    right_average, right_standard_deviation = calculate_stats(
        right_step_width)
    
    return {
        "left_average": left_average,
        "right_average": right_average,
        "average": average,
        "left_standard_deviation": left_standard_deviation,
        "right_standard_deviation": right_standard_deviation,
        "left_min_value": min(left_step_width) if left_step_width else 0,
        "left_max_value": max(left_step_width) if left_step_width else 0,
        "right_min_value": min(right_step_width) if right_step_width else 0,
        "right_max_value": max(right_step_width) if right_step_width else 0,
        "min_value": min(step_width) if step_width else 0,
        "max_value": max(step_width) if step_width else 0,
        "standard_deviation": standard_deviation,
        "chart_url": f"/dashboard/step_width/{action_id}"
    }

@router.get("/step_speed/{action_id}")
def get_average_step_speed(action_id: int, session: SessionDep = SessionDep):
    left_step_speed = []
    right_step_speed = []
    step_speed = []

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
            step_speed.append(step_info.step_speed)

    left_average, left_standard_deviation = calculate_stats(left_step_speed)
    right_average, right_standard_deviation = calculate_stats(right_step_speed)
    average, standard_deviation = calculate_stats(step_speed)
    left_min_value = min(left_step_speed) if left_step_speed else 0
    left_max_value = max(left_step_speed) if left_step_speed else 0
    right_min_value = min(right_step_speed) if right_step_speed else 0
    right_max_value = max(right_step_speed) if right_step_speed else 0
    min_value = min(step_speed) if step_speed else 0
    max_value = max(step_speed) if step_speed else 0

    all_speeds = left_step_speed + right_step_speed
    average, standard_deviation = calculate_stats(all_speeds)

    return {
        "left_average": left_average,
        "right_average": right_average,
        "average": average,
        "left_standard_deviation": left_standard_deviation,
        "right_standard_deviation": right_standard_deviation,
        "standard_deviation": standard_deviation,  # Corrected: uses combined data
        "left_min_value": left_min_value,
        "left_max_value": left_max_value,
        "right_min_value": right_min_value,
        "right_max_value": right_max_value,
        "min_value": min_value,
        "max_value": max_value,
        "chart_url": f"/dashboard/step_speed/{action_id}"
    }


@router.get("/step_stride/{action_id}")
def get_average_step_stride(action_id: int, session: SessionDep = SessionDep):
    step_stride = []
    left_step_stride = []
    right_step_stride = []

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
            if step_info.front_leg == "left":
                left_step_stride.append(step_info.stride_length)
            elif step_info.front_leg == "right":
                right_step_stride.append(step_info.stride_length)
            step_stride.append(step_info.stride_length)

    average, standard_deviation = calculate_stats(step_stride)
    left_average, left_standard_deviation = calculate_stats(
        left_step_stride)
    right_average, right_standard_deviation = calculate_stats(
        right_step_stride)
    left_min_value = min(left_step_stride) if left_step_stride else 0
    left_max_value = max(left_step_stride) if left_step_stride else 0
    right_min_value = min(right_step_stride) if right_step_stride else 0
    right_max_value = max(right_step_stride) if right_step_stride else 0
    min_value = min(step_stride) if step_stride else 0
    max_value = max(step_stride) if step_stride else 0

    return {
        "left_average": left_average,
        "right_average": right_average,
        "average": average,
        "left_standard_deviation": left_standard_deviation,
        "right_standard_deviation": right_standard_deviation,
        "standard_deviation": standard_deviation,
        "left_min_value": left_min_value,
        "left_max_value": left_max_value,
        "right_min_value": right_min_value,
        "right_max_value": right_max_value,
        "min_value": min_value,
        "max_value": max_value,
        "chart_url": f"/dashboard/step_stride/{action_id}"
    }


@router.get("/step_difference/{action_id}")
def get_average_step_difference(action_id: int, session: SessionDep = SessionDep):
    step_difference = []
    left_step_difference = []
    right_step_difference = []

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
            if step_info.front_leg == "left":
                left_step_difference.append(step_info.steps_diff)
            elif step_info.front_leg == "right":
                right_step_difference.append(step_info.steps_diff)
            step_difference.append(step_info.steps_diff)

    average, standard_deviation = calculate_stats(step_difference)
    left_average, left_standard_deviation = calculate_stats(
        left_step_difference)
    right_average, right_standard_deviation = calculate_stats(
        right_step_difference)
    left_min_value = min(left_step_difference) if left_step_difference else 0
    left_max_value = max(left_step_difference) if left_step_difference else 0
    right_min_value = min(right_step_difference) if right_step_difference else 0
    right_max_value = max(right_step_difference) if right_step_difference else 0
    min_value = min(step_difference) if step_difference else 0
    max_value = max(step_difference) if step_difference else 0

    return {
        "left_average": left_average,
        "right_average": right_average,
        "average": average,
        "left_standard_deviation": left_standard_deviation,
        "right_standard_deviation": right_standard_deviation,
        "standard_deviation": standard_deviation,
        "left_min_value": left_min_value,
        "left_max_value": left_max_value,
        "right_min_value": right_min_value,
        "right_max_value": right_max_value,
        "min_value": min_value,
        "max_value": max_value,
        "chart_url": f"/dashboard/step_difference/{action_id}"
    }


@router.get("/support_time/{action_id}")
def get_average_support_time(action_id: int, session: SessionDep = SessionDep):
    left_support_time = []
    right_support_time = []
    support_time = []

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
            support_time.append(step_info.support_time)

    left_average, left_standard_deviation = calculate_stats(left_support_time)
    right_average, right_standard_deviation = calculate_stats(
        right_support_time)
    average, standard_deviation = calculate_stats(support_time)
    left_min_value = min(left_support_time) if left_support_time else 0
    left_max_value = max(left_support_time) if left_support_time else 0
    right_min_value = min(right_support_time) if right_support_time else 0
    right_max_value = max(right_support_time) if right_support_time else 0
    min_value = min(support_time) if support_time else 0
    max_value = max(support_time) if support_time else 0

    return {
        "left_average": left_average,
        "right_average": right_average,
        "average": average,
        "left_standard_deviation": left_standard_deviation,
        "right_standard_deviation": right_standard_deviation,
        "standard_deviation": standard_deviation,
        "left_min_value": left_min_value,
        "left_max_value": left_max_value,
        "right_min_value": right_min_value,
        "right_max_value": right_max_value,
        "min_value": min_value,
        "max_value": max_value,
        "chart_url": f"/dashboard/support_time/{action_id}"
    }


@router.get("/liftoff_height/{action_id}")
def get_average_liftoff_height(action_id: int, session: SessionDep = SessionDep):
    left_liftoff_height = []
    right_liftoff_height = []
    liftoff_height = []

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
            liftoff_height.append(step_info.liftoff_height)

    left_average, left_standard_deviation = calculate_stats(
        left_liftoff_height)
    right_average, right_standard_deviation = calculate_stats(
        right_liftoff_height)
    average, standard_deviation = calculate_stats(liftoff_height)
    left_min_value = min(left_liftoff_height) if left_liftoff_height else 0
    left_max_value = max(left_liftoff_height) if left_liftoff_height else 0
    right_min_value = min(right_liftoff_height) if right_liftoff_height else 0
    right_max_value = max(right_liftoff_height) if right_liftoff_height else 0
    min_value = min(liftoff_height) if liftoff_height else 0
    max_value = max(liftoff_height) if liftoff_height else 0

    return {
        "left_average": left_average,
        "right_average": right_average,
        "average": average,
        "left_standard_deviation": left_standard_deviation,
        "right_standard_deviation": right_standard_deviation,
        "standard_deviation": standard_deviation,
        "left_min_value": left_min_value,
        "left_max_value": left_max_value,
        "right_min_value": right_min_value,
        "right_max_value": right_max_value,
        "min_value": min_value,
        "max_value": max_value,
        "chart_url": f"/dashboard/liftoff_height/{action_id}"
    }
