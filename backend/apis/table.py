import io
from datetime import datetime

import pandas as pd
from common.utils import calculate_stats
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from models import SessionDep, Stage, StepsInfo
from sqlalchemy import select

router = APIRouter(tags=["tables"], prefix="/table")


@router.get("/step_hip_degree/{action_id}")
async def get_average_step_hip_degree(action_id: int, session: SessionDep = SessionDep):
    data = {
        "left": {"low": [], "high": []},
        "right": {"low": [], "high": []},
        "all": {"low": [], "high": []}
    }

    # Optimized Query: Get relevant stage IDs first
    result = await session.execute(select(Stage.id).where(
        Stage.action_id == action_id, Stage.is_deleted == False
    ))
    stage_ids = [id_tuple[0] for id_tuple in result.all()]

    if not stage_ids:
        # Return early if no relevant stages found
        return {"low_average": 0, "high_average": 0, "average": 0,
                "low_standard_deviation": 0, "high_standard_deviation": 0,
                "standard_deviation": 0, "chart_url": f"/dashboard/step_hip_degree/{action_id}"}

    # Single query for all steps info
    result = await session.execute(select(StepsInfo).where(
        StepsInfo.stage_id.in_(stage_ids),
        StepsInfo.is_deleted == False
    ))
    steps_info_list = result.scalars().all()

    for step_info in steps_info_list:
        if step_info.hip_min_degree is not None:
            data[step_info.front_leg]["low"].append(step_info.hip_min_degree)
            data["all"]["low"].append(step_info.hip_min_degree)
        if step_info.hip_max_degree is not None:
            data[step_info.front_leg]["high"].append(step_info.hip_max_degree)
            data["all"]["high"].append(step_info.hip_max_degree)

    result = {}
    for leg in ["left", "right"]:
        for level in ["low", "high"]:
            avg, std = calculate_stats(data[leg][level])
            result[f"{leg}_{level}_average"] = avg
            result[f"{leg}_{level}_standard_deviation"] = std
        all_degrees = data[leg]["low"] + data[leg]["high"]
        avg, std = calculate_stats(all_degrees)
        result[f"{leg}_average"] = avg
        result[f"{leg}_standard_deviation"] = std
        result[f"{leg}_min_value"] = min(all_degrees, default=0)
        result[f"{leg}_max_value"] = max(all_degrees, default=0)

    # 处理 all
    all_degrees = data["all"]["low"] + data["all"]["high"]
    avg, std = calculate_stats(all_degrees)
    result["average"] = avg
    result["standard_deviation"] = std
    result["min_value"] = min(all_degrees, default=0)
    result["max_value"] = max(all_degrees, default=0)

    return {
        "low_average": result["left_low_average"],
        "high_average": result["left_high_average"],
        "left_low_average": result["left_low_average"],
        "left_high_average": result["left_high_average"],
        "right_low_average": result["right_low_average"],
        "right_high_average": result["right_high_average"],
        "left_average": result["left_average"],
        "right_average": result["right_average"],
        "average": result["average"],
        "min_value": result["left_min_value"],
        "max_value": result["left_max_value"],
        "left_min_value": result["left_min_value"],
        "left_max_value": result["left_max_value"],
        "right_min_value": result["right_min_value"],
        "right_max_value": result["right_max_value"],
        "left_standard_deviation": result["left_standard_deviation"],
        "right_standard_deviation": result["right_standard_deviation"],
        "left_low_standard_deviation": result["left_low_standard_deviation"],
        "left_high_standard_deviation": result["left_high_standard_deviation"],
        "right_low_standard_deviation": result["right_low_standard_deviation"],
        "right_high_standard_deviation": result["right_high_standard_deviation"],
        "low_standard_deviation": result["left_low_standard_deviation"],
        "high_standard_deviation": result["left_high_standard_deviation"],
        "standard_deviation": result["standard_deviation"],
        "chart_url": f"/dashboard/step_hip_degree/{action_id}",
        "download_url": f"/table/step_hip_degree/{action_id}/download"
    }


@router.get("/step_length/{action_id}")
async def get_average_step_length(action_id: int, session: SessionDep = SessionDep):
    left_step_length = []
    right_step_length = []
    step_length = []

    result = await session.execute(select(Stage.id).where(
        Stage.action_id == action_id, Stage.is_deleted == False
    ))
    stage_ids = [id_tuple[0] for id_tuple in result.all()]

    if not stage_ids:
        return {"left_average": 0, "right_average": 0, "average": 0,
                "left_standard_deviation": 0, "right_standard_deviation": 0,
                "standard_deviation": 0, "chart_url": f"/dashboard/step_length/{action_id}"}

    result = await session.execute(select(StepsInfo).where(
        StepsInfo.stage_id.in_(stage_ids),
        StepsInfo.is_deleted == False
    ))
    steps_info_list = result.scalars().all()

    for step_info in steps_info_list:
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
    left_min_value = min(left_step_length, default=0)
    left_max_value = max(left_step_length, default=0)
    right_min_value = min(right_step_length, default=0)
    right_max_value = max(right_step_length, default=0)
    min_value = min(step_length, default=0)
    max_value = max(step_length, default=0)

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
        "chart_url": f"/dashboard/step_length/{action_id}",
        "download_url": f"/table/step_length/{action_id}/download"
    }


@router.get("/step_width/{action_id}")
async def get_average_step_width(action_id: int, session: SessionDep = SessionDep):
    step_width = []
    left_step_width = []
    right_step_width = []

    result = await session.execute(select(Stage.id).where(
        Stage.action_id == action_id, Stage.is_deleted == False
    ))
    stage_ids = [id_tuple[0] for id_tuple in result.all()]

    if not stage_ids:
        return {"average": 0, "standard_deviation": 0,
                "chart_url": f"/dashboard/step_width/{action_id}"}

    result = await session.execute(select(StepsInfo).where(
        StepsInfo.stage_id.in_(stage_ids),
        StepsInfo.is_deleted == False
    ))
    steps_info_list = result.scalars().all()

    for step_info in steps_info_list:
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
        "left_min_value": min(left_step_width, default=0),
        "left_max_value": max(left_step_width, default=0),
        "right_min_value": min(right_step_width, default=0),
        "right_max_value": max(right_step_width, default=0),
        "min_value": min(step_width, default=0),
        "max_value": max(step_width, default=0),
        "standard_deviation": standard_deviation,
        "chart_url": f"/dashboard/step_width/{action_id}",
        "download_url": f"/table/step_width/{action_id}/download"
    }


@router.get("/step_speed/{action_id}")
async def get_average_step_speed(action_id: int, session: SessionDep = SessionDep):
    left_step_speed = []
    right_step_speed = []
    step_speed = []

    result = await session.execute(select(Stage.id).where(
        Stage.action_id == action_id, Stage.is_deleted == False
    ))
    stage_ids = [id_tuple[0] for id_tuple in result.all()]

    if not stage_ids:
        return {"left_average": 0, "right_average": 0, "average": 0,
                "left_standard_deviation": 0, "right_standard_deviation": 0,
                "standard_deviation": 0, "chart_url": f"/dashboard/step_speed/{action_id}"}

    result = await session.execute(select(StepsInfo).where(
        StepsInfo.stage_id.in_(stage_ids),
        StepsInfo.is_deleted == False
    ))
    steps_info_list = result.scalars().all()

    for step_info in steps_info_list:
        if step_info.step_speed is not None:  # Check for None
            if step_info.front_leg == "left":
                left_step_speed.append(step_info.step_speed)
            elif step_info.front_leg == "right":
                right_step_speed.append(step_info.step_speed)
            step_speed.append(step_info.step_speed)

    left_average, left_standard_deviation = calculate_stats(left_step_speed)
    right_average, right_standard_deviation = calculate_stats(right_step_speed)
    average, standard_deviation = calculate_stats(step_speed)
    left_min_value = min(left_step_speed, default=0)
    left_max_value = max(left_step_speed, default=0)
    right_min_value = min(right_step_speed, default=0)
    right_max_value = max(right_step_speed, default=0)
    min_value = min(step_speed, default=0)
    max_value = max(step_speed, default=0)

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
        "chart_url": f"/dashboard/step_speed/{action_id}",
        "download_url": f"/table/step_speed/{action_id}/download"
    }


@router.get("/step_stride/{action_id}")
async def get_average_step_stride(action_id: int, session: SessionDep = SessionDep):
    step_stride = []
    left_step_stride = []
    right_step_stride = []

    result = await session.execute(select(Stage.id).where(
        Stage.action_id == action_id, Stage.is_deleted == False
    ))
    stage_ids = [id_tuple[0] for id_tuple in result.all()]

    if not stage_ids:
        return {"average": 0, "standard_deviation": 0,
                "chart_url": f"/dashboard/step_stride/{action_id}"}

    result = await session.execute(select(StepsInfo).where(
        StepsInfo.stage_id.in_(stage_ids),
        StepsInfo.is_deleted == False,
        StepsInfo.first_step == False  # Keep this filter if intended
    ))
    steps_info_list = result.scalars().all()

    for step_info in steps_info_list:
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
    left_min_value = min(left_step_stride, default=0)
    left_max_value = max(left_step_stride, default=0)
    right_min_value = min(right_step_stride, default=0)
    right_max_value = max(right_step_stride, default=0)
    min_value = min(step_stride, default=0)
    max_value = max(step_stride, default=0)

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
        "chart_url": f"/dashboard/step_stride/{action_id}",
        "download_url": f"/table/step_stride/{action_id}/download"
    }


@router.get("/step_difference/{action_id}")
async def get_average_step_difference(action_id: int, session: SessionDep = SessionDep):
    step_difference = []
    left_step_difference = []
    right_step_difference = []

    result = await session.execute(select(Stage.id).where(
        Stage.action_id == action_id, Stage.is_deleted == False
    ))
    stage_ids = [id_tuple[0] for id_tuple in result.all()]

    if not stage_ids:
        return {"average": 0, "standard_deviation": 0,
                "chart_url": f"/dashboard/step_difference/{action_id}"}

    result = await session.execute(select(StepsInfo).where(
        StepsInfo.stage_id.in_(stage_ids),
        StepsInfo.is_deleted == False,
        StepsInfo.first_step == False
    ))
    steps_info_list = result.scalars().all()

    for step_info in steps_info_list:
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
    left_min_value = min(left_step_difference, default=0)
    left_max_value = max(left_step_difference, default=0)
    right_min_value = min(right_step_difference, default=0)
    right_max_value = max(right_step_difference, default=0)
    min_value = min(step_difference, default=0)
    max_value = max(step_difference, default=0)

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
        "chart_url": f"/dashboard/step_difference/{action_id}",
        "download_url": f"/table/step_difference/{action_id}/download"
    }


@router.get("/support_time/{action_id}")
async def get_average_support_time(action_id: int, session: SessionDep = SessionDep):
    left_support_time = []
    right_support_time = []
    support_time = []

    result = await session.execute(select(Stage.id).where(
        Stage.action_id == action_id, Stage.is_deleted == False
    ))
    stage_ids = [id_tuple[0] for id_tuple in result.all()]

    if not stage_ids:
        return {"left_average": 0, "right_average": 0, "average": 0,
                "left_standard_deviation": 0, "right_standard_deviation": 0,
                "standard_deviation": 0, "chart_url": f"/dashboard/support_time/{action_id}"}

    result = await session.execute(select(StepsInfo).where(
        StepsInfo.stage_id.in_(stage_ids),
        StepsInfo.is_deleted == False
    ))
    steps_info_list = result.scalars().all()

    for step_info in steps_info_list:
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
    left_min_value = min(left_support_time, default=0)
    left_max_value = max(left_support_time, default=0)
    right_min_value = min(right_support_time, default=0)
    right_max_value = max(right_support_time, default=0)
    min_value = min(support_time, default=0)
    max_value = max(support_time, default=0)

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
        "chart_url": f"/dashboard/support_time/{action_id}",
        "download_url": f"/table/support_time/{action_id}/download"
    }


@router.get("/liftoff_height/{action_id}")
async def get_average_liftoff_height(action_id: int, session: SessionDep = SessionDep):
    left_liftoff_height = []
    right_liftoff_height = []
    liftoff_height = []

    result = await session.execute(select(Stage.id).where(
        Stage.action_id == action_id, Stage.is_deleted == False
    ))
    stage_ids = [id_tuple[0] for id_tuple in result.all()]

    if not stage_ids:
        return {"left_average": 0, "right_average": 0, "average": 0,
                "left_standard_deviation": 0, "right_standard_deviation": 0,
                "standard_deviation": 0, "chart_url": f"/dashboard/liftoff_height/{action_id}"}

    result = await session.execute(select(StepsInfo).where(
        StepsInfo.stage_id.in_(stage_ids),
        StepsInfo.is_deleted == False
    ))
    steps_info_list = result.scalars().all()

    for step_info in steps_info_list:
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
    left_min_value = min(left_liftoff_height, default=0)
    left_max_value = max(left_liftoff_height, default=0)
    right_min_value = min(right_liftoff_height, default=0)
    right_max_value = max(right_liftoff_height, default=0)
    min_value = min(liftoff_height, default=0)
    max_value = max(liftoff_height, default=0)

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
        "chart_url": f"/dashboard/liftoff_height/{action_id}",
        "download_url": f"/table/liftoff_height/{action_id}/download"
    }


# Helper function to create Excel file from data
async def create_excel_response(data: dict, filename: str) -> StreamingResponse:
    """Create an Excel file from data and return as StreamingResponse"""
    # Create a pandas DataFrame from the data
    df_data = []

    # Extract the main statistics
    df_data.extend(
        {"Metric": key, "Value": value}
        for key, value in data.items()
        if key not in ["chart_url", "download_url"]
    )
    df = pd.DataFrame(df_data)

    # Create Excel file in memory
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Statistics', index=False)

    output.seek(0)

    return StreamingResponse(
        io.BytesIO(output.read()),
        media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


# Download APIs for each table endpoint

@router.get("/step_hip_degree/{action_id}/download")
async def download_step_hip_degree(action_id: int, session: SessionDep = SessionDep):
    """Download step hip degree data as Excel file"""
    # Get the data from the original API
    data = await get_average_step_hip_degree(action_id, session)

    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"step_hip_degree_action_{action_id}_{timestamp}.xlsx"

    return await create_excel_response(data, filename)


@router.get("/step_length/{action_id}/download")
async def download_step_length(action_id: int, session: SessionDep = SessionDep):
    """Download step length data as Excel file"""
    data = await get_average_step_length(action_id, session)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"step_length_action_{action_id}_{timestamp}.xlsx"

    return await create_excel_response(data, filename)


@router.get("/step_width/{action_id}/download")
async def download_step_width(action_id: int, session: SessionDep = SessionDep):
    """Download step width data as Excel file"""
    data = await get_average_step_width(action_id, session)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"step_width_action_{action_id}_{timestamp}.xlsx"

    return await create_excel_response(data, filename)


@router.get("/step_speed/{action_id}/download")
async def download_step_speed(action_id: int, session: SessionDep = SessionDep):
    """Download step speed data as Excel file"""
    data = await get_average_step_speed(action_id, session)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"step_speed_action_{action_id}_{timestamp}.xlsx"

    return await create_excel_response(data, filename)


@router.get("/step_stride/{action_id}/download")
async def download_step_stride(action_id: int, session: SessionDep = SessionDep):
    """Download step stride data as Excel file"""
    data = await get_average_step_stride(action_id, session)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"step_stride_action_{action_id}_{timestamp}.xlsx"

    return await create_excel_response(data, filename)


@router.get("/step_difference/{action_id}/download")
async def download_step_difference(action_id: int, session: SessionDep = SessionDep):
    """Download step difference data as Excel file"""
    data = await get_average_step_difference(action_id, session)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"step_difference_action_{action_id}_{timestamp}.xlsx"

    return await create_excel_response(data, filename)


@router.get("/support_time/{action_id}/download")
async def download_support_time(action_id: int, session: SessionDep = SessionDep):
    """Download support time data as Excel file"""
    data = await get_average_support_time(action_id, session)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"support_time_action_{action_id}_{timestamp}.xlsx"

    return await create_excel_response(data, filename)


@router.get("/liftoff_height/{action_id}/download")
async def download_liftoff_height(action_id: int, session: SessionDep = SessionDep):
    """Download liftoff height data as Excel file"""
    data = await get_average_liftoff_height(action_id, session)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"liftoff_height_action_{action_id}_{timestamp}.xlsx"

    return await create_excel_response(data, filename)


@router.get("/all_metrics/{action_id}/download")
async def download_all_metrics(action_id: int, session: SessionDep = SessionDep):
    """Download all metrics data as Excel file with multiple sheets"""

    # Get all data from different APIs
    hip_degree_data = await get_average_step_hip_degree(action_id, session)
    step_length_data = await get_average_step_length(action_id, session)
    step_width_data = await get_average_step_width(action_id, session)
    step_speed_data = await get_average_step_speed(action_id, session)
    step_stride_data = await get_average_step_stride(action_id, session)
    step_difference_data = await get_average_step_difference(action_id, session)
    support_time_data = await get_average_support_time(action_id, session)
    liftoff_height_data = await get_average_liftoff_height(action_id, session)

    # Create Excel file with multiple sheets
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Helper function to convert data to DataFrame
        def data_to_df(data, sheet_name):
            df_data = []
            for key, value in data.items():
                if key not in ["chart_url", "download_url"]:
                    df_data.append({"Metric": key, "Value": value})
            df = pd.DataFrame(df_data)
            df.to_excel(writer, sheet_name=sheet_name, index=False)

        # Create sheets for each metric
        data_to_df(hip_degree_data, "Hip Degree")
        data_to_df(step_length_data, "Step Length")
        data_to_df(step_width_data, "Step Width")
        data_to_df(step_speed_data, "Step Speed")
        data_to_df(step_stride_data, "Step Stride")
        data_to_df(step_difference_data, "Step Difference")
        data_to_df(support_time_data, "Support Time")
        data_to_df(liftoff_height_data, "Liftoff Height")

        # Create summary sheet
        summary_data = []
        summary_data.append({"Metric": "Hip Degree Average",
                            "Value": hip_degree_data.get("average", 0)})
        summary_data.append(
            {"Metric": "Step Length Average", "Value": step_length_data.get("average", 0)})
        summary_data.append({"Metric": "Step Width Average",
                            "Value": step_width_data.get("average", 0)})
        summary_data.append({"Metric": "Step Speed Average",
                            "Value": step_speed_data.get("average", 0)})
        summary_data.append(
            {"Metric": "Step Stride Average", "Value": step_stride_data.get("average", 0)})
        summary_data.append({"Metric": "Step Difference Average",
                            "Value": step_difference_data.get("average", 0)})
        summary_data.append({"Metric": "Support Time Average",
                            "Value": support_time_data.get("average", 0)})
        summary_data.append({"Metric": "Liftoff Height Average",
                            "Value": liftoff_height_data.get("average", 0)})

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

    output.seek(0)

    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"all_metrics_action_{action_id}_{timestamp}.xlsx"

    # Create streaming response
    response = StreamingResponse(
        io.BytesIO(output.read()),
        media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

    return response
