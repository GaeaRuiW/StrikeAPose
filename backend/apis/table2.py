from common.utils import calculate_stats
from fastapi import APIRouter
from models import SessionDep, Stage, StepsInfo, Objects

router = APIRouter(tags=["tables"], prefix="/table")

def generic_stat_api(field, url_path, dashboard_path, extra_filter=None, need_front_leg=True):
    @router.get(url_path)
    def stat_api(action_id: int, session: SessionDep = SessionDep):
        final_results = []
        objects_query = session.query(Objects).filter(
            Objects.action_id == action_id, Objects.is_deleted == False
        )
        if not objects_query.first():
            return {"data": [{"unknown": {
                "left_average": 0, "right_average": 0, "average": 0,
                "left_standard_deviation": 0, "right_standard_deviation": 0,
                "standard_deviation": 0, "chart_url": f"{dashboard_path}/{action_id}"
            }}], "message": "No objects found for the given action_id."}
        objects = list(objects_query.all())
        for obj in objects:
            left_values, right_values, all_values = [], [], []
            stages_ids_query = session.query(Stage.id).filter(
                Stage.action_id == action_id, Stage.is_deleted == False
            )
            stage_ids = [id_tuple[0] for id_tuple in stages_ids_query.all()]
            if not stage_ids:
                continue
            steps_info_query = session.query(StepsInfo).filter(
                StepsInfo.stage_id.in_(stage_ids),
                StepsInfo.is_deleted == False
            )
            if extra_filter is not None:
                steps_info_query = steps_info_query.filter(extra_filter)
            for step_info in steps_info_query.all():
                value = getattr(step_info, field, None)
                if value is not None:
                    if need_front_leg:
                        if step_info.front_leg == "left":
                            left_values.append(value)
                        elif step_info.front_leg == "right":
                            right_values.append(value)
                    all_values.append(value)
            left_avg, left_std = calculate_stats(left_values)
            right_avg, right_std = calculate_stats(right_values)
            avg, std = calculate_stats(all_values)
            left_min = min(left_values, default=0)
            left_max = max(left_values, default=0)
            right_min = min(right_values, default=0)
            right_max = max(right_values, default=0)
            min_value = min(all_values, default=0)
            max_value = max(all_values, default=0)
            final_results.append({obj.name: {
                "left_average": left_avg,
                "right_average": right_avg,
                "average": avg,
                "left_standard_deviation": left_std,
                "right_standard_deviation": right_std,
                "standard_deviation": std,
                "left_min_value": left_min,
                "left_max_value": left_max,
                "right_min_value": right_min,
                "right_max_value": right_max,
                "min_value": min_value,
                "max_value": max_value,
                "chart_url": f"{dashboard_path}/{action_id}/{obj.id}"
            }})
        return {"data": final_results, "message": "Success"}
    return stat_api

# 单独处理髋关节角度（有low/high）
@router.get("/step_hip_degree/{action_id}")
def get_average_step_hip_degree(action_id: int, session: SessionDep = SessionDep):
    final_results = []
    objects_query = session.query(Objects).filter(
        Objects.action_id == action_id, Objects.is_deleted == False
    )
    if not objects_query.first():
        return {"data": [{"unknown": {"low_average": 0, "high_average": 0, "average": 0,
                "low_standard_deviation": 0, "high_standard_deviation": 0,
                                      "standard_deviation": 0, "chart_url": f"/dashboard/step_hip_degree/{action_id}"}}], "message": "No objects found for the given action_id."}
    objects = list(objects_query.all())
    for obj in objects:
        data = {
            "left": {"low": [], "high": []},
            "right": {"low": [], "high": []},
            "all": {"low": [], "high": []}
        }
        stages_ids_query = session.query(Stage.id).filter(
            Stage.object_id == obj.id, Stage.is_deleted == False
        )
        stage_ids = [id_tuple[0] for id_tuple in stages_ids_query.all()]
        if not stage_ids:
            continue
        steps_info_query = session.query(StepsInfo).filter(
            StepsInfo.stage_id.in_(stage_ids),
            StepsInfo.is_deleted == False
        )
        for step_info in steps_info_query.all():
            if step_info.hip_min_degree is not None:
                data[step_info.front_leg]["low"].append(
                    step_info.hip_min_degree)
                data["all"]["low"].append(step_info.hip_min_degree)
            if step_info.hip_max_degree is not None:
                data[step_info.front_leg]["high"].append(
                    step_info.hip_max_degree)
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
        all_degrees = data["all"]["low"] + data["all"]["high"]
        avg, std = calculate_stats(all_degrees)
        result["average"] = avg
        result["standard_deviation"] = std
        result["min_value"] = min(all_degrees, default=0)
        result["max_value"] = max(all_degrees, default=0)
        result["chart_url"] = f"/dashboard/step_hip_degree/{action_id}/{obj.id}"
        final_results.append({obj.name: result})
    return {"data": final_results, "message": "Success"}

# 其它统计型接口复用通用函数
generic_stat_api("step_length", "/step_length/{action_id}", "/dashboard/step_length")
generic_stat_api("step_width", "/step_width/{action_id}", "/dashboard/step_width")
generic_stat_api("step_speed", "/step_speed/{action_id}", "/dashboard/step_speed")
generic_stat_api("stride_length", "/step_stride/{action_id}", "/dashboard/step_stride", extra_filter=(getattr(StepsInfo, "first_step") == False))
generic_stat_api("steps_diff", "/step_difference/{action_id}", "/dashboard/step_difference", extra_filter=(getattr(StepsInfo, "first_step") == False))
generic_stat_api("support_time", "/support_time/{action_id}", "/dashboard/support_time")
generic_stat_api("liftoff_height", "/liftoff_height/{action_id}", "/dashboard/liftoff_height")