import math
from common.utils import get_length_to_show
from fastapi import APIRouter
from models import SessionDep, Stage, StepsInfo, Action, Objects
from pyecharts import options as opts
from pyecharts.charts import Line
from pyecharts.globals import ThemeType

router = APIRouter(tags=["dashboard"], prefix="/dashboard")

toolbox_opts = opts.global_options.ToolBoxFeatureOpts(
    save_as_image={"show": True, "title": "save as image", "type": "png"}
)

# 通用数据查询
def get_objects_and_stages(action_id, session, object_id=None):
    if object_id:
        objects = session.query(Objects).filter(
            Objects.id == object_id, Objects.is_deleted == False, Objects.action_id == action_id
        ).all()
    else:
        objects = session.query(Objects).filter(
            Objects.action_id == action_id, Objects.is_deleted == False
        ).all()
    if not objects:
        return []
    result = []
    for obj in objects:
        stages = session.query(Stage).filter(
            Stage.object_id == obj.id, Stage.is_deleted == False
        ).all()
        result.append((obj, stages))
    return result

# 通用曲线生成
def build_line_chart(x_data, y_datas, y_names, title, y_unit, theme="light", y_colors=None):
    init_opts = opts.InitOpts(theme=ThemeType.DARK) if theme == "dark" else {}
    line = Line(init_opts=init_opts)
    line.add_xaxis(xaxis_data=x_data)
    for idx, y_data in enumerate(y_datas):
        kwargs = dict(series_name=y_names[idx], y_axis=y_data, is_smooth=True)
        if y_colors and idx < len(y_colors):
            kwargs["color"] = y_colors[idx]
        if y_names[idx] in ["最小值", "最大值"]:
            kwargs["areastyle_opts"] = opts.AreaStyleOpts(opacity=0.3)
            kwargs["stack"] = "stack1"
        if "脚" in y_names[idx]:
            kwargs["is_connect_nones"] = True
            kwargs["symbol_size"] = 8
        line.add_yaxis(**kwargs)
    line.set_global_opts(
        title_opts=opts.TitleOpts(title=title),
        tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
        xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=False, axislabel_opts=opts.LabelOpts(rotate=90)),
        yaxis_opts=opts.AxisOpts(type_="value", name=y_unit, name_location="end", name_gap=15, axislabel_opts=opts.LabelOpts(rotate=90)),
        toolbox_opts=opts.ToolboxOpts(feature=toolbox_opts),
    )
    return line

# 通用接口生成
def generate_curve_api(
    path, 
    y_fields, 
    y_names, 
    title, 
    y_unit, 
    theme="light", 
    y_colors=None, 
    x_label_func=None, 
    y_postprocess=None
):
    @router.get(path)
    def api(action_id: int, object_id: int = None, session: SessionDep = SessionDep):
        action = session.query(Action).filter(Action.id == action_id, Action.is_deleted == False).first()
        if not action:
            return {"message": "Action not found", "data": []}, 404
        obj_stage_list = get_objects_and_stages(action_id, session, object_id)
        if not obj_stage_list:
            return {"message": "No objects found for this action", "data": []}, 404
        final_results = []
        for obj, stages in obj_stage_list:
            if not stages:
                continue
            x_data = []
            y_datas = [[] for _ in y_fields]
            step = 1
            for stage in stages:
                steps_info = session.query(StepsInfo).filter(
                    StepsInfo.stage_id == stage.id, StepsInfo.is_deleted == False
                ).all()
                for step_info in steps_info:
                    x_data.append(x_label_func(step) if x_label_func else f"第{step}步")
                    for idx, field in enumerate(y_fields):
                        value = getattr(step_info, field)
                        if y_postprocess:
                            value = y_postprocess(field, value, step_info)
                        y_datas[idx].append(value)
                    step += 1
            chart = build_line_chart(x_data, y_datas, y_names, title, y_unit, theme, y_colors)
            if object_id:
                return {obj.name: chart.dump_options_with_quotes()}
            final_results.append({obj.name: chart.dump_options_with_quotes()})
        return {"data": final_results, "message": "Success"}
    return api

# 生成各类接口
generate_curve_api(
    "/step_hip_degree/{action_id}", 
    ["hip_min_degree", "hip_max_degree"], 
    ["最小值", "最大值"], 
    "髋关节角度范围", "度"
)
generate_curve_api(
    "/step_hip_degree/{action_id}/{object_id}", 
    ["hip_min_degree", "hip_max_degree"], 
    ["最小值", "最大值"], 
    "髋关节角度范围", "度"
)
generate_curve_api(
    "/step_width/{action_id}", 
    ["step_width"], 
    ["步宽"], 
    "步宽", "厘米", 
    y_postprocess=lambda f, v, _: round(v * 100, 2)
)
generate_curve_api(
    "/step_width/{action_id}/{object_id}", 
    ["step_width"], 
    ["步宽"], 
    "步宽", "厘米", 
    y_postprocess=lambda f, v, _: round(v * 100, 2)
)
generate_curve_api(
    "/step_length/{action_id}", 
    ["step_length", "step_length"], 
    ["左脚", "右脚"], 
    "步长", "厘米", "dark", 
    y_postprocess=lambda f, v, s: round(v * 100, 2) if (f == "step_length" and s.front_leg == ("left" if s.front_leg == "left" else "right")) else None,
    x_label_func=None
)
generate_curve_api(
    "/step_length/{action_id}/{object_id}", 
    ["step_length", "step_length"], 
    ["左脚", "右脚"], 
    "步长", "厘米", "dark", 
    y_postprocess=lambda f, v, s: round(v * 100, 2) if (f == "step_length" and s.front_leg == ("left" if s.front_leg == "left" else "right")) else None,
    x_label_func=None
)
generate_curve_api(
    "/step_speed/{action_id}", 
    ["step_speed", "step_speed"], 
    ["=左脚", "=右脚"], 
    "步速", "厘米/秒", "dark", ["blue", "red"],
    y_postprocess=lambda f, v, s: round(v * 100, 2) if (f == "step_speed" and s.front_leg == ("left" if s.front_leg == "left" else "right")) else None
)
generate_curve_api(
    "/step_speed/{action_id}/{object_id}", 
    ["step_speed", "step_speed"], 
    ["=左脚", "=右脚"], 
    "步速", "厘米/秒", "dark", ["blue", "red"],
    y_postprocess=lambda f, v, s: round(v * 100, 2) if (f == "step_speed" and s.front_leg == ("left" if s.front_leg == "left" else "right")) else None
)
generate_curve_api(
    "/step_stride/{action_id}", 
    ["stride_length"], 
    ["步幅"], 
    "步幅", "厘米", 
    y_postprocess=lambda f, v, _: round(v * 100, 2)
)
generate_curve_api(
    "/step_stride/{action_id}/{object_id}", 
    ["stride_length"], 
    ["步幅"], 
    "步幅", "厘米", 
    y_postprocess=lambda f, v, _: round(v * 100, 2)
)
generate_curve_api(
    "/step_difference/{action_id}", 
    ["steps_diff"], 
    ["步长差"], 
    "步长差", "厘米", 
    x_label_func=lambda step: f"第{step} - {step + 1}步",
    y_postprocess=lambda f, v, _: round(v * 100, 2)
)
generate_curve_api(
    "/step_difference/{action_id}/{object_id}", 
    ["steps_diff"], 
    ["步长差"], 
    "步长差", "厘米", 
    x_label_func=lambda step: f"第{step} - {step + 1}步",
    y_postprocess=lambda f, v, _: round(v * 100, 2)
)
generate_curve_api(
    "/support_time/{action_id}", 
    ["support_time"], 
    ["支撑时间"], 
    "支撑时间", "秒"
)
generate_curve_api(
    "/support_time/{action_id}/{object_id}", 
    ["support_time"], 
    ["支撑时间"], 
    "支撑时间", "秒"
)
generate_curve_api(
    "/liftoff_height/{action_id}", 
    ["liftoff_height"], 
    ["离地距离"], 
    "离地距离", "厘米", 
    y_postprocess=lambda f, v, _: round(v * 100, 2)
)
generate_curve_api(
    "/liftoff_height/{action_id}/{object_id}", 
    ["liftoff_height"], 
    ["离地距离"], 
    "离地距离", "厘米", 
    y_postprocess=lambda f, v, _: round(v * 100, 2)
)