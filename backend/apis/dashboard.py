from fastapi import APIRouter
from models import SessionDep, Stage, StepsInfo
from pyecharts import options as opts
from pyecharts.charts import Line
from pyecharts.globals import ThemeType
from common.utils import get_length_to_show

router = APIRouter(tags=["dashboard"], prefix="/dashboard")

toolbox_opts = opts.global_options.ToolBoxFeatureOpts(
    save_as_image={"show": True, "title": "save as image", "type": "png"})



@router.get("/step_hip_degree/{action_id}")
def get_step_hip_degree_overlap(action_id: int, session: SessionDep = SessionDep):
    x_data = []
    y_low_data = []
    y_high_data = []
    step = 1
    stages = session.query(Stage).filter(
        Stage.action_id == action_id, Stage.is_deleted == False).all()
    for stage in stages:
        steps_info = session.query(StepsInfo).filter(
            StepsInfo.stage_id == stage.id, StepsInfo.is_deleted == False).all()
        for step_info in steps_info:
            x_data.append(f"第{step}步")
            y_low_data.append(round(step_info.hip_min_degree, 2))
            y_high_data.append(round(step_info.hip_max_degree, 2))
            step += 1

    line = Line({"theme": "dark"})
    line.add_xaxis(xaxis_data=x_data)

    line.add_yaxis(
        series_name="Low",
        y_axis=y_low_data,
        is_smooth=True,
        areastyle_opts=opts.AreaStyleOpts(opacity=0.3),
        stack="stack1"
    )

    line.add_yaxis(
        series_name="High",
        y_axis=y_high_data,
        is_smooth=True,
        areastyle_opts=opts.AreaStyleOpts(opacity=0.3),
        stack="stack1"
    )

    line.set_global_opts(
        title_opts=opts.TitleOpts(title="髋关节角度范围"),
        tooltip_opts=opts.TooltipOpts(
            trigger="axis", axis_pointer_type="cross"),
        xaxis_opts=opts.AxisOpts(
            type_="category", boundary_gap=False, axislabel_opts=opts.LabelOpts(rotate=90)
        ),
        yaxis_opts=opts.AxisOpts(
            type_="value",
            name="度",
            name_location="end",
            name_gap=15
        ),
        toolbox_opts=opts.ToolboxOpts(feature=toolbox_opts),
        datazoom_opts=[
            opts.DataZoomOpts(
                type_="slider",
                xaxis_index=0,
                start_value=0,
                end_value=min(get_length_to_show() - 1, len(x_data)),
                range_start=0,
                range_end=min(get_length_to_show() - 1, len(x_data)),
            ),
            opts.DataZoomOpts(
                type_="inside",
                xaxis_index=0,
                start_value=0,
                end_value=min(get_length_to_show() - 1, len(x_data)),
                range_start=0,
                range_end=min(get_length_to_show() - 1, len(x_data)),
            )
        ]
    )
    return line.dump_options_with_quotes()

@router.get("/step_hip_degree/raw/{action_id}")
def get_step_hip_degree_raw(action_id: int, session: SessionDep = SessionDep):
    x_data = []
    y_low_data = []
    y_high_data = []
    step = 1
    stages = session.query(Stage).filter(
        Stage.action_id == action_id, Stage.is_deleted == False).all()
    for stage in stages:
        steps_info = session.query(StepsInfo).filter(
            StepsInfo.stage_id == stage.id, StepsInfo.is_deleted == False).all()
        for step_info in steps_info:
            x_data.append(f"第{step}步")
            y_low_data.append(round(step_info.hip_min_degree, 2))
            y_high_data.append(round(step_info.hip_max_degree, 2))
            step += 1
    return {"x_data": x_data, "y_low_data": y_low_data, "y_high_data": y_high_data}

@router.get("/step_length/{action_id}")
def get_step_length(action_id: int, session: SessionDep = SessionDep):
    x_data = []
    y_left = []
    y_right = []
    step = 1
    stages = session.query(Stage).filter(
        Stage.action_id == action_id, Stage.is_deleted == False).all()
    for stage in stages:
        steps_info = session.query(StepsInfo).filter(
            StepsInfo.stage_id == stage.id, StepsInfo.is_deleted == False).all()
        for step_info in steps_info:
            x_data.append(f"第{step}步")
            if step_info.front_leg == "left":
                y_left.append(round(step_info.step_length, 2))
                y_right.append(None)
            else:
                y_left.append(None)
                y_right.append(round(step_info.step_length, 2))
            step += 1
    line = Line(init_opts=opts.InitOpts(theme=ThemeType.DARK))
    line.add_xaxis(xaxis_data=x_data)
    line.add_yaxis(series_name="左脚", y_axis=y_left,
                   is_smooth=True, is_connect_nones=True,  symbol_size=8)
    line.add_yaxis(series_name="右脚", y_axis=y_right,
                   is_smooth=True, is_connect_nones=True,  symbol_size=8)
    line.set_global_opts(
        title_opts=opts.TitleOpts(title="步长"),
        xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=90)),
        toolbox_opts=opts.ToolboxOpts(feature=toolbox_opts),
        datazoom_opts=[
            opts.DataZoomOpts(
                type_="slider",
                xaxis_index=0,
                start_value=0,
                end_value=min(get_length_to_show() - 1, len(x_data)),
                range_start=0,
                range_end=min(get_length_to_show() - 1, len(x_data)),
            ),
            opts.DataZoomOpts(
                type_="inside",
                xaxis_index=0,
                start_value=0,
                end_value=min(get_length_to_show() - 1, len(x_data)),
                range_start=0,
                range_end=min(get_length_to_show() - 1, len(x_data)),
            )
        ],
        yaxis_opts=opts.AxisOpts(
            name="像素",
            name_location="end",
            name_gap=15
        )
    )
    return line.dump_options_with_quotes()

@router.get("/step_length/raw/{action_id}")
def get_step_length_raw(action_id: int, session: SessionDep = SessionDep):
    x_data = []
    y_left = []
    y_right = []
    step = 1
    stages = session.query(Stage).filter(
        Stage.action_id == action_id, Stage.is_deleted == False).all()
    for stage in stages:
        steps_info = session.query(StepsInfo).filter(
            StepsInfo.stage_id == stage.id, StepsInfo.is_deleted == False).all()
        for step_info in steps_info:
            x_data.append(f"第{step}步")
            if step_info.front_leg == "left":
                y_left.append(round(step_info.step_length, 2))
                y_right.append(None)
            else:
                y_left.append(None)
                y_right.append(round(step_info.step_length, 2))
            step += 1
    return {"x_data": x_data, "y_left": y_left, "y_right": y_right}


@router.get("/step_speed/{action_id}")
def get_speed(action_id: int, session: SessionDep = SessionDep):
    x_data = []
    y_left_data = []
    y_right_data = []
    step = 1
    stages = session.query(Stage).filter(
        Stage.action_id == action_id, Stage.is_deleted == False).all()
    for stage in stages:
        steps_info = session.query(StepsInfo).filter(
            StepsInfo.stage_id == stage.id, StepsInfo.is_deleted == False).all()
        for step_info in steps_info:
            x_data.append(f"第{step}步")
            if step_info.front_leg == "left":
                y_left_data.append(round(step_info.step_speed, 2))
                y_right_data.append(None)
            else:
                y_left_data.append(None)
                y_right_data.append(round(step_info.step_speed, 2))
            step += 1
    line = Line(init_opts=opts.InitOpts(theme=ThemeType.DARK))
    line.add_xaxis(xaxis_data=x_data)
    line.add_yaxis(series_name="=左脚", y_axis=y_left_data, color="blue",
                   is_smooth=True, is_connect_nones=True,  symbol_size=8)
    line.add_yaxis(series_name="=右脚", y_axis=y_right_data, color="red",
                   is_smooth=True, is_connect_nones=True,  symbol_size=8)
    line.set_global_opts(
        title_opts=opts.TitleOpts(title="步速"),
        xaxis_opts=opts.AxisOpts(
            axislabel_opts=opts.LabelOpts(rotate=90)),
        toolbox_opts=opts.ToolboxOpts(feature=toolbox_opts),
        datazoom_opts=[
            opts.DataZoomOpts(
                type_="slider",
                xaxis_index=0,
                start_value=0,
                end_value=min(get_length_to_show() - 1, len(x_data)),
                range_start=0,
                range_end=min(get_length_to_show() - 1, len(x_data)),
            ),
            opts.DataZoomOpts(
                type_="inside",
                xaxis_index=0,
                start_value=0,
                end_value=min(get_length_to_show() - 1, len(x_data)),
                range_start=0,
                range_end=min(get_length_to_show() - 1, len(x_data)),
            )
        ],
        yaxis_opts=opts.AxisOpts(
            name="像素/秒",
            name_location="end",
            name_gap=15
        )
    )
    return line.dump_options_with_quotes()

@router.get("/step_speed/raw/{action_id}")
def get_speed_raw(action_id: int, session: SessionDep = SessionDep):
    x_data = []
    y_left_data = []
    y_right_data = []
    step = 1
    stages = session.query(Stage).filter(
        Stage.action_id == action_id, Stage.is_deleted == False).all()
    for stage in stages:
        steps_info = session.query(StepsInfo).filter(
            StepsInfo.stage_id == stage.id, StepsInfo.is_deleted == False).all()
        for step_info in steps_info:
            x_data.append(f"第{step}步")
            if step_info.front_leg == "left":
                y_left_data.append(round(step_info.step_speed, 2))
                y_right_data.append(None)
            else:
                y_left_data.append(None)
                y_right_data.append(round(step_info.step_speed, 2))
            step += 1
    return {"x_data": x_data, "y_left_data": y_left_data, "y_right_data": y_right_data}


@router.get("/step_stride/{action_id}")
def get_step_stride(action_id: int, session: SessionDep = SessionDep):
    x_data = []
    y_data = []
    step = 1
    stages = session.query(Stage).filter(
        Stage.action_id == action_id, Stage.is_deleted == False).all()
    for stage in stages:
        steps_info = session.query(StepsInfo).filter(
            StepsInfo.stage_id == stage.id, StepsInfo.is_deleted == False).all()
        for step_info in steps_info:
            # if step_info.first_step:
            #     step += 1
            #     continue
            x_data.append(f"第{step}步")
            y_data.append(round(step_info.stride_length, 2))
            step += 1
    line = Line()
    line.add_xaxis(xaxis_data=x_data)
    line.add_yaxis(series_name="步幅", y_axis=y_data, is_smooth=True)
    line.set_global_opts(
        title_opts=opts.TitleOpts(title="=步幅"),
        xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=90)),
        toolbox_opts=opts.ToolboxOpts(feature=toolbox_opts),
        datazoom_opts=[
            opts.DataZoomOpts(
                type_="slider",
                xaxis_index=0,
                start_value=0,
                end_value=min(get_length_to_show() - 1, len(x_data)),
                range_start=0,
                range_end=min(get_length_to_show() - 1, len(x_data)),
            ),
            opts.DataZoomOpts(
                type_="inside",
                xaxis_index=0,
                start_value=0,
                end_value=min(get_length_to_show() - 1, len(x_data)),
                range_start=0,
                range_end=min(get_length_to_show() - 1, len(x_data)),
            )
        ],
        yaxis_opts=opts.AxisOpts(
            name="像素",
            name_location="end",
            name_gap=15,
            axislabel_opts=opts.LabelOpts(rotate=90)
        )
    )
    return line.dump_options_with_quotes()

@router.get("/step_stride/raw/{action_id}")
def get_step_stride_raw(action_id: int, session: SessionDep = SessionDep):
    x_data = []
    y_data = []
    step = 1
    stages = session.query(Stage).filter(
        Stage.action_id == action_id, Stage.is_deleted == False).all()
    for stage in stages:
        steps_info = session.query(StepsInfo).filter(
            StepsInfo.stage_id == stage.id, StepsInfo.is_deleted == False).all()
        for step_info in steps_info:
            # if step_info.first_step:
            #     step += 1
            #     continue
            x_data.append(f"第{step}步")
            y_data.append(round(step_info.stride_length, 2))
            step += 1
    return {"x_data": x_data, "y_data": y_data}


@router.get("/step_difference/{action_id}")
def get_step_difference(action_id: int, session: SessionDep = SessionDep):
    x_data = []
    y_data = []
    step = 1
    stages = session.query(Stage).filter(
        Stage.action_id == action_id, Stage.is_deleted == False).all()
    for stage in stages:
        steps_info = session.query(StepsInfo).filter(
            StepsInfo.stage_id == stage.id, StepsInfo.is_deleted == False).all()
        for step_info in steps_info:
            # if step_info.first_step:
            #     step += 1
            #     continue
            x_data.append(f"第{step} - {step + 1}步")
            y_data.append(round(step_info.steps_diff, 2))
            step += 1
    line = Line()
    line.add_xaxis(xaxis_data=x_data)
    line.add_yaxis(series_name="步长差", y_axis=y_data, is_smooth=True)
    line.set_global_opts(
        title_opts=opts.TitleOpts(title="步长差"),
        xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=90)),
        toolbox_opts=opts.ToolboxOpts(feature=toolbox_opts),
        datazoom_opts=[
            opts.DataZoomOpts(
                type_="slider",
                xaxis_index=0,
                start_value=0,
                end_value=min(get_length_to_show() - 1, len(x_data)),
                range_start=0,
                range_end=min(get_length_to_show() - 1, len(x_data)),
            ),
            opts.DataZoomOpts(
                type_="inside",
                xaxis_index=0,
                start_value=0,
                end_value=min(get_length_to_show() - 1, len(x_data)),
                range_start=0,
                range_end=min(get_length_to_show() - 1, len(x_data)),
            )
        ],
        yaxis_opts=opts.AxisOpts(
            name="像素",
            name_location="end",
            name_gap=15,
            axislabel_opts=opts.LabelOpts(rotate=90)
        )
    )
    return line.dump_options_with_quotes()

@router.get("/step_difference/raw/{action_id}")
def get_step_difference_raw(action_id: int, session: SessionDep = SessionDep):
    x_data = []
    y_data = []
    step = 1
    stages = session.query(Stage).filter(
        Stage.action_id == action_id, Stage.is_deleted == False).all()
    for stage in stages:
        steps_info = session.query(StepsInfo).filter(
            StepsInfo.stage_id == stage.id, StepsInfo.is_deleted == False).all()
        for step_info in steps_info:
            # if step_info.first_step:
            #     step += 1
            #     continue
            x_data.append(f"第{step} - {step + 1}步")
            y_data.append(round(step_info.steps_diff, 2))
            step += 1
    return {"x_data": x_data, "y_data": y_data}


@router.get("/average/step_hip_degree/{action_id}")
def get_average_step_hip_degree(action_id: int, session: SessionDep = SessionDep):
    step_hip_degree_low = []
    step_hip_degree_high = []
    stages = session.query(Stage).filter(
        Stage.action_id == action_id, Stage.is_deleted == False).all()
    for stage in stages:
        steps_info = session.query(StepsInfo).filter(
            StepsInfo.stage_id == stage.id, StepsInfo.is_deleted == False).all()
        for step_info in steps_info:
            step_hip_degree_low.append(step_info.hip_min_degree)
            step_hip_degree_high.append(step_info.hip_max_degree)
    low_average = round(sum(step_hip_degree_low) / len(step_hip_degree_low), 2)
    high_average = round(sum(step_hip_degree_high) /
                         len(step_hip_degree_high), 2)
    average = round((sum(step_hip_degree_low) + sum(step_hip_degree_high)) /
                    (len(step_hip_degree_low) + len(step_hip_degree_high)), 2)
    return {"low_average": low_average, "high_average": high_average, "average": average}


@router.get("/average/step_length/{action_id}")
def get_average_step_length(action_id: int, session: SessionDep = SessionDep):
    left_step_length = []
    right_step_length = []
    stages = session.query(Stage).filter(
        Stage.action_id == action_id, Stage.is_deleted == False).all()
    for stage in stages:
        steps_info = session.query(StepsInfo).filter(
            StepsInfo.stage_id == stage.id, StepsInfo.is_deleted == False).all()
        for step_info in steps_info:
            if step_info.front_leg == "left":
                left_step_length.append(step_info.step_length)
            else:
                right_step_length.append(step_info.step_length)
    left_average = round(sum(left_step_length) / len(left_step_length), 2)
    right_average = round(sum(right_step_length) / len(right_step_length), 2)
    average = round((sum(left_step_length) + sum(right_step_length)) /
                    (len(left_step_length) + len(right_step_length)), 2)
    return {"left_average": left_average, "right_average": right_average, "average": average}


@router.get("/average/step_speed/{action_id}")
def get_average_step_speed(action_id: int, session: SessionDep = SessionDep):
    left_step_speed = []
    right_step_speed = []
    stages = session.query(Stage).filter(
        Stage.action_id == action_id, Stage.is_deleted == False).all()
    for stage in stages:
        steps_info = session.query(StepsInfo).filter(
            StepsInfo.stage_id == stage.id, StepsInfo.is_deleted == False).all()
        for step_info in steps_info:
            if step_info.front_leg == "left":
                left_step_speed.append(step_info.step_speed)
            else:
                right_step_speed.append(step_info.step_speed)
    left_average = round(sum(left_step_speed) / len(left_step_speed), 2)
    right_average = round(sum(right_step_speed) / len(right_step_speed), 2)
    average = round((sum(left_step_speed) + sum(right_step_speed)) /
                    (len(left_step_speed) + len(right_step_speed)), 2)
    return {"left_average": left_average, "right_average": right_average, "average": average}


@router.get("/average/step_stride/{action_id}")
def get_average_step_stride(action_id: int, session: SessionDep = SessionDep):
    step_stride = []
    stages = session.query(Stage).filter(
        Stage.action_id == action_id, Stage.is_deleted == False).all()
    for stage in stages:
        steps_info = session.query(StepsInfo).filter(
            StepsInfo.stage_id == stage.id, StepsInfo.is_deleted == False, StepsInfo.first_step == False).all()
        step_stride.extend(step_info.stride_length for step_info in steps_info)
    average = round(sum(step_stride) / len(step_stride), 2)
    return {"average": average}


@router.get("/average/step_difference/{action_id}")
def get_average_step_difference(action_id: int, session: SessionDep = SessionDep):
    step_difference = []
    stages = session.query(Stage).filter(
        Stage.action_id == action_id, Stage.is_deleted == False).all()
    for stage in stages:
        steps_info = session.query(StepsInfo).filter(
            StepsInfo.stage_id == stage.id, StepsInfo.is_deleted == False, StepsInfo.first_step == False).all()
        step_difference.extend(
            step_info.steps_diff for step_info in steps_info)
    average = round(sum(step_difference) / len(step_difference), 2)
    return {"average": average}
