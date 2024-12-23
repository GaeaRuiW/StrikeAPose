from fastapi import APIRouter
from models import (SessionDep, StepHipDegree, StepLength,
                    StepSpeed, StepStride)
from pyecharts import options as opts
from pyecharts.charts import Line
from pyecharts.globals import ThemeType

router = APIRouter(tags=["dashboard"], prefix="/dashboard")

toolbox_opts = opts.global_options.ToolBoxFeatureOpts(
    save_as_image={"show": True, "title": "save as image", "type": "png"})


@router.get("/step_hip_degree/{action_id}")
def get_step_hip_degree_overlap(action_id: int, session: SessionDep = SessionDep):
    step_hip_degree_db = session.query(StepHipDegree).filter(
        StepHipDegree.action_id == action_id, StepHipDegree.is_deleted == False).all()
    step_hip_degree_data = sorted(step_hip_degree_db, key=lambda x: x.step_id)

    x_data = [f"第{x.step_id + 1}步" for x in step_hip_degree_data]
    y_low_data = [round(x.low, 2) for x in step_hip_degree_data]
    y_high_data = [round(x.high, 2) for x in step_hip_degree_data]

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
        yaxis_opts=opts.AxisOpts(type_="value"),
        toolbox_opts=opts.ToolboxOpts(feature=toolbox_opts)
    )

    return line.dump_options_with_quotes()


@router.get("/step_length/{action_id}")
def get_step_length(action_id: int, session: SessionDep = SessionDep):
    step_length_db = session.query(StepLength).filter(
        StepLength.action_id == action_id, StepLength.is_deleted == False).all()
    step_length_data = sorted(step_length_db, key=lambda x: x.step_id)
    x_data = [f"第{x.step_id + 1}步" for x in step_length_data]
    y_left_dict = {x.step_id: round(x.size, 2)
                   for x in step_length_data if x.left_feet}
    y_right_dict = {x.step_id: round(x.size, 2)
                    for x in step_length_data if x.right_feet}
    y_left = [y_left_dict.get(i, None)
              for i in range(len(step_length_data) + 1)]
    y_right = [y_right_dict.get(i, None)
               for i in range(len(step_length_data) + 1)]
    line = Line(init_opts=opts.InitOpts(theme=ThemeType.DARK))
    line.add_xaxis(xaxis_data=x_data)
    line.add_yaxis(series_name="左脚", y_axis=y_left,
                   is_smooth=True, is_connect_nones=True,  symbol_size=8)
    line.add_yaxis(series_name="右脚", y_axis=y_right,
                   is_smooth=True, is_connect_nones=True,  symbol_size=8)
    line.set_global_opts(
        title_opts=opts.TitleOpts(title="步长"),
        xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=90)),
        toolbox_opts=opts.ToolboxOpts(feature=toolbox_opts))
    return line.dump_options_with_quotes()


@router.get("/step_speed/{action_id}")
def get_speed(action_id: int, session: SessionDep = SessionDep):
    speed_db = session.query(StepSpeed).filter(
        StepSpeed.action_id == action_id, StepSpeed.is_deleted == False).all()
    speed_data = sorted(speed_db, key=lambda x: x.step_id)
    x_data = [f"第{x.step_id + 1}步" for x in speed_data]
    y_left_dict = {x.step_id: round(x.size, 2)
                   for x in speed_data if x.left_feet}
    y_right_dict = {x.step_id: round(x.size, 2)
                    for x in speed_data if x.right_feet}
    y_left_data = [y_left_dict.get(i, None)
                   for i in range(len(speed_data) + 1)]
    y_right_data = [y_right_dict.get(i, None)
                    for i in range(len(speed_data) + 1)]
    line = Line(init_opts=opts.InitOpts(theme=ThemeType.DARK))
    line.add_xaxis(xaxis_data=x_data)
    line.add_yaxis(series_name="=左脚", y_axis=y_left_data, color="blue",
                   is_smooth=True, is_connect_nones=True,  symbol_size=8)
    line.add_yaxis(series_name="=右脚", y_axis=y_right_data, color="red",
                   is_smooth=True, is_connect_nones=True,  symbol_size=8)
    line.set_global_opts(title_opts=opts.TitleOpts(title="步速"),
                         xaxis_opts=opts.AxisOpts(
                             axislabel_opts=opts.LabelOpts(rotate=90)),
                         toolbox_opts=opts.ToolboxOpts(feature=toolbox_opts))
    return line.dump_options_with_quotes()


@router.get("/step_stride/{action_id}")
def get_step_stride(action_id: int, session: SessionDep = SessionDep):
    step_stride_db = session.query(StepStride).filter(
        StepStride.action_id == action_id, StepStride.is_deleted == False).all()
    step_stride_data = sorted(step_stride_db, key=lambda x: x.stride_id)
    x_data = [f"第{x.stride_id + 1}步" for x in step_stride_data]
    y_data = [y.size for y in step_stride_data]
    y_data = [round(x, 2) for x in y_data]
    line = Line()
    line.add_xaxis(xaxis_data=x_data)
    line.add_yaxis(series_name="步幅", y_axis=y_data, is_smooth=True)
    line.set_global_opts(
        title_opts=opts.TitleOpts(title="=步幅"),
        xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=90)),
        toolbox_opts=opts.ToolboxOpts(feature=toolbox_opts))
    return line.dump_options_with_quotes()


@router.get("/step_difference/{action_id}")
def get_step_differece(action_id: int, session: SessionDep = SessionDep):
    step_length_db = session.query(StepLength).filter(
        StepLength.action_id == action_id, StepLength.is_deleted == False).all()
    step_length_data = sorted(step_length_db, key=lambda x: x.step_id)
    x_data = [
        f"第{x.step_id + 1} - {x.step_id + 2}步" for x in step_length_data[:-1]]
    y_data = [abs(step_length_data[i + 1].size - step_length_data[i].size)
              for i in range(len(step_length_data) - 1)]
    y_data = [round(x, 2) for x in y_data]
    line = Line()
    line.add_xaxis(xaxis_data=x_data)
    line.add_yaxis(series_name="步长差", y_axis=y_data, is_smooth=True)
    line.set_global_opts(
        title_opts=opts.TitleOpts(title="步长差异"),
        xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=90)),
        toolbox_opts=opts.ToolboxOpts(feature=toolbox_opts)
    )
    return line.dump_options_with_quotes()


@router.get("/average/step_hip_degree/{action_id}")
def get_average_step_hip_degree(action_id: int, session: SessionDep = SessionDep):
    step_hip_degree_db = session.query(StepHipDegree).filter(
        StepHipDegree.action_id == action_id, StepHipDegree.is_deleted == False).all()
    step_hip_degree_low = [x.low for x in step_hip_degree_db]
    step_hip_degree_high = [x.high for x in step_hip_degree_db]
    low_average = round(sum(step_hip_degree_low) / len(step_hip_degree_low), 2)
    high_average = round(sum(step_hip_degree_high) /
                         len(step_hip_degree_high), 2)
    return {"low_average": low_average, "high_average": high_average}


@router.get("/average/step_length/{action_id}")
def get_average_step_length(action_id: int, session: SessionDep = SessionDep):
    step_length_db = session.query(StepLength).filter(
        StepLength.action_id == action_id, StepLength.is_deleted == False).all()
    left_step_length = [x.size for x in step_length_db if x.left_feet]
    right_step_length = [x.size for x in step_length_db if x.right_feet]
    left_average = round(sum(left_step_length) / len(left_step_length), 2)
    right_average = round(sum(right_step_length) / len(right_step_length), 2)
    average = round(sum(x.size for x in step_length_db) /
                    len(step_length_db), 2)
    return {"left_average": left_average, "right_average": right_average, "average": average}


@router.get("/average/step_speed/{action_id}")
def get_average_step_speed(action_id: int, session: SessionDep = SessionDep):
    speed_db = session.query(StepSpeed).filter(
        StepSpeed.action_id == action_id, StepSpeed.is_deleted == False).all()
    left_step_speed = [x.size for x in speed_db if x.left_feet]
    right_step_speed = [x.size for x in speed_db if x.right_feet]
    left_average = round(sum(left_step_speed) / len(left_step_speed), 2)
    right_average = round(sum(right_step_speed) / len(right_step_speed), 2)
    average = round(sum(x.size for x in speed_db) / len(speed_db), 2)
    return {"left_average": left_average, "right_average": right_average, "average": average}


@router.get("/average/step_stride/{action_id}")
def get_average_step_stride(action_id: int, session: SessionDep = SessionDep):
    step_stride_db = session.query(StepStride).filter(
        StepStride.action_id == action_id, StepStride.is_deleted == False).all()
    step_stride = [x.size for x in step_stride_db]
    average = round(sum(step_stride) / len(step_stride), 2)
    return {"average": average}


@router.get("/average/step_difference/{action_id}")
def get_average_step_difference(action_id: int, session: SessionDep = SessionDep):
    step_length_db = session.query(StepLength).filter(
        StepLength.action_id == action_id, StepLength.is_deleted == False).all()
    step_length_data = sorted(step_length_db, key=lambda x: x.step_id)
    step_difference = [abs(step_length_data[i + 1].size - step_length_data[i].size)
                       for i in range(len(step_length_data) - 1)]
    average = round(sum(step_difference) / len(step_difference), 2)
    return {"average": average}
