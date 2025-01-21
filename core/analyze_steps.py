from flask import Flask, request, jsonify
from rtmpose_trt_inference import inference
import os

app = Flask(__name__)


# REST API for step analysis
@app.route('/analyze_steps', methods=['POST'])
def analyze_steps():
    """
    REST API for analyzing video steps.
    Accepts form-data input with video and parameters.
    """
    try:
        # 获取上传的文件和参数
        if 'video' not in request.files:
            return jsonify({"status": "error", "message": "No video file uploaded"}), 400

        video_file = request.files['video']
        action_id = request.form.get('action_id', 'unknown')
        output_path = request.form.get('output_path', './output')
        diff = int(request.form.get('diff', 3))
        num_circle = int(request.form.get('num_circle', 3))
        smooth_sigma = int(request.form.get('smooth_sigma', 16))
        vis = request.form.get('vis', 'false').lower() == 'true'

        # 确保输出目录存在
        os.makedirs(output_path, exist_ok=True)

        # 保存上传的视频
        video_path = os.path.join(output_path, video_file.filename)
        video_file.save(video_path)

        # 调用 inference 函数
        result = inference(video_path, output_path, diff, smooth_sigma, num_circle, vis)

        if result is not None:
            return jsonify({"status": "success", "action_id": action_id, "result": result})
        else:
            return jsonify({"status": "error", "action_id": action_id,
                            "message": "No result! Please check the csv and log file."}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
