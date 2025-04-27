from flask import Flask, send_file # EDIT HERE
from flask import jsonify
from flask import request
from flask import send_from_directory
from Model_API import end_to_end
import os
import json

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ANNOTATIONS_FILE = 'annotations.json'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(ANNOTATIONS_FILE):
    with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as file:
        json.load(file)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.config["ANNOTATIONS_FILE"] = ANNOTATIONS_FILE

@app.route("/upload", methods=['POST'])
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file"}), 400

    file = request.files['video']

    if file.filename == '':
        return jsonify({"error": "Video file not selected"}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    return ({"message": "Video upload successful!","filename": file.filename}), 200

@app.route("/videos", methods=['GET'])
def list_videos():
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    return jsonify({"videos": files})

@app.route("/videos/<filename>", methods=['GET'])
def get_video(filename):
    #return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename) # EDIT HERE
    return send_file(video_path, mimetype='video/mp4') # EDIT HERE

def load_annotations():
    if not os.path.exists(ANNOTATIONS_FILE):
        return {}
    with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as file:
        return json.load(file)
    
def save_annotations(data):
    with open(ANNOTATIONS_FILE, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)

@app.route("/annotations/<video_id>", methods=['POST'])
def add_annotations(video_id):
    data = request.json
    timestamp = data.get("timestamp")
    text = data.get("text")
    style = data.get("style", {})

    if timestamp is None or not text:
        return jsonify({"error": "Timestamp and text are required"}), 400

    annotations = load_annotations()
    if video_id not in annotations:
        annotations[video_id] = []

    annotations[video_id].append({"timestamp": timestamp, "text": text, "style": style})
    save_annotations(annotations)

    return jsonify({"message": "Annotation added", "annotation": data})

@app.route("/annotations/<video_id>", methods=["GET"])
def get_annotations(video_id):
    annotations = load_annotations()
    return jsonify(annotations.get(video_id, []))

@app.route("/inference", methods=["POST"])
def run_inference():
    content = request.json
    video_name = content.get("video_name")
    model_path = "C:/Users/ridas/Desktop/TAD/flask-server/videomae_temporal_detector_Big1.pth"
    pickle_path = "C:/Users/ridas/Desktop/TAD/flask-server/mini.pkl"

    if not all([video_name, model_path, pickle_path]):
        return jsonify({"error": "Missing required fields: video_name, model_path, pickle_path"}), 400

    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_name)
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], "annotated" + video_name)

    try:
        end_to_end(
            video_path=video_path,
            model_path=model_path,
            pickle_path=pickle_path,
            outputpath=output_path
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "message": "Inference complete.",
        "outputvideo": "annotated" + video_name
    })

if __name__ == "__main__":
    app.run(debug=True)