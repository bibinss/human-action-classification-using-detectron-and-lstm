import os
import sys
import time

from flask import Flask
from flask import render_template, Response, request, send_from_directory, flash, url_for
from flask import current_app as app
from werkzeug.utils import secure_filename

from src.lstm import ActionClassificationLSTM
from src.video_analyzer import analyse_video

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

app = Flask(__name__)
UPLOAD_FOLDER = './'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "secret key" 


# detectron config
start = time.time()
cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
pose_detector = DefaultPredictor(cfg)
model_load_done = time.time()
print("model_load", model_load_done - start)

#lstm
lstm_classifier = ActionClassificationLSTM.load_from_checkpoint("models/saved_model.ckpt")
lstm_classifier.eval()


class DataObject():
    pass


def checkFileType(f: str):
    return f.split('.')[-1] in ['mp4']


def cleanString(v: str):
    out_str = v
    delm = ['_', '-', '.']
    for d in delm:
        out_str = out_str.split(d)
        out_str = " ".join(out_str)
    return out_str

@app.route('/', methods=['GET', 'POST'])
def upload():
    obj = DataObject
    obj.is_video_display = False
    obj.video = ""
    print("files", request.files)
    if request.method == 'POST' and 'video' in request.files:
        video_file = request.files['video']
        if checkFileType(video_file.filename):
            filename = secure_filename(video_file.filename)
            print("filename", filename)
            # save file to /static/uploads
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print("filepath", filepath)
            video_file.save(filepath)
            obj.video = filename
            obj.is_video_display = True
            obj.is_predicted = False
            return render_template('/index.html', obj=obj)
        else:
            if video_file.filename:
                msg = f"{video_file.filename} is not a video file"
            else:
                msg = "Please select a video file"
            flash(msg)
        return render_template('/index.html', obj=obj)
    return render_template('/index.html', obj=obj)

@app.route('/files/<filename>')
def get_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/analyze/<filename>')
def analyze(filename):
    file = analyse_video(pose_detector, lstm_classifier, filename)
    return send_from_directory(app.config['UPLOAD_FOLDER'], file)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
