import os
import sys

from flask import Flask
from flask import render_template, request, send_from_directory, flash, url_for
from flask import current_app as app
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = './'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "secret key" 

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

if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
