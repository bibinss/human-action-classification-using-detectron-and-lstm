import cv2
import ntpath
from .utils import filter_persons, draw_keypoints
from .lstm import WINDOW_SIZE

import numpy as np
import torch
import torch.nn.functional as F

# lstm was trained using the keypoints detected using openpose.
# So we need map detectron2 keypoints to OpenPose before feeding into lstm.
openpose_coco_mapping = [0, 6, 8, 10, 5, 7,
                         9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]

LABELS = {
    0: "JUMPING",
    1: "JUMPING_JACKS",
    2: "BOXING",
    3: "WAVING_2HANDS",
    4: "WAVING_1HAND",
    5: "CLAPPING_HANDS"
}

def analyse_video(pose_detector, lstm_classifier, video_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print("fps ", fps)
    print("width height ", width, height)
    tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("tot_frames", tot_frames)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    file_name = ntpath.basename(video_path)
    vid_writer = cv2.VideoWriter('res_{}'.format(
        file_name), fourcc, 30, (width, height))
    counter = 0
    buffer_window = []
    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        counter += 1
        outputs = pose_detector(frame)
        img = frame.copy()
        persons, pIndicies = filter_persons(outputs)
        if len(persons) >= 1:
            p = persons[0]
            draw_keypoints(p, img)

            features = []
            for i, row in enumerate(p):
                features.append(row[0])
                features.append(row[1])

            if len(buffer_window) < WINDOW_SIZE:
                buffer_window.append(features)
            else:
                # convert input to tensor
                model_input = torch.Tensor(np.array(buffer_window, dtype=np.float32))
                model_input = torch.unsqueeze(model_input, dim=0)
                y_pred = lstm_classifier(model_input)
                prob = F.softmax(y_pred, dim=1)
                # get the index of the max probability
                pred_index = prob.data.max(dim=1)[1]
                # pop the first value from buffer_window and add the new entry in FIFO fashion, to have a sliding window of size 32.
                buffer_window.pop(0)
                buffer_window.append(features)
                label = LABELS[pred_index.numpy()[0]]
                #print("Label detected ", label)
                cv2.putText(img, 'Action: {}'.format(label),
                            (int(width-400), height-50), cv2.FONT_HERSHEY_COMPLEX, 0.9, (102, 255, 255), 2)
        vid_writer.write(img)
        percentage = int(counter*100/tot_frames)
        yield "data:" + str(percentage) + "\n\n"
    print("finished video analysis")

def stream_video(video_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print("fps ", fps)
    print("width height", width, height)
    tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("tot_frames", tot_frames)
    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        out_frame = cv2.imencode('.jpg', frame)[1].tobytes()
        result = (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' +
                  out_frame + b'\r\n')
        yield result
    print("finished video streaming")
