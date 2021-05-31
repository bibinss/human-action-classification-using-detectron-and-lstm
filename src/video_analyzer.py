import cv2
import ntpath
from .utils import filter_persons, draw_keypoints

import numpy as np
import torch
import torch.nn.functional as F

coco_mapping = [0, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
WINDOW_SIZE = 32
LABELS = {
    0:"JUMPING",
    1:"JUMPING_JACKS",
    2:"BOXING",
    3:"WAVING_2HANDS",
    4:"WAVING_1HAND",
    5:"CLAPPING_HANDS"
}

def analyse_video(pose_detector, lstm_classifier, video_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print("fps ", fps)
    print("width height", width, height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    file_name = ntpath.basename(video_path)
    vid_writer = cv2.VideoWriter('/content/out_res1_{}'.format(file_name), fourcc, 30, (width, height))
    skip_count = 0
    window = []
    while True:
        ret, frame = cap.read()
        if ret == False:
          break
        # skip_count += 1
        # if(skip_count > 2):
        #     continue
        # skip_count = 0
        outputs = pose_detector(frame)
        img = frame.copy()
        persons, pIndicies = filter_persons(outputs)
        if len(persons) >= 1:
            p = persons[0]
            draw_keypoints(p, img)
            p = p[coco_mapping]

            if (len(p) != 17):
              print("len(persons[0]) ", len(p))

            features = []
            for i, row in enumerate(p):
                features.append(row[0])
                features.append(row[1])
               
            if len(window) < WINDOW_SIZE:
                window.append(features)
            else:
                #print("window shape", window.shape)
                temp = np.array(window, dtype=np.float32)
                temp = torch.Tensor(temp)
                temp = torch.unsqueeze(temp, dim=0)
                #print("temp shape", temp.shape)
                y_pred = lstm_classifier(temp)
                #print("y_pred", y_pred)
                prob = F.softmax(y_pred, dim=1)    
        
                # get the max probability
                pred_prob = prob.data.max(dim=1)[0]
                
                # get the index of the max probability
                pred_index = prob.data.max(dim=1)[1]  
                window.pop(0)
                window.append(features)
                label = LABELS[pred_index.numpy()[0]]
                print("label ", label)
                cv2.putText(img, 'Action: {}'.format(label),
                    (int(width-300), 50), cv2.FONT_HERSHEY_COMPLEX, 0.7, (102, 255, 255), 2)

                
        out_frame = cv2.imencode('.jpg', img)[1].tobytes()
        result = (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + out_frame + b'\r\n')
        yield result
        
        vid_writer.write(img)

