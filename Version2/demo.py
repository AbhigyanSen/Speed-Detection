import cv2
import os
import pandas as pd
from ultralytics import YOLO
from tracker import Tracker
import time

model = YOLO('yolov8m.pt')

# Use correct path for the video
cap = cv2.VideoCapture('/root/Documents/SpeedDetection/Speed-Detection/Version2/InputVideos/highway.mp4')

class_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

count = 0
tracker = Tracker()
down = {}
up = {}
counter_down = []
counter_up = []

red_line_y = 198
blue_line_y = 268
offset = 6

# Create a folder to save frames
if not os.path.exists('DetectedFrames'):
    os.makedirs('DetectedFrames')

# Update VideoWriter to save in MP4 format
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'H264'
out = cv2.VideoWriter('DEMO.mp4', fourcc, 20.0, (1020, 500))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)
    a = results[0].boxes.data
    a = a.detach().cpu().numpy()
    px = pd.DataFrame(a).astype("float")
    list = []

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'car' in c:
            list.append([x1, y1, x2, y2])
    bbox_id = tracker.update(list)

    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2

        if red_line_y < (cy + offset) and red_line_y > (cy - offset):
            down[id] = time.time()  # current time when vehicle touches the first line
        if id in down:
            if blue_line_y < (cy + offset) and blue_line_y > (cy - offset):
                elapsed_time = time.time() - down[id]  # current time when vehicle touches the second line
                if id not in counter_down:
                    counter_down.append(id)
                    distance = 10  # meters
                    a_speed_ms = distance / elapsed_time
                    a_speed_kh = a_speed_ms * 3.6  # km/h
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                    cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)  # Draw bounding box
                    cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(frame, str(int(a_speed_kh)) + ' km/h', (x4, y4), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2)

        if blue_line_y < (cy + offset) and blue_line_y > (cy - offset):
            up[id] = time.time()
        if id in up:
            if red_line_y < (cy + offset) and red_line_y > (cy - offset):
                elapsed1_time = time.time() - up[id]
                if id not in counter_up:
                    counter_up.append(id)
                    distance1 = 10  # meters
                    a_speed_ms1 = distance1 / elapsed1_time
                    a_speed_kh1 = a_speed_ms1 * 3.6
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                    cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)  # Draw bounding box
                    cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(frame, str(int(a_speed_kh1)) + ' km/h', (x4, y4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    text_color = (0, 0, 0)  # Black color for text
    yellow_color = (0, 255, 255)  # Yellow color for background
    red_color = (0, 0, 255)  # Red color for lines
    blue_color = (255, 0, 0)  # Blue color for lines
    gray_color = (128,128,128) # Gray color for rectangle

    cv2.rectangle(frame, (347, 2), (471, 28), gray_color, -1)
    cv2.rectangle(frame, (517, 2), (641, 28), gray_color, -1)

    # LINES
    cv2.line(frame, (172, 198), (774, 198), red_color, 2)
    cv2.putText(frame, ('Red Line'), (172, 198), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
    cv2.line(frame, (8, 268), (927, 268), blue_color, 2)
    cv2.putText(frame, ('Blue Line'), (8, 268), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)


    cv2.putText(frame, ('Down Lane: ' + str(len(counter_down))), (350, 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, text_color, 1, cv2.LINE_AA)
    cv2.putText(frame, ('Up Lane: ' + str(len(counter_up))), (520, 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, text_color, 1, cv2.LINE_AA)

    # Save frame
    frame_filename = f'DetectedFrames/frame_{count}.jpg'
    cv2.imwrite(frame_filename, frame)

    out.write(frame)

cap.release()
out.release()

# Print the size of the output video
video_size = os.path.getsize('DEMO.mp4')
print(f'Size of DEMO.mp4: {video_size / (1024 * 1024):.2f} MB')

cv2.destroyAllWindows()