import cv2
import os
import pandas as pd
from ultralytics import YOLO
from tracker import Tracker
import time

model = YOLO('yolov8s.pt')

cap = cv2.VideoCapture('Speed-detection-of-vehicles\\highway_mini.mp4')

class_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

count = 0
tracker = Tracker()
down = {}
up = {}
counter_down = []
counter_up = []
vehicle_speeds = {}

red_line_y = 198
blue_line_y = 268
offset = 6

# Create a folder to save frames and violated images
if not os.path.exists('detected_frames2'):
    os.makedirs('detected_frames2')

if not os.path.exists('violate'):
    os.makedirs('violate')

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('Save.avi', fourcc, 20.0, (1020, 500))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 2 != 0:  # Skip every other frame for efficiency
        continue
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
        cx = int((x3 + x4) // 2)
        cy = int((y3 + y4) // 2)

        # GOING DOWN
        if red_line_y < (cy + offset) and red_line_y > (cy - offset):
            down[id] = time.time()  # Current time when vehicle touches the first line
        if id in down:
            if blue_line_y < (cy + offset) and blue_line_y > (cy - offset):
                elapsed_time = time.time() - down[id]  # Current time when vehicle touches the second line minus the time at the first line
                if counter_down.count(id) == 0:
                    counter_down.append(id)
                    distance = 10  # Meters - distance between the 2 lines is 10 meters
                    a_speed_ms = distance / elapsed_time
                    a_speed_kh = a_speed_ms * 3.6  # Convert to kilometers per hour
                    vehicle_speeds[id] = a_speed_kh
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                    cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)  # Draw bounding box
                    cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(frame, str(int(a_speed_kh)) + ' km/h', (x4, y4), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2)
                    if a_speed_kh > 4:  # Speed limit check
                        crop_img = frame[y3:y4, x3:x4]
                        cv2.imwrite(f'violate/vehicle_{id}.jpg', crop_img)

        # GOING UP     
        if blue_line_y < (cy + offset) and blue_line_y > (cy - offset):
            up[id] = time.time()
        if id in up:
            if red_line_y < (cy + offset) and red_line_y > (cy - offset):
                elapsed1_time = time.time() - up[id]
                if counter_up.count(id) == 0:
                    counter_up.append(id)
                    distance1 = 10  # Meters - distance between the 2 lines is 10 meters
                    a_speed_ms1 = distance1 / elapsed1_time
                    a_speed_kh1 = a_speed_ms1 * 3.6
                    vehicle_speeds[id] = a_speed_kh1
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                    cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)  # Draw bounding box
                    cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(frame, str(int(a_speed_kh1)) + ' km/h', (x4, y4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    if a_speed_kh1 > 4:  # Speed limit check
                        crop_img = frame[y3:y4, x3:x4]
                        cv2.imwrite(f'violate/vehicle_{id}.jpg', crop_img)

    text_color = (0, 0, 0)  # Black color for text
    yellow_color = (0, 255, 255)  # Yellow color for background
    red_color = (0, 0, 255)  # Red color for lines
    blue_color = (255, 0, 0)  # Blue color for lines
    gray_color = (128, 128, 128)  # Gray color for rectangle

    # DISPLAY UNIT
    cv2.rectangle(frame, (347, 2), (471, 28), gray_color, -1)
    cv2.rectangle(frame, (517, 2), (641, 28), gray_color, -1)
    # Vehicle Count
    cv2.putText(frame, 'Down Lane: ' + str(len(counter_down)), (350, 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, text_color, 1, cv2.LINE_AA)
    cv2.putText(frame, 'Up Lane: ' + str(len(counter_up)), (520, 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, text_color, 1, cv2.LINE_AA)

    # Save frame
    frame_filename = f'detected_frames2/frame_{count}.jpg'
    cv2.imwrite(frame_filename, frame)

    out.write(frame)

    cv2.imshow("frames", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Auto Next Frame
        break

# Save vehicle speeds to a CSV file
# speeds_df = pd.DataFrame(list(vehicle_speeds.items()), columns=['Vehicle_ID', 'Speed_km_h'])
# speeds_df.to_csv('vehicle_speeds.csv', index=False)

cap.release()
out.release()
cv2.destroyAllWindows()


# APPEND LICENSE PLATE REOCGNITION
# SPEED ISSUE FOR DETECTION