import cv2
import os
import pandas as pd
from ultralytics import YOLO
from tracker import Tracker
import time
import shutil  # Import shutil for directory operations

VIDEOPATH = "/root/Documents/SpeedDetection/Speed-Detection/Version2/InputVideos/highway.mp4"
Filename = input("Enter VideoFile Name: ")
model = YOLO('yolov8m.pt')

# Use correct path for the video
cap = cv2.VideoCapture(VIDEOPATH)

class_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

count = 0
tracker = Tracker()
down = {}
up = {}
counter_down = []
counter_up = []

# Initialize counters for trucks, cars, and buses
truck_count = 0
car_count = 0
bus_count = 0

red_line_y = 198
blue_line_y = 268
offset = 6

# Create a folder to save frames
if not os.path.exists('DetectedFrames'):
    os.makedirs('DetectedFrames')

# Update VideoWriter to save in MP4 format
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'H264'
out = cv2.VideoWriter(f'{Filename}.mp4', fourcc, 20.0, (1020, 500))

# Set to keep track of vehicles already counted
counted_vehicles = set()

def check_crossing_down(cy, cx, x1, y1, x2, y2, id, frame):
    """Check if a vehicle crosses the down lane and calculate speed."""
    global truck_count, car_count, bus_count

    if red_line_y < (cy + offset) and red_line_y > (cy - offset):
        down[id] = time.time()  # Record time at red line
    if id in down:
        if blue_line_y < (cy + offset) and blue_line_y > (cy - offset):
            elapsed_time = time.time() - down[id]
            if id not in counter_down:
                counter_down.append(id)
                distance = 10  # meters
                a_speed_ms = distance / elapsed_time
                a_speed_kh = a_speed_ms * 3.6  # km/h
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box
                cv2.putText(frame, str(id), (x1, y1), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame, f'{int(a_speed_kh)} km/h', (x2, y2), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2)

                # Increment counts based on vehicle type
                class_id = int(px.loc[px[0] == x1].iloc[0][5])
                if 'car' in class_list[class_id]:
                    car_count += 1
                elif 'truck' in class_list[class_id]:
                    truck_count += 1
                elif 'bus' in class_list[class_id]:
                    bus_count += 1

def check_crossing_up(cy, cx, x1, y1, x2, y2, id, frame):
    """Check if a vehicle crosses the up lane and calculate speed."""
    if blue_line_y < (cy + offset) and blue_line_y > (cy - offset):
        up[id] = time.time()
    if id in up:
        if red_line_y < (cy + offset) and red_line_y > (cy - offset):
            elapsed_time = time.time() - up[id]
            if id not in counter_up:
                counter_up.append(id)
                distance = 10  # meters
                a_speed_ms = distance / elapsed_time
                a_speed_kh = a_speed_ms * 3.6
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box
                cv2.putText(frame, str(id), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame, f'{int(a_speed_kh)} km/h', (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

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
    detections = []

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        print(f'Detected: {c} at {x1}, {y1}, {x2}, {y2}')
        if 'car' in c or 'truck' in c or 'bus' in c:
            detections.append([x1, y1, x2, y2])

    bbox_id = tracker.update(detections)

    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2

        if id not in counted_vehicles:
            check_crossing_down(cy, cx, x3, y3, x4, y4, id, frame)
            check_crossing_up(cy, cx, x3, y3, x4, y4, id, frame)
            counted_vehicles.add(id)  # Mark this vehicle as processed

    # Colour Codes
    text_color = (0, 0, 0)  # Black color for text
    gray_color = (128, 128, 128)  # Gray color for rectangle
    red_color = (0, 0, 255)  # Red color for lines
    blue_color = (255, 0, 0)  # Blue color for lines

    # Display
    cv2.rectangle(frame, (347, 2), (471, 28), gray_color, -1)
    cv2.rectangle(frame, (517, 2), (641, 28), gray_color, -1)
    cv2.putText(frame, f'Down Lane: {len(counter_down)}', (350, 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, text_color, 1, cv2.LINE_AA)
    cv2.putText(frame, f'Up Lane: {len(counter_up)}', (520, 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, text_color, 1, cv2.LINE_AA)

    # LINES
    cv2.line(frame, (172, 198), (774, 198), red_color, 2)
    cv2.line(frame, (8, 268), (927, 268), blue_color, 2)

    # Save frame
    frame_filename = f'DetectedFrames/frame_{count}.jpg'
    cv2.imwrite(frame_filename, frame)

    out.write(frame)

# Ensure proper closure
out.release()
cap.release()

# Now check the video file size
video_size = os.path.getsize(f'{Filename}.mp4')
print(f'Size of {Filename}.mp4: {video_size / (1024 * 1024):.2f} MB')

# Print the number of Trucks, Cars, and Buses
print(f'Number of Trucks: {truck_count}')
print(f'Number of Cars: {car_count}')
print(f'Number of Buses: {bus_count}')

# Delete the DetectedFrames folder
# shutil.rmtree('DetectedFrames')

cv2.destroyAllWindows()