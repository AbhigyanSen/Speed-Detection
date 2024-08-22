import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import*
import time

model=YOLO('yolov8m.pt')

class_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

tracker=Tracker()
count=0

cap=cv2.VideoCapture('Speed-detection-of-vehicles\\highway.mp4')


while True:    
    ret,frame = cap.read()
    if not ret:
        break
    count += 1
    frame=cv2.resize(frame,(1020,500))
   

    results=model.predict(frame)
 #   print(results)
    a=results[0].boxes.data
    a = a.detach().cpu().numpy() 
    px=pd.DataFrame(a).astype("float")
    #print(px)

    list=[]
             
    for index,row in px.iterrows():
#        print(row) 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        if 'car' in c:
            list.append([x1,y1,x2,y2])
            #print(c)

    bbox_id=tracker.update(list)
    #print(bbox_id)
    for bbox in bbox_id:
        x3,y3,x4,y4,id=bbox
        cx=int(x3+x4)//2
        cy=int(y3+y4)//2

        red_line_y=198
        blue_line_y=268   
        offset = 7

        cv2.circle(frame,(cx,cy),4,(0,0,255),-1) #draw ceter points of bounding box
        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)  # Draw bounding box
        cv2.putText(frame,str(id),(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
                 
    

    text_color = (255,255,255)  # white color for text
    red_color = (0, 0, 255)  # (B, G, R)   
    blue_color = (255, 0, 0)  # (B, G, R)
    green_color = (0, 255, 0)  # (B, G, R)  

    cv2.line(frame,(172,198),(774,198),red_color,3)  #  starting cordinates and end of line cordinates
    cv2.putText(frame,('red line'),(172,198),cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
    
    cv2.line(frame,(8,268),(927,268),blue_color,3)  # seconde line
    cv2.putText(frame,('blue line'),(8,268),cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)    


    cv2.imshow("frames", frame)
    if cv2.waitKey(0)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()