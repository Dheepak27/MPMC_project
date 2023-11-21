import os
from ultralytics import YOLO
import cv2
import time
cap = cv2.VideoCapture('test1.mp4')

model_path = os.path.join('.', 'runs', 'detect', 'train2', 'weights', 'last.pt')

# Load a model
model = YOLO(model_path)  # load a custom model

threshold = 0.4
c=0
t=0
start=0
end=0
while True:    
    ret, frame = cap.read()
    if not ret:
        break
    frame=cv2.resize(frame,(500,300),interpolation=cv2.INTER_AREA)
    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 250), 4)
            cv2.putText(frame,'Working', (int(x1+10), int(y1+30)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)  
            cv2.putText(frame, str(score*100)[0:3], (int(x1), int(y1+70)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
            if c==0:
                start=time.time()
                c=1
        else:
            end=time.time()
            t+=start-end
            c=0
    cv2.imshow('',frame)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
if t==0:
    print((start-end)/3.6e+9," hrs")
else:
    print(t)
cap.release()
cv2.destroyAllWindows()