import os
from ultralytics import YOLO
import numpy as np
import cv2
cap = cv2.VideoCapture('vid2.mp4')
ret,frame=cap.read()
frame=cv2.resize(frame,(500,400),interpolation=cv2.INTER_AREA)
h,w=frame.shape[0],frame.shape[1]

model_path = os.path.join('.', 'runs', 'detect', 'train12', 'weights', 'last.pt')

# Load a model
model = YOLO(model_path)  # load a custom model
c=0
threshold = 0.4
img=np.ones([int(h),int(w)],dtype='uint32')

while ret:    
    results = model(frame)[0]
    c1=0
    if not ret:
        break
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > threshold:
            frame=cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)
            # frame=cv2.putText(frame,'', (int(x1), int(y1 - 10)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,255,0), 3, cv2.LINE_AA)  
            # frame=cv2.putText(frame, str(score*100)[0:3], (int(x1), int(y1 - 10+50)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
            c1+=1
        c=max(c,c1)
        cv2.putText(frame,'COUNT: '+str(c),(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,255,), 3, cv2.LINE_AA)
        img[int(y1):int(y2),int(x1):int(x2)]+=1
        img_norm=(img-img.min())/(img.max()-img)
        img_norm=img_norm.astype('uint8')
        img_norm=cv2.GaussianBlur(img_norm,(5,5),0)
        heatmap=cv2.applyColorMap(img_norm,cv2.COLORMAP_JET)
        final=cv2.addWeighted(heatmap,0.5,frame,0.5,0)
        cv2.imshow("heatmap",heatmap)
        cv2.imshow("final",final)

    # cv2.imshow('',frame)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
    ret, frame = cap.read()
    frame=cv2.resize(frame,(500,400),interpolation=cv2.INTER_AREA)

cap.release()
cv2.destroyAllWindows()