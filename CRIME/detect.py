from keras.models import load_model  
import cv2 
import numpy as np
from pushbullet import Pushbullet






def notif(msg):
    ph_key="o.4hzgD5SzyhSd7g8JPn4SuV7uj5E2NBZU"
    pb=Pushbullet(ph_key)
    phone = pb.devices[0]
    pb.push_sms(phone, "+917338935190",msg)


np.set_printoptions(suppress=True)


model = load_model("keras_Model.h5", compile=False)


class_names = open("labels.txt", "r").readlines()


camera = cv2.VideoCapture('test2.mp4')
crime=0
normal=0

while True:


    ret, image = camera.read()
    if not ret:
        break
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)


    cv2.imshow("Webcam Image", image)

    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)


    image = (image / 127.5) - 1


    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]


    if(int(str(np.round(confidence_score * 100))[:-2])>=98):
        print("Class:","CRIME", end="")
        crime+=1
    else:
        print("Class:","NORMAL", end="")
        normal+=1
    print()
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
    if cv2.waitKey(2) & 0xFF==ord('q'):
        break

if crime<normal:
    print('RESULT:- NORMAL VIDEO')
else:
    print('RESULT:- SUSPICIOUS ACTIVITY DETECTED')
    notif("CRIME DETECTED ALERT!")
camera.release()
cv2.destroyAllWindows()
