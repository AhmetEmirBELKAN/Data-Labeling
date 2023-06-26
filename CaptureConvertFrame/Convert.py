
import cv2
import os 
import time


IMAGES_PATH='CollectedImages/'

labels=["engine","led","insurance"]
number_imgs=60

vid = cv2.VideoCapture(2)

for label in labels:
    os.mkdir(f"CollectedImages/{label}")
    while(True):
    
        print(f"{label} için kamera açılıyor")
        for imgnum in range(number_imgs):
            ret, frame = vid.read()
            time.sleep(1)
            print(f"imgnum {imgnum}")
            img_name=f"{label}_{imgnum}.jpg"
            
            cv2.imwrite(os.path.join(IMAGES_PATH+label,img_name),frame)
            cv2.imshow('frame', frame)
            cv2.waitKey(1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        break    
            