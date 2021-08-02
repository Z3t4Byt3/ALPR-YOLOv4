import YOLNP
from plate_manager import consult_plate,treat_plate,make_draw
import cv2
import time
import numpy as np
from sys import exit as quitt
from os import system as sy

all_infos = []

video_path = "VIDEO_PATH"

cap = cv2.VideoCapture(video_path)
ret, img = cap.read()
mean_time = []
ss = time.time()

try:
    if ret:
        frameSize = (img.shape[1], img.shape[0])
        out = cv2.VideoWriter('output_video.avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, frameSize)
        detector = YOLNP.Detector()
        
        while (cv2.waitKey(1) < 0):
            start = time.time()
            ret, image = cap.read()
            if not ret: break
            
            image_copy = image.copy()
            image_h, image_w, image_c = image.shape
            scale_percent = 10
            width = int(image_w * scale_percent / 100)
            height = int(image_h * scale_percent / 100)
            dim = (width,height)
            image_copy = cv2.resize(image_copy,dim)
            plate_crops, positions = detector.plate(image,False,True)
            
            plates_content = []
            original_text = []
            img_plate_index = 0
            
            if plate_crops:
                for crop in plate_crops:
                    plate_content, score, coordinates = detector.paddle_ocr(crop)
                    plates_content.append((treat_plate(plate_content),img_plate_index))
                    original_text.append(plate_content)
                    img_plate_index += 1
                    
            for plate_content,index in plates_content:
                info = consult_plate(plate_content)
                roi = ((positions[index][0],positions[index][1]),( positions[index][0] + positions[index][-2],positions[index][1] + positions[index][-1]))
                cv2.rectangle(image,roi[0],roi[1],(0,255,0),5)
                if info:
                    all_infos.append(info)
                    plate_resource = info[-1]
                    venal = info[0].replace(".",",")
                    if venal == "Aliquota:": venal = "Não Estimado"
                    else: venal = "R$ " + info[0].replace(".",",")
                    marca = info[1]
                    cor = info[5]
                    image = make_draw(image,positions[index],plate_resource, venal, marca, cor)
                        
            out.write(image)   
            end = time.time()
            mean_time.append(end-start)
            sy("cls")
            print('frame processed in: {:.2f} seconds\naverage processing time: {:.2f} seconds'.format(end - start,np.mean(mean_time)))
       
        out.release()
        
    else:
        raise ValueError("Could not open the file, check the path!")
except:
    out.release()
    quitt()
finally:
    print("Time processed: {:.2f}".format(time.time()-ss))    