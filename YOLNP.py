import cv2
import numpy as np
import os
import re
import wget
import requests
import pytesseract
import matplotlib.pyplot as plt
from paddleocr import PaddleOCR


class Detector:
    def __init__(self):
        super().__init__()
        self.image = 0
        self.configPath = 0
        self.weightsPath = 0
        self.threshold = 0.8
        self.threshold_NMS = 0.3
        self.pytesseract_config = '--tessdata-dir tessdata -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 2'
        cfg = requests.get("https://raw.githubusercontent.com/Guiflayrom/ALPR-YOLOv4/master/yolov4_files/yolov4_plate.cfg").text
        with open("config_plate.cfg","w",encoding="utf8") as f:
            f.write(cfg)
        cfg = requests.get("https://raw.githubusercontent.com/Guiflayrom/ALPR-YOLOv4/master/yolov4_files/yolov4_ocr.cfg").text
        with open("config_ocr.cfg","w",encoding="utf8") as f:
            f.write(cfg)   
            
    def paddle_ocr(self,image):
        try:
            self.__set_image(image,read=True if type(image) == np.ndarray else False)
            ocr = PaddleOCR(use_angle_cls=True, lang='en') # need to run only once to download and load model into memory
            result = ocr.ocr(self.image, cls=True)
            plate_content = result[0][1][0]
            score = result[0][1][1]
            coordinates = result[0][0]
            return plate_content, score, coordinates
        except:
            return None,None,None
    
    def plate(self,image,plot=False,pos=False):
        model = "yolov4_files/plate.weights"
        self.__set_image(image,read=True if type(image) == np.ndarray else False)
        crops, positions = self.__model_detect("plate",model)
        if plot and len(crops):
            for crop in crops:
                self.__show_plot(crop)
        return crops, positions
    
    def tesseracr_ocr(self,image,get_only_crops=False,plot=False):    
        model = "yolov4_files/ocr.weights"
        self.__set_image(image,read=True if type(image) == np.ndarray else False)
        crops, _ = self.__model_detect("ocr",model)
        letters = []
        if len(crops) > 0:
            if not get_only_crops:
                for crop in crops:  
                    if plot: self.__show_plot(crop)
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    guassian = cv2.GaussianBlur(gray, (5, 5), 0) 
                    #val, otsu = cv2.threshold(guassian,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                    text = pytesseract.image_to_string(guassian,lang='por',config=self.pytesseract_config)
                    clean_txt = re.sub('[\W_]+','',text)
                    if clean_txt == "" or clean_txt == " ": clean_txt = "1"
                    letters.append(clean_txt)      
                    
                return crops,''.join(letters)
            else:
                return crops
        else: return None

    def __show_plot(self,img):
        try:
            fig = plt.gcf()
            fig.set_size_inches(16, 10)
            plt.axis("off")
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.show()  
        except: pass
    
    def __model_detect(self,cfg_type,model):
  
        self.configPath = "config_plate.cfg" if cfg_type == "plate" else "config_ocr.cfg"
        self.weightsPath = model
        objs, boxes = self.__get_objs_boxes(self.threshold, self.threshold_NMS)
    
        img = cv2.imread(self.image) if type(self.image) != np.ndarray else self.image
        crops = []
        positions = []
        if len(objs) > 0:
          for i in objs.flatten():
            x, y, w, h = self.__get_crop(i, boxes)
            positions.append([x,y,w,h])
        positions.sort()
        for position in positions:
            x, y, w, h = position
            objeto = img[y:y + h, x:x + w]
            crops.append(objeto)
        return crops, positions

    def __get_crop(self,i, boxes):
      (x, y) = (boxes[i][0], boxes[i][1])
      (w, h) = (boxes[i][2], boxes[i][3])
      return x,y,w,h
        
    def __detections(self,detection, _threshold, boxes, confidences, IDclasses, dim):
      (H,W) = dim
      
      scores = detection[5:]      
      classeID = np.argmax(scores)
      conf = scores[classeID]
      
      if conf > _threshold:
          boxe = detection[0:4] * np.array([W, H, W, H])
          (centerX, centerY, width, height) = boxe.astype("int")
          x = int(centerX - (width / 2))
          y = int(centerY - (height / 2))
          boxes.append([x, y, int(width), int(height)])
          confidences.append(float(conf))
          IDclasses.append(classeID)
      return boxes, confidences, IDclasses      

    def __get_objs_boxes(self,threshold,threshold_NMS):
        net = cv2.dnn.readNet(self.configPath, self.weightsPath)
        np.random.seed(42)
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        image = cv2.imread(self.image) if type(self.image) != np.ndarray else self.image
        dim = image.shape[:2] 
        net, image, layerOutputs = self.__blob_image(net,image,ln)
        _threshold = threshold
        _threshold_NMS = threshold_NMS
        boxes = []   
        confidences = []   
        IDclasses = []    
        
        for output in layerOutputs:
            for detection in output:
                boxes, confidences, IDclasses = self.__detections(detection, _threshold, boxes, confidences, IDclasses, dim)    
        self.objs = cv2.dnn.NMSBoxes(boxes, confidences, _threshold, _threshold_NMS)
        
        self.__len_plates_detected = len(self.objs)
        return self.objs,boxes            

    def __blob_image(self,net,image,ln):
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)
        return net, image, layerOutputs           
     
    def __set_image(self,image,read=False):
        if read and type(image) == np.ndarray:
            self.image = image            
        else:
            if not os.path.isfile(image) or re.match('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',image):
                print(image)
                raise ValueError("O arquivo não existe")
    
            self.image = wget.download(image) \
                if re.match(
                  'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                   image) \
                else image
                
def main():
    image = "car3.jpg"
    detector = Detector()
    results = []

    plate_crops = detector.plate(image,plot=True)

    if plate_crops:
        for crop in plate_crops:
            plate_content, score, coordinates = detector.paddle_ocr(crop)
            results.append((plate_content,score,coordinates))
            print(plate_content,score)

    print(results)