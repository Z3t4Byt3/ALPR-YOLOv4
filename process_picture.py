from plate_manager import consult_plate, treat_plate, make_draw
import YOLNP
import cv2

all_infos = []
image_path = "IMAGE PATH"

image = cv2.imread(image_path)

detector = YOLNP.Detector()
plate_crops, positions = detector.plate(image,True,False)

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

image_h, image_w, image_c = image.shape
scale_percent = 25
width = int(image_w * scale_percent / 100)
height = int(image_h * scale_percent / 100)
dim = (width,height)
image_copy = cv2.resize(image,dim)

cv2.imshow("Image Resized",image_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()