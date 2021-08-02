from bs4 import BeautifulSoup as bs
import requests
import re
import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw

def consult_plate(placa,save_csv=False):
    def __get_info():
        infos_to_get = ["Marca:","Modelo:","Importado:","Ano:","Cor:","Cilindrada:",
                  "Potencia:","Combustível:","UF:","Município:"]
        infos_obtained = []
        for info in infos_to_get:
            try:
                ret = tds_list[tds_list.index(info) + 1]  
                infos_obtained.append(ret)
            except: infos_obtained.append(None)
        return infos_obtained
    try:
        url = "https://www.keplaca.com/placa/"
            
        headers = {"user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36 Edg/92.0.902.55"}
        req = requests.get(url + placa,headers=headers)
        
        cs = bs(req.text,"lxml")
        tds_list = []
        tds = cs.findAll("td")
        
        for td in tds: tds_list.append(td.text)
        
        venal = cs.findAll("table")[1].findAll("td")[2].text.split(" ")[-1].replace(".","").replace(",",".")
        marca, modelo, importado, ano, cor, cilindrada, potencia, combustivel, uf, municipio = __get_info()
        modelo =  modelo.replace(",",".")
        cilindrada = cilindrada.replace("cc","").strip() if cilindrada != None else None
        potencia = potencia.replace("cv","").strip() if potencia != None else None
        placa_resource = cs.findAll("img")[1]['data-src']
        
        info = [venal,marca,modelo,importado,ano,cor,cilindrada,potencia,combustivel,uf,municipio,placa,placa_resource]
        return info
    except: return False
    
def treat_plate(plate_content):
    try:
        forbiden_symbols = [":",",",".","'",'"',";","[","{","]","}","\\","/"," ","-","="]
        for symbol in forbiden_symbols:
            plate_content = plate_content.replace(symbol,"")
        plate_content = plate_content.strip()
        re_plate = "[A-Z]{3}[0-9][0-9A-Z][0-9]{2}"
        if re.match(re_plate, plate_content) and len(plate_content) == 7:
            return plate_content
        else: None
    except: return None

def make_draw(image,position,plate_resource,venal,marca,cor):
    #reading images
    square = 'resource/square.png'
    headers = {"user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36 Edg/92.0.902.55"}
    req = requests.get(plate_resource,headers=headers).content
    plate = 'plate.webp'
    with open(plate, 'wb+') as f: f.write(req)
    square = cv2.imread(square)
    plate = cv2.imread(plate)
    
    #scale
    gen_scale_percent = 300
    
    gen_width = int(square.shape[1] * gen_scale_percent / 100)
    gen_height = int(square.shape[0] * gen_scale_percent / 100)    
    square = cv2.resize(square,(gen_width,gen_height))
    
    gen_width = int(plate.shape[1] * gen_scale_percent / 100)
    gen_height = int(plate.shape[0] * gen_scale_percent / 100)    
    plate = cv2.resize(plate,(gen_width,gen_height))        
    
    #reference x,y objs
    reference_square_x_y = [600 * -1,500 * -1] 
    reference_plate_x_y = (25,25)
    
    #boxe - plate detected
    starting_point = (position[0], position[1])
    ending_point = (position[0] + position[-2], position[1]+position[-1])
    
    #position content
    position_content_hor = "left"
    position_content_ver = "top"
    
    image_h, image_w, image_c = image.shape
    half_image_h = int(image_w / 2)
    half_image_v = int(image_h / 2)
    if starting_point[0] < half_image_h: position_content_hor = "right"
    if starting_point[1] < half_image_v: position_content_ver = "bottom"
    
    #start and end arrow
    start_arrow_x = starting_point[0] - 10 if position_content_hor == "left" else ending_point[0] + 10
    start_arrow_y = starting_point[1] - 10 if position_content_ver == "top" else  ending_point[1] + 10 
    
    arrow_size = 130
    end_arrow_x = starting_point[0] - arrow_size if position_content_hor == "left" else ending_point[0] + arrow_size
    end_arrow_y = starting_point[1] - arrow_size if position_content_ver == "top" else ending_point[1] + arrow_size
    
    start_arrow = (start_arrow_x,start_arrow_y) 
    end_arrow = (end_arrow_x,end_arrow_y)
    
    #position objs
    x_square = end_arrow[0] + reference_square_x_y[0] if position_content_hor == "left" else end_arrow[0] + 35
    y_square = end_arrow[1] + reference_square_x_y[1] if position_content_ver == "top" else end_arrow[1] - 130
    x_y_square = (x_square, y_square)
    x_y_plate = (x_y_square[0] + reference_plate_x_y[0],x_y_square[1] + reference_plate_x_y[1])
    
    #resize plate
    square_h, square_w, square_c = square.shape
    scale_percent = 29
    plate_h, plate_w, plate_c = plate.shape
    width = int(plate_w * scale_percent / 100)
    height = int(plate_h * scale_percent / 100)
    dim = (width,height)
    plate = cv2.resize(plate,dim)
    plate_h, plate_w, plate_c = plate.shape
    
    #inserting text
    def put_text(image,position,text,size=72,color_text=(0,0,0)):
        im_pil = Image.fromarray(image)
        font = ImageFont.truetype("resource/font.ttf", size=size)
        draw = ImageDraw.Draw(im_pil)
        draw.text(position, text, fill=color_text, font=font)
        image = np.asarray(im_pil)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    square = put_text(square, (25,300),marca,size=64 if len(marca) <=20 else 50)
    square = put_text(square, (25,400), venal)
    square = put_text(square, (25,500), cor)
    
    #drawing
    cv2.arrowedLine(image, start_arrow,end_arrow, (0,255,0), 7,tipLength = 0.2)
    try:
        image[x_y_square[1]:x_y_square[1]+square_h,x_y_square[0]:x_y_square[0]+square_w] = square
        image[x_y_plate[1]:x_y_plate[1]+plate_h,x_y_plate[0]:x_y_plate[0]+plate_w] = plate
    except:
        pass
    return image


# plate_content = treat_plate("SKD4ll8")
# if plate_content: print("OK")

# plates = ['CKI1191',"EUB3011","FTC1J32","CZI7787","GFP8G08","EPR1425"]
# plates_info = []
# for plate in plates:    
#     info = consult_plate(plate)
#     plates_info.append(info)
    
    
# df = pd.DataFrame(plates_info,columns=["venal","marca","modelo","importado","ano",
#                                  "cor","cilindrada","potencia",
#                                  "combustivel","uf","municipio","placa","placa_resouce"
#                                  ])

# df.to_csv("consult_car.csv",sep=",",index=False,encoding="utf-8-sig")
    
    
    
    
    
    