#author: Guilherme D. Alves

"""

Deve ser executado dentro da pasta darknet no Google Colab

"""

import requests
import argparse
import os

class Arquivos():
    def __init__(self,classes,train_file_path,valid_file_path,obj_names_file_path,backup_file_path):
        super().__init__()
        self.train_file_path = self.__train_file_path(train_file_path)
        self.valid_file_path = self.__set_valid_file_path(valid_file_path)
        self.__classes = self.__set_classes(classes)
        self.__obj_data = self.__set_obj_data(train_file_path,valid_file_path,obj_names_file_path,backup_file_path)

    def __train_file_path(self,train_file_path):
        if not type(train_file_path) == str:
            raise ValueError('Para definir os caminhos você deve especificar os mesmos em str como: "caminho/do/arquivo"')
        self.train_file_path = train_file_path
        return train_file_path

    def __set_valid_file_path(self,valid_file_path):
        if not type(valid_file_path) == str:
            raise ValueError('Para definir os caminhos você deve especificar os mesmos em str como: "caminho/do/arquivo"')
        self.valid_file_path = valid_file_path
        return valid_file_path

    def __set_obj_data(self,train,valid,names,backup):
        if not type(train) == str and type(valid) == str and type(names) == str and type(backup) == str:
            raise ValueError('Para definir os caminhos você deve especificar os mesmos em str como: "caminho/do/arquivo"')
        qtd_classes = len(self.__classes)
        self.__obj_data = f'classes = {qtd_classes}\ntrain = {train}\nvalid = {valid}\nnames = {names}\nbackup = {backup}'
        return self.__obj_data

    def __set_classes(self,classes):
        if not type(classes) == str:
            raise ValueError("Para definir classes você deve especificar como uma str, separando as mesmas com virgula.")
        self.__classes = classes.split(',')
        return self.__classes

    def get_classes(self):
        return self.__classes

    def get_obj_data(self):
        return self.__obj_data

    def get_yolov4_cfg_custom(self,batch_size=32,width=608,height=608,learning_rate=0.0013):
        max_batches = int(2000 * len(self.__classes))
        steps = (int(max_batches*0.80),int(max_batches*0.90))
        filtros = (len(self.__classes) + 5) * 3

        yolov4_cfg = requests.get("https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg").text.split("\n")

        yolov4_cfg[1] = f'batch={batch_size}'
        yolov4_cfg[2] = 'subdivisions=64'
        yolov4_cfg[6] = f'width={width}'
        yolov4_cfg[7] = f'height={height}'
        yolov4_cfg[16] = f'learning_rate={learning_rate}'
        yolov4_cfg[18] = f'max_batches = {max_batches}'
        yolov4_cfg[20] = f'steps={steps[0]},{steps[1]}'

        yolov4_cfg[967] = f'classes={len(self.__classes)}'
        yolov4_cfg[960] = f'filters={filtros}'
        
        yolov4_cfg[1048] = f'filters={filtros}'
        yolov4_cfg[1055] = f'classes={len(self.__classes)}'

        yolov4_cfg[1136] = f'filters={filtros}'
        yolov4_cfg[1143] = f'classes={len(self.__classes)}'

        return yolov4_cfg

    def criar_arquivos_train_valid_txt(self,valid=False):
        imagens = []
        os.chdir(os.path.join("data", "obj" if not valid else "valid"))
        for filename in os.listdir(os.getcwd()):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                imagens.append(f"data/obj/{filename}" if not valid else f"data/valid/{filename}")
        os.chdir("..")

        with open(self.train_file_path.split("/")[-1] if not valid else self.valid_file_path.split("/")[-1], "w") as outfile:
            for img in imagens:
                outfile.write(img)
                outfile.write("\n")
            outfile.close()
        os.chdir("..")
    
parser = argparse.ArgumentParser(description="Darknet Configuration Files")

parser.add_argument("classes",type=str,help="Classes do seu projeto, separe por virgulas")
parser.add_argument("train_path",type=str,help="Onde será salvo o arquivo train.txt")
parser.add_argument("test_path",type=str,help="Onde será salvo o arquivo test.txt")
parser.add_argument("obj_names",type=str,help="Onde será salvo o arquivo obj.names")
parser.add_argument("obj_data",type=str,help="Onde será salvo o arquivo obj.data")
parser.add_argument("cfg_custom",type=str,help="Onde será salvo o arquivo yolov4_custom.cfg")
parser.add_argument("model_backup",type=str,help="Especifique onde será a pasta de backup para o Darknet salvar seus modelos")

args = parser.parse_args()

classes = args.classes
train_path = args.train_path
test_path = args.test_path
obj_names = args.obj_names
obj_data_path = args.obj_data
cfg_custom = args.cfg_custom
model_backup = args.model_backup

#python manager.py plate data/train.txt data/valid.txt data/obj.names data/obj.data cfg/yolov4_custom.cfg /yolo_backup/

gerenciador = Arquivos(classes,train_path,test_path,obj_names,model_backup)

classes = gerenciador.get_classes()

obj_data = gerenciador.get_obj_data()

yolov4_cfg_custom = gerenciador.get_yolov4_cfg_custom()

with open(obj_names,"w",encoding="utf8") as f:
    for y in classes:
        f.write(y+"\n")

with open(obj_data_path,"w",encoding="utf8") as f:
    f.write(obj_data)

with open(cfg_custom,"w",encoding="utf8") as f:
    for line in yolov4_cfg_custom:
        f.write(line+"\n")

gerenciador.criar_arquivos_train_valid_txt()
gerenciador.criar_arquivos_train_valid_txt(True)