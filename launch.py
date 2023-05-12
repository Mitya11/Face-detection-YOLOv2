from PIL import Image,ImageDraw
import torchvision
import torch
import time
from torchvision import transforms
from BoundingBoxDataSet import BoundingBoxDataSet
from Custom_transforms import *
def show_bounding_box(img,rect):
    width,height = img.size
    draw = ImageDraw.Draw(img,"RGBA")
    draw.rectangle(((rect[0]-rect[2]/2)*width,(rect[1]-rect[3]/2)*height,(rect[0]+rect[2]/2)*width,(rect[1]+rect[3]/2)*height))
    img.show()


#show_bounding_box('1--Handshaking/1_Handshaking_Handshaking_1_275.jpg',train_data['1--Handshaking/1_Handshaking_Handshaking_1_275.jpg'])

img = Image.open("C://Users/mitya/Pictures/Camera Roll/WIN_20230428_20_19_14_Pro.jpg")
#img = Image.open("C://Users/mitya/PycharmProjects/Face-detection-YOLO/ImageDataSet/0--Parade/0_Parade_Parade_0_829.jpg")
width, height = img.size
img = img.resize((224,224),Image.Resampling.LANCZOS)
tens = transforms.ToTensor()(img)
tens = torch.unsqueeze(tens,dim=0)

dataset = BoundingBoxDataSet(folder ="ImageDataSet/",transform= transforms.Compose([ToTensor(),RandomCrop(0.7),Resize((224,224))]))

data = torch.utils.data.DataLoader(dataset,batch_size=32,shuffle=True)
im ,label = next(iter(data))
from YOLO import YOLO
a = YOLO()
a.cuda()

torch.set_printoptions(sci_mode=False,precision=2)

a.load_state_dict(torch.load("YOLO224x224"))

a.train(5,data)
torch.save(a.state_dict(),"YOLO224x224")

otv = a(tens)
print(otv.size())
print(torch.reshape(otv[0,:,-1],(7,7)))
mor = torch.argmax(otv[0,:,-1])
show_bounding_box(img,otv[0,mor,:4])
