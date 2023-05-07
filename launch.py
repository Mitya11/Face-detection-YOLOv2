from PIL import Image,ImageDraw
import torchvision
import torch
import time
from torchvision import transforms
from BoundingBoxDataSet import BoundingBoxDataSet
def show_bounding_box(path,rect):
    img = Image.open("ImageDataSet/"+path)
    draw = ImageDraw.Draw(img,"RGBA")
    draw.rectangle((rect[0]-rect[2]/2,rect[1]-rect[3]/2,rect[0]+rect[2]/2,rect[1]+rect[3]/2))
    img.show()


#show_bounding_box('1--Handshaking/1_Handshaking_Handshaking_1_275.jpg',train_data['1--Handshaking/1_Handshaking_Handshaking_1_275.jpg'])

img = Image.open("ImageDataSet/"+'1--Handshaking/1_Handshaking_Handshaking_1_275.jpg')
width, height = img.size
img = img.resize((224,224),Image.Resampling.LANCZOS)
tens = transforms.ToTensor()(img)
tens = torch.unsqueeze(tens,dim=0)

dataset = BoundingBoxDataSet(folder ="ImageDataSet/",transform= transforms.Compose([transforms.ToTensor(),transforms.Resize((224,224))]))

data = torch.utils.data.DataLoader(dataset,batch_size=32,shuffle=True)
im ,label = next(iter(data))
from YOLO import YOLO
a = YOLO()
a.cuda()
a.train(15,data)
torch.save(a.state_dict(),"YOLO224x224")
