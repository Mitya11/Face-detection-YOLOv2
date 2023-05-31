import torchvision.transforms
from PIL import Image
import torch
from BoundingBoxDataSet import BoundingBoxDataSet
from Custom_transforms import *
from utils import show_bounding_box , found_best_boxes
from YOLO import YOLO
from YOLOv2 import YOLOv2
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

torch.set_printoptions(sci_mode=False, precision=2)
# Test image
img = Image.open("C://Users/mitya/Pictures/Camera Roll/WIN_20230511_23_10_00_Pro.jpg")
#img = Image.open("C://Users/mitya/PycharmProjects/Face-detection-YOLO/ImageDataSet/0--Parade/0_Parade_Parade_0_829.jpg")
#img = Image.open("C://Users/mitya/PycharmProjects/Face-detection-YOLO/ImageDataSet/1--Handshaking/1_Handshaking_Handshaking_1_94.jpg")
width, height = img.size
#img = img.resize((224, 224), Image.Resampling.LANCZOS)
tens = transforms.ToTensor()(img)
tens = torchvision.transforms.Resize((224,224))(tens)
tens = torch.unsqueeze(tens, dim=0)

# load train data from txt file
dataset = BoundingBoxDataSet(folder="ImageDataSet/",
                             transform=transforms.Compose([ToTensor(),RandomCrop(0.8),Resize((224, 224)),RandomRotate() ]))
data = torch.utils.data.DataLoader(dataset, batch_size=48, shuffle=True)

# initialize network and load weights
a = YOLOv2()
a.load_state_dict(torch.load("YOLOv2224x224"))
a.cuda()

a.train(5, data)
torch.save(a.state_dict(), "YOLOv2224x224")

# check test image
otv = a(tens)
print(otv[0][11])
print(torch.reshape(otv[0, :, 9], (a.grid, a.grid)))
print(torch.reshape(otv[0, :, -1], (a.grid, a.grid)))

o = torch.reshape(otv[0, :, -2], (a.grid, a.grid))
mor = torch.argmax(otv[0,:,-2])


boxes = found_best_boxes(otv,a.grid,a.anchors)
print(boxes)

output_img = show_bounding_box(img, boxes)

for i in range(0):
    output_img = show_bounding_box(img, otv[0, i, :4])
    output_img.show()

output_img.show()
