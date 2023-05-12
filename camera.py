import cv2
from YOLO import YOLO
import torch
import torchvision.transforms as T
from PIL import Image,ImageDraw
from torchvision import transforms
import numpy as np

def show_bounding_box(img,rect):
    width,height = img.size
    draw = ImageDraw.Draw(img,"RGBA")
    draw.rectangle(((rect[0]-rect[2]/2)*width,(rect[1]-rect[3]/2)*height,(rect[0]+rect[2]/2)*width,(rect[1]+rect[3]/2)*height))
    return img

a = YOLO()
a.cuda()
a.load_state_dict(torch.load("YOLO224x224"))

cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=1, fy=1, interpolation=cv2.INTER_AREA)

    img = Image.fromarray(frame)
    img = img.resize((224, 224), Image.Resampling.LANCZOS)


    img_tens = transforms.ToTensor()(img)
    img_tens = torch.unsqueeze(img_tens, dim=0)

    bounding_box = a(img_tens)
    mor = torch.argmax(bounding_box[0, :, -1])
    output_img = show_bounding_box(img,bounding_box[0, mor, :4])
    output_img = output_img.resize((448, 448), Image.Resampling.LANCZOS)

    cv2.imshow('Input', np.array(output_img))

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()