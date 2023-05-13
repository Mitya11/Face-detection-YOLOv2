import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from utils import show_bounding_box
from YOLO import YOLO

a = YOLO()
a.cuda()
torch.no_grad()
a.load_state_dict(torch.load("YOLO224x224"))

cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")
delay = 0
bounding_box =torch.tensor([[[0,0,0,0,0]]])
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=1, fy=1, interpolation=cv2.INTER_AREA)

    img = Image.fromarray(frame)
    img = img.resize((224, 224), Image.Resampling.LANCZOS)

    if delay%8==0:
        img_tens = transforms.ToTensor()(img)
        img_tens = torch.unsqueeze(img_tens, dim=0)
        bounding_box = a(img_tens)
        mor = torch.argmax(bounding_box[0, :, -1])
        print(delay,bounding_box[0,mor,-1].item())

    output_img = show_bounding_box(img, bounding_box[0, mor, :4])
    output_img = output_img.resize((448, 448), Image.Resampling.LANCZOS)

    cv2.imshow('Input', np.array(output_img))
    delay+=1

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()
