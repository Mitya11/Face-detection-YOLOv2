from PIL import Image
import torch
from BoundingBoxDataSet import BoundingBoxDataSet
from Custom_transforms import *
from utils import show_bounding_box
from YOLO import YOLO

torch.set_printoptions(sci_mode=False, precision=2)

# Test image
img = Image.open("C://Users/mitya/Pictures/Camera Roll/WIN_20230428_20_19_14_Pro.jpg")
img = Image.open("C://Users/mitya/PycharmProjects/Face-detection-YOLO/ImageDataSet/0--Parade/0_Parade_Parade_0_829.jpg")
width, height = img.size
img = img.resize((224, 224), Image.Resampling.LANCZOS)
tens = transforms.ToTensor()(img)
tens = torch.unsqueeze(tens, dim=0)

# load train data from txt file
dataset = BoundingBoxDataSet(folder="ImageDataSet/",
                             transform=transforms.Compose([ToTensor(), RandomCrop(0.7), Resize((224, 224)),RandomRotate()]))
data = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# initialize network and load weights
a = YOLO()
a.cuda()
a.load_state_dict(torch.load("YOLO224x224"))

a.train(5, data)
torch.save(a.state_dict(), "YOLO224x224")

# check test image
otv = a(tens)
print(otv.size())
print(torch.reshape(otv[0, :, -1], (7, 7)))
mor = torch.argmax(otv[0, :, -1])
output_img = show_bounding_box(img, otv[0, mor, :4])
output_img.show()
