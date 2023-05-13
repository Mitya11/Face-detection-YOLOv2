import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class BoundingBoxDataSet(Dataset):
    def __init__(self, folder, labels_file="ImageDataSet/wider_face_val_bbx_gt.txt", transform=None):
        self.folder = folder
        self.dict = dict()
        dan = 0
        with open(labels_file) as file:
            while i := file.readline():
                dan += 1
                if dan > 10000: break
                count = file.readline()
                if int(count) == 1:
                    self.dict[i.rstrip()] = [int(h) for h in file.readline().split(" ")[:4]]
                    self.dict[i.rstrip()][0] += self.dict[i.rstrip()][2] / 2
                    self.dict[i.rstrip()][1] += self.dict[i.rstrip()][3] / 2
                else:
                    for j in range(max(int(count), 1)):
                        file.readline()
        self.transform = transform

    def __len__(self):
        return len(self.dict.keys())

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = Image.open(self.folder + list(self.dict.keys())[idx])
        width, height = image.size
        bounds = torch.tensor(self.dict[list(self.dict.keys())[idx]])
        bounds[0] /= width
        bounds[2] /= width
        bounds[1] /= height
        bounds[3] /= height
        bounds = torch.cat((bounds, torch.tensor([0])))
        if self.transform:
            image, bounds = self.transform((image, bounds))

        """from launch import show_bounding_box
        from torchvision import transforms
        show_bounding_box(transforms.ToPILImage()(image),bounds[:4])"""

        return image, bounds
