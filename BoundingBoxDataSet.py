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
                if dan >=10000: break
                count = file.readline()
                if int(count) >= 1 and int(count) <= 1:
                    dan += 1
                    self.dict[i.rstrip()] = []
                    for j in range(int(count)):
                        labels = [int(h) for h in file.readline().split(" ")[:4]]
                        labels[0] += labels[2] / 2.0
                        labels[1] += labels[3] / 2.0
                        self.dict[i.rstrip()].append(labels)
                    self.dict[i.rstrip()] = torch.tensor(self.dict[i.rstrip()])
                else:
                    for j in range(max(int(count), 1)):
                        file.readline()
        self.transform = transform

    def __len__(self):
        return len(self.dict.keys())

    def __getitem__(self, idx):
        S = 7
        B = 3
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = Image.open(self.folder + list(self.dict.keys())[idx])
        width, height = image.size
        bounds = torch.tensor(self.dict[list(self.dict.keys())[idx]])
        bounds[:,0] /= width
        bounds[:,2] /= width
        bounds[:,1] /= height
        bounds[:,3] /= height

        if self.transform:
            image, bounds = self.transform((image, bounds))

        coords = torch.zeros(S*S,B,4)
        conf_mask_i = torch.zeros((S*S))
        conf_mask_ij = torch.zeros((S * S, B))
        for h in range(bounds.size(dim=0)):
            for i in range(S):
                for j in range(S):
                    if (bounds[h, 0] - bounds[h, 2] / 2 < (1 / S) * j + 1 / S/2
                            and (1 / S) * j + 1 / S/2 < bounds[h, 0] + bounds[h, 2] / 2
                            and bounds[h, 1] - bounds[h, 3] / 2 < (1 / S) * i + 1 / S/2
                            and (1 / S) * i + 1 / S/2 < bounds[h, 1] + bounds[h, 3] / 2):
                        conf_mask_i[i*S+j] = 1
                        conf_mask_ij[i*S+j] = torch.tensor([1.] * B)
                        anchor_cur = torch.clone(bounds[h])
                        anchor_cur[0] = (anchor_cur[0] - j * (1 / S)) * S
                        anchor_cur[1] = (anchor_cur[1] - i * (1 / S)) * S
                        coords[i*S+j] = anchor_cur.expand(B, -1)

            center = bounds[h,0] // (1/S) + bounds[h,1] // (1/S) * S
            if int(center) >48 or int(center) <0:
                continue
            conf_mask_i[int(center.item())] = 1
            conf_mask_ij[int(center.item())] = torch.tensor([1.]*B)
            bounds[h,0]= (bounds[h,0] -center%S *(1/S))*S
            bounds[h, 1] = (bounds[h, 1] - center // S * (1 / S)) * S
            coords[int(center)] = bounds[h].expand(B,-1)
        #bounds = torch.cat((bounds, torch.tensor([0])))


        target = {"coords":coords,"i":conf_mask_i,"ij":conf_mask_ij}
        """from launch import show_bounding_box
        from torchvision import transforms
        show_bounding_box(transforms.ToPILImage()(image),bounds[:4])"""

        return image, target
