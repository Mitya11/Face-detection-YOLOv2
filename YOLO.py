import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy
from intersection_over_union import iou


def lossF(output, target, grid):
    batch_size = output.size(dim=0)
    target_coords = target["coords"].cuda()
    B = 2
    all_iou = torch.zeros((batch_size, grid * grid, B)).cuda()
    for i in range(B):
        all_iou[:, :, i] = iou(output[:, :, i * 4 + i:(i + 1) * 4 + i], target_coords[:, :, i, :])

    best_iou = torch.eq(all_iou, torch.unsqueeze(torch.max(all_iou, dim=2).values, dim=2).repeat(1, 1, B))


    conf_ij = target["ij"].cuda() * best_iou
    localization_error = 0
    confidence_error = 0
    for i in range(B):
        localization_error += 5 * torch.sum(
            torch.sum((output[:, :, i * 4 + i:(i + 1) * 4 + i] - target_coords[:, :, i, :]) ** 2, dim=2) * conf_ij[:, :, i])
    for i in range(B):
        confidence_error += torch.sum((output[:, :, i * 4 + i + 4] - all_iou[:, :, i]) ** 2 * conf_ij[:, :, i])
        confidence_error += 0.5 * torch.sum(
            (output[:, :, i * 4 + i + 4] - all_iou[:, :, i]) ** 2 * (1 - conf_ij[:, :, i]))

    #classification_error = torch.sum((output[:, :, -1] - conf_i) ** 2 * conf_i)

    loss = localization_error + confidence_error# + classification_error

    return loss


class YOLO(torch.nn.Module):
    def __init__(self, grid=7):
        super(YOLO, self).__init__()
        self.grid = grid

        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 7, padding=(3, 3)),

            torch.nn.MaxPool2d((2,2)),

            torch.nn.Conv2d(64, 192, 3, padding=(1, 1)),

            torch.nn.MaxPool2d((2,2)),

            torch.nn.Conv2d(192, 128, 3, padding=(1, 1)),

            torch.nn.MaxPool2d((2,2)),

            torch.nn.Conv2d(128, 128, 3, padding=(1, 1)),


            torch.nn.Conv2d(128, 256, 3, padding=(1, 1)),

            torch.nn.MaxPool2d((2,2)),

            torch.nn.Conv2d(256, 512, 3, padding=(1, 1)),

            torch.nn.MaxPool2d((2, 2)),

            torch.nn.Conv2d(512, 512,7, padding=(3, 3)),



            torch.nn.Flatten(),
            torch.nn.Linear(25088, self.grid ** 2 * 10)  # 7*7 from grid map
        )
    def forward(self, x):
        """l =x[0][0].cpu().detach().numpy()
        l[:, ::224//self.grid] = 0
        l[::224//self.grid, :] = 0
        plt.imshow(l)
        plt.colorbar()
        plt.show()"""
        x = x.cuda()

        x= self.net(x)
        return torch.reshape(x, (-1, self.grid ** 2, 10))

    def train(self, epochesCount, dataset):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.000005)
        torch.autograd.set_detect_anomaly(True)
        for i in range(epochesCount):
            print("Epoch:", i + 1)
            if 0.0002>self.epoch(dataset, optimizer):
                break

    def epoch(self, data, optimizer):
        it = iter(data)
        sr = 0
        for i in range(len(data)):
            data, target = next(it)

            output = self(data)

            optimizer.zero_grad()

            loss = lossF(output, target, self.grid)
            loss.backward()
            #torch.nn.utils.clip_grad_norm(self.parameters(),1)
            optimizer.step()
            sr += float(loss)
        print("Loss:", sr / len(data))
        return sr / len(data)
