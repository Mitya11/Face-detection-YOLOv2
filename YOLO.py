import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy


def iou(a, b):  # intersection over union
    x1 = a[:, :, 0:1] - a[:, :, 2:3] / 2
    y1 = a[:, :, 1:2] - a[:, :, 3:4] / 2
    x2 = a[:, :, 0:1] + a[:, :, 2:3] / 2
    y2 = a[:, :, 1:2] + a[:, :, 3:4] / 2

    x3 = b[:, :, 0:1] - b[:, :, 2:3] / 2
    y3 = b[:, :, 1:2] - b[:, :, 3:4] / 2
    x4 = b[:, :, 0:1] + b[:, :, 2:3] / 2
    y4 = b[:, :, 1:2] + b[:, :, 3:4] / 2

    x5 = torch.max(torch.cat((x1, x3), dim=2), dim=2).values
    y5 = torch.max(torch.cat((y1, y3), dim=2), dim=2).values
    x6 = torch.max(torch.cat((x2, x4), dim=2), dim=2).values
    y6 = torch.max(torch.cat((y2, y4), dim=2), dim=2).values

    interS = F.relu6(x6 - x5) * F.relu6(y6 - y5)
    unionS = (a[:, :, 2] * a[:, :, 3] + b[:, :, 2] * b[:, :, 3]) - interS
    iou = interS - unionS
    return iou


def lossF(output, target):
    loss = 0
    for h in range(target.size(dim=0)):
        for i in range(7):
            for j in range(7):
                if (target[h, i * 7 + j, 0] - target[h, i * 7 + j, 2] / 2 < (1 / 7) * j + 1 / 14
                        and (1 / 7) * j + 1 / 14 < target[h, i * 7 + j, 0] + target[h, i * 7 + j, 2] / 2
                        and target[h, i * 7 + j, 1] - target[h, i * 7 + j, 3] / 2 < (1 / 7) * i + 1 / 14
                        and (1 / 7) * i + 1 / 14 < target[h, i * 7 + j, 1] + target[h, i * 7 + j, 3] / 2):
                    target[h, i * 7 + j, -1] = 1
                else:
                    target[h, i * 7 + j, -1] = 0

                """output[:, i*7+j, 0]= (1/7*i) + (1/7)* output[:, i*7+j, 0]
                output[:, i*7+j, 1] = (1/7*j) + (1/7)* output[:, i*7+j, 1]
                output[:, i*7+j, 4] = (1/7*i) + (1/7)* output[:, i*7+j, 4]
                output[:, i*7+j, 5] = (1/7*j) + (1/7)* output[:, i*7+j, 5]
                output[:, i*7+j, :] = torch.where(output[:, i*7+j, :] < 1,output[:, i*7+j, :],1)
                output[:, i*7+j, :] = torch.where(output[:, i*7+j, :] > 0,output[:, i*7+j, :],0)"""
        center = target[h, 0, 0].item() // (1 / 7) + target[h, 0, 1].item() // (1 / 7) * 7
        center = min(center,48)
        target[h, int(center), -1] = 1  # cell containing center of rectangle always equal 1
    loss += torch.sum(
        ((target[:, :, 0] - output[:, :, 0]) ** 2 + (target[:, :, 1] - output[:, :, 1]) ** 2) * target[:, :, -1])

    loss += torch.sum(
        ((target[:, :, 2] - output[:, :, 2]) ** 2 + (target[:, :, 3] - output[:, :, 3]) ** 2) * target[:, :, -1])

    loss += 0.7 * torch.sum(((target[:, :, -1] - output[:, :, -1]) ** 2) * target[:, :, -1])

    loss += torch.sum(((output[:, :, -1] - target[:, :, -1]) ** 2))
    return loss


class YOLO(torch.nn.Module):
    def __init__(self):
        super(YOLO, self).__init__()

        self.conv1 = torch.nn.Conv2d(3, 16, 7, padding=(3, 3))
        self.conv2 = torch.nn.Conv2d(16, 32, 5, padding=(2, 2))
        self.batchN1 = torch.nn.BatchNorm2d(32)
        self.conv3 = torch.nn.Conv2d(32, 50, 5, padding=(2, 2))
        self.conv4 = torch.nn.Conv2d(50, 100, 5, padding=(2, 2))
        self.conv5 = torch.nn.Conv2d(100, 150, 5, padding=(2, 2))
        self.conv6 = torch.nn.Conv2d(150, 250, 5, padding=(2, 2))
        self.batchN2 = torch.nn.BatchNorm2d(250)
        self.conv7 = torch.nn.Conv2d(250, 300, 3, padding=(1, 1))
        self.conv8 = torch.nn.Conv2d(300, 400, 3, padding=(1, 1))

        self.fc1 = torch.nn.Linear(19600, 512)  # 7*7 from grid map
        self.fc2 = torch.nn.Linear(512, 49 * 6) # 7*7 from grid map
    def forward(self, x):
        """l =x[0][0].cpu().detach().numpy()
        l[:, ::32] = 0
        l[::32, :] = 0
        plt.imshow(l)
        plt.colorbar()
        plt.show()"""

        x = x.cuda()
        x = F.leaky_relu(self.conv1(x), 0.1)
        x = F.max_pool2d(F.relu(x), (2, 2))

        x = F.leaky_relu(self.conv2(x), 0.1)
        x = self.batchN1(x)
        x = F.leaky_relu(self.conv3(x), 0.1)
        x = F.max_pool2d(F.relu(x), (2, 2))

        x = F.leaky_relu(self.conv4(x), 0.1)
        x = F.leaky_relu(self.conv5(x), 0.1)

        x = F.max_pool2d(F.relu(x), (2, 2))

        x = F.leaky_relu(self.conv6(x), 0.1)
        x = self.batchN2(x)
        x = F.leaky_relu(self.conv7(x), 0.1)
        x = F.max_pool2d(F.relu(x), (2, 2))

        x = F.leaky_relu(self.conv8(x), 0.1)
        x = F.max_pool2d(F.relu(x), (2, 2))

        x = torch.flatten(x, 1)
        x = F.relu(x)
        #x = F.dropout(x,0.05)
        x = F.relu(self.fc1(x))
        #x = F.dropout(x,0.05)
        x = self.fc2(x)

        return torch.reshape(x, (-1, 49, 6))

    def train(self, epochesCount, dataset):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        for i in range(epochesCount):
            print("Epoch:",i+1)
            self.epoch(dataset, optimizer)

    def epoch(self, data, optimizer):
        it = iter(data)
        sr = 0
        for i in range(len(data)):
            data, target = next(it)
            data = data.cuda()
            target_v = []
            for j in range(data.size(dim=0)):
                target_v.append(target[j].expand(49, -1))
            target = torch.stack(target_v).cuda()
            output = self(data)

            optimizer.zero_grad()

            loss = lossF(output, target)
            loss.backward()
            optimizer.step()
            sr += loss
        print("Loss:", sr / len(data))
