import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy
from intersection_over_union import iou
from loss import calc_loc_loss,calc_conf_loss
def non_maximum_supression():
    pass
def reorg_layer(x):
    batch_size = x.size(dim=0)
    grid = x.size(dim=2)
    result = torch.zeros((batch_size,x.size(dim=1)*4,grid//2,grid//2))
    for i in range(4):
        result[:,i*x.size(dim=1):(i+1)*x.size(dim=1)] = x[:,:,0::2,0::2]
    return result.cuda()
def lossF(output, target, grid,anchors):
    batch_size = output.size(dim=0)
    target_coords = target["coords"].cuda()
    B = 3
    all_iou = torch.autograd.Variable(torch.zeros((batch_size, grid * grid, B)).cuda())
    for i in range(B):
        anchor_B = torch.clone(output[:, :, i * 4 + i:(i + 1) * 4 + i])
        anchor_B[:,:,:2] = torch.nn.functional.sigmoid(anchor_B[:,:,:2])
        anchor_B[:, :, 2:] = torch.tanh(anchor_B[:, :, 2:])
        anchor_B[:,:,2] = anchors[i,0] * torch.exp(anchor_B[:, :, 2])
        anchor_B[:, :, 3] = anchors[i,1] * torch.exp(anchor_B[:, :, 3])
        all_iou[:, :, i] = iou(anchor_B, target_coords[:, :, i, :])

    best_iou = torch.eq(all_iou, torch.unsqueeze(torch.max(all_iou, dim=2).values, dim=2).repeat(1, 1, B))
    conf_ij = target["ij"].cuda()# * best_iou

    Lnoobj = (all_iou < 0.6).float() * (1-conf_ij)

    localization_error = calc_loc_loss(output,target_coords,anchors,conf_ij)

    # classification_error = torch.sum((output[:, :, -1] - conf_i) ** 2 * conf_i)

    confidence_error = calc_conf_loss(output,all_iou,conf_ij,Lnoobj) # + classification_error
    loss = localization_error + confidence_error
    return loss


class YOLOv2(torch.nn.Module):
    def __init__(self, grid=7):
        super(YOLOv2, self).__init__()
        self.grid = grid
        self.anchors = torch.tensor([[0.6,0.7],[0.3,0.35],[0.06,0.08]])
        self.net1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, padding=(1, 1)),
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU(0.1),
            torch.nn.MaxPool2d((2, 2)),

            torch.nn.Conv2d(32, 64, 3, padding=(1, 1)),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(0.1),
            torch.nn.MaxPool2d((2, 2)),

            torch.nn.Conv2d(64, 128,  3, padding=(1, 1)),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Conv2d(128, 64, 1),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Conv2d(64, 128, 3, padding=(1, 1)),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(0.1),
            torch.nn.MaxPool2d((2, 2)),


            torch.nn.Conv2d(128, 256,  3, padding=(1, 1)),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Conv2d(256, 128, 1),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Conv2d(128, 256, 3, padding=(1, 1)),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(0.1),
            torch.nn.MaxPool2d((2, 2)),

            torch.nn.Conv2d(256, 512, 3, padding=(1, 1)),
            torch.nn.BatchNorm2d(512),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Conv2d(512, 256, 1),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Conv2d(256, 512, 3, padding=(1, 1)),
            torch.nn.BatchNorm2d(512),
            torch.nn.LeakyReLU(0.1),



        )

        self.net2 = torch.nn.Sequential(


            torch.nn.Conv2d(512, 1024,  3, padding=(1, 1)),
            torch.nn.BatchNorm2d(1024),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Conv2d(1024, 512, 1),
            torch.nn.BatchNorm2d(512),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Conv2d(512, 1024, 3, padding=(1, 1)),
            torch.nn.BatchNorm2d(1024),
            torch.nn.LeakyReLU(0.1),


            torch.nn.MaxPool2d((2, 2))

        )
        self.last = torch.nn.Conv2d(3072, 15, 7,padding=(3,3))  # 7*7 from grid map

    def forward(self, x):
        """l =x[0][0].cpu().detach().numpy()
        l[:, ::224//self.grid] = 0
        l[::224//self.grid, :] = 0
        plt.imshow(l)
        plt.colorbar()
        plt.show()"""
        x = x.cuda()

        x_conc = self.net1(x)
        x_2 = self.net2(x_conc)
        x_1 = reorg_layer(x_conc)
        x_added = torch.cat((x_1,x_2),dim=1)
        x_res = x_added
        x_res = self.last(x_res)
        #print(x[0,0,0,:])
        #print(x[1, 0, 0, :])
        x_res = torch.reshape(x_res, (-1, self.anchors.size(dim=0)*5, self.grid*self.grid))
        #x_res = F.tanh(x_res)
        return torch.transpose(x_res,1,2)

    def train(self, epochesCount, dataset):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.00001)
        torch.autograd.set_detect_anomaly(True)
        for i in range(epochesCount):
            print("Epoch:", i + 1)
            if 0.0002 > self.epoch(dataset, optimizer):
                break

    def epoch(self, dataset, optimizer):
        it = iter(dataset)
        sr = 0
        for i in range(len(dataset)):
            data, target = next(it)
            data = data.cuda()
            output = self(data)

            optimizer.zero_grad()
            loss = lossF(output, target, self.grid, self.anchors)
            loss.backward()
            # torch.nn.utils.clip_grad_norm(self.parameters(),1)
            optimizer.step()
            sr += float(loss)
        print("Loss:", sr / len(data))
        return sr / len(dataset)
