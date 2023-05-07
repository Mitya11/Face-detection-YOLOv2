import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy

def iou(a,b):
    x1 = a[:,:,0:1] - a[:,:,2:3]/2
    y1 = a[:,:,1:2] - a[:,:,3:4]/2
    x2 = a[:,:,0:1] + a[:,:,2:3] / 2
    y2 = a[:,:,1:2] + a[:,:,3:4] / 2

    x3 = b[:,:,0:1] - b[:,:,2:3] / 2
    y3 = b[:,:,1:2] - b[:,:,3:4] / 2
    x4 = b[:,:,0:1] + b[:,:,2:3] / 2
    y4 = b[:,:,1:2] + b[:,:,3:4] / 2

    k =torch.cat((x1,x3),dim=2)
    x5 = torch.max(torch.cat((x1,x3),dim=2), dim=2).values
    y5 = torch.max(torch.cat((y1,y3),dim=2), dim=2).values
    x6 = torch.max(torch.cat((x2,x4),dim=2), dim=2).values
    y6 = torch.max(torch.cat((y2,y4),dim=2), dim=2).values

    interS = F.relu6(x6 - x5) * F.relu6(y6-y5)
    unionS = (a[:,:,2] * a[:,:,3] + b[:,:,2] * b[:,:,3])-interS
    iou = interS - unionS
    return iou
def lossF(output,target):
    loss = 0
    for i in range(7):
        for j in range(7):
            target[:,i+j*7,-1] = 1 - (((1/7)*i+1/14)-target[:,i+j*7,1])**2 + (((1/7)*j+1/14)-target[:,i+j*7,0])**2

    loss+=torch.sum(((target[:,:,0]-output[:,:,0])**2 +(target[:,:,1]-output[:,:,1])**2)*target[:,:,-1])
    loss+= torch.sum(((target[:,:,2]-output[:,:,2])**2 +(target[:,:,3]-output[:,:,3])**2)*target[:,:,-1])
    loss+= torch.sum((target[:,:,-1]-output[:,:,-1])**2)
    return torch.sum(loss)

class YOLO(torch.nn.Module):
    def __init__(self):
        super(YOLO, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 7,padding=(3,3))
        self.conv2 = torch.nn.Conv2d(16, 32, 5,padding=(2,2))
        self.conv3 = torch.nn.Conv2d(32,64, 5,padding=(2,2))
        self.conv4 = torch.nn.Conv2d(64, 100,3,padding=(1,1))
        self.conv5 = torch.nn.Conv2d(100, 150, 3,padding=(1,1))


        self.fc1 = torch.nn.Linear(7350, 490)  # 5*5 from image dimension
    def forward(self,x):
        x = x.cuda()
        x = self.conv1(x)
        x = F.max_pool2d(F.relu(x),(2,2))

        x = self.conv2(x)
        x = F.max_pool2d(F.relu(x),(2,2))

        x = self.conv3(x)
        x = F.max_pool2d(F.relu(x), (2, 2))

        x = self.conv4(x)
        x = F.max_pool2d(F.relu(x), (2, 2))

        x = self.conv5(x)
        x = F.max_pool2d(F.relu(x), (2, 2))
        x = torch.flatten(x,1)

        x = self.fc1(x)
        return torch.reshape(x, (-1,49, 10))

    def train(self,epochesCount,dataset):
        optimizer = torch.optim.Adam(self.parameters(),lr=0.001)
        for i in range(epochesCount):
            print(i)
            self.epoch(dataset,optimizer)
    def epoch(self,data,optimizer):
        it = iter(data)
        sr = 0
        for i in range(len(data)):
            data , target = next(it)
            data =data.cuda()
            target_v=[]
            for j in range(data.size(dim=0)):
                target_v.append(target[j].expand(49,-1))
            target = torch.stack(target_v).cuda()
            output = self(data)

            optimizer.zero_grad()

            loss = lossF(output,target)
            loss.backward()
            optimizer.step()
            #print("Output:",output)
            #print(loss)
            sr += loss
        print("Loss:",sr/len(data))