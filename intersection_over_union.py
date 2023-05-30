import torch
import torch.nn.functional as F


def iou(a, b):  # intersection over union
    x1 = a[:, :, 0:1] - a[:, :, 2:3] / 2 * 7
    y1 = a[:, :, 1:2] - a[:, :, 3:4] / 2 * 7
    x2 = a[:, :, 0:1] + a[:, :, 2:3] / 2 * 7
    y2 = a[:, :, 1:2] + a[:, :, 3:4] / 2 * 7

    x3 = b[:, :, 0:1] - b[:, :, 2:3] / 2 * 7
    y3 = b[:, :, 1:2] - b[:, :, 3:4] / 2 * 7
    x4 = b[:, :, 0:1] + b[:, :, 2:3] / 2 * 7
    y4 = b[:, :, 1:2] + b[:, :, 3:4] / 2 * 7

    x5 = torch.max(torch.cat((x1, x3), dim=2), dim=2).values
    y5 = torch.max(torch.cat((y1, y3), dim=2), dim=2).values
    x6 = torch.min(torch.cat((x2, x4), dim=2), dim=2).values
    y6 = torch.min(torch.cat((y2, y4), dim=2), dim=2).values

    zero_coefs = torch.where(b[:, :, 2] * b[:, :, 3] == 0, 0, 1)
    interS = F.relu6(x6 - x5) * F.relu6(y6 - y5)

    unionS = (a[:, :, 2] * 7 * a[:, :, 3] * 7 + b[:, :, 2] * 7 * b[:, :, 3] * 7) - interS

    iou = (interS / (abs(unionS + 0.0001))) * zero_coefs
    # iou = torch.where(iou>1,1,iou)
    return F.relu6(iou)
