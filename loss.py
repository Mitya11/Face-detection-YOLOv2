import torch
from intersection_over_union import iou
B = 3
def calc_loc_loss(output ,target ,anchors ,Lobj ,koef = 5):
    loss = 0
    target[:, :, :, 2:] = torch.sqrt(target[:, :, :, 2:])
    Nobj = torch.sum(Lobj)
    for i in range(B):
        anchor_B = torch.clone(output[:, :, i * 4 + i:(i + 1) * 4 + i])
        anchor_B[:, :, 2] = torch.sqrt(anchors[i, 0] * torch.exp(anchor_B[:, :, 2]))
        anchor_B[:, :, 3] = torch.sqrt(anchors[i, 1] * torch.exp(anchor_B[:, :, 3]))

        loss += torch.sum(
            torch.sum((anchor_B - target[:, :, i, :]) ** 2, dim=2) * Lobj[:, :, i])
    loss *= koef/Nobj
    return loss
def calc_conf_loss(output,target,Lobj,Lnoobj,koef_obj = 1, koef_noobj = 0.2):
    loss = 0
    Nconf = torch.sum(Lobj + Lnoobj*(1-Lobj))/48
    for i in range(B):
        k = torch.reshape(output[0, :, i * 4 + i + 4],(7,7))
        loss += koef_obj* torch.sum((output[:, :, i * 4 + i + 4] - target[:,:,i]*1.5) ** 2 *Lobj[:,:,i])
        loss += koef_noobj * torch.sum(
            (0 -output[:, :, i * 4 + i + 4])**2 * Lnoobj[:,:,i])
    loss /= Nconf
    return loss