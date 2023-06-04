from PIL import Image,ImageDraw
import torch
import torchvision
def show_bounding_box(img,rect):
    width,height = img.size
    draw = ImageDraw.Draw(img,"RGBA")
    for i in range(len(rect)):
        try:
            draw.rectangle(((rect[i][0])*width,(rect[i][1])*height,(rect[i][2])*width,(rect[i][3])*height))
        except:
            pass
    return img

def found_best_boxes(nn_out,grid,anchor):
    nn_out.cpu()
    B = nn_out.size(dim=2)//5
    predicts = []
    scores = []
    for i in range(B):
        for j in range(nn_out.size(dim=1)):
            if nn_out[0,j,(i+1)*4 + i] >0.4:
                scores.append(nn_out[0,j,(i+1)*4 + i])
                box=nn_out[0,j,i * 4 + i:(i + 1) * 4 + i]
                box[:2] = torch.nn.functional.sigmoid(box[:2])
                box[2:] = torch.tanh(box[2:])

                box[0] = box[0] * (1 / grid) + (1 / grid) * (j % grid)
                box[1] = box[1] * (1 / grid) + (1 / grid) * (j // grid)
                box[2]= anchor[i, 0] * torch.exp(box[2])
                box[3]= anchor[i, 1] * torch.exp(box[3])

                center = (box[0].item(),box[1].item())
                box[0] = center[0] - box[2]/2
                box[1] = center[1] - box[3]/2
                box[2] = center[0] + box[2]/2
                box[3]= center[1] + box[3]/2
                predicts.append(box.tolist())
    predicts = torch.tensor(predicts)
    scores = torch.tensor(scores)
    if predicts.size(dim=0) == 0:
        return [],[]
    best_predicts = torchvision.ops.nms(predicts,scores,0.1)
    return predicts[best_predicts].tolist(), scores[best_predicts].tolist()