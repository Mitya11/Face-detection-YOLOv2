from PIL import Image,ImageDraw
import torch
def show_bounding_box(img,rect):
    width,height = img.size
    draw = ImageDraw.Draw(img,"RGBA")
    for i in range(len(rect)):
        try:
            draw.rectangle(((rect[i][0]-rect[i][2]/2)*width,(rect[i][1]-rect[i][3]/2)*height,(rect[i][0]+rect[i][2]/2)*width,(rect[i][1]+rect[i][3]/2)*height))
        except:
            pass
    return img

def found_best_boxes(nn_out,grid,anchor):
    nn_out.cpu()
    B = nn_out.size(dim=2)//5
    predicts = []
    for i in range(B):
        for j in range(nn_out.size(dim=1)):
            if nn_out[0,j,(i+1)*4 + i] >0.4:
                box=nn_out[0,j,i * 4 + i:(i + 1) * 4 + i]
                box[0] = box[0] * (1 / grid) + (1 / grid) * (j % grid)
                box[1] = box[1] * (1 / grid) + (1 / grid) * (j // grid)
                box[2]= anchor[i, 0] * torch.exp(box[2])
                box[3]= anchor[i, 1] * torch.exp(box[3])

                predicts.append(box.tolist())
                nn_out[0, j,:] = torch.tensor([0.]*(B*5))
    return predicts