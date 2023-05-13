from PIL import Image,ImageDraw

def show_bounding_box(img,rect):
    rect = abs(rect)
    width,height = img.size
    draw = ImageDraw.Draw(img,"RGBA")
    draw.rectangle(((rect[0]-rect[2]/2)*width,(rect[1]-rect[3]/2)*height,(rect[0]+rect[2]/2)*width,(rect[1]+rect[3]/2)*height))
    return img