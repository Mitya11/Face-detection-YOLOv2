import random
from torchvision import transforms
import math

class RandomCrop(object):
    def __init__(self, size):
        if isinstance(size, float):
            self.output_size = (size, size)
        else:
            self.output_size = size

    def __call__(self, sample):
        image, bounds = sample

        height, width = image.size()[1:]
        w, h = random.uniform(max(0, bounds[0][0] - self.output_size[0]),
                              min(1 - self.output_size[0], bounds[0][0])), random.uniform(
            max(0, bounds[0][1] - self.output_size[1]), min(1 - self.output_size[1], bounds[0][1]))
        cropped_image = image[:, int(h * height):int((h + self.output_size[1]) * height),
                        int(w * width):int((w + self.output_size[0]) * width)]
        for i in range(len(bounds)):

            bounds[i][0] = (bounds[i][0] - w) / self.output_size[0]
            bounds[i][1] = (bounds[i][1] - h) / self.output_size[1]
            bounds[i][2] /= self.output_size[0]
            bounds[i][3] /= self.output_size[1]
        return cropped_image, bounds


class RandomRotate(object):
    def __call__(self, sample):
        image, bounds = sample

        angle = random.choice([0,90,180,270])

        image = transforms.functional.rotate(image, angle)
        for i in range(len(bounds)):
            x = (bounds[i][0]-0.5) * math.cos(-1*angle/180*math.pi) - (bounds[i][1]-0.5) * math.sin(-1*angle/180*math.pi) +0.5
            y = (bounds[i][0]-0.5) * math.sin(-1*angle/180*math.pi) + (bounds[i][1]-0.5) * math.cos(-1*angle/180*math.pi) +0.5

            bounds[i][0] = x
            bounds[i][1] = y
            if angle == 90 or angle == 270:
                bounds[i][2], bounds[i][3] = bounds[i][3],bounds[i][2]

        return image, bounds


class Resize(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, bounds = sample
        image = transforms.Resize(self.output_size)(image)
        return image, bounds


class ToTensor(object):
    def __call__(self, sample):
        image, bounds = sample
        image = transforms.ToTensor()(image)
        return image, bounds
