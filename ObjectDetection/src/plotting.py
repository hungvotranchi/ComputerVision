import os
from torch.utils.data import Dataset
import PIL.Image
import json
import torch
from torchvision import datasets
from torchvision.transforms import transforms
import sys
from preprocess import *


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p
        self.hf = transforms.RandomHorizontalFlip(1)
        
    def __call__(self, img, bboxes):
        
        if torch.rand(1)[0] < self.p:            
            img = self.hf.forward(img)
            bboxes = self.hf.forward(bboxes)
        
        return img, bboxes
    
    
class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p
        self.vf = transforms.RandomVerticalFlip(1)
        
    def __call__(self, img, bboxes):
        if torch.rand(1)[0] < self.p:                    
            img = self.vf.forward(img)
            bboxes = self.vf.forward(bboxes)
        
        return img, bboxes

class Resize(object):
    def __init__(self, size):
        self.size = size
        self.resize = transforms.Resize(self.size, antialias=True)
        
    def __call__(self, img, bboxes):
        img = self.resize.forward(img)
        
        bboxes = self.resize.forward(bboxes)

        return img, bboxes
    
    
def show(sample):
    import matplotlib.pyplot as plt

    from torchvision.transforms.v2 import functional as F
    from torchvision.utils import draw_bounding_boxes
    
    resize = Resize((300, 300))
    
    rhf = RandomHorizontalFlip()
    rvf = RandomVerticalFlip()
    image, target = sample
    
    image, bboxes = image,target["boxes"] 
    
    image, bboxes = resize(image, bboxes)
    image, bboxes = rhf(image, bboxes)
    image, bboxes = rvf(image, bboxes)
    
    if isinstance(image, PIL.Image.Image):
        image = F.to_image_tensor(image)
        
    image = F.convert_dtype(image, torch.uint8)
    annotated_image = draw_bounding_boxes(image, bboxes, colors="yellow", width=3)

    fig, ax = plt.subplots()
    ax.imshow(annotated_image.permute(1, 2, 0).numpy())
    ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    fig.tight_layout()

    fig.show()