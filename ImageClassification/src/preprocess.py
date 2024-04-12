import torch
from torchvision.transforms import transforms

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
    
transform = transforms.Compose(
    [
        transforms.RandomPhotometricDistort(),        
        transforms.RandomAutocontrast(),
        transforms.RandomEqualize(),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToImageTensor(),
        transforms.ConvertImageDtype(torch.float32),
    ]
)