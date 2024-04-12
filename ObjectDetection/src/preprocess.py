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
    
class CustomBatchs:
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.inp = torch.stack(transposed_data[0], 0)
        self.tgt = transposed_data[1]

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.inp = self.inp.pin_memory()
        return (self.inp, self.tgt)
    
def collate_wrapper(batch):
    if torch.cuda.is_available():
        return CustomBatchs(batch)
    else:
        return tuple(zip(*batch))