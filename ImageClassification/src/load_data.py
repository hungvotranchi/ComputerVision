import torchvision.transforms.v2 as transforms
import torch
from torchvision.transforms.v2 import functional as F
import json
from torchvision import datasets


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

def load_COCO(transform, data_path):
    coco = datasets.CocoDetection(root=f"{data_path}/train2017/", annFile = f"{data_path}/annotations/instances_train2017.json", \
                               transform= transform)
    coco_val = datasets.CocoDetection(root=f"{data_path}/val2017/", annFile = f"{data_path}/annotations/instances_val2017.json", \
                                transform= transform)
    return coco, coco_val

class NewCocoDataset(Dataset):    
    def __init__(self, coco_dataset, image_size=(312, 312)):
        """
        Arguments:
            coco_dataset (dataset): The coco dataset containing all the expected transforms.
            image_size (tuple): Target image size. Default is (512, 512)
        """
        
        self.coco_dataset = coco_dataset
        self.resize = Resize(image_size)
        self.rhf = RandomHorizontalFlip()
        self.rvf = RandomVerticalFlip()   
        self.transformer = transforms.Compose([
            transforms.ToImageTensor(),
            transforms.ConvertImageDtype(torch.float32),
        ])

        
    def __len__(self):
        return len(self.coco_dataset)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        new_target = {}
        
        image, target = self.coco_dataset[idx]
        
        if 'boxes' not in target:    
            new_idx = idx-1
            _img, _t = self.coco_dataset[new_idx]
            while 'boxes' not in _t :
                new_idx -= 1
                _img, _t = self.coco_dataset[new_idx]
                
            image, target = self.coco_dataset[new_idx]
        
        
        image, bboxes = image, target["boxes"] 
            
        image, bboxes = self.resize(image, bboxes)
        image, bboxes = self.rhf(image, bboxes)
        image, bboxes = self.rvf(image, bboxes)
        
        image = self.transformer(image)
        
        new_boxes = []
        for box in bboxes:
            if box[0] < box[2] and box[1] < box[3]:
                new_boxes.append(box)
        
        new_target["boxes"] = torch.stack(new_boxes)
        new_target["labels"] = target["labels"]
    
        return (image, new_target)