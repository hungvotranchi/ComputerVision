import torchvision.datasets as dset
import os
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as transforms
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from preprocess import *

def load_dataset(IMG_DIR, ANN_FILE):
    coco_train = dset.CocoDetection(root = IMG_DIR, 
                              annFile = ANN_FILE)
    return dset.wrap_dataset_for_transforms_v2(coco_train)

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
            transforms.PILToTensor(),
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