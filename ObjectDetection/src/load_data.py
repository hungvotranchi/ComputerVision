import os
from torch.utils.data import Dataset
from PIL import Image
import json
from torchvision import datasets
from torchvision.transforms import transforms

transform_sub = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def load_COCO(transform, data_path):
    coco = datasets.CocoDetection(root=f"{data_path}/train2017/", annFile = f"{data_path}/annotations/instances_train2017.json", \
                               transform= transform)
    coco_val = datasets.CocoDetection(root=f"{data_path}/val2017/", annFile = f"{data_path}/annotations/instances_val2017.json", \
                                transform= transform)
    return coco, coco_val
