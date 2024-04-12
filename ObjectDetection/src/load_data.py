import torchvision.transforms.v2 as transforms
import os
import sys
import torch
from torchvision import datasets, tv_tensors
from torchvision.transforms import v2
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from plotting import *

transform = v2.Compose(
    [
        v2.ToImage(),
        v2.RandomPhotometricDistort(p=1),
        v2.RandomZoomOut(fill={tv_tensors.Image: (123, 117, 104), "others": 0}),
        v2.RandomIoUCrop(),
        v2.RandomHorizontalFlip(p=1),
        v2.SanitizeBoundingBoxes(),
        v2.ToDtype(torch.float32, scale=True),
    ]
)


def load_dataset(TRAIN_IMG_DIR, TRAIN_ANN_FILE):
    data =  datasets.CocoDetection(root = TRAIN_IMG_DIR, 
                              annFile = TRAIN_ANN_FILE,
                              transforms = transform)
    data = datasets.wrap_dataset_for_transforms_v2(data, target_keys=["boxes", "labels", "masks"])
    return data

def load_datasetNone(TRAIN_IMG_DIR, TRAIN_ANN_FILE):
    data = datasets.CocoDetection(root = TRAIN_IMG_DIR, 
                              annFile = TRAIN_ANN_FILE)
    data = datasets.wrap_dataset_for_transforms_v2(data, target_keys=["boxes", "labels", "masks"])
    return data
