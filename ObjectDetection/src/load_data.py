import os
from torch.utils.data import Dataset
from PIL import Image
import json
from torchvision import datasets
from torchvision.transforms import transforms

transform_sub = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def load_COCO(transform):
    data_path = f"..\datasets\COCO"
    data_path_linux = f"../datasets/COCO"
    try:
        coco = datasets.CocoDetection(root=data_path, train = True, \
                                download= True, transform= transform)
    except:
        coco = datasets.CocoDetection(root=data_path_linux, train = True, \
                                download= False, transform= transform)
    return coco
