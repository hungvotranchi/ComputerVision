import torchvision.transforms.v2 as transforms
import os
import sys
import torchvision.datasets as dset
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from plotting import *

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
def load_dataset(TRAIN_IMG_DIR, TRAIN_ANN_FILE):
    return dset.CocoDetection(root = TRAIN_IMG_DIR, 
                              annFile = TRAIN_ANN_FILE,
                              transforms = transform)
