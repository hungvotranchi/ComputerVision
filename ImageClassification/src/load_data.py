import os
from torch.utils.data import Dataset
from PIL import Image
import json
from torchvision import datasets
from src.preprocess import transform_CIFAR10
from torchvision.transforms import transforms

transform_sub = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def load_CIFAR10(transform):
    data_path = r"..\datasets\CIFAR10"
    data_path_linux = r"../datasets/CIFAR10"
    try:
        cifar10 = datasets.CIFAR10(root=data_path, train = True, \
                                download= False, transform= transform)
        cifar10_val = datasets.CIFAR10(root=data_path, train = False, \
                                download= False, transform= transform)
    except:
        cifar10 = datasets.CIFAR10(root=data_path_linux, train = True, \
                                download= False, transform= transform)
        cifar10_val = datasets.CIFAR10(root=data_path_linux, train = False, \
                                download= False, transform= transform)
    return cifar10, cifar10_val

def load_fgvc(transform):
    data_path = r"..\datasets\fgvc"
    data_path_linux = r"../datasets/fgvc"
    try:
        fgvc = datasets.FGVCAircraft(root=data_path, split = "train", \
                                download= False, transform= transform)
        fgvc_val = datasets.FGVCAircraft(root=data_path, split = "val", \
                                download= False, transform= transform)
    except:
        fgvc = datasets.FGVCAircraft(root=data_path_linux, split = "train", \
                                download= False, transform= transform)
        fgvc_val = datasets.FGVCAircraft(root=data_path_linux, split = "val", \
                                download= False, transform= transform)
    return fgvc, fgvc_val




class ImageNetKaggle(Dataset):
    def __init__(self, root, split, transform=None):
        self.samples = []
        self.targets = []
        self.transform = transform
        self.syn_to_class = {}
        with open(os.path.join(root, "imagenet_class_index.json"), "rb") as f:
                    json_file = json.load(f)
                    for class_id, v in json_file.items():
                        self.syn_to_class[v[0]] = int(class_id)
        with open(os.path.join(root, "ILSVRC2012_val_labels.json"), "rb") as f:
                    self.val_to_syn = json.load(f)
        samples_dir = os.path.join(root, "ILSVRC/Data/CLS-LOC", split)
        for entry in os.listdir(samples_dir):
            if split == "train":
                syn_id = entry
                target = self.syn_to_class[syn_id]
                syn_folder = os.path.join(samples_dir, syn_id)
                for sample in os.listdir(syn_folder):
                    sample_path = os.path.join(syn_folder, sample)
                    self.samples.append(sample_path)
                    self.targets.append(target)
            elif split == "val":
                syn_id = self.val_to_syn[entry]
                target = self.syn_to_class[syn_id]
                sample_path = os.path.join(samples_dir, entry)
                self.samples.append(sample_path)
                self.targets.append(target)
    def __len__(self):
            return len(self.samples)
    def __getitem__(self, idx):
            x = Image.open(self.samples[idx]).convert("RGB")
            if self.transform:
                x = self.transform(x)
            return x, self.targets[idx]