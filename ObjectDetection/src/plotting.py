import torchvision.datasets as dset
import os
from torch.utils.data import Dataset, DataLoader
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from preprocess import *


def show(sample):
    import matplotlib.pyplot as plt

    from torchvision.transforms.v2 import functional as F
    from torchvision.utils import draw_bounding_boxes
    import torchvision.transforms.v2 as v2
    
    resize = Resize((300, 300))
    
    rhf = RandomHorizontalFlip()
    rvf = RandomVerticalFlip()
    image, target = sample
    
    image, bboxes = image,target["boxes"] 
    
    image, bboxes = resize(image, bboxes)
    image, bboxes = rhf(image, bboxes)
    image, bboxes = rvf(image, bboxes)
    
    if isinstance(image, PIL.Image.Image):
        img_transform = v2.PILToTensor()
        image = img_transform(image)
    
    toNumber = v2.ToDtype(torch.uint8)
    image = toNumber(image)
    annotated_image = draw_bounding_boxes(image, bboxes, colors="yellow", width=3)

    fig, ax = plt.subplots()
    ax.imshow(annotated_image.permute(1, 2, 0).numpy())
    ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    fig.tight_layout()

    fig.show()