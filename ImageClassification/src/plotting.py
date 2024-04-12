import PIL.Image
from preprocess import *

def show(sample):
    import matplotlib.pyplot as plt

    from torchvision.transforms.v2 import functional as F
    from torchvision.utils import draw_bounding_boxes
    
    resize = Resize((300, 300))
    
    rhf = RandomHorizontalFlip()
    rvf = RandomVerticalFlip()
    image, target = sample
    
    image, bboxes = image,target["boxes"] 
    
    image, bboxes = resize(image, bboxes)
    image, bboxes = rhf(image, bboxes)
    image, bboxes = rvf(image, bboxes)
    
    if isinstance(image, PIL.Image.Image):
        image = F.to_image_tensor(image)
        
    image = F.convert_dtype(image, torch.uint8)
    annotated_image = draw_bounding_boxes(image, bboxes, colors="yellow", width=3)

    fig, ax = plt.subplots()
    ax.imshow(annotated_image.permute(1, 2, 0).numpy())
    ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    fig.tight_layout()

    fig.show()