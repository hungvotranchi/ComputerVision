import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt



def plot_obj(data_loader, device, classes):
    images, targets= next(iter(data_loader))
    images = list(image.to(device) for image in images)
    targets = [{k: torch.tensor([v]).to(device) if k == "image_id" else v.to(device) for k, v in t.items()} for t in targets]

    plt.figure(figsize=(20,20))
    for i, (image, target) in enumerate(zip(images, targets)):
        plt.subplot(2,2, i+1)
        boxes = targets[i]['boxes'].cpu().numpy().astype(np.int32)
        sample = images[i].permute(1,2,0).cpu().numpy()
        names = targets[i]['labels'].cpu().numpy().astype(np.int64)
        for i,box in enumerate(boxes):
            cv2.rectangle(sample,
                        (box[0], box[1]),
                        (box[2], box[3]),
                        (0, 0, 220), 2)
            cv2.putText(sample, classes[names[i]], (box[0],box[1]+15),cv2.FONT_HERSHEY_COMPLEX ,0.5,(0,220,0),1,cv2.LINE_AA)  

        plt.axis('off')
        plt.imshow(sample)
    
def obj_detector(img, model, device):
    img = cv2.imread(img, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)


    img /= 255.0
    img = torch.from_numpy(img)
    img = img.unsqueeze(0)
    img = img.permute(0,3,1,2)
    
    model.eval()

    detection_threshold = 0.70
    
    img = list(im.to(device) for im in img)
    output = model(img)

    for i , im in enumerate(img):
        boxes = output[i]['boxes'].data.cpu().numpy()
        scores = output[i]['scores'].data.cpu().numpy()
        labels = output[i]['labels'].data.cpu().numpy()

        labels = labels[scores >= detection_threshold]
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        scores = scores[scores >= detection_threshold]

        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
    
    sample = img[0].permute(1,2,0).cpu().numpy()
    sample = np.array(sample)
    boxes = output[0]['boxes'].data.cpu().numpy()
    name = output[0]['labels'].data.cpu().numpy()
    scores = output[0]['scores'].data.cpu().numpy()
    boxes = boxes[scores >= detection_threshold].astype(np.int32)
    names = name.tolist()
    
    return names, boxes, sample
    
    