import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as transforms
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import gc

def train(num_loops: int, optim: optim, model: nn.Module, \
          train_dataloader: torch.utils.data.DataLoader, device: torch.device):
    model.train()
    for epoch in tqdm(range(num_loops)):
        running_classifier_loss = 0.0
        running_bbox_loss = 0.0
        running_loss = 0.0
        
        counter = 0
        model.train()
    
        for data_point in tqdm(train_dataloader):
            _i, _t = data_point[0], data_point[1]
            
            if device != "cuda":
                _i = torch.stack(_i)

    #         _t = torch.from_numpy(np.asarray(_t))
            
            _i = _i.to(device)
            _t = [{k: v.to(device) for k, v in __t.items()} for __t in _t]

            optim.zero_grad()


            loss_dict = model(_i, _t)
            
    #         running_bbox_loss += torch.mean(loss_dict['bbox_regression']).item()
    #         running_classifier_loss += torch.mean(loss_dict['classification']).item()

            losses = sum(loss for loss in loss_dict.values())
        
            losses.backward()
            optim.step()
            
            running_loss += losses.item()
            
            del loss_dict, losses
            
            counter += 1
            
            if counter % 500 == 499:
                last_classifier_loss = running_classifier_loss / 500 # loss per batch
                last_bbox_loss = running_bbox_loss / 500 # loss per batch
                last_loss = running_loss / 500 # loss per batch
    #             print(f'batch {counter + 1} Classification Loss: {last_classifier_loss}', end='')
    #             print(f', BBox Loss: {last_bbox_loss}')
                print(f'Epoch {epoch}, Batch {counter + 1}, Running Loss: {last_loss}')
                running_classifier_loss = 0.0
                running_bbox_loss = 0.0
                running_loss = 0.0
            
        gc.collect()

def predict(model: nn.Module, \
          data: tuple, device: torch.device, name_idx: dict):
    img_dtype_converter = transforms.ConvertImageDtype(torch.uint8)
    _i = data[0]

    threshold = 0.5
    idx = 3

    if device != "cuda":
        _i = torch.stack(_i)

    _i = _i.to(device)
    model.eval()
    p_t = model(_i)

    confidence_length = len(np.argwhere(p_t[idx]['scores'] > threshold)[0])

    p_boxes = p_t[idx]['boxes'][: confidence_length]
    p_labels = [name_idx[i] for i in p_t[idx]['labels'][: confidence_length].tolist()]
    i_img = img_dtype_converter(_i[idx])

    annotated_image = draw_bounding_boxes(i_img, p_boxes, p_labels, colors="yellow", width=3)
    fig, ax = plt.subplots()
    ax.imshow(annotated_image.permute(1, 2, 0).numpy())
    ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    fig.tight_layout()


    fig.show()



