import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
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
        
        if device != "cpu":
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



