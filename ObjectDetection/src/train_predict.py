import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train(num_loops: int, criterion: nn, optim: optim, model: nn.Module, \
          train_dataloader: torch.utils.data.DataLoader, device: torch.device, \
            path, criterion_cls, criterion_reg):
    model.train()
    for epoch in tqdm(range(num_loops)):
        run_loss = 0.0
        len_batch = 0
        for images, targets in enumerate(train_dataloader):

            
            #Zero parameter gradients
            optim.zero_grad()

            rois = torch.cat([t['boxes'] for t in targets])
            roi_indices = torch.cat([torch.full_like(t['labels'], i) for i, t in enumerate(targets)])
            #Forward
            class_logits, bbox_regressions = model(images, rois, roi_indices)
            
            # Assuming targets are properly formatted for each ROI
            loss_cls = criterion_cls(class_logits, torch.cat([t['labels'] for t in targets]))
            # Adjust targets['boxes'] as needed to match the predicted format
            loss_reg = criterion_reg(bbox_regressions, torch.cat([t['boxes'] for t in targets]).float())
        
            # Total loss
            loss = loss_cls + loss_reg
            #output = torch.round(torch.sigmoid(output_logits))
            #Calculate the loss
            loss = criterion(output_logits, labels)
            #Compute gradient
            loss.backward()
            #Optimize the parameters based on gradient
            optim.step()

            run_loss += loss.item()
            len_batch +=1
            del inputs, labels
        
        print(f"Epoch: {epoch} | Loss: {run_loss/len_batch}")
    torch.save(model.state_dict(), path)
    print("Finished Training")


def test_classification(model: nn.Module, test_dataloader: torch.utils.data.DataLoader,
          device: torch.device, classes):
    correct = 0
    total = 0
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    with torch.no_grad():
        model.to(device)
        for data in test_dataloader:
            img, labels = data[0].to(device), data[1].to(device)

            outputs = model(img)
            _, predicts = torch.max(outputs, 1)
            total +=labels.size(0)
            correct += (predicts == labels).sum().item()
            for label, predict in zip(labels, predicts):
                if label == predict:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] +=1
    print(f"Accuracy of the network in the test dataset: {100 * correct // total} %")

    for class_name, correct_count in correct_pred.items():
        accuracy = 100* float(correct_count) / total_pred[class_name]
        print(f"Accuracy for class {class_name}: {accuracy}")
  


