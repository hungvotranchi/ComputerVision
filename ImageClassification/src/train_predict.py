import torch
import torch.nn as nn
import torch.optim as optim

def train(num_loops: int, criterion: nn, optim: optim, model: nn.Module, \
          train_dataloader: torch.utils.data.DataLoader, device: torch.device, \
            path):
    model.train()
    for epoch in range(num_loops):
        run_loss = 0.0
        for i, data in enumerate(train_dataloader):
            inputs, labels = data[0].to(device), (data[1].type(torch.LongTensor)).to(device)

            
            #Zero parameter gradients
            optim.zero_grad()

            #Forward
            output_logits = model(inputs)
            #output = torch.round(torch.sigmoid(output_logits))
            #Calculate the loss
            loss = criterion(output_logits, labels)
            #Compute gradient
            loss.backward()
            #Optimize the parameters based on gradient
            optim.step()

            run_loss += loss.item()
            if i % 200 == 199:    # print every 200 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {run_loss / 2000:.3f}')
                run_loss = 0.0
            del inputs, labels
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
  


