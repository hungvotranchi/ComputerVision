import torch
import torchvision
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.blockCNN1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=96, kernel_size= 11, stride= 4, padding= 0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 3, stride= 2)
        )

        self.blockCNN2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels= 256, kernel_size= 5, stride= 1, padding= 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 3, stride= 2)
        )
        self.blockCNN3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels= 384, kernel_size= 3, stride= 1, padding= 1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels= 384, kernel_size= 3, stride= 1, padding= 1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels= 256, kernel_size= 3, stride= 1, padding= 1),
            nn.MaxPool2d(kernel_size= 3, stride= 2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features= 256 * 5 * 5, out_features= 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features = 4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features= 4096, out_features= num_class)
        )

    
    def forward(self, X):
        X = self.blockCNN1(X)
        X = self.blockCNN2(X)
        X = self.blockCNN3(X)
        X = self.classifier(torch.flatten(X, 1))
        return X
    