import torch
import torchvision
import torch.nn as nn


class VGG16(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.blockCNN1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64, kernel_size= 3, padding= 1),
            nn.ReLU(),
            nn.Conv2d(in_channels= 64,out_channels=64,kernel_size=3, padding= 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 2, stride= 2)
        )

        self.blockCNN2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels= 128, kernel_size= 3, padding= 1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels= 128, kernel_size= 3, padding= 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 2, stride= 2)
        )
        self.blockCNN3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels= 256, kernel_size= 3, padding= 1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels= 256, kernel_size= 3, padding= 1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels= 256, kernel_size= 3, padding= 1),
            nn.MaxPool2d(kernel_size= 2, stride= 2)
        )
        self.blockCNN4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels= 512, kernel_size= 3, padding= 1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels= 512, kernel_size= 3, padding= 1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels= 512, kernel_size= 3, padding= 1),
            nn.MaxPool2d(kernel_size= 2, stride= 2)
        )
        self.blockCNN5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels= 512, kernel_size= 3, padding= 1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels= 512, kernel_size= 3, padding= 1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels= 512, kernel_size= 3, padding= 1),
            nn.MaxPool2d(kernel_size= 2, stride= 2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features= 512 * 7 * 7, out_features= 4096),
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
        X = self.blockCNN4(X)
        X = self.blockCNN5(X)
        X = self.classifier(torch.flatten(X, 1))
        return X