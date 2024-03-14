import torch
import torch.nn as nn
import torch.nn.functional as F


def Conv2D_fw(input_channels, output_channel, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels=input_channels, out_channels= output_channel, kernel_size= 3, stride= stride, padding=1),
        nn.BatchNorm2d(output_channel),
        nn.ReLU()
    )

def Conv2D_dw(input_channels, output_channel, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels=input_channels, out_channels= input_channels, kernel_size= 3, \
                  stride= stride, padding=1, groups= input_channels),
        nn.BatchNorm2d(input_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels=input_channels, out_channels= input_channels, kernel_size= 1, \
                  stride= 1),
        nn.BatchNorm2d(output_channel),
        nn.ReLU(),
    )
class MobileNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.convfw = Conv2D_fw(input_channels=3, output_channel=32, stride= 2),
        self.convdw = nn.Sequential(
            Conv2D_dw(input_channels=32, output_channel= 64, stride = 1),
            Conv2D_dw(input_channels=64, output_channel=128, stride = 2),
            *[Conv2D_dw(input_channels= 128, output_channel= 128*i, stride = i) for i in range(1,3)],
            *[Conv2D_dw(input_channels= 256, output_channel= 256*i, stride = i) for i in range(1,3)],
            *[Conv2D_dw(input_channels= 512, output_channel= 512, stride = 1) for i in range(1,6)],
            Conv2D_dw(input_channels=512, output_channel=1024, stride = 2),
            Conv2D_dw(input_channels=1024, output_channel=1024, stride = 2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1),
        self.fc = nn.Linear(in_features= 1024, out_features= num_classes)
        
    def forward(self, X):
        X = self.convfw(X)
        X = self.convdw(X)
        X = self.avgpool(X)
        X = self.fc(X)
        return X

