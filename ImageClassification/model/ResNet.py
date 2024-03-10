import torch
import torchvision
import torch.nn as nn
import torch.functional as F


class ResidualBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size= 3, \
                               stride= stride, padding= 1)
        self.bn1 = nn.BatchNorm2d(num_features= out_channels)

        self.conv2 = nn.Conv2d(in_channels= out_channels, out_channels= out_channels, kernel_size= 3, \
                               stride= 1, padding= 1)
        self.bn2 = nn.BatchNorm2d(num_features= out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels= in_channels, out_channels= out_channels, kernel_size= 1,stride= stride),
                nn.BatchNorm2d(num_features= out_channels)
            )

    def forward(self, X):
        residual = self.downsample(X)
        out_cnn = self.bn1(self.conv1(X))
        out_cnn = self.bn2(self.conv2(out_cnn))
        return nn.functional.relu(residual + out_cnn)
    

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size= 1, bias= False)
        self.bn1 = nn.BatchNorm2d(num_features= out_channels)

        self.conv2 = nn.Conv2d(in_channels= out_channels, out_channels= out_channels, kernel_size= 3, \
                               stride= stride, padding= 1, bias= False)
        self.bn2 = nn.BatchNorm2d(num_features= out_channels)
        self.conv3 = nn.Conv2d(in_channels= out_channels, out_channels= out_channels*self.expansion, kernel_size= 1, bias= False)
        self.bn3 = nn.BatchNorm2d(num_features= out_channels * self.expansion)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels= in_channels, out_channels= out_channels * self.expansion, kernel_size= 1,stride= stride, bias= False),
                nn.BatchNorm2d(num_features= out_channels)
            )

    def forward(self, X):
        residual = self.downsample(X)
        out_cnn = self.bn1(self.conv1(X))
        out_cnn = self.bn2(self.conv2(out_cnn))
        out_cnn = self.bn3(self.conv3(out_cnn))
        return nn.functional.relu(residual + out_cnn)
    
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super().__init__()
        self.in_channel = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(num_features=64)
        )
        self.layer_1 = self._make_layers(block=block, out_channels=64, num_blocks= num_blocks[0], stride=1 )
        self.layer_2 = self._make_layers(block=block, out_channels=128, num_blocks= num_blocks[1], stride=2 )
        self.layer_3 = self._make_layers(block=block, out_channels=256, num_blocks= num_blocks[2], stride=2 )
        self.layer_4 = self._make_layers(block=block, out_channels=512, num_blocks= num_blocks[3], stride=2 )
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)


    def _make_layers(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(in_channels = self.in_channel, out_channels = out_channels, \
                                stride = stride))
            self.in_channel = out_channels * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, X):
        X = nn.functional.relu(self.conv1(X))
        X = nn.functional.max_pool2d(X, kernel_size=3, stride=2, padding=1)
        X = self.layer_4(self.layer_3(self.layer_2(self.layer_1(X))))
        X = self.avgpool(X)
        return self.fc(torch.flatten(X, 1))
    



    
