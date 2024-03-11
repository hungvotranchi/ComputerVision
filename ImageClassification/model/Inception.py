import torch
import torch.nn as nn
import torch.nn.functional as F

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, num1x1, num3x3R, num3x3, num5x5R, num5x5, pool_prj):
        super().__init__()
        #1x1 branch
        self.branch_1 = nn.Sequential(
            nn.Conv2d(in_channels= in_channels, out_channels= num1x1, kernel_size= 1),
            nn.BatchNorm2d(num1x1),
            nn.ReLU(True)
        )

        #1x1 -> 3x3 branch
        self.branch_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels= num3x3R, kernel_size=1),
            nn.BatchNorm2d(num3x3R),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num3x3R,out_channels= num3x3, kernel_size=3, padding= 1),
            nn.BatchNorm2d(num3x3),
            nn.ReLU(True)

        )
        
        #1x1 -> 5x5 branch
        self.branch_3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels= num5x5R, kernel_size=1),
            nn.BatchNorm2d(num5x5R),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num5x5R,out_channels= num5x5, kernel_size=5, padding= 2),
            nn.BatchNorm2d(num5x5),
            nn.ReLU(True)

        )

        self.branch_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size= 3, stride= 1, padding= 1),
            nn.Conv2d(in_channels=in_channels, out_channels= pool_prj, kernel_size=1),
            nn.BatchNorm2d(pool_prj),
            nn.ReLU(True)
        )

    def forward(self, X):
        return torch.cat([self.branch_1(X), self.branch_2(X), self.branch_3(X), self.branch_4(X)], 1)

class InceptionNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.pre_inception = nn.Sequential(
            nn.Conv2d(in_channels= 3, out_channels= 64, kernel_size= 7, stride= 2, padding= 3),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size= 3, stride= 2, padding= 1),
            nn.Conv2d(in_channels= 64, out_channels= 192, kernel_size=3, stride= 1, padding = 1),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride= 2, padding = 1)
        )

        self.a3 = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.b3 = InceptionBlock(256, 128, 128, 192, 32, 96, 64)
        self.maxpool = nn.MaxPool2d(kernel_size= 3, stride= 2, padding= 1)

        self.a4 = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
        self.b4 = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
        self.c4 = InceptionBlock(512, 128, 128, 256, 24, 64, 64)
        self.d4 = InceptionBlock(512, 112, 144, 288, 32, 64, 64)
        self.e4 = InceptionBlock(528, 256, 160, 320, 32, 128, 128)

        self.a5 = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
        self.b5 = InceptionBlock(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(in_features= 1024, out_features= num_classes)


    def forward(self, X):
        X = self.pre_inception(X)
        X = self.a3(X)
        X = self.b3(X)
        X = self.maxpool(X)
        X = self.maxpool(self.e4(self.d4(self.c4(self.b4(self.a4(X))))))
        X = self.maxpool(self.b5(self.a5(X)))
        X = self.avgpool(X)
        X = torch.flatten(X, 1)
        X = self.dropout(X)
        X = self.fc(X)
        return X


