import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, input_feature, hidden_units, output_feature):
        super().__init__()
        #in_channels -> RGB | out_channels -> Total of filer conv | kernel_size -> Size of filter
        self.conv1 = nn.Conv2d(input_feature, hidden_units, 5)
        self.pool = nn.MaxPool2d(2, 2)
        #in_channels = prev_out_channels | out_channels -> Total of filer conv | kernel_size -> Size of filter
        self.conv2 = nn.Conv2d(hidden_units, hidden_units * 2, 5)
        #in_channels = size of filter * num of out_channels | out_channels -> self-defined
        self.fully_connected1 = nn.Linear(hidden_units * 5 * 5 * 2, 256)
        self.fully_connected2 = nn.Linear(256, 64)
        self.output = nn.Linear(64, out_features= output_feature)

    def forward(self, X):
        x = F.relu(self.pool(self.conv1(X)))
        x = F.relu(self.pool(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.fully_connected1(x)
        x = self.fully_connected2(x)
        x = self.output(x)
        return x

