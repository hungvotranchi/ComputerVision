import torch
import torch.nn as nn
import torch.nn.functional as F

class ANN(nn.Module):
    def __init__(self, input_feature, output_feature):
        super().__init__()
        self.fully_connected1 = nn.Linear(in_features= input_feature, out_features=256)
        self.fully_connected2 = nn.Linear(256, 128)
        self.fully_connected3 = nn.Linear(128, 64)
        self.output = nn.Linear(64, out_features= output_feature)

    def forward(self, X):
        return self.output(self.fully_connected3(self.fully_connected2(self.fully_connected1(torch.flatten(X, 1)))))
    

