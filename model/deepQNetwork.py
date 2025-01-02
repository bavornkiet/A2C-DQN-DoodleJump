import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class DQNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=2, bias=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.layer2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.layer3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.fc_layer = nn.Linear(256, 3)

    def forward(self, input_tensor):
        input_tensor = input_tensor.view(-1, 1, 80, 80)
        out1 = F.relu(self.layer1(input_tensor))
        pooled1 = self.pool1(out1)
        out2 = F.relu(self.layer2(pooled1))
        pooled2 = self.pool2(out2)
        out3 = F.relu(self.layer3(pooled2))
        pooled3 = self.pool3(out3)
        flattened = pooled3.view(-1, 256)
        output = self.fc_layer(F.relu(flattened))
        return output

    def save(self, file_name='model.pth', folder_path='./model_dqn'):
        os.makedirs(folder_path, exist_ok=True)
        full_path = os.path.join(folder_path, file_name)
        torch.save(self.state_dict(), full_path)