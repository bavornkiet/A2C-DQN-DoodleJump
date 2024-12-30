import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class Deep_QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 8, 4, bias=True, padding=2)
        self.maxpool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv2 = nn.Conv2d(32, 64, 4, 2, bias=True, padding=1)
        self.maxpool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, bias=True, padding=1)
        self.maxpool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.fc1 = nn.Linear(256, 3)

    def forward(self, x):
        x = x.view(-1,1,80,80)
        conv1_res = F.relu(self.conv1(x))
        maxpool1_res = self.maxpool1(conv1_res)
        conv2_res = F.relu(self.conv2(maxpool1_res))
        maxpool2_res = self.maxpool2(conv2_res)
        conv3_res = F.relu(self.conv3(maxpool2_res))
        maxpool3_res = self.maxpool3(conv3_res)
        flattened_res = torch.reshape(maxpool3_res, (-1, 256))
        fc1_res = self.fc1(F.relu(flattened_res))
        return fc1_res

    def save(self, file_name='model.pth', model_folder_path='./model_dqn'):
        os.makedirs(model_folder_path, exist_ok=True)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class Deep_RQNet(nn.Module):
    def __init__(self):
        super().__init__()

        # In case of LSTM hidden state is a tuple containing both cell state and hidden state
        # self.hidden = (Variable(torch.zeros(1, 1, 256).float()), Variable(torch.zeros(1, 1, 256).float()))

        # GRU has a single hidden state
        # self.hidden = Variable(torch.randn(1, 1, 256).float())
        self.conv1 = nn.Conv2d(1, 32, 8, 4, bias=True, padding=2)
        self.maxpool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv2 = nn.Conv2d(32, 64, 4, 2, bias=True, padding=1)
        self.maxpool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, bias=True, padding=1)
        self.maxpool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rnn = nn.GRU(256, 256, 1)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = x.view(-1,1,80,80)
        conv1_res = F.relu(self.conv1(x))
        maxpool1_res = self.maxpool1(conv1_res)
        conv2_res = F.relu(self.conv2(maxpool1_res))
        maxpool2_res = self.maxpool2(conv2_res)
        conv3_res = F.relu(self.conv3(maxpool2_res))
        maxpool3_res = self.maxpool3(conv3_res)
        flattened_res = torch.reshape(maxpool3_res, (-1, 256))
        flattened_res = flattened_res.unsqueeze(1)
        rnn_res, last_hidden = self.rnn(flattened_res)
        fc1_res = self.fc1(rnn_res)
        fc2_res = self.fc2(fc1_res)
        fc2_res = fc2_res.squeeze(1)
        return fc2_res

    def save(self, file_name='model.pth', model_folder_path='./model_drqn'):
        os.makedirs(model_folder_path, exist_ok=True)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)