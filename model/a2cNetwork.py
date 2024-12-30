import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super(ActorNetwork, self).__init__()
        """
        A separate Actor network for policy.
        :param state_dim: number of input features from the environment
        :param action_dim: number of discrete actions
        :param hidden_size: size of hidden layers
        """
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.logits = nn.Linear(hidden_size, action_dim)

    def forward(self, x):
        """
        Forward pass to get the raw action logits.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.logits(x)  # raw scores for each action
        return out


class CriticNetwork(nn.Module):
    def __init__(self, state_dim, hidden_size=128):
        super(CriticNetwork, self).__init__()
        """
        A separate Critic network for state-value estimation.
        :param state_dim: number of input features from the environment
        :param hidden_size: size of hidden layers
        """
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        Forward pass to get the state value.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.value_head(x)
        return value
