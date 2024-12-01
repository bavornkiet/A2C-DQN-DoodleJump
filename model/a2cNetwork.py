import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class ActorCritic(nn.Module):
    def __init__(self, state_dim, n_actions, activation=nn.Tanh):
        super().__init__()
        self.n_actions = n_actions

        # Common network
        self.fc_common = nn.Sequential(
            nn.Linear(state_dim, 128),
            activation(),
            nn.Linear(128, 128),
            activation(),
        )

        # Actor network
        self.fc_actor = nn.Sequential(
            nn.Linear(128, 64),
            activation(),
            nn.Linear(64, n_actions),
        )

        # Critic network
        self.fc_critic = nn.Sequential(
            nn.Linear(128, 64),
            activation(),
            nn.Linear(64, 1),
        )

        # Log standard deviations for action distribution
        logstds_param = nn.Parameter(torch.full((n_actions,), 0.1))
        self.register_parameter("logstds", logstds_param)

    def forward(self, X):
        x = X
        common_res = self.fc_common(x)
        # Actor output
        means = self.fc_actor(common_res)
        stds = torch.clamp(self.logstds.exp(), 1e-3, 50)
        # Critic output
        value = self.fc_critic(common_res)
        # Return action distribution and value estimate
        return torch.distributions.Normal(means, stds), value

    def save(self, file_name='model.pth', model_folder_path=None):
        if model_folder_path is None:
            # Default to saving in the 'Parameters' folder in the root directory
            model_folder_path = os.path.join(
                os.path.dirname(__file__), '..', 'Parameters')
        else:
            # Use the provided path, adjusted relative to the root directory
            model_folder_path = os.path.join(
                os.path.dirname(__file__), '..', model_folder_path)

        os.makedirs(model_folder_path, exist_ok=True)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
