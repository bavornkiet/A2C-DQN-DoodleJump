import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class A2CNetwork(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_activation=nn.Tanh):
        super().__init__()

        # Store action dimension
        self.action_dim = action_dim
        
        # Convolutional feature extractor
        self.conv_layer1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4, padding=2)
        self.pool_layer1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.pool_layer2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        
        self.conv_layer3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool_layer3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        
        # A small fully-connected layer to reduce dimensionality after convolutions
        self.feature_transform = nn.Linear(in_features=256, out_features=64)
        
        # Actor branch
        self.actor_head = nn.Sequential(
            nn.Linear(256, 64),
            hidden_activation(),
            nn.Linear(64, 64),
            hidden_activation(),
            nn.Linear(64, action_dim)
        )
        
        # Critic branch
        self.critic_head = nn.Sequential(
            nn.Linear(64, 64),
            hidden_activation(),
            nn.Linear(64, 64),
            hidden_activation(),
            nn.Linear(64, 1)
        )
        
        # Log std parameters for Gaussian policy
        init_log_stds = torch.full((action_dim,), 0.1)
        self.log_sigmas = nn.Parameter(init_log_stds)

    def forward(self, inputs):
        batch_data = inputs.view(-1, 1, 80, 80)
        
        # Convolution layers + pooling
        out = F.relu(self.conv_layer1(batch_data))
        out = self.pool_layer1(out)
        out = F.relu(self.conv_layer2(out))
        out = self.pool_layer2(out)
        out = F.relu(self.conv_layer3(out))
        out = self.pool_layer3(out)

        # Flatten the feature maps
        features = out.view(out.size(0), -1)  # (N, 256)

        # Produce the mean (mu) of the action distribution
        mu = self.actor_head(features)

        # Convert log std parameters to a valid std range
        sigma = torch.clamp(self.log_sigmas.exp(), min=1e-3, max=50.0)

        # Get a reduced representation of the features
        reduced_features = self.feature_transform(features)
        value = self.critic_head(reduced_features)

        # Return a Normal distribution and the predicted value
        policy_dist = torch.distributions.Normal(mu, sigma)
        return policy_dist, value

    def save_model(self, filename='actor_critic_model.pth', directory='./saved_models'):
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, filename)
        torch.save(self.state_dict(), filepath)
        print(f"Model saved to {filepath}")