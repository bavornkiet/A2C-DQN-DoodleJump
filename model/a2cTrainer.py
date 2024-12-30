import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time


class A2CTrainer:

    def __init__(self, actor, critic, lr=1e-3, gamma=0.99, beta_entropy=0.001, device=torch.device('cpu')):

        self.actor = actor
        self.critic = critic
        self.gamma = gamma
        self.beta_entropy = beta_entropy
        self.device = device
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.loss_fn = nn.SmoothL1Loss()

    def compute_returns(self, rewards, dones, last_value):
        """
        Compute discounted returns for each timestep in a rollout/episode.
        :param rewards: list of rewards at each timestep
        :param dones: list of done booleans at each timestep
        :param last_value: the predicted value for the last state
        :return: array of discounted returns
        """
        returns = []
        R = last_value
        for r, done in zip(reversed(rewards), reversed(dones)):
            if done:
                R = 0
            R = r + self.gamma * R
            returns.insert(0, R)
        return returns

    def update(self, states, actions, returns, values, log_probs):
        """
        Update actor and critic given the rollout buffers.
        :param states: tensor of states
        :param actions: tensor of actions taken
        :param returns: tensor of discounted returns
        :param values: tensor of critic values
        :param log_probs: tensor of log(pi(a|s))
        """

        returns = torch.FloatTensor(returns).to(self.device)
        values = torch.cat(values, dim=0).to(self.device).view(-1)
        log_probs = torch.cat(log_probs, dim=0).to(self.device).view(-1)
        actions = torch.LongTensor(actions).to(self.device)

        critic_loss = self.loss_fn(values, returns)

        advantages = returns - values
        actor_loss = -(log_probs * advantages.detach()).mean()

        entropy_loss = 0.0  # or implement if needed

        total_actor_loss = actor_loss + self.beta_entropy * entropy_loss

        # ====== BACKPROP ======
        # Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor
        self.actor_optimizer.zero_grad()
        total_actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item(), critic_loss.item()
