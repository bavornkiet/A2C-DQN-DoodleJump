import numpy as np
import torch
import os
import random
from torch import nn
from torch.nn import functional as F
# from torch.utils.tensorboard import SummaryWriter
# This module defines the learning process of the A2C algorithm.


def t(x):
    x = np.array(x) if not isinstance(x, np.ndarray) else x
    return torch.from_numpy(x).float()

# discounted_rewards: Calculates the discounted rewards for each time step.


def discounted_rewards(rewards, dones, gamma):
    ret = 0
    discounted = []
    for reward, done in zip(rewards[::-1], dones[::-1]):
        ret = reward + ret * gamma * (1 - done)
        discounted.append(ret)

    return discounted[::-1]


def process_memory(memory, batch_size, device, gamma, discount_rewards=True):
    actions = []
    states = []
    next_states = []
    rewards = []
    dones = []
    mini_sample = []
    if len(memory) > batch_size:
        mini_sample = random.sample(memory, batch_size)  # list of tuples
    else:
        mini_sample = memory
    for action, reward, state, next_state, done in mini_sample:
        actions.append(action if action is not None else np.zeros(3))
        rewards.append(reward if reward is not None else 0)
        states.append(state if state is not None else np.zeros_like(state))
        next_states.append(
            next_state if next_state is not None else np.zeros_like(state))
        dones.append(done if done is not None else False)

    if discount_rewards:
        # if False and dones[-1] == 0:
        #     rewards = discounted_rewards(rewards + [last_value], dones + [0], gamma)[:-1]
        # else:
        rewards = discounted_rewards(rewards, dones, gamma)

    actions = t(actions).to(device)
    states = t(states).to(device)
    next_states = t(next_states).to(device)
    rewards = t(rewards).view(-1, 1).to(device)
    dones = t(dones).view(-1, 1).to(device)
    return actions, rewards, states, next_states, dones

# clip_grad_norm_: Clips gradients to prevent exploding gradients.


def clip_grad_norm_(optimizer, max_grad_norm):
    nn.utils.clip_grad_norm_(
        [p for g in optimizer.param_groups for p in g["params"]], max_grad_norm)


###########
# Hyperparameters: Sets learning rates, discount factor, entropy coefficient, and batch size.
# Optimizer: Uses Adam optimizer for updating the actor-critic network.
class A2CLearner():
    def __init__(self, actorcritic, device, gamma=0.9, entropy_beta=0,
                 actor_lr=4e-4, critic_lr=4e-3, max_grad_norm=0.5, batch_size=1000):
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.actorcritic = actorcritic
        self.entropy_beta = entropy_beta
        self.optimizer = torch.optim.Adam(
            actorcritic.parameters(), lr=actor_lr)
        self.device = device
        self.batch_size = batch_size

    def learn(self, memory, steps, writer, discount_rewards=True):
        actions, rewards, states, next_states, dones = process_memory(
            memory, self.batch_size, self.device, self.gamma, discount_rewards)

        if discount_rewards:
            td_target = rewards
        else:
            with torch.no_grad():
                td_target = rewards + self.gamma * \
                    self.actorcritic(next_states)[1] * (1 - dones)
        value = self.actorcritic(states)[1]
        advantage = td_target - value

        # Actor loss
        norm_dists = self.actorcritic(states)[0]
        logs_probs = norm_dists.log_prob(actions)
        entropy = norm_dists.entropy().mean()

        actor_loss = (-logs_probs * advantage.detach()).mean() - \
            entropy * self.entropy_beta

        # Critic loss
        critic_loss = F.mse_loss(td_target, value)

        total_loss = actor_loss + critic_loss

        # Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        clip_grad_norm_(self.optimizer, self.max_grad_norm)
        self.optimizer.step()

        # Logging
        writer.add_scalar("losses/log_probs", -
                          logs_probs.mean(), global_step=steps)
        writer.add_scalar("losses/entropy", entropy, global_step=steps)
        writer.add_scalar("losses/entropy_beta",
                          self.entropy_beta, global_step=steps)
        writer.add_scalar("losses/actor", actor_loss, global_step=steps)
        writer.add_scalar("losses/advantage",
                          advantage.mean(), global_step=steps)
        writer.add_scalar("losses/critic", critic_loss, global_step=steps)
        writer.add_scalar("losses/total", total_loss, global_step=steps)
