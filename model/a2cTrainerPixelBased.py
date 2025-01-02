import numpy as np
import torch
import os
import random
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

def convert_to_tensor(array):
    if not isinstance(array, np.ndarray):
        array = np.array(array)
    return torch.from_numpy(array).float()

def experience_buffer(
    memory,
    batch_size,
    device,
    gamma,
    discount_flag=True,
    default_state_shape=(1, 80, 80),
    default_action_shape=(1, 3)
):
    
    if len(memory) > batch_size:
        sample_batch = random.sample(memory, batch_size)
    else:
        sample_batch = memory
        
    action_list, reward_list = [], []
    state_list, next_state_list = [], []
    done_list = []

    for (action, reward, state, next_state, done) in sample_batch:
        action_list.append(action if action is not None else np.zeros(default_action_shape))
        reward_list.append(reward if reward is not None else 0.0)
        state_list.append(state if state is not None else np.zeros(default_state_shape))
        next_state_list.append(next_state if next_state is not None else np.zeros(default_state_shape))
        done_list.append(done if done is not None else False)

    if discount_flag:
        discounted_values = []
        running_return = 0.0

        # Process rewards backwards
        for r, d in zip(reward_list[::-1], done_list[::-1]):
            running_return = r + gamma * running_return * (1.0 - d)
            discounted_values.append(running_return)

        # Reverse them to restore chronological order
        reward_list = discounted_values[::-1]

    actions = convert_to_tensor(action_list).to(device)
    states = convert_to_tensor(state_list).to(device)
    next_states = convert_to_tensor(next_state_list).to(device)
    rewards = convert_to_tensor(reward_list).view(-1, 1).to(device)
    dones = convert_to_tensor(done_list).view(-1, 1).to(device)

    return actions, rewards, states, next_states, dones

class A2CTrainer():
    def __init__(
        self,
        model,
        device,
        gamma=0.9,
        entropy_beta=0.0,
        actor_lr=4e-4,
        critic_lr=4e-3,
        max_grad_norm=0.5,
        batch_size=1000
    ):
        self.gamma = gamma
        self.entropy_beta = entropy_beta
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.device = device

        # Combined model (actor & critic)
        self.actor_critic = model

        # Using a single optimizer for all parameters
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=actor_lr)

    def train_model(self, memory, steps, writer, discount_flag=True):
        # 1. Process memory into batched tensors (with optional reward discounting)
        actions, rewards, states, next_states, dones = experience_buffer(
            memory,
            self.batch_size,
            self.device,
            self.gamma,
            discount_flag
        )
        
        # 2. Compute the target (TD target or discounted reward)
        if discount_flag:
            td_target = rewards
        else:
            td_target = rewards + self.gamma * self.actor_critic(next_states)[1] * (1 - dones)
        
        # 3. Compute current value from the critic
        current_values = self.actor_critic(states)[1]
        advantages = td_target - current_values

        # 4. Compute log probabilities & entropy from the actor
        dist = self.actor_critic(states)[0]
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        # 5. Calculate losses
        actor_loss = (-log_probs * advantages.detach()).mean() - entropy * self.entropy_beta
        critic_loss = F.mse_loss(td_target, current_values)
        total_loss = actor_loss + critic_loss

        # 6. Backpropagation & gradient clipping
        total_loss.backward()
        self.optimizer.zero_grad()
        self.optimizer.step()

        writer.add_scalar("loss/log_probs", -log_probs.mean().item(), steps)
        writer.add_scalar("loss/entropy", entropy.item(), steps)
        writer.add_scalar("loss/actor", actor_loss.item(), steps)
        writer.add_scalar("loss/critic", critic_loss.item(), steps)
        writer.add_scalar("loss/total", total_loss.item(), steps)
        writer.add_scalar("loss/advantage_mean", advantages.mean().item(), steps)
        writer.add_scalar("config/entropy_beta", self.entropy_beta, steps)
