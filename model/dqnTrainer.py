import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


class QLearningTrainer:
    def __init__(self, model, learning_rate, discount_rate, device, channel_count, epsilon):
        super().__init__()
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.network = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        self.loss_function = nn.MSELoss()
        self.device = device
        self.network.to(self.device)
        
        # Image normalization parameters
        self.mean_values = (0.485, 0.456, 0.406)
        self.std_devs = (0.229, 0.224, 0.225)
        if channel_count == 1:
            self.mean_values = np.mean(self.mean_values)
            self.std_devs = np.mean(self.std_devs)
            
        self.mean_tensor = torch.tensor(self.mean_values).view(channel_count, 1, 1).cpu()
        self.std_tensor = torch.tensor(self.std_devs).view(channel_count, 1, 1).cpu()
        self.upper_bound = ((1 - self.mean_tensor) / self.std_tensor)
        self.lower_bound = ((0 - self.mean_tensor) / self.std_tensor)
        self.epsilon_tensor = epsilon / self.std_tensor
        self.step_size = (1.25 * epsilon) / self.std_tensor
        self.step_counter = 0 

    def perform_training_step(self, current_state, actions, rewards, next_states, done_flags):
        current_state = torch.tensor(current_state, dtype=torch.float).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        
        # Expand dimensions for single batch
        if current_state.shape[0] == 1:
            current_state = torch.unsqueeze(current_state, 0)
            next_states = torch.unsqueeze(next_states, 0)
            actions = torch.unsqueeze(actions, 0)
            rewards = torch.unsqueeze(rewards, 0)
            done_flags = (done_flags, )

        # Forward pass for next state predictions
        self.network.eval()
        with torch.no_grad():
            predicted_next = self.network(next_states)
        
        # Forward pass for current state predictions
        self.network.train()
        predicted_current = self.network(current_state)
        
        # Update Q values
        target_values = predicted_current.clone().detach()
        for idx in range(len(done_flags)):
            updated_Q = rewards[idx]
            if not done_flags[idx] and idx != len(done_flags) - 1:
                updated_Q = rewards[idx] + self.discount_rate * torch.max(predicted_next[idx])
            target_values[idx][torch.argmax(actions[idx]).item()] = updated_Q

        # Compute loss and backpropagate
        self.optimizer.zero_grad()
        loss = self.loss_function(predicted_current, target_values)
        loss.backward()
        self.optimizer.step()
        return float(loss)

    @staticmethod
    def clamp_tensor(tensor, lower_bound, upper_bound):
        return torch.max(torch.min(tensor, upper_bound), lower_bound)
    
    def generate_adversarial_state(self, current_state):
        current_state = torch.tensor(current_state, dtype=torch.float).to(self.device)
        current_state = torch.unsqueeze(current_state, 0)

        self.network.eval()
        with torch.no_grad():
            predicted = self.network(current_state)
        
        # Create adversarial perturbation
        action_vector = [0, 0, 0]
        selected_action = torch.argmax(predicted).item()
        action_vector[selected_action] = 1
        action_vector = torch.tensor(action_vector, dtype=torch.float).to(self.device)
        
        perturbation = torch.zeros_like(current_state).to(self.device)
        for i in range(len(self.epsilon_tensor)):
            perturbation[:, i, :, :].uniform_(-self.epsilon_tensor[i][0][0].item(), self.epsilon_tensor[i][0][0].item())
        perturbation.data = self.clamp_tensor(perturbation, self.lower_bound - current_state, self.upper_bound - current_state)
        perturbation.requires_grad = True

        self.network.train()
        output = self.network((current_state + perturbation[:current_state.size(0)]).float())
        
        self.optimizer.zero_grad()
        loss = self.loss_function(output, action_vector)
        loss.backward()
        gradient = perturbation.grad.detach()
        perturbation.data = self.clamp_tensor(perturbation + self.step_size * torch.sign(gradient), -self.epsilon_tensor, self.epsilon_tensor)
        perturbation.data[:current_state.size(0)] = self.clamp_tensor(perturbation[:current_state.size(0)], self.lower_bound - current_state, self.upper_bound - current_state)
        perturbation = perturbation.detach()
        self.step_counter += 1

        return perturbation[:current_state.size(0)]