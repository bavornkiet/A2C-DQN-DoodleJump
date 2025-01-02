
import copy
import torch
import numpy as np
import psutil
import GPUtil
from scipy.stats import trim_mean
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import random
import os
import datetime
import argparse
import json
from collections import namedtuple, deque
import time
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from game.game import DoodleJump
import sys
import os
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))

class Network(nn.Module):
    def __init__(self,state_dim, action_dim, hidden_layer=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim,hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer,hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer,action_dim)
        )
    def forward(self, features):
        logits = self.network(features)
        return logits
    
Transition = namedtuple("Transition", ("S","A","S_new","R"))
class ReplayMemory:
    def __init__(self,size):
        self.memory = deque([],maxlen=size)
    def append(self,*data):
        self.memory.append(Transition(*data))
    def randomSample(self,n):
        return random.sample(self.memory,n)
    def __len__(self):
        return len(self.memory)

class DQNAgent:
    def __init__(self, network, lr = 1e-3, gamma = 0.99, e_start = 0.9, e_end = 0.05, batchsize = 128, memorysize = 10000, device = torch.device('cpu')):
        print(f"agent using device {device}")
        self.network = network.to(device)
        self.target_network = copy.deepcopy(network).to(device)
        self.lr = lr
        self.tau = 1e-2
        self.e_0 = e_start
        self.e_t = e_end
        self.e_decay = 1000
        self.gamma = gamma
        self.device = device
        self.batchsize = batchsize
        self.output_shape = None
        self.memory = ReplayMemory(memorysize)
        self.optimizer = torch.optim.AdamW(network.parameters(), lr = self.lr, amsgrad=True)
        self.lossfn = nn.SmoothL1Loss()
        self.ep_steps = 0
    def getAction(self,state):
        rng = random.random()
        state = torch.from_numpy(state).view(1,-1).to(device)
        epsilon = self.e_t + (self.e_0 - self.e_t)* np.exp(-self.ep_steps/self.e_decay)
        if self.output_shape == None:
            self.output_shape = self.network(state).shape
            print(f"output shape: {self.output_shape}")
        action = np.zeros(self.output_shape[1])
        #print(action)
        if rng < epsilon:
            #random
            action[random.randint(0,len(action)-1)] = 1
        else:
            with torch.no_grad():
                #print(self.network(state).max(1))
                action[self.network(state).max(1).indices.view(1,1).item()] = 1 #error: <...>.indices has no attribute 'view'
        self.ep_steps += 1
        #print(action)
        return action
    def update(self,state,action,statenew,reward):
        """
        calculate reward with regards to new state?
        
        update memory
        replay in batch
        
        """
        state = torch.from_numpy(state)
        #print(f"state {state}")
        statenew = torch.from_numpy(statenew)
        action = torch.from_numpy(action).max(0).indices
        #print(f"action {action} of type {type(action)} <")
        reward = torch.tensor(reward)
        self.memory.append(state,action,statenew,reward)
        self.replay()
    def replay(self):
        if len(self.memory) < self.batchsize:
            return
        batch = self.memory.randomSample(self.batchsize)
        #[print(t) for t in batch]
        batch = Transition(*zip(*batch))
        batch_state = torch.stack(list(batch.S)).to(device)
        #print(f"batch state shape {batch_state.shape}")
        batch_actions = torch.tensor(batch.A).unsqueeze(-1).to(device)
        #print(f"batch actions shape {batch_actions.shape}")
        batch_rewards = torch.tensor(batch.R).unsqueeze(-1).to(device)
        #print(f"batch rewards shape {batch_rewards.shape}")

        output = model(batch_state).gather(1,batch_actions).to(device)
        with torch.no_grad():
            expected  = batch_rewards + self.gamma*self.target_network(batch_state).max(1).values.unsqueeze(-1).to(device)
        loss = self.lossfn(output,expected).to(device)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    def soft_update_target(self):
        target_net_state = self.target_network.state_dict()
        policy_net_state = self.network.state_dict()
        for key in target_net_state:
            target_net_state[key] = policy_net_state[key]*self.tau + target_net_state[key]*(1-self.tau)
        self.target_network.load_state_dict(target_net_state)


    


        
"""
for episodes:
    game = game()
    while not done:
        game.getstate()
        agent.getaction()
        game.resolveaction()
        agent.add(state,action,reward,newstate)
        if game = done():
            done
            agent.update()?

"""


def get_available_device():
    gpus = GPUtil.getGPUs()
    if not gpus:
        print("No GPU found. Using CPU.")
        return torch.device('cpu')

    # Find RTX3060 by name
    for gpu in gpus:
        if '3060' in gpu.name:
            print(f"Using GPU: {gpu.name}")
            return torch.device(f'cuda:{gpu.id}')

    # If RTX3060 not found, use the first available GPU
    print(f"Using GPU: {gpus[0].name}")
    return torch.device(f'cuda:{gpus[0].id}')


def log_resource_usage(writer, episode):
    # CPU and RAM usage
    cpu_percent = psutil.cpu_percent()
    ram = psutil.virtual_memory()
    ram_percent = ram.percent

    # GPU usage
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu = gpus[0]  # Assuming RTX3060 is first
        gpu_load = gpu.load * 100  # Convert to percentage
        gpu_memory_used = gpu.memoryUsed  # In MB
        gpu_memory_total = gpu.memoryTotal  # In MB
    else:
        gpu_load = 0
        gpu_memory_used = 0
        gpu_memory_total = 0

    # Log to TensorBoard
    writer.add_scalar('Resource/CPU_Usage', cpu_percent, episode)
    writer.add_scalar('Resource/RAM_Usage', ram_percent, episode)
    writer.add_scalar('Resource/GPU_Usage', gpu_load, episode)
    writer.add_scalar('Resource/GPU_Memory_Used', gpu_memory_used, episode)


def parse_args():
    parser = argparse.ArgumentParser(description="DQN Agent for DoodleJump")
    parser.add_argument("--reward_type", type=int,
                        default=2, choices=[1, 2, 3])
    parser.add_argument("--server", action="store_true")
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--max_games', type=int, default=1000)
    parser.add_argument('--experiment_name', type=str, default='exp1')
    parser.add_argument('--test', action='store_true',
                        help="If set, runs the agent in test mode")
    return parser.parse_args()

if __name__ == '__main__':
    #parse args
    args = parse_args()
    device = get_available_device()
    # Create folder structure for saving parameters
    parameters_folder = os.path.join(
        "Parameters", "gamma_testing", args.experiment_name)
    if not os.path.exists(parameters_folder):
        os.makedirs(parameters_folder)
    env = DoodleJump(
        server=args.server,
        reward_type=args.reward_type,
        FPS=0,
        render_skip=0
        # add other env configs as needed
    )
    # Initialize TensorBoard SummaryWriter
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_log_dir = os.path.join(
        "runs", "gamma_testing", f"{args.experiment_name}")
    writer = SummaryWriter(log_dir=tensorboard_log_dir)
    #initialise network
     # We need to figure out the dimension of your state and action
    # By default, from your environment code, you get: getFeatures returns a 1D array
    sample_state = env.getFeatures()
    state_dim = sample_state.shape[0]
    action_dim = 3
    model = Network(state_dim, action_dim).to(device)
    agent = DQNAgent(model, lr = args.learning_rate, gamma = args.gamma, device = device)
    best_score = float("-inf")
    all_scores = []
    all_rewards = []
    if not args.test:
        t0 = time.time()
        for episode in range(args.max_games):
            state = env.getFeatures()  # Reset-like logic? how does this even work
            done = False
            episode_rewards = []
            episode_states = []
            episode_actions = []
            episode_log_probs = []
            episode_values = []
            episode_dones = []
            state = env.getFeatures()
            while not done:
                """
                current state = get state
                agent.getaction(current state)
                getfeedback()
                agent.update(state,action,nextstate,reward)
                agent.replay()
                agent.calculateloss()
                """
                
                action = agent.getAction(state)
                reward,done,score = env.agentPlay(action)
                nextstate = env.getFeatures()
                agent.update(state,action,nextstate,reward)

                episode_states.append(state)
                episode_actions.append(action.max(-1))
                episode_rewards.append(reward)
                episode_dones.append(done)
                state = nextstate
                agent.soft_update_target()

            # Some logging
            ep_score = score
            all_scores.append(ep_score)
            ep_reward = sum(episode_rewards)
            all_rewards.append(ep_reward)
            best_score = max(best_score, ep_score)

            log_resource_usage(writer, episode)
            mean_score = np.mean(all_scores)

            mean_reward = np.mean(all_rewards)
            writer.add_scalar(
                'Training/Mean_Score', mean_score, episode)
            writer.add_scalar(
                'Training/Mean_Reward', mean_reward, episode)
            writer.add_scalar(
                'Training/High_Score', best_score, episode)

            if (episode + 1) % 10 == 0:
                print(f"Episode: {episode+1}, Score: {ep_score}, Reward: {ep_reward}")

        total_train_time = time.time() - t0
        trimmed_avg = trim_mean(all_scores, 0.1)

        # ============ Save final model =============
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        network_save_path = os.path.join(
            parameters_folder, f"model_dqnfeatures_{timestamp}.pth")

        torch.save(agent.network.state_dict(), network_save_path)

        # ============ Save summary =============
        summary_file = os.path.join(parameters_folder, "training_summary.txt")
        with open(summary_file, "w") as f:
            f.write("===== Training Summary =====\n")
            f.write(f"Experiment Name: {args.experiment_name}\n")
            f.write(f"Total Training Time: {total_train_time:.2f} s\n")
            f.write(f"Max Episodes: {args.max_games}\n")
            f.write(f"Learning Rate: {args.learning_rate}\n")
            f.write(f"Reward Type: {args.reward_type} \n")
            f.write(f"Gamma: {args.gamma}\n")
            f.write(f"Best Score Achieved: {best_score}\n")
            f.write(f"Mean Score: {np.mean(all_scores):.2f}\n")
            f.write(f"Trimmed Mean Score: {trimmed_avg:.2f}\n")
            f.write(f"Mean Reward: {np.mean(all_rewards):.2f}\n")
            f.write(f"Number of Episodes: {len(all_scores)}\n")
            f.write("===========================\n")

        print("Training complete. Models and summary saved in:", parameters_folder)
    else:
        print("Running in test mode. Evaluate agent's performance...")
        model_load_path = "Parameters/reward_testing/expR3-2/model_actor_20241228-210124.pth"
        model.load_state_dict(torch.load(model_load_path, map_location=device))
        model.eval()
        for _ in range(10):
            state = env.getFeatures()
            done = False
            while not done:
                action = agent.getAction(state)
                reward, done, score = env.agentPlay(action)
                state = env.getFeatures()
            print("Finished test episode with score:", score)

    # For discrete actions: let's assume you have 3 actions: [Left, Idle, Right].
    # Adjust accordingly if your environment has a different action space.
    
    #start game etc
    #do some extras?
    