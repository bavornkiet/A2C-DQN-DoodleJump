from model.a2cTrainer import A2CTrainer
from model.a2cNetwork import ActorNetwork, CriticNetwork
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
import time
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from game.game import DoodleJump
import sys
import os
import numpy as np
import torch
# from model.a2cNetwork import ActorCritic
# from model.a2cTrainer import A2CLearner

sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))


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
    parser = argparse.ArgumentParser(description="A2C Agent for DoodleJump")
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


def main():
    args = parse_args()
    device = get_available_device()
    # Create folder structure for saving parameters
    parameters_folder = os.path.join(
        "Parameters", "gamma_testing", args.experiment_name)
    if not os.path.exists(parameters_folder):
        os.makedirs(parameters_folder)

    # Initialize TensorBoard SummaryWriter
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_log_dir = os.path.join(
        "runs", "gamma_testing", f"{args.experiment_name}")
    writer = SummaryWriter(log_dir=tensorboard_log_dir)

    # Initialize Environment
    env = DoodleJump(
        server=args.server,
        reward_type=args.reward_type,
        FPS=1000,
        render_skip=0
        # add other env configs as needed
    )

    # We need to figure out the dimension of your state and action
    # By default, from your environment code, you get: getFeatures returns a 1D array
    sample_state = env.getFeatures()
    state_dim = sample_state.shape[0]

    # For discrete actions: let's assume you have 3 actions: [Left, Idle, Right].
    # Adjust accordingly if your environment has a different action space.
    action_dim = 3

    # Initialize Actor-Critic networks
    actor = ActorNetwork(state_dim, action_dim, hidden_size=128).to(device)
    critic = CriticNetwork(state_dim, hidden_size=128).to(device)

    # Initialize A2C trainer
    trainer = A2CTrainer(
        actor=actor,
        critic=critic,
        lr=args.learning_rate,
        gamma=args.gamma,
        beta_entropy=0.001,  # optional
        device=device
    )
    total_train_time = 0.0
    best_score = float("-inf")
    all_scores = []
    all_rewards = []

    if not args.test:
        # --------------- TRAIN MODE ---------------
        start_time = time.time()
        for episode in range(args.max_games):
            state = env.getFeatures()  # Reset-like logic
            done = False

            episode_rewards = []
            episode_states = []
            episode_actions = []
            episode_log_probs = []
            episode_values = []
            episode_dones = []

            while not done:
                # 1) Convert state to torch tensor
                state_t = torch.FloatTensor(state).unsqueeze(
                    0).to(device)  # shape [1, state_dim]

                # 2) Actor forward pass -> raw logits
                logits = actor(state_t)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()  # sample an action
                log_prob = dist.log_prob(action)

                # 3) Critic forward pass -> value
                value = critic(state_t)  # shape []

                # 4) Step the environment
                action_index = action.item()

                action_list = [0, 0, 0]
                action_list[action_index] = 1

                reward, done, score = env.agentPlay(action_list)
                # env.agentPlay returns reward, terminal, score

                # 5) Collect transitions
                episode_states.append(state)
                episode_actions.append(action_index)
                episode_values.append(value)
                episode_log_probs.append(log_prob)
                episode_rewards.append(reward)
                episode_dones.append(done)

                # 6) Next state
                next_state = env.getFeatures()
                state = next_state

            # ============ End of Episode ==============
            if not env.is_terminal_state():
                # If not terminal, get critic value for next_state
                next_state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                last_value = critic(next_state_t)
            else:
                last_value = 0.0

            # 7) Compute returns
            returns = trainer.compute_returns(
                episode_rewards, episode_dones, last_value)

            # 8) Update the networks
            actor_loss, critic_loss = trainer.update(
                states=episode_states,
                actions=episode_actions,
                returns=returns,
                values=episode_values,
                log_probs=episode_log_probs
            )

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
                print(f"Episode: {episode+1}, Score: {ep_score}, Reward: {ep_reward}, "

                      f"Actor Loss: {actor_loss:.3f}, Critic Loss: {critic_loss:.3f}")

            if (episode + 1) % 200 == 0:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                x = episode+1
                actor_save_path = os.path.join(
                    parameters_folder, f"model_actor_{timestamp}_{x}.pth")
                critic_save_path = os.path.join(
                    parameters_folder, f"model_critic_{timestamp}_{x}.pth")
                torch.save(actor.state_dict(), actor_save_path)
                torch.save(critic.state_dict(), critic_save_path)

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        actor_save_path = os.path.join(
            parameters_folder, f"model_actor_{timestamp}_final.pth")
        critic_save_path = os.path.join(
            parameters_folder, f"model_critic_{timestamp}_final.pth")

        total_train_time = time.time() - start_time
        trimmed_avg = trim_mean(all_scores, 0.1)

        # ============ Save final model =============

        torch.save(actor.state_dict(), actor_save_path)
        torch.save(critic.state_dict(), critic_save_path)

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
        actor_load_path = "Parameters/reward_testing/99R3-2/model_actor_20241230-014653_final.pth"
        critic_load_path = "Parameters/gamma_testing/99R3-2/model_critic_20241230-014653_final.pth"
        actor.load_state_dict(torch.load(actor_load_path, map_location=device))
        critic.load_state_dict(torch.load(
            critic_load_path, map_location=device))
        actor.eval()
        critic.eval()
        for _ in range(10):
            state = env.getFeatures()
            done = False
            while not done:
                state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                logits = actor(state_t)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample().item()

                action_list = [0, 0, 0]
                action_list[action] = 1

                reward, done, score = env.agentPlay(action_list)
                state = env.getFeatures()
            print("Finished test episode with score:", score)


if __name__ == "__main__":
    main()
