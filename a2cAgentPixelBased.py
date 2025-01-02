import argparse
import datetime
import json
import os
import random
import sys
import time

import cv2
import numpy as np
import psutil
import torch
import torch.nn as nn
import GPUtil

from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from scipy.stats import trim_mean

from game.game import DoodleJump
from model.a2cNetworkPixelBased import A2CNetwork
from model.a2cTrainerPixelBased import A2CTrainer

def parse_args():
    parser = argparse.ArgumentParser(description='A2C Pixel-Based Doodle Jump')

    # 1. System
    parser.add_argument("--test", action="store_true", 
                        help="Play the game with a trained agent.")
    parser.add_argument("--server", action="store_true", 
                        help="Train on the server without rendering the game.")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Randomize the game by changing the seed value.")
    parser.add_argument("--max_games", type=int, default=500, 
                        help="Set the maximum number of games to be played.")
    parser.add_argument('--experiment_name', type=str, default='exp1',
                        help="Name for the experiment to store results.")

    # 2. Hyperparameters
    parser.add_argument("-d", "--difficulty", type=str, default="EASY", 
                        choices=["EASY", "MEDIUM", "HARD"], 
                        help="Select the game difficulty.")
    parser.add_argument("-m", "--model", type=str, default="a2c", 
                        choices=["a2c"], 
                        help="Select the RL model.")
    parser.add_argument("-lr", "--learning_rate", type=float, default=4e-4, 
                        help="Set the learning rate.")
    parser.add_argument("-g", "--gamma", type=float, default=0.9, 
                        help="Set the discount factor.")
    parser.add_argument("--max_memory", type=int, default=10000, 
                        help="Set the memory size of the buffer.")
    parser.add_argument("--batch_size", type=int, default=1000, 
                        help="Set the batch size.")
    parser.add_argument("--exploration", type=int, default=40, 
                        help="Set the number of games to explore.")
    parser.add_argument("--reward_type", type=int, default=1, 
                        choices=[1, 2, 3], 
                        help="Set the reward type.")
    parser.add_argument("--store_frames", action="store_true", 
                        help="Store the frames encountered during gameplay by the agent.")

    # 3. Image
    parser.add_argument("--height", type=int, default=80, 
                        help="Set the height of the image after resizing.")
    parser.add_argument("--width", type=int, default=80, 
                        help="Set the width of the image after resizing.")
    parser.add_argument("--channels", type=int, default=1, 
                        help="Set the number of channels for preprocessing.")

    return parser.parse_args()

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

def convert_to_tensor(array):
    if not isinstance(array, np.ndarray):
        array = np.array(array)
    return torch.from_numpy(array).float()

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_):
        return input_ * torch.tanh(F.softplus(input_))

class EpisodeManager:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        
        # Episode state
        self.current_frame = None
        self.is_done = True
        self.total_steps = 0
        self.episode_reward = 0.0
        self.cumulative_episode_rewards = []
        self.mean_reward = 0.0
        self.scores = []
        self.best_score = 0
        self.num_episodes = 0
        self.frame_counter = 1  # For storing frames

        # Image and training hyperparameters
        self.discount_factor = config.gamma
        self.sample_batch_size = config.batch_size
        self.store_frames = config.store_frames
        self.img_height = config.height
        self.img_width = config.width
        self.img_channels = config.channels

        # Set random seeds
        self._set_random_seeds(config.seed)

        # Choose device (CPU or GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = get_available_device()

    def reset_environment(self):
        self.episode_reward = 0.0
        self.is_done = False
        self.current_frame = None
        self.env.gameReboot()

    @staticmethod
    def _set_random_seeds(seed_value):
        os.environ['PYTHONHASHSEED'] = str(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        np.random.seed(seed_value)
        random.seed(seed_value)

    def _rotate_image(self, image, angle=270):
        center = (self.img_width / 2, self.img_height / 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, rotation_matrix, (self.img_height, self.img_width))

    def _store_image(self, image, file_index, folder="./image_dump"):
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{file_index}.jpg")
        cv2.imwrite(path, image)

    def _normalize_gray_image(self, image, mean_values, std_values):
        img_norm = image / 255.0
        mean = np.mean(mean_values)
        std = np.mean(std_values)
        return (img_norm - mean) / std

    def _normalize_rgb_image(self, image, mean_values, std_values):
        img_norm = image / 255.0
        return (img_norm - mean_values) / std_values

    def _preprocess_frame(self, frame):
        # 1. Resize
        resized_img = cv2.resize(frame, (self.img_width, self.img_height))

        # 2. Rotate
        rotated_img = self._rotate_image(resized_img, angle=270)

        # 3. Optionally store frame
        if self.store_frames:
            self._store_image(rotated_img, self.frame_counter)
            self.frame_counter += 1

        # 4/5. Convert to grayscale + normalization OR RGB normalization
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]

        if self.img_channels == 1:
            # Convert to grayscale
            gray_img = cv2.cvtColor(rotated_img, cv2.COLOR_BGR2GRAY)
            normalized_img = self._normalize_gray_image(gray_img, imagenet_mean, imagenet_std)
            normalized_img = np.expand_dims(normalized_img, axis=0)  # (1, H, W)
        else:
            # Keep RGB
            normalized_img = self._normalize_rgb_image(rotated_img, imagenet_mean, imagenet_std)
            normalized_img = normalized_img.transpose((2, 0, 1))  # (C, H, W)

        # 6. Add batch dimension → (1, C, H, W)
        batch_frame = np.expand_dims(normalized_img, axis=0)
        return batch_frame

    def get_preprocessed_state(self):
        raw_frame = self.env.getPixelFrame()
        preprocessed = self._preprocess_frame(raw_frame)
        return preprocessed

    @staticmethod
    def convert_continuous_to_discrete_actions(actions_cont):
        discrete_action = [0, 0, 0]
        max_index = np.argmax(actions_cont)
        discrete_action[max_index] = 1
        return discrete_action

    @staticmethod
    def _check_experience_buffer(buffer, max_size):
        if len(buffer) > max_size:
            buffer.clear()

    def _log_episode_progress(self, writer, final_score):
        self.num_episodes += 1
        self.cumulative_episode_rewards.append(self.episode_reward)

        # Update the best score if needed
        if final_score > self.best_score:
            self.best_score = final_score

        self.scores.append(final_score)
        mean_score = np.mean(self.scores)

        print(
            f"Episode: {self.num_episodes} | "
            f"Episode Reward: {self.episode_reward:.2f} | "
            f"Score: {final_score} | "
            f"Best Score: {self.best_score} | "
            f"Steps So Far: {self.total_steps}"
        )

        # TensorBoard logging
        writer.add_scalar('Training/High_Score', self.best_score, self.num_episodes)
        writer.add_scalar('Training/Mean_Score', mean_score, self.num_episodes)

    def _process_single_step(self, model, writer, experience_buffer):
        # Get current state
        current_state_img = self.get_preprocessed_state()
        current_state_tensor = convert_to_tensor(current_state_img).to(self.device)

        # Sample action from the model
        action_dist = model(current_state_tensor)[0]
        sampled_action_tensor = action_dist.sample()
        continuous_actions = sampled_action_tensor.detach().cpu().numpy()
        clipped_actions = np.clip(continuous_actions, -1, 1)
        discrete_action = self.convert_continuous_to_discrete_actions(clipped_actions)

        # Step in the environment
        reward, self.is_done, score = self.env.agentPlay(discrete_action)

        # Get next state
        next_state_img = self.get_preprocessed_state()

        # Store transition
        experience_buffer.append(
            (continuous_actions, reward, current_state_img, next_state_img, self.is_done)
        )

        # Update stats
        self.episode_reward += reward
        self.total_steps += 1
        self.mean_reward = self.episode_reward / max(1, self.total_steps)

        # Log incremental reward to TensorBoard
        writer.add_scalar("Training/Mean_Reward", self.mean_reward, self.total_steps)

        # If episode finishes, log progress and reset
        if self.is_done:
            self._log_episode_progress(writer, score)
            self.reset_environment()

    def run_n_steps(self, model, writer, max_steps=16, experience_buffer=None):
        if experience_buffer is None:
            experience_buffer = []

        # Ensure the buffer doesn't exceed config.max_memory
        self._check_experience_buffer(experience_buffer, self.config.max_memory)

        # Collect steps
        for _ in range(max_steps):
            if self.is_done:
                self.reset_environment()

            self._process_single_step(model, writer, experience_buffer)

        return experience_buffer

def train_loop(config, writer, env, model):
    manager = EpisodeManager(env, config)
    print(f"Using device: {manager.device}")

    # A2C parameters
    entropy_beta = 0.0
    max_grad_norm = 0.5

    # Create the A2CLearner
    trainer = A2CTrainer(
        model=model,
        device=manager.device,
        gamma=config.gamma,
        entropy_beta=entropy_beta,
        actor_lr=config.learning_rate,
        critic_lr=config.learning_rate,
        max_grad_norm=max_grad_norm,
        batch_size=config.batch_size
    )

    print("TRAINING...")
    max_steps_per_iteration = 16
    start_time = time.time()

    # Roll out episodes until max_games is reached
    while manager.num_episodes < config.max_games:
        # Collect transitions
        transitions = manager.run_n_steps(model, writer, max_steps=max_steps_per_iteration)
        # Perform one A2C update step
        trainer.train_model(transitions, manager.total_steps, writer, discount_flag=False)

    total_time = time.time() - start_time

    # Create directory for saving model parameters
    parameters_folder = os.path.join("Parameters", config.experiment_name)
    os.makedirs(parameters_folder, exist_ok=True)
    model_save_path = os.path.join(parameters_folder, "a2c_model_final.pth")
    torch.save(model.state_dict(), model_save_path)

    # Write a summary of training
    best_score = manager.best_score
    all_scores = manager.scores
    trimmed_avg = trim_mean(all_scores, 0.1)
    summary_file_path = os.path.join(parameters_folder, "training_summary.txt")

    with open(summary_file_path, "w") as f:
        f.write("===== Training Summary =====\n")
        f.write(f"Experiment Name: {config.experiment_name}\n")
        f.write(f"Total Training Time: {total_time:.2f} s\n")
        f.write(f"Max Games: {config.max_games}\n")
        f.write(f"Learning Rate: {config.learning_rate}\n")
        f.write(f"Reward Type: {config.reward_type}\n")
        f.write(f"Gamma: {config.gamma}\n")
        f.write(f"Best Score Achieved: {best_score}\n")
        f.write(f"Mean Score: {np.mean(all_scores):.2f}\n")
        f.write(f"Trimmed Mean Score (10%): {trimmed_avg:.2f}\n")
        f.write(f"Mean Reward: {manager.mean_reward:.2f}\n")
        f.write(f"Number of Games Played: {manager.num_episodes}\n")
        f.write("===========================\n")

    print("Training complete.")
    print(f"Models and summary saved in: {parameters_folder}")

def test_loop(config, writer, env, model):
    print("TESTING...")
    manager = EpisodeManager(env, config)
    best_score_ever = 0

    while manager.num_episodes < config.max_games:
        # 1. Get current preprocessed state
        current_state_img = manager.get_preprocessed_state()
        current_state_tensor = convert_to_tensor(current_state_img).to(manager.device)

        # 2. Sample action from the trained model
        action_dist = model(current_state_tensor)[0]
        action_tensor = action_dist.sample()
        continuous_actions = action_tensor.detach().cpu().numpy()
        clipped_actions = np.clip(continuous_actions, -1, 1)
        discrete_action = manager.convert_continuous_to_discrete_actions(clipped_actions)

        # 3. Step the environment
        _, is_done, score = env.agentPlay(discrete_action)

        # 4. If episode is done, log and reset
        if is_done:
            manager.num_episodes += 1
            env.gameReboot()
            if score > best_score_ever:
                best_score_ever = score

            # Example logging to console (you can use TensorBoard if desired)
            print(
                f"Test Episode: {manager.num_episodes} | "
                f"Score: {score} | "
                f"Best Score So Far: {best_score_ever}"
            )

def main():
    config = parse_args()

    # Create experiment directories
    parameters_folder = os.path.join("Parameters", config.experiment_name)
    os.makedirs(parameters_folder, exist_ok=True)

    # TensorBoard
    tensorboard_log_dir = os.path.join("runs", f"{config.experiment_name}")
    writer = SummaryWriter(log_dir=tensorboard_log_dir)
    writer.add_text('Model Parameters:', str(vars(config)), 0)

    # Create environment
    env = DoodleJump(
        server=config.server,
        reward_type=config.reward_type,
        FPS=30000,  # Speed up the game if desired
        render_skip=0
    )

    # Create model
    # For pixel-based states, shape is typically (N, C, H, W).
    # For a discrete action space of size 3:
    temp_manager = EpisodeManager(env, config)
    state_dim = temp_manager.get_preprocessed_state().shape[0]  # C dimension
    action_dim = 3
    a2c_model = A2CNetwork(
        state_dim,
        action_dim,
        hidden_activation=Mish
    ).to(temp_manager.device)

    # If in test mode, load the trained model from disk
    if config.test:
        model_path = ""
        if os.path.isfile(model_path):
            a2c_model.load_state_dict(torch.load(model_path, map_location=temp_manager.device))
            a2c_model.eval()
        else:
            print(f"Model path '{model_path}' not found. Exiting.")
            sys.exit(1)

        # Testing
        test_loop(config, writer, env, a2c_model)
    else:
        # Training
        train_loop(config, writer, env, a2c_model)

    writer.close()

if __name__ == '__main__':
    main()
    