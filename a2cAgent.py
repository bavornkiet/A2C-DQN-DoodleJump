import torch
import numpy as np
import torch.nn as nn
import random
import os
import datetime
import argparse
import json
import time
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from game.game import DoodleJump  # The game environment.
from model.a2cNetwork import ActorCritic  # The neural network model.
from model.a2cTrainer import A2CLearner  # The A2C algorithm implementation.
# This script contains the main logic for training and testing the agent.

# Mish: A smooth, non-monotonic activation function that can improve
# neural network performance. It's defined and wrapped in an nn.Module
# for use in the network.


def mish(input):
    return input * torch.tanh(F.softplus(input))


class Mish(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, input): return mish(input)

# t(x): Converts a NumPy array or any input x into a PyTorch tensor of type float.


def t(x):
    x = np.array(x) if not isinstance(x, np.ndarray) else x
    return torch.from_numpy(x).float()  # .reshape(6400, 1)

# The Runner class is responsible for interacting with the game
# environment, processing observations, and collecting experiences.


class Runner():
    def __init__(self, game, hyper_params, dstr, args):
        self.game = game
        self.state = None
        self.done = True
        self.steps = 0
        self.episode_reward = 0
        self.mean_reward = 0
        self.mean_score = 0
        self.N_PLATFORMS = 10
        self.N_SPRINGS = 3
        self.N_MONSTERS = 2
        self.episode_rewards = []
        self.total_score = 0
        ''''''
        self.game_counter = 0
        self.epsilon = 0
        self.ctr = 1
        seed = args.seed
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.hyper_params = hyper_params
        self.dstr = dstr
        self.record = 0
    # Reset: Resets the game state for a new episode.

    def reset(self):
        self.episode_reward = 0
        self.done = False
        self.state = None
        self.game.gameReboot()
    # Image Processing
#     Resize: Adjusts the image to the desired dimensions.
# Rotate: Rotates the image by 270 degrees to match the orientation expected by the model.
# Normalization: Applies ImageNet mean and standard deviation normalization.
# Color Channels: Converts to grayscale if image_c == 1.
# Reshape: Adds a batch dimension.

    def convert_state_to_vector(self, state_dict):
        # Flatten the state dictionary into a feature vector
        state_vector = []

        # Player features
        N_PLATFORMS = 15
        N_SPRINGS = 3
        N_MONSTERS = 1
        player_x = state_dict['player_x']
        player_y = state_dict['player_y']
        player_xmovement = state_dict['player_xmovement']
        player_ymovement = state_dict['player_ymovement']
        score = state_dict['score']

        state_vector.extend([
            player_x,
            player_y,
            player_xmovement,
            player_ymovement,
            score
        ])

        # Helper function to compute distance
        def distance_to_player(obj):
            dx = obj['x'] - player_x
            dy = obj['y'] - player_y
            return np.sqrt(dx ** 2 + dy ** 2)

        # Select N nearest platforms
        platforms = state_dict['platforms']
        platforms.sort(key=lambda p: distance_to_player(p))
        selected_platforms = platforms[:self.N_PLATFORMS]

        for platform in selected_platforms:
            state_vector.extend([
                platform['x'],
                platform['y'],
                platform['type'],
                platform['state']
            ])

        # Pad if fewer platforms
        for _ in range(self.N_PLATFORMS - len(selected_platforms)):
            state_vector.extend([0, 0, 0, 0])

        # Springs
        springs = state_dict['springs']
        springs.sort(key=lambda s: distance_to_player(s))
        selected_springs = springs[:self.N_SPRINGS]

        for spring in selected_springs:
            state_vector.extend([
                spring['x'],
                spring['y'],
                spring['state']
            ])

        # Pad if fewer springs
        for _ in range(self.N_SPRINGS - len(selected_springs)):
            state_vector.extend([0, 0, 0])

        # Monsters
        monsters = state_dict['monsters']
        monsters.sort(key=lambda m: distance_to_player(m))
        selected_monsters = monsters[:self.N_MONSTERS]

        for monster in selected_monsters:
            state_vector.extend([
                monster['x'],
                monster['y'],
                monster['state']
            ])

        # Pad if fewer monsters
        for _ in range(self.N_MONSTERS - len(selected_monsters)):
            state_vector.extend([0, 0, 0])

        # Normalize features
        state_vector = self.normalize_features(state_vector)

        return np.array(state_vector, dtype=np.float32)


# Get Current State: Retrieves and preprocesses the current game frame.


    def get_state(self):
        state_dict = self.game.getFeatures()
        state_vector = self.convert_state_to_vector(state_dict)
        return state_vector
# Main Loop:
# Reset Check: Resets the game if the episode is done.
# State Retrieval: Gets the current state.
# Action Selection:
# Policy Forward Pass: Passes the state through the actor-critic network to get action distributions.
# Action Sampling: Samples an action from the distribution.
# Action Clipping: Clips actions to the valid range.
# Action Formatting: Converts the action into the game's expected format.
# Environment Interaction: Takes a step in the game using playStep.
# Memory Storage: Stores experiences for training.
# Reward and Score Tracking: Updates rewards and scores.
# Model Saving: Saves the model periodically and when a new record is achieved.
# Logging: Writes metrics to TensorBoard and prints progress.

    def normalize_features(self, features):
        # Example normalization (you may need to adjust based on your game's specifics)
        features[0] /= 800  # player_x
        features[1] /= 800  # player_y
        features[2] /= 10  # player_xmovement
        features[3] /= 35  # player_ymovement
        features[4] /= 30000  # score

        # Normalize platform positions and types
        # index = 5
        # # 4 features per platform * max_platforms
        # num_platform_features = 4 * self.N_PLATFORMS
        # for i in range(index, index + num_platform_features, 4):
        #     features[i] /= self.game.screen_width  # platform_x
        #     features[i + 1] /= self.game.screen_height  # platform_y

        # for i in range(num_Platforms):
        #     # platform_x
        #     features['platforms'][i]['x'] /= self.game.screen_width
        #     # platform_y
        #     features['platforms'][i]['y'] /= self.game.screen_height
        #     # Type and state may not need normalization if they are categorical/integers

        # for i in range(num_Springs):
        #     features['springs'][i]['x'] /= self.game.screen_width
        #     features['springs'][i]['y'] /= self.game.screen_height
        #     # Type and state may not need normalization if they are categorical/integers

        # for i in range(num_monsters):
        #     features['monsters'][i]['x'] /= self.game.screen_width  # platform_x
        #     # platform_y
        #     features['monsters'][i]['y'] /= self.game.screen_height
        #     # Type and state may not need normalization if they are categorical/integers
        index = 5  # Starting index for platforms

    # Platforms normalization
        num_platform_features = 4 * self.N_PLATFORMS
        for i in range(self.N_PLATFORMS):
            # platform_x
            features[index] /= self.game.screen_width
            index += 1
            # platform_y
            features[index] /= self.game.screen_height
            index += 3
            # type (categorical, can be left as is)
            # index += 2
            # state (categorical, can be left as is)
            # index += 1

        # Springs normalization
        num_spring_features = 3 * self.N_SPRINGS
        for i in range(self.N_SPRINGS):
            # spring_x
            features[index] /= self.game.screen_width
            index += 1
            # spring_y
            features[index] /= self.game.screen_height
            index += 2
            # state (categorical)
            # index += 1

        # Monsters normalization
        num_monster_features = 3 * self.N_MONSTERS
        for i in range(self.N_MONSTERS):
            # monster_x
            features[index] /= self.game.screen_width
            index += 1
            # monster_y
            features[index] /= self.game.screen_height
            index += 2
            # state (categorical)
            # index += 1
        return features

    def run(self, max_steps, memory=None):
        if not memory or len(memory) > args.max_memory:
            memory = []
        for _ in range(max_steps):
            if self.done:
                self.reset()
            state_old = self.get_state()
            state_tensor = t(state_old).unsqueeze(0).to(self.device)
            dists = actorcritic(state_tensor)[0]
            actions = dists.sample().detach().cpu().numpy()[0]
            actions_clipped = np.clip(actions, -1, 1)

            # Prepare action for the game
            final_move = [0, 0, 0]
            final_move[np.argmax(actions_clipped)] = 1
            reward, self.done, score = self.game.playStep(final_move)

            next_state = self.get_state()
            memory.append(
                (actions, reward, state_old, next_state, self.done))

            self.state = next_state
            self.steps += 1
            self.episode_reward += reward
            self.mean_reward = self.episode_reward / self.steps

            writer.add_scalar("Reward/mean_reward",
                              self.mean_reward, global_step=self.steps)

            if self.done:
                self.game_counter += 1
                self.episode_rewards.append(self.episode_reward)
                if len(self.episode_rewards) % 10 == 0:
                    print("Episode:", len(self.episode_rewards),
                          ", Episode Reward:", self.episode_reward)
                writer.add_scalar("Reward/episode_reward",
                                  self.episode_reward, global_step=self.steps)

                # Inside the run method of Runner class
                if score > self.record:
                    self.record = score
                    # Save the best model with hyperparameters and dstr
                    model_filename = (
                        f"a2c_model_best" +
                        f"{self.hyper_params}" +
                        f"{self.dstr}.pth"
                    )
                    actorcritic.save(file_name=model_filename,
                                     model_folder_path="./Parameters")

                if self.game_counter % 100 == 0:
                    # Save the model periodically
                    model_filename = (
                        f"a2c_model_{self.game_counter}" +
                        f"{self.hyper_params}" +
                        f"{self.dstr}.pth"
                    )
                    actorcritic.save(file_name=model_filename,
                                     model_folder_path="./Parameters")

                print('Game', self.game_counter, 'Score',
                      score, 'Record:', self.record)
                writer.add_scalar('Score/High_Score',
                                  self.record, self.game_counter)

                self.total_score += score
                self.mean_score = self.total_score / agent.game_counter
                writer.add_scalar('Score/Mean_Score',
                                  self.mean_score, self.game_counter)
        return memory


# Testing Loop:
# State Retrieval: Gets the current state.
# Action Selection: Similar to the training loop but without storing experiences.
# Game Interaction: Takes a step in the game.
# Score Tracking: Keeps track of scores and records.


def test(game, args):
    record = 0
    agent = Runner(game)
    print("Now testing")

    while agent.game_counter != args.max_games:
        state_old = agent.get_state()
        state_tensor = t(state_old).unsqueeze(0).to(agent.device)
        dists = actorcritic(state_tensor)[0]
        actions = dists.sample().detach().cpu().numpy()[0]
        actions_clipped = np.clip(actions, -1, 1)

        final_move = [0, 0, 0]
        final_move[np.argmax(actions_clipped)] = 1
        reward, done, score = game.playStep(final_move)
        if done:
            agent.game_counter += 1
            game.gameReboot()
            if score > record:
                record = score
            print('Game', agent.game_counter, 'Score', score, 'Record:', record)
# Argument Parsing: Parses command-line arguments for configuration.
# Game and Agent Initialization: Sets up the game and agent.
# Logging Setup: Configures TensorBoard logging.
# Model Initialization: Creates the actor-critic network.
# Model Loading: Loads a pre-trained model if specified.
# Testing or Training: Decides whether to run in test mode or train the agent.
# Training Loop:
# Experience Collection: Runs the agent to collect experiences.
# Learning Step: Updates the model using the collected experiences.
# Logging: Records hyperparameters and final metrics.


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL Agent for Doodle Jump')
    parser.add_argument("--macos", action="store_true",
                        help="select model to train the agent")
    parser.add_argument("--human", action="store_true",
                        help="playing the game manually without agent")
    parser.add_argument("--test", action="store_true",
                        help="playing the game with a trained agent")
    parser.add_argument("-d", "--difficulty", type=str, default="EASY",
                        choices=["EASY", "MEDIUM", "HARD"], help="select difficulty of the game")
    parser.add_argument("-m", "--model", type=str, default="a2c",
                        choices=["a2c"], help="select model to train the agent")
    parser.add_argument("-p", "--model_path", type=str,
                        help="path to weights of an earlier trained model")
    # parser.add_argument("-cp", "--critic_path", type=str, help="path to weights of an earlier trained model")
    parser.add_argument("-alr", "--actor_lr", type=float, default=1e-4,
                        help="set learning rate for training the model")
    parser.add_argument("-clr", "--critic_lr", type=float, default=1e-3,
                        help="set learning rate for training the model")
    parser.add_argument("-g", "--gamma", type=float, default=0.99,
                        help="set discount factor for q learning")
    parser.add_argument("--max_memory", type=int, default=10000,
                        help="Buffer memory size for long training")
    parser.add_argument("--store_frames", action="store_true",
                        help="store frames encountered during game play by agent")
    parser.add_argument("--batch_size", type=int, default=1000,
                        help="Batch size for long training")
    parser.add_argument("--reward_type", type=int, default=5,
                        choices=[1, 2, 3, 4, 5, 6], help="types of rewards formulation")
    parser.add_argument("--exploration", type=int, default=40,
                        help="number of games to explore")
    parser.add_argument("--channels", type=int, default=1,
                        help="set the image channels for preprocessing")
    parser.add_argument("--height", type=int, default=80,
                        help="set the image height post resize")
    parser.add_argument("--width", type=int, default=80,
                        help="set the image width post resize")
    parser.add_argument("--server", action="store_true",
                        help="when training on server add this flag")
    parser.add_argument("--seed", type=int, default=42,
                        help="change seed value for creating game randomness")
    parser.add_argument("--max_games", type=int, default=2000,
                        help="set the max number of games to be played by the agent")
    args = parser.parse_args()

    game = DoodleJump(difficulty=args.difficulty,
                      server=args.server, reward_type=args.reward_type)

    hyper_params = "_d_" + args.difficulty + "_m_" + args.model + "_alr_" + str(args.actor_lr) + "_clr_" + str(
        args.critic_lr) + "_g_" + str(args.gamma) + "_mem_" + str(args.max_memory) + "_batch_" + str(args.batch_size)
    dstr = datetime.datetime.now().strftime("_dt-%Y-%m-%d-%H-%M-%S")

    # Save hyperparameters and dstr to JSON file
    hyperparameters = vars(args)
    hyperparameters['dstr'] = dstr
    os.makedirs('./Parameters', exist_ok=True)
    # with open(f'./Parameters/hyperparameters_{dstr}.json', 'w') as f:
    #     json.dump(hyperparameters, f, indent=4)

    # Update the log directory to include hyperparameters and dstr
    writer = SummaryWriter(log_dir=os.path.join(
        "./Parameters", "logs", hyper_params + dstr))
    arg_dict = vars(args)
    writer.add_text('Model Parameters', str(arg_dict), 0)
    writer.add_text('Datetime String', dstr, 0)
    agent = Runner(game, hyper_params, dstr, args)
    print(f"Using device: {agent.device}")
    # Configuration
    state_example = agent.get_state()
    state_dim = len(state_example)
    n_actions = 3  # Number of possible actions

    actorcritic = ActorCritic(state_dim, n_actions,
                              activation=nn.Tanh).to(agent.device)
    if args.model_path or args.test:
        actorcritic.load_state_dict(torch.load(args.model_path))

    if args.test:
        test(game, args)
    else:
        learner = A2CLearner(actorcritic, agent.device, gamma=args.gamma, entropy_beta=0.01,
                             actor_lr=args.actor_lr, critic_lr=args.critic_lr, max_grad_norm=0.5,
                             batch_size=args.batch_size)

        steps_on_memory = 16
        start_time = time.time()
        while agent.game_counter != args.max_games:
            memory = agent.run(steps_on_memory)
            learner.learn(memory, agent.steps, writer, discount_rewards=False)
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Training completed in {total_time:.2f} seconds.")

        writer.add_hparams(hparam_dict=vars(args),
                           metric_dict={'mean_reward': agent.mean_reward,
                                        'high_score': agent.record,
                                        'mean_score': agent.mean_score,
                                        'total_time_sec': total_time})

        # Optionally, Print Total Time in Hours, Minutes, Seconds
        hours, rem = divmod(total_time, 3600)
        minutes, seconds = divmod(rem, 60)
        # print(hours+":"+)
        # print(minutes)

    # Close the SummaryWriter
    writer.close()
