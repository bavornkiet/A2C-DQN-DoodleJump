import os
import datetime
import argparse
import random
import cv2
import torch
from scipy.stats import trim_mean
import numpy as np
from collections import deque
from game.game import DoodleJump
from model.deepQNetwork import Deep_QNet, Deep_RQNet
from model.dqnTrainer import QTrainer
from helper import write_model_params
from torch.utils.tensorboard import SummaryWriter

class Agent:
    def __init__(self, config):
        self.game_counter = 0
        self.step_counter = 1
        random_seed = config.seed
        self.explore_limit = config.exploration
        os.environ['PYTHONHASHSEED'] = str(random_seed)
        
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        self.save_frames = config.store_frames
        self.img_height = config.height
        self.img_width = config.width
        self.img_channels = config.channels
        self.replay_memory = deque(maxlen=config.max_memory)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.discount_factor = config.gamma
        self.batch_size = config.batch_size
        self.learning_rate = config.learning_rate
        self.steps_done = 0
        self.exploration_strategy = config.explore
        self.decay_rate = config.decay_factor
        self.epsilon_value = config.epsilon
        self.e_constant = 2.71828
        
        if config.explore == "epsilon_g_decay_exp":
            self.epsilon_value = 1
            
        if config.model == "dqn":
            self.network = Deep_QNet()
        elif config.model == "drqn":
            self.network = Deep_RQNet()

        self.network = self.network.to(self.device)
        
        if config.model_path or config.test:
            self.network.load_state_dict(torch.load(config.model_path, map_location=self.device), strict=False)
            
        self.trainer = QTrainer(model=self.network, lr=self.learning_rate, gamma=self.discount_factor, device=self.device, 
                                num_channels=self.img_channels, attack_eps=config.attack_eps)
        
    def process_frame(self, frame):
        resized_frame = cv2.resize(frame, (self.img_width, self.img_height))
        rotation_matrix = cv2.getRotationMatrix2D((self.img_width / 2, self.img_height / 2), 270, 1.0)
        rotated_frame = cv2.warpAffine(resized_frame, rotation_matrix, (self.img_height, self.img_width))

        if self.save_frames:
            os.makedirs("./image_dump", exist_ok=True)
            cv2.imwrite(f"./image_dump/{self.step_counter}.jpg", rotated_frame)
            self.step_counter += 1
            
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]
        
        if self.img_channels == 1:
            grayscale_img = cv2.cvtColor(rotated_frame, cv2.COLOR_BGR2GRAY)
            normalized_img = ((grayscale_img / 255.0) - np.mean(imagenet_mean)) / np.mean(imagenet_std)
        else:
            normalized_img = ((rotated_frame / 255.0) - imagenet_mean) / imagenet_std
            normalized_img = normalized_img.transpose((2, 0, 1))
        
        final_img = np.expand_dims(normalized_img, axis=0)
        return final_img

    def get_game_state(self, game_instance):
        current_frame = game_instance.getPixelFrame()
        return self.process_frame(current_frame)

    def evaluate_exploration(self, testing_mode):
        self.steps_done += 1
        random_chance = random.random()
        if testing_mode:
            return False
        if self.exploration_strategy == "epsilon_g":
            pass
        elif self.exploration_strategy == "epsilon_g_decay_exp":
            self.epsilon_value *= pow((1.0 - self.decay_rate), self.steps_done)
        elif self.exploration_strategy == "epsilon_g_decay_exp_cur":
            self.epsilon_value = self.decay_rate * pow(self.e_constant, -self.steps_done)
    
        if random_chance > self.epsilon_value:
            return True
        return False
    
    def train_replay_memory(self):
        if len(self.replay_memory) > self.batch_size:
            mini_batch = random.sample(self.replay_memory, self.batch_size)
        else:
            mini_batch = self.replay_memory
        
        states, actions, rewards, next_states, game_done_flags = zip(*mini_batch)
        self.network.train()
        return self.trainer.train_step(states, actions, rewards, next_states, game_done_flags)

    def train_single_step(self, state, action, reward, next_state, game_done):
        self.network.train()
        return self.trainer.train_step(state, action, reward, next_state, game_done)
    
    def choose_action(self, state, testing_mode=False):
        action_vector = [0, 0, 0]  
        if self.evaluate_exploration(testing_mode):
            selected_action = random.randint(0, 2)
            action_vector[selected_action] = 1
        else:
            state_tensor = torch.tensor(state, dtype=torch.float).to(self.device)
            self.network.eval()
            with torch.no_grad():
                predictions = self.network(state_tensor)
            selected_action = torch.argmax(predictions).item()
            action_vector[selected_action] = 1

        return action_vector
    
    def save_experience(self, current_state, action, reward, next_state, game_done):
        self.replay_memory.append((current_state, action, reward, next_state, game_done))

def train(game, args, writer):
    if args.macos:
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    sum_rewards = 0
    sum_short_loss = 0
    total_score = 0
    score_array = []
    reward_array = []
    record = 0
    loop_ctr = 0
    agent = Agent(args)
    
    dummy_input = torch.rand(1, args.channels, args.height, args.width).to(agent.device)
    writer.add_graph(agent.network, dummy_input)
    print("Now playing")
    
    while agent.game_counter != args.max_games:
        loop_ctr += 1
        
        # Get old state
        state_old = agent.get_game_state(game_instance=game)

        # Get move
        final_move = agent.choose_action(state_old)

        # Perform move and get new state
        reward, done, score = game.agentPlay(final_move)
        state_new = agent.get_game_state(game_instance=game)

        reward_array.append(reward)
        sum_rewards += reward

        # Train short memory
        short_loss = agent.train_single_step(state_old, final_move, reward, state_new, done)
        sum_short_loss += short_loss

        # Save experience to memory
        agent.save_experience(state_old, final_move, reward, state_new, done)

        # Log metrics every 25 steps
        if loop_ctr % 25 == 0:
            writer.add_scalar('Loss/Short_train', sum_short_loss / loop_ctr, loop_ctr)
            writer.add_scalar('Reward/mean_reward', sum_rewards / loop_ctr, loop_ctr)

        if done:
            # Train long memory, reset the game, and update metrics
            game.gameReboot()
            agent.game_counter += 1
            long_loss = agent.train_replay_memory()

            # Log metrics for the episode
            writer.add_scalar('Loss/Long_train', long_loss, agent.game_counter)
            writer.add_scalar('Reward/Total_rewards', sum_rewards, agent.game_counter)
            writer.add_scalar('Game/High_Score', record, agent.game_counter)

            if score > record:
                record = score
                # Save the best model yet
                agent.network.save(file_name="model_best.pth", model_folder_path="./model")
            
            if agent.game_counter % 100 == 0:
                # Save model every 100 games
                agent.network.save(file_name=f"model_{agent.game_counter}.pth", model_folder_path="./model")

            print(f'Game {agent.game_counter} | Score: {score} | High Score: {record}')
            score_array.append(score)
            total_score += score

            # Log mean score
            mean_score = total_score / agent.game_counter
            writer.add_scalar('Game/Mean_Score', mean_score, agent.game_counter)
            writer.add_scalars('Game/Scores', {
                'Current_Score': score,
                'Mean_Score': mean_score,
                'High_Score': record
            }, agent.game_counter)

    # Final metrics logging
    trimmed_avg_score = trim_mean(score_array, 0.1)
    trimmed_avg_reward = trim_mean(reward_array, 0.1)
    print('Mean trimmed score: ', trimmed_avg_score)
    print('Mean trimmed reward: ', trimmed_avg_reward)
    writer.add_hparams(
        hparam_dict=vars(args),
        metric_dict={
            'Mean_Trimmed_Score': trimmed_avg_score,
            'Mean_Trimmed_Reward': trimmed_avg_reward,
            'High_Score': record,
            'Mean_Score': total_score / agent.game_counter
        }
    )

def test(game, args):
    if args.macos:
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    record = 0
    cum_score = 0
    agent = Agent(args)
    print("Now playing")

    with open("test_logs.txt", "w") as f:
        f.write("Now playing\n")
    
    while agent.game_counter < args.max_games:
        if args.attack:
            state = agent.get_game_state(game)  # Original state
            adv_manip = agent.trainer.create_adv_state(state)  # Manipulated state
            final_move = agent.choose_action(state + adv_manip, test_mode=True)
            reward, done, score = game.agentPlay(final_move)
        else:
            state_old = agent.get_game_state(game)
            final_move = agent.choose_action(state_old, test_mode=True)
            reward, done, score = game.agentPlay(final_move)

        if done:
            agent.game_counter += 1
            cum_score += score
            game.gameReboot()

            if score > record:
                record = score
            
            with open("test_logs.txt", "a") as f:
                f.write(
                    f"Game: {agent.game_counter} Score: {score} Record: {record} "
                    f"Mean Score: {cum_score / agent.game_counter}\n"
                )
            print(
                f"Game {agent.game_counter} Score {score} Record {record} "
                f"Mean Score: {cum_score / agent.game_counter}"
            )

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='RL Agent for Doodle Jump')
    parser.add_argument("--macos", action="store_true", help="select model to train the agent")
    parser.add_argument("--human", action="store_true", help="playing the game manually without agent")
    parser.add_argument("--test", action="store_true", help="playing the game with a trained agent")
    parser.add_argument("-m", "--model", type=str, default="dqn", choices=["dqn", "drqn", "resnet", "mobilenet", "mnasnet"], help="select model to train the agent")
    parser.add_argument("-p", "--model_path", type=str, help="path to weights of an earlier trained model")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="set learning rate for training the model")
    parser.add_argument("-g", "--gamma", type=float, default=0.9, help="set discount factor for q learning")
    parser.add_argument("--max_memory", type=int, default=10000, help="Buffer memory size for long training")
    parser.add_argument("--store_frames", action="store_true", help="store frames encountered during game play by agent")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for long training")
    parser.add_argument("--reward_type", type=int, default=1, choices=[1, 2, 3, 4, 5, 6], help="types of rewards formulation")
    parser.add_argument("--exploration", type=int, default=40, help="number of games to explore")
    parser.add_argument("--channels", type=int, default=1, help="set the image channels for preprocessing")
    parser.add_argument("--height", type=int, default=80, help="set the image height post resize")
    parser.add_argument("--width", type=int, default=80, help="set the image width post resize")
    parser.add_argument("--server", action="store_true", help="when training on server add this flag")
    parser.add_argument("--seed", type=int, default=42, help="change seed value for creating game randomness")
    parser.add_argument("--max_games", type=int, default=1000, help="set the max number of games to be played by the agent")
    parser.add_argument("--explore", type=str, default="epsilon_g", choices=["epsilon_g","epsilon_g_decay_exp","epsilon_g_decay_exp_cur"], help="select the exploration vs exploitation tradeoff")
    parser.add_argument("--decay_factor", type=float, default=0.9, help="set the decay factor for exploration")
    parser.add_argument("--epsilon", type=float, default=0.8, help="set the epsilon value for exploration")
    parser.add_argument("--attack", action="store_true", help="use fast fgsm attack to manipulate the input state")
    parser.add_argument("--attack_eps", type=float, default=0.3, help="epsilon value for the fgsm attack")
    args = parser.parse_args()
    
    game = DoodleJump(server=args.server, reward_type=args.reward_type)

    if args.human:
        game.run()
    elif args.test:
        test(game, args)
    else:
        hyper_params = "_m_"+args.model+"_lr_"+str(args.learning_rate)+"_g_"+str(args.gamma)+"_mem_"+str(args.max_memory)+"_batch_"+str(args.batch_size)
        arg_dict = vars(args)

        dstr = datetime.datetime.now().strftime("_dt-%Y-%m-%d-%H-%M-%S")
        writer = SummaryWriter(log_dir="model"+hyper_params+dstr)
        writer.add_text('Model Parameters: ', str(arg_dict), 0)
        
        train(game, args, writer)
            