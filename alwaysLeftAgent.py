import os
import datetime
import argparse
import random
import cv2
import torch
import numpy as np
from collections import deque
from game.game import DoodleJump
from model.deepQNetwork import Deep_QNet, Deep_RQNet, DQ_Resnet18, DQ_Mobilenet, DQ_Mnasnet
from model.dqnTrainer import QTrainer
from helper import write_model_params
from torch.utils.tensorboard import SummaryWriter

class AlwaysLeftAgent:
    def __init__(self, args):
        # Initialization placeholders for compatibility
        self.n_games = 0
        self.steps = 0
        self.args = args

    def preprocess(self, state):
        """
        Preprocess the game state to maintain compatibility.
        """
        return state  # For the always-left agent, no preprocessing is required.

    def get_state(self, game):
        """
        Retrieve the current state from the game.
        """
        state = game.getCurrentFrame()
        return self.preprocess(state)

    def get_action(self, state, test_mode=False):
        """
        Always return the action to move left.
        """
        # Assume actions are represented as [LEFT, NO_MOVE, RIGHT].
        # Action [1, 0, 0] corresponds to moving left.
        return [1, 0, 0]


def train(game, args, writer):
    if args.macos:
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    sum_rewards = 0
    total_score = 0
    record = 0
    loop_ctr = 0
    dummy_input = torch.rand(1, args.channels, args.height, args.width).to('cpu')  # Update device if necessary
    print("Now playing with Always Left Agent")

    while loop_ctr < args.max_games:
        loop_ctr += 1
        # get move (always left)
        final_move = [1, 0, 0]  # Assuming "left" corresponds to this one-hot encoded action

        # perform move and get new state
        reward, done, score = game.playStep(final_move)
        sum_rewards += reward

        if done:
            # reset the game
            game.gameReboot()
            total_score += score
            mean_score = total_score / loop_ctr
            print(f'Game {loop_ctr}, Score: {score}, Record: {record}, Mean Score: {mean_score}')

            if score > record:
                record = score

    print(f"Training complete. Record: {record}, Mean Score: {total_score / loop_ctr}")

def test(game, args):
    if args.macos:
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    record = 0
    cum_score = 0
    print("Testing with Always Left Agent")

    with open("test_logs.txt", "w") as f:
        f.write("Now playing with Always Left Agent\n")

    for game_count in range(args.max_games):
        # Always take the "left" action
        final_move = [1, 0, 0]  # Assuming "left" corresponds to this one-hot encoded action
        reward, done, score = game.playStep(final_move)

        if done:
            cum_score += score
            game.gameReboot()
            if score > record:
                record = score
            mean_score = cum_score / (game_count + 1)
            with open("test_logs.txt", "a") as f:
                f.write(f'Game: {game_count + 1} Score: {score} Record: {record} Mean Score: {mean_score}\n')
            print(f'Game {game_count + 1}, Score: {score}, Record: {record}, Mean Score: {mean_score}')

    print(f"Testing complete. Record: {record}, Mean Score: {cum_score / args.max_games}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='RL Agent for Doodle Jump')
    parser.add_argument("--macos", action="store_true", help="select model to train the agent")
    parser.add_argument("--human", action="store_true", help="playing the game manually without agent")
    parser.add_argument("--test", action="store_true", help="playing the game with a trained agent")
    parser.add_argument("-d", "--difficulty", type=str, default="HARD", choices=["EASY", "MEDIUM", "HARD"], help="select difficulty of the game")
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
    parser.add_argument("--max_games", type=int, default=100000, help="set the max number of games to be played by the agent")
    parser.add_argument("--explore", type=str, default="epsilon_g", choices=["epsilon_g","epsilon_g_decay_exp","epsilon_g_decay_exp_cur"], help="select the exploration vs exploitation tradeoff")
    parser.add_argument("--decay_factor", type=float, default=0.9, help="set the decay factor for exploration")
    parser.add_argument("--epsilon", type=float, default=0.8, help="set the epsilon value for exploration")
    parser.add_argument("--attack", action="store_true", help="use fast fgsm attack to manipulate the input state")
    parser.add_argument("--attack_eps", type=float, default=0.3, help="epsilon value for the fgsm attack")
    args = parser.parse_args()
    
    game = DoodleJump(difficulty=args.difficulty, server=args.server, reward_type=args.reward_type)

    if args.human:
        game.run()
    elif args.test:
        test(game, args)
    else:
        hyper_params = "_d_"+args.difficulty+"_m_"+args.model+"_lr_"+str(args.learning_rate)+"_g_"+str(args.gamma)+"_mem_"+str(args.max_memory)+"_batch_"+str(args.batch_size)
        arg_dict = vars(args)

        dstr = datetime.datetime.now().strftime("_dt-%Y-%m-%d-%H-%M-%S")
        writer = SummaryWriter(log_dir="model"+hyper_params+dstr)
        writer.add_text('Model Parameters: ', str(arg_dict), 0)
        
        train(game, args, writer)