# Doodle Jump Reinforcement Learning Agents

## Overview

This project implements and compares various reinforcement learning (RL) agents to play the classic **Doodle Jump** game. Leveraging both pixel-based and feature-based approaches, the project utilizes the following algorithms:

- **Deep Q-Network (DQN)**
- **Advantage Actor-Critic (A2C)**

By employing these algorithms on both pixel data and engineered features, the project aims to analyze and compare their performance across different metrics.

## Getting Started

### Prerequisites

- **Python**: Version 3.10+ is recommended.
- **Package Manager**: Miniconda or pip must be installed on your machine.
- **PyTorch**: Install the version compatible with your CUDA setup. Find the appropriate installation command [here](https://pytorch.org/get-started/previous-versions/).

### Installation

1. **Clone the Repository**

    ```bash
    git clone https://github.com/yourusername/ReinforcementGRP86.git
    cd doodle-jump-rl
    ```
2. **Set Up the Environment**

    Using **pip**:

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

    Or using **Miniconda**:

    ```bash
    conda create -n doodle-jump-rl python=3.10
    conda activate doodle-jump-rl
    ```

3. **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

Each agent has a dedicated training script. Use the following commands to train each agent:

### Feature-Based A2C

```bash
python a2cAgent.py

  --human               playing the game manually without agent
  --test                playing the game with a trained agent
  -m {dqn,drqn,resnet,mobilenet,mnasnet}, --model {dqn,drqn,resnet,mobilenet,mnasnet}
                        select model to train the agent
  -p MODEL_PATH, --model_path MODEL_PATH
                        path to weights of an earlier trained model
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        set learning rate for training the model
  -gamma GAMMA, --gamma GAMMA
                        set discount factor for q learning
  --reward_type {1,2,3,4,5,6}
                        types of rewards formulation
  --exploration EXPLORATION
                        number of games to explore
  --channels CHANNELS   set the image channels for preprocessing
  --height HEIGHT       set the image height post resize
  --width WIDTH         set the image width post resize
  --server              when training on server add this flag
  --seed SEED           change seed value for creating game randomness
  --max_games MAX_GAMES
                        set the max number of games to be played by the agent
  --explore {epsilon_g,epsilon_g_decay_exp,epsilon_g_decay_exp_cur}
                        select the exploration vs exploitation tradeoff
  --decay_factor DECAY_FACTOR
                        set the decay factor for exploration
  --epsilon EPSILON     set the epsilon value for exploration
  --attack              use fast fgsm attack to manipulate the input state
  --attack_eps ATTACK_EPS
                        epsilon value for the fgsm attack
```
