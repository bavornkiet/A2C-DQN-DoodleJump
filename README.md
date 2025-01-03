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
   mkdir doodle-jump-rl
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

Each agent has a dedicated training/testing script. Use the following commands to train/test each agent:

## Feature-Based A2C

To test the Feature-Based A2C

```bash
python a2cAgent.py --test
```

Other Commands for Training

```bash
python a2cAgent.py

  --learning_rate LEARNING_RATE,
                        set learning rate for training the model
  -gamma GAMMA,
                        set discount factor for q learning
  --reward_type {1,2,3}
                        types of rewards formulation
  --server              when training on server add this flag to stop the game from rendering
  --max_games MAX_GAMES
                        set the max number of games to be played by the agent
  --experiment_name EXPERIMENT_NAME
                        set the experiment name (for training)
```

## Feature-Based DQN

To test the Feature-Based DQN

```bash
python dqnAgent.py --test
```

Other Commands for Training

```bash
python a2cAgent.py

  --learning_rate LEARNING_RATE,
                        set learning rate for training the model
  -gamma GAMMA,
                        set discount factor for q learning
  --reward_type {1,2,3}
                        types of rewards formulation
  --server              when training on server add this flag to stop the game from rendering
  --max_games MAX_GAMES
                        set the max number of games to be played by the agent
  --experiment_name EXPERIMENT_NAME
                        set the experiment name (for training)
```

## Pixel-Based A2C

To test the Pixel-Based A2C

```bash
python a2cAgent.py --test
```

Other Commands for Training

```bash
python a2cAgent.py

  --learning_rate LEARNING_RATE,
                        set learning rate for training the model
  -gamma GAMMA,
                        set discount factor for q learning
  --reward_type {1,2,3}
                        types of rewards formulation
  --server              when training on server add this flag to stop the game from rendering
  --max_games MAX_GAMES
                        set the max number of games to be played by the agent
  --experiment_name EXPERIMENT_NAME
                        set the experiment name (for training)
```

## Pixel-Based DQN

To test the Pixel-Based DQN

```bash
python dqnAgent.py --test
```

Other Commands for Training

```bash
python dqnAgent.py

  --macos                      select model to train the agent on macOS
  --human                      play the game manually without the agent
  --test                       play the game with a trained agent
  -m, --model {dqn}
                               select the model to train the agent
  -p, --model_path MODEL_PATH  path to weights of an earlier trained model
  -lr, --learning_rate LEARNING_RATE
                               set learning rate for training the model
  -g, --gamma GAMMA            set discount factor for Q-learning
  --max_memory MAX_MEMORY      buffer memory size for long training
  --store_frames               store frames encountered during gameplay
  --batch_size BATCH_SIZE      batch size for training
  --reward_type {1,2,3,4,5,6}  types of rewards formulation
  --exploration EXPLORATION    number of games for exploration
  --channels CHANNELS          set the image channels for preprocessing
  --height HEIGHT              set the image height post-resizing
  --width WIDTH                set the image width post-resizing
  --server                     when training on a server, add this flag to stop the game from rendering
  --seed SEED                  random seed for reproducibility
  --max_games MAX_GAMES        set the max number of games to be played by the agent
  --explore {epsilon_g,epsilon_g_decay_exp,epsilon_g_decay_exp_cur}
                               select the exploration vs. exploitation tradeoff
  --decay_factor DECAY_FACTOR  set the decay factor for exploration
  --epsilon EPSILON            set the epsilon value for exploration
  --attack                     enable adversarial training with FGSM attacks
  --attack_eps ATTACK_EPS      epsilon value for FGSM adversarial attacks
```

## References


