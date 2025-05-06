# Doodle Jump Reinforcement Learning Agents
## Disclaimer

This project utilizes the **Doodle Jump** environment developed by Katti et al. (2021) with modifications to the platform generation mechanism to enhance platform spawning. Additionally, a `getFeature` function was implemented to enable feature-based models to extract the state from the environment. For both the DQN pixel-based agent, we drew inspiration from their image preprocessing techniques and algorithms. For A2C pixel-based-agent  `a2cPixel.py` , we also drew inspiration from their image preprocessing techniques, but other parts of the codebase are developed by our team. The codebase for `a2cFeature.py` and `dqnFeature.py` was fully developed by our team from scratch, with no code borrowed from external sources.

**Citation:**

Katti, A. R., Sarkar, A., Swamy, P. M., Kothari, R., & Gurudatt, S. P. (2021). *Reinforcement Learning and Adversarial Attacks on Player Model with Doodle Jump*. Retrieved from [https://pranavmswamy.github.io/doodle-jump-website/assets/CSCI527_Technical_Paper.pdf](https://pranavmswamy.github.io/doodle-jump-website/assets/CSCI527_Technical_Paper.pdf) & (https://github.com/USC-CSCI527-Spring2021/Doodle-Jump)

## Overview
## Overview

This project implements and compares various reinforcement learning (RL) agents to play the classic **Doodle Jump** game. Leveraging both pixel-based and feature-based approaches, the project utilizes the following algorithms:

- **Deep Q-Network (DQN)**
- **Advantage Actor-Critic (A2C)**

By employing these algorithms on both pixel data and engineered features, the project aims to analyze and compare their performance across different metrics.

## Getting Started

### Prerequisites

- **Python**: Version 3.10+ is recommended.
- **Package Manager**: Miniconda or pip must be installed on your machine.

### Installation

1. **Clone the Repository**

   ```bash
   mkdir doodle-jump-rl
   cd doodle-jump-rl
   git clone https://github.com/yourusername/ReinforcementGRP86.git
   cd ReinforcementGRP86
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
4. **Install numpy**

   ```bash
   pip install "numpy<2"
   ```

5. **Install Pytorch and CUDA**

   - **PyTorch**: Install the version compatible with your CUDA setup. Find the appropriate installation command [here](https://pytorch.org/get-started/previous-versions/).
   - We recommend pytorch version 2.1.0, torchvision 0.16.0 and torchaudio 2.1.0
   - For CUDA, you would need to find a version that match with your Nvidia GPU card.

```bash
#example command
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```
## Usage

Each agent has a dedicated training/testing script. Use the following commands to train/test each agent:

## Feature-Based A2C

To test the Feature-Based A2C

```bash
python a2cFeature.py --test
```

Other Commands for Training

```bash
python a2cFeature.py

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
python dqnFeature.py --test
```

Other Commands for Training

```bash
python dqnFeature.py

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
python a2cFeature.py --test
```

Other Commands for Training

```bash
python a2cFeature.py

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
python dqnPixel.py --test
```

Other Commands for Training

```bash
python dqnPixel.py

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
  --reward_type {1,2,3}  types of rewards formulation
  --exploration EXPLORATION    number of games for exploration
  --server                     when training on a server, add this flag to stop the game from rendering
  --max_games MAX_GAMES        set the max number of games to be played by the agent
  --explore {epsilon_g,epsilon_g_decay_exp,epsilon_g_decay_exp_cur}
                               select the exploration vs. exploitation tradeoff
  --decay_factor DECAY_FACTOR  set the decay factor for exploration
  --epsilon EPSILON            set the epsilon value for exploration

```

## References


# DSAC for Inventory Management Simulation

This project implements **Distributional Soft Actor Critic (DSAC)** for an inventory management problem using reinforcement learning. The environment simulates a dynamic inventory system, adapted from OR-Gym, and trains a DSAC agent to learn optimal inventory policies in a risk-sensitive setting.

## 🧠 Core Components

### `dsac_agent.py`

Implements the **DSAC (Distributional Soft Actor Critic)** algorithm, including:

- Distributional Q-value estimation
- Entropy-regularized actor updates
- Risk-sensitive policy optimization

This is based on the algorithm proposed by Ma et al. (2020) [see citation below].

### `main.py`

Handles:

- Environment initialization
- Agent creation
- Training loop and evaluation
- Logging and saving results

To Train the agent

```bash
python main.py
```


### `invm.py`
A simulation environment for managing inventory under uncertainty. It is a **modified version** of the original environment from OR-Gym, adapted to better suit distributional reinforcement learning and risk-sensitive policies.

### `models/`
Contains all the PyTorch-based network architectures:
- Policy (actor) networks
- Critic networks with distributional output
- Value networks


## Citation
```

@misc{HubbsOR-Gym,
author={Christian D. Hubbs and Hector D. Perez and Owais Sarwar and Nikolaos V. Sahinidis and Ignacio E. Grossmann and John M. Wassick},
title={OR-Gym: A Reinforcement Learning Library for Operations Research Problems},
year={2020},
Eprint={arXiv:2008.06319}
}

@misc{ma2020dsacdistributionalsoftactor,
title={DSAC: Distributional Soft Actor Critic for Risk-Sensitive Reinforcement Learning},
author={Xiaoteng Ma and Li Xia and Zhengyuan Zhou and Jun Yang and Qianchuan Zhao},
year={2020},
eprint={2004.14547},
archivePrefix={arXiv},
primaryClass={cs.LG},
url={https://arxiv.org/abs/2004.14547},
}

```



