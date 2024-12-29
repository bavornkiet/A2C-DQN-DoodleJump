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

### Pixel-Based DQN

```bash
python train_pixel_dqn.py --config configs/pixel_dqn_config.yaml
