<div align="center">

# Deep Q-Network for Autonomous Lunar Landing

### A from-scratch implementation of Deep Reinforcement Learning for OpenAI Gym's LunarLander-v2

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io)
[![OpenAI Gym](https://img.shields.io/badge/OpenAI%20Gym-0081A5?style=for-the-badge)](https://www.gymlibrary.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

*Developed by [**Mohammad Asadolahi**](https://github.com/MohammadAsadolahi) — Senior Agentic AI Engineer | Focus: Agentic AI Architectures In The Wild*

---

<img src="https://gymnasium.farama.org/_images/lunar_lander.gif" alt="LunarLander-v2" width="480">

</div>

---

## Overview

This repository presents a **Deep Q-Network (DQN)** implementation that teaches an agent to autonomously land a spacecraft on the lunar surface. The agent learns entirely from raw 8-dimensional state observations through trial-and-error interaction with the environment — no hand-crafted heuristics, no human demonstrations.

The project implements the foundational algorithm from DeepMind's landmark paper [*"Human-level control through deep reinforcement learning"*](https://www.nature.com/articles/nature14236) (Mnih et al., Nature 2015), adapted for continuous-state, discrete-action control.

> **The environment is considered "solved" when the agent achieves an average reward of ≥ 200 over 100 consecutive episodes.**

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     DQN Agent Pipeline                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌───────────┐    ┌──────────┐    ┌───────────────────────┐   │
│   │  LunarLa- │    │ ε-Greedy │    │     Q-Network         │   │
│   │  nder-v2  │───▶│  Policy  │───▶│  ┌─────────────────┐  │   │
│   │  (Gym)    │    │          │    │  │ Input:   8 dims  │  │   │
│   └─────┬─────┘    └──────────┘    │  │ Hidden: 256×ReLU │  │   │
│         │                          │  │ Hidden: 256×ReLU │  │   │
│         │  (s, a, r, s', done)     │  │ Output: 4 actions│  │   │
│         │                          │  └─────────────────┘  │   │
│         ▼                          └───────────┬───────────┘   │
│   ┌─────────────┐                              │               │
│   │   Replay    │◀─────── Sample Batch ────────┘               │
│   │   Buffer    │         (batch=64)                           │
│   │  (1M trans) │                                              │
│   └─────────────┘                                              │
│                                                                 │
│   Loss = MSE( Q(s,a) , r + γ·max_a' Q(s',a')·(1-done) )      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Component | Choice | Rationale |
|---|---|---|
| **Q-Network** | 2-layer MLP (256 units each) | Sufficient capacity for 8D→4 mapping without overfitting |
| **Activation** | ReLU | Efficient gradients, avoids vanishing gradient in shallow nets |
| **Optimizer** | Adam (lr=0.001) | Adaptive learning rate, fast convergence |
| **Replay Buffer** | 1M transitions, circular | Breaks temporal correlation, improves sample efficiency |
| **Exploration** | ε-greedy, exponential decay (0.9995) | Smooth transition from exploration to exploitation |
| **Discount (γ)** | 0.99 | Long planning horizon — landing requires sustained strategy |

---

## State & Action Space

### Observation (8-dimensional continuous vector)

| Index | Feature | Description |
|:---:|---|---|
| 0 | `x` | Horizontal position |
| 1 | `y` | Vertical position |
| 2 | `vx` | Horizontal velocity |
| 3 | `vy` | Vertical velocity |
| 4 | `θ` | Angle |
| 5 | `ω` | Angular velocity |
| 6 | `left_leg` | Left leg ground contact (bool) |
| 7 | `right_leg` | Right leg ground contact (bool) |

### Actions (4 discrete)

| Action | Description |
|:---:|---|
| 0 | Do nothing |
| 1 | Fire left engine |
| 2 | Fire main engine |
| 3 | Fire right engine |

---

## Training Results

The plots below show the agent's learning progress over approximately 80 training episodes. The average reward trends upward over time, demonstrating the agent is learning, though it has not yet reached the solved threshold of +200 average reward in this training run:

<div align="center">

| Total Episode Rewards | Average Reward |
|:---:|:---:|
| ![Total Rewards](LunarLanderV2_DQN_Total%20Rewards.png) | ![Average Rewards](LunarLanderV2_DQN_Average%20Rewards.png) |

</div>

---

## Project Structure

```
.
├── dqn/                          # Core DQN package
│   ├── __init__.py               # Package exports
│   ├── agent.py                  # DQNAgent — network, action selection, learning
│   ├── replay_buffer.py          # ReplayBuffer — circular experience storage
│   └── config.py                 # DQNConfig — centralized hyperparameters
│
├── train.py                      # CLI training script with full arg parsing
├── evaluate.py                   # CLI evaluation script with statistics
│
├── Deep-Q-Learning-for-solving-OpenAi-Gym-LunarLander-v2.py
│                                 # Original monolithic training script
├── DQN_for_Gym_LunarLander.ipynb # Interactive Jupyter notebook version
│
├── requirements.txt              # Pinned dependencies
├── pyproject.toml                # Modern Python packaging (PEP 621)
├── .gitignore                    # Git ignore rules
├── LICENSE                       # MIT License
├── CITATION.cff                  # Academic citation metadata
└── README.md                     # ← You are here
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- [SWIG](http://www.swig.org/) (required for Box2D compilation)

### Installation

```bash
# Clone the repository
git clone https://github.com/MohammadAsadolahi/Deep-Q-Learning-for-solving-OpenAi-Gym-LunarLander-v2-in-python.git
cd Deep-Q-Learning-for-solving-OpenAi-Gym-LunarLander-v2-in-python

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate    # Linux/macOS
.venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt
```

### Train

```bash
# Default training (500 episodes)
python train.py

# Custom configuration
python train.py --episodes 1000 --batch-size 128 --gamma 0.995

# Train with live rendering
python train.py --render --episodes 200
```

### Evaluate

```bash
# Run 10 greedy evaluation episodes
python evaluate.py

# Evaluate with visual rendering
python evaluate.py --episodes 50 --render

# Use specific weights
python evaluate.py --weights results/DQN_LunarLanderV2.weights.h5
```

---

## Algorithm Deep Dive

### The Bellman Equation at the Core

The Q-function is updated toward the **temporal difference target**:

$$Q(s, a) \leftarrow r + \gamma \cdot \max_{a'} Q(s', a') \cdot (1 - \text{done})$$

where $r$ is the immediate reward, $\gamma$ is the discount factor, and the $(1 - \text{done})$ term zeroes out future value at terminal states.

### Experience Replay

Instead of learning from sequential transitions (which are highly correlated), we store all transitions in a **circular buffer** of 1M capacity and sample **uniformly at random** in mini-batches of 64. This provides:

1. **Decorrelation** — breaks the temporal dependency between consecutive samples
2. **Data efficiency** — each transition can be reused across multiple gradient updates
3. **Stability** — smooths out the non-stationary distribution of incoming data

### Exploration Schedule

The exploration rate $\varepsilon$ decays exponentially:

$$\varepsilon_{t+1} = \max(\varepsilon_t \times 0.9995, \ 0.01)$$

Starting from $\varepsilon = 1.0$ (fully random), the agent gradually shifts to exploitation while maintaining a 1% exploration floor to prevent convergence to suboptimal deterministic policies.

---

## Hyperparameter Reference

| Parameter | Value | CLI Flag |
|---|---|---|
| Discount factor (γ) | `0.99` | `--gamma` |
| Initial ε | `1.0` | — |
| ε decay rate | `0.9995` | — |
| Minimum ε | `0.01` | — |
| Batch size | `64` | `--batch-size` |
| Replay buffer size | `1,000,000` | — |
| Hidden layers | `2 × 256` | — |
| Learning rate | `0.001` | `--lr` |
| Episodes | `500` | `--episodes` |

---

## Citation

If you use this work in your research, please cite:

```bibtex
@software{asadolahi2022dqn,
  author    = {Mohammad Asadolahi},
  title     = {Deep Q-Network for Solving OpenAI Gym LunarLander-v2},
  year      = {2022},
  url       = {https://github.com/MohammadAsadolahi/Deep-Q-Learning-for-solving-OpenAi-Gym-LunarLander-v2-in-python},
  license   = {MIT}
}
```

---

## References

1. Mnih, V. et al. (2015). [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236). *Nature*, 518(7540), 529–533.
2. Lin, L.-J. (1992). Self-improving reactive agents based on reinforcement learning, planning and teaching. *Machine Learning*, 8(3), 293–321.
3. [Gymnasium Documentation — LunarLander-v2](https://gymnasium.farama.org/environments/box2d/lunar_lander/)

---

<div align="center">

MIT License &copy; 2022 Mohammad Asadolahi

</div>

this readme is AI assisted generated, so check for mistakes
