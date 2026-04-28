"""
Hyperparameter configuration for the DQN agent.

All tunable parameters are centralized here for reproducibility
and systematic ablation studies.
"""

from dataclasses import dataclass, field


@dataclass
class DQNConfig:
    """Configuration dataclass for DQN hyperparameters.

    Attributes:
        gamma: Discount factor for future rewards. Controls the agent's
            planning horizon — values closer to 1.0 prioritize long-term reward.
        explore_rate: Initial probability of selecting a random action
            (epsilon in ε-greedy policy).
        explore_rate_decay: Multiplicative decay applied to explore_rate
            after each learning step.
        min_explore_rate: Floor value for exploration rate to ensure the
            agent never fully stops exploring.
        replay_buffer_size: Maximum number of transitions stored in the
            experience replay buffer (circular buffer).
        batch_size: Number of transitions sampled per learning step.
        hidden_units: Number of neurons in each hidden layer of the
            Q-network.
        num_hidden_layers: Number of hidden layers in the Q-network.
        learning_rate: Learning rate for the Adam optimizer.
        num_episodes: Total number of episodes to train.
        model_name: Filename for saving/loading model weights.
    """

    gamma: float = 0.99
    explore_rate: float = 1.0
    explore_rate_decay: float = 0.9995
    min_explore_rate: float = 0.01
    replay_buffer_size: int = 1_000_000
    batch_size: int = 64
    hidden_units: int = 256
    num_hidden_layers: int = 2
    learning_rate: float = 0.001
    num_episodes: int = 500
    model_name: str = "DQN_LunarLanderV2.h"
