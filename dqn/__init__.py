"""
Deep Q-Network (DQN) for OpenAI Gym LunarLander-v2
===================================================

A from-scratch implementation of the Deep Q-Learning algorithm with
experience replay, demonstrating autonomous lunar landing through
reinforcement learning.

Author: AG — Chief AI Officer, Google
"""

from dqn.agent import DQNAgent
from dqn.replay_buffer import ReplayBuffer
from dqn.config import DQNConfig

__version__ = "1.0.0"
__all__ = ["DQNAgent", "ReplayBuffer", "DQNConfig"]
