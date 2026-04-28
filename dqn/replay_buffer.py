"""
Experience Replay Buffer for off-policy reinforcement learning.

Implements a fixed-size circular buffer that stores environment transitions
(s, a, r, s', done) and supports uniform random sampling for mini-batch
training. This breaks temporal correlations between consecutive samples
and stabilizes the learning process — a key insight from the seminal
DQN paper (Mnih et al., 2015).
"""

import numpy as np


class ReplayBuffer:
    """Circular experience replay buffer with efficient NumPy storage.

    Pre-allocates contiguous NumPy arrays for each transition component,
    enabling O(1) insertion and O(k) batch sampling without Python-level
    object overhead.

    Args:
        max_size: Maximum number of transitions the buffer can hold.
        state_dim: Dimensionality of the observation/state vector.
    """

    def __init__(self, max_size: int, state_dim: int) -> None:
        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros(max_size, dtype=np.int8)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.dones = np.zeros(max_size, dtype=np.int8)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.max_size = max_size
        self.cursor = 0
        self.size = 0

    def store(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a single transition in the buffer.

        Overwrites the oldest transition when the buffer is full (circular).
        """
        idx = self.cursor
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = int(done)
        self.cursor = (self.cursor + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int):
        """Sample a random mini-batch of transitions.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones) arrays.
        """
        effective_batch = min(self.size, batch_size)
        indices = np.random.choice(
            self.size, size=effective_batch, replace=False)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )

    def __len__(self) -> int:
        return self.size
