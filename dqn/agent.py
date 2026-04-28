"""
Deep Q-Network Agent.

Implements the core DQN algorithm with:
  - Fully-connected Q-network (state → Q-values for all actions)
  - ε-greedy exploration with exponential decay
  - Experience replay for sample-efficient, stable learning
  - MSE loss between predicted and bootstrapped target Q-values

Reference:
    Mnih, V. et al. (2015). "Human-level control through deep
    reinforcement learning." Nature, 518(7540), 529–533.
"""

import numpy as np
import keras
from keras.layers import Dense

from dqn.replay_buffer import ReplayBuffer
from dqn.config import DQNConfig


class DQNAgent:
    """Deep Q-Learning agent for discrete action spaces.

    Args:
        state_dim: Dimensionality of the observation space.
        action_dim: Number of discrete actions available.
        config: Hyperparameter configuration object.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: DQNConfig | None = None,
    ) -> None:
        self.config = config or DQNConfig()
        self.action_dim = action_dim
        self.explore_rate = self.config.explore_rate
        self.memory = ReplayBuffer(self.config.replay_buffer_size, state_dim)
        self.model = self._build_network(state_dim, action_dim)

    def _build_network(self, input_dim: int, output_dim: int) -> keras.Model:
        """Construct the Q-network as a fully-connected MLP.

        Architecture:  Input → [256-ReLU] × 2 → Linear(output_dim)
        """
        inputs = keras.Input(shape=(input_dim,))
        x = inputs
        for _ in range(self.config.num_hidden_layers):
            x = Dense(self.config.hidden_units, activation="relu")(x)
        outputs = Dense(output_dim)(x)

        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer="adam", loss="mse")
        return model

    def select_action(self, state: np.ndarray) -> int:
        """Select an action using ε-greedy exploration.

        With probability ε, a uniformly random action is chosen;
        otherwise the action with the highest Q-value is selected.
        """
        if np.random.random() <= self.explore_rate:
            return np.random.randint(self.action_dim)
        q_values = self.model.predict(
            np.expand_dims(state, axis=0), verbose=0
        )
        return int(np.argmax(q_values[0]))

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store an environment transition in the replay buffer."""
        self.memory.store(state, action, reward, next_state, done)

    def learn(self) -> None:
        """Perform one learning step from a mini-batch of replay data.

        Computes Q-learning targets using the Bellman equation:
            Q_target(s,a) = r + γ · max_a' Q(s', a') · (1 − done)
        and fits the Q-network via MSE regression.
        """
        if len(self.memory) < self.config.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(
            self.config.batch_size
        )

        q_current = self.model.predict(states, verbose=0)
        q_next = self.model.predict(next_states, verbose=0)

        batch_idx = np.arange(len(states), dtype=np.int32)
        q_current[batch_idx, actions] = rewards + (
            self.config.gamma * np.max(q_next, axis=1) * (1 - dones)
        )

        self.model.fit(x=states, y=q_current, verbose=0)
        self._decay_exploration()

    def _decay_exploration(self) -> None:
        """Exponentially decay the exploration rate."""
        self.explore_rate = max(
            self.explore_rate * self.config.explore_rate_decay,
            self.config.min_explore_rate,
        )

    def save(self, path: str | None = None) -> None:
        """Persist model weights to disk."""
        self.model.save_weights(path or self.config.model_name)

    def load(self, path: str | None = None) -> None:
        """Load model weights from disk."""
        self.model.load_weights(path or self.config.model_name)
