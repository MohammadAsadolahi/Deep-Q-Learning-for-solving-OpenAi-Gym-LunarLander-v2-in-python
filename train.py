#!/usr/bin/env python3
"""
Training script for the DQN LunarLander-v2 agent.

Usage:
    python train.py                          # Train with default hyperparameters
    python train.py --episodes 1000          # Train for 1000 episodes
    python train.py --render                 # Visualize during training

Author: AG — Chief AI Officer, Google
"""

import argparse
import os

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from dqn import DQNAgent, DQNConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a DQN agent on LunarLander-v2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--episodes", type=int, default=500, help="Number of training episodes"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Mini-batch size for replay sampling"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="Discount factor"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument(
        "--render", action="store_true", help="Render the environment during training"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save plots and model weights",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="DQN_LunarLanderV2.weights.h5",
        help="Filename for saved model weights",
    )
    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
    """Execute the full DQN training loop."""
    os.makedirs(args.output_dir, exist_ok=True)

    config = DQNConfig(
        gamma=args.gamma,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_episodes=args.episodes,
        model_name=os.path.join(args.output_dir, args.model_name),
    )

    render_mode = "human" if args.render else None
    env = gym.make("LunarLander-v2", render_mode=render_mode)

    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        config=config,
    )

    episode_rewards: list[float] = []
    avg_rewards: list[float] = []
    best_avg = -np.inf

    print("=" * 60)
    print("  DQN Training — LunarLander-v2")
    print("=" * 60)
    print(f"  Episodes:      {config.num_episodes}")
    print(f"  Batch size:    {config.batch_size}")
    print(f"  γ (gamma):     {config.gamma}")
    print(f"  ε decay:       {config.explore_rate_decay}")
    print(f"  Output dir:    {args.output_dir}")
    print("=" * 60)

    for episode in range(1, config.num_episodes + 1):
        state, _ = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_transition(state, action, reward, next_state, done)
            agent.learn()

            state = next_state
            total_reward += reward

        episode_rewards.append(total_reward)
        running_avg = np.mean(episode_rewards[-100:])
        avg_rewards.append(running_avg)

        if running_avg > best_avg:
            best_avg = running_avg
            agent.save()

        if episode % 10 == 0 or episode == 1:
            print(
                f"  Episode {episode:4d} | "
                f"Reward: {total_reward:8.2f} | "
                f"Avg(100): {running_avg:8.2f} | "
                f"ε: {agent.explore_rate:.4f}"
            )

    env.close()

    # ── Save training curves ──────────────────────────────────────
    _plot_results(episode_rewards, avg_rewards, args.output_dir)
    print(f"\n  Training complete. Model saved to {args.output_dir}/")


def _plot_results(
    episode_rewards: list[float],
    avg_rewards: list[float],
    output_dir: str,
) -> None:
    """Generate and save reward plots."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(episode_rewards, alpha=0.6, linewidth=0.8)
    axes[0].set_title("Episode Rewards")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total Reward")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(avg_rewards, color="darkorange", linewidth=2)
    axes[1].set_title("Running Average Reward (window=100)")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Average Reward")
    axes[1].axhline(y=200, color="green", linestyle="--",
                    alpha=0.7, label="Solved (200)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"), dpi=200)
    plt.close()


if __name__ == "__main__":
    train(parse_args())
