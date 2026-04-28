#!/usr/bin/env python3
"""
Evaluation script for a trained DQN LunarLander-v2 agent.

Loads saved model weights and runs deterministic (greedy) episodes
with optional rendering for visual inspection.

Usage:
    python evaluate.py                                 # 10 episodes, no render
    python evaluate.py --episodes 50 --render          # 50 episodes, visual
    python evaluate.py --weights results/best.weights.h5

Author: AG — Chief AI Officer, Google
"""

import argparse
import os

import gymnasium as gym
import numpy as np

from dqn import DQNAgent, DQNConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained DQN agent on LunarLander-v2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--episodes", type=int, default=10, help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="results/DQN_LunarLanderV2.weights.h5",
        help="Path to saved model weights",
    )
    parser.add_argument(
        "--render", action="store_true", help="Render the environment"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    return parser.parse_args()


def evaluate(args: argparse.Namespace) -> None:
    """Run greedy evaluation episodes and report statistics."""
    np.random.seed(args.seed)

    render_mode = "human" if args.render else None
    env = gym.make("LunarLander-v2", render_mode=render_mode)

    config = DQNConfig(explore_rate=0.0)  # Fully greedy
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        config=config,
    )
    agent.explore_rate = 0.0  # Ensure no exploration during eval
    agent.load(args.weights)

    rewards: list[float] = []

    print("=" * 50)
    print("  DQN Evaluation — LunarLander-v2")
    print("=" * 50)

    for ep in range(1, args.episodes + 1):
        state, _ = env.reset(seed=args.seed + ep)
        total_reward = 0.0
        done = False
        steps = 0

        while not done:
            action = agent.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

        rewards.append(total_reward)
        status = "LANDED" if total_reward >= 200 else "CRASHED"
        print(
            f"  Episode {ep:3d} | Reward: {total_reward:8.2f} | Steps: {steps:4d} | {status}")

    env.close()

    print("=" * 50)
    print(f"  Mean Reward:   {np.mean(rewards):8.2f}")
    print(f"  Std Reward:    {np.std(rewards):8.2f}")
    print(f"  Min / Max:     {np.min(rewards):8.2f} / {np.max(rewards):8.2f}")
    print(
        f"  Success Rate:  {sum(1 for r in rewards if r >= 200) / len(rewards) * 100:.1f}%")
    print("=" * 50)


if __name__ == "__main__":
    evaluate(parse_args())
