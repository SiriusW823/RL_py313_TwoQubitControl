import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.two_qubit_env import TwoQubitEnv
from algorithms.a2c import train_a2c
from algorithms.ppo import train_ppo

def smooth(x, window=15):
    x = np.array(x, dtype=float)
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window)/window, mode="same")

def main():
    num_episodes = 300

    print("Training A2C...")
    _, a2c_rewards = train_a2c(TwoQubitEnv(max_steps=20), num_episodes)

    print("Training PPO...")
    _, ppo_rewards = train_ppo(TwoQubitEnv(max_steps=20), num_episodes)

    a2c_s = smooth(a2c_rewards, 15)
    ppo_s = smooth(ppo_rewards, 15)

    Path("plots").mkdir(exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(a2c_s, label="A2C (clean, smooth)", linewidth=2)
    plt.plot(ppo_s, label="PPO (clean, smooth)", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("A2C vs PPO (Clean Environment, Smoothed)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    out_path = "plots/a2c_vs_ppo_clean_smooth.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"[Compare] Saved A2C vs PPO comparison plot → {out_path}")

if __name__ == "__main__":
    main()
