import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from envs.two_qubit_env import TwoQubitEnv
from algorithms.ppo import train_ppo

def smooth(x, window=15):
    x = np.array(x, dtype=float)
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window)/window, mode="same")

def main():
    env = TwoQubitEnv(max_steps=20)
    model, rewards = train_ppo(env, num_episodes=300)

    rewards_s = smooth(rewards, window=15)

    Path("plots").mkdir(exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(rewards_s, label="PPO (clean, smooth)", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("PPO on Two-Qubit Control (Smoothed)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    out_path = "plots/ppo_clean_smooth.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[PPO] Saved clean smooth plot → {out_path}")

if __name__ == "__main__":
    main()
