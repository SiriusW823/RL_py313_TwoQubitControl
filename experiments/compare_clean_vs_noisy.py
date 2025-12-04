import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.two_qubit_env import TwoQubitEnv
from algorithms.a2c import train_a2c
from algorithms.ppo import train_ppo

# 平滑化函式
def smooth(data, window=15):
    data = np.array(data, dtype=float)
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode="same")

def main():
    num_episodes = 300
    sigma = 0.05

    print("Training A2C (clean)...")
    _, a2c_clean = train_a2c(TwoQubitEnv(max_steps=20, noise_sigma=0.0), num_episodes)

    print("Training PPO (clean)...")
    _, ppo_clean = train_ppo(TwoQubitEnv(max_steps=20, noise_sigma=0.0), num_episodes)

    print(f"Training A2C (noisy, sigma={sigma})...")
    _, a2c_noisy = train_a2c(TwoQubitEnv(max_steps=20, noise_sigma=sigma), num_episodes)

    print(f"Training PPO (noisy, sigma={sigma})...")
    _, ppo_noisy = train_ppo(TwoQubitEnv(max_steps=20, noise_sigma=sigma), num_episodes)

    # 平滑化
    a2c_clean_s = smooth(a2c_clean, 15)
    ppo_clean_s = smooth(ppo_clean, 15)
    a2c_noisy_s = smooth(a2c_noisy, 15)
    ppo_noisy_s = smooth(ppo_noisy, 15)

    # 建立資料夾
    Path("plots").mkdir(exist_ok=True)

    plt.figure(figsize=(10, 6))

    # Clean
    plt.plot(a2c_clean_s, label="A2C (clean)", linewidth=2)
    plt.plot(ppo_clean_s, label="PPO (clean)", linewidth=2)

    # Noisy
    plt.plot(a2c_noisy_s, label=f"A2C (noisy, σ={sigma})", linewidth=2)
    plt.plot(ppo_noisy_s, label=f"PPO (noisy, σ={sigma})", linewidth=2)

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"Clean vs Noisy Environment (σ={sigma})", fontsize=16)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=11)
    plt.tight_layout()

    out_path = f"plots/clean_vs_noisy_sigma_{sigma}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"[Compare] Saved clean vs noisy plot → {out_path}")

if __name__ == "__main__":
    main()
