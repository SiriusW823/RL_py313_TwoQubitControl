import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from envs.two_qubit_env import TwoQubitEnv
from algorithms.a2c import train_a2c

# 平滑函數
def smooth(x, window=15):
    x = np.array(x, dtype=float)
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window)/window, mode="same")

def main():
    env = TwoQubitEnv(max_steps=20)
    model, rewards = train_a2c(env, num_episodes=300)

    # 平滑後 reward
    rewards_s = smooth(rewards, window=15)

    # 產生 plots/ 資料夾
    Path("plots").mkdir(exist_ok=True)

    # 畫圖
    plt.figure(figsize=(10, 6))
    plt.plot(rewards_s, label="A2C (clean, smooth)", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("A2C on Two-Qubit Control (Smoothed)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    out_path = "plots/a2c_clean_smooth.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[A2C] Saved clean smooth plot → {out_path}")

if __name__ == "__main__":
    main()
