import numpy as np
from gymnasium import spaces
from envs.two_qubit_env import TwoQubitEnv


class TwoQubitNoisyEnv(TwoQubitEnv):
    """
    在 TwoQubitEnv 的基礎上加入「量子雜訊」的版本。

    雜訊模型（簡化近似）：
    - 每一步在更新 self.current 後，加上一個 Gaussian noise：
      current ← current + N(0, noise_std^2)
    - 之後再重新計算 fidelity 與 reward。
    """

    def __init__(self, max_steps: int = 20, noise_std: float = 0.05):
        super().__init__(max_steps=max_steps)
        self.noise_std = noise_std

    def step(self, action):
        # 先讓原本的動力學更新 current（但不要信任原本的 reward / terminated）
        obs, _, _, _, _ = super().step(action)

        # 在 current 上加入高斯雜訊，模擬 decoherence / control noise
        noise = self.np_random.normal(
            loc=0.0,
            scale=self.noise_std,
            size=self.current.shape,
        )
        self.current = np.clip(self.current + noise, -1.0, 1.0)

        # 用「加雜訊後」的 current 重新計算 fidelity / reward / 終止條件
        fidelity = self._fidelity()
        reward = float(fidelity)
        terminated = fidelity > 0.95
        truncated = self.current_step >= self.max_steps

        obs = self._get_obs()
        info = {
            "fidelity": float(fidelity),
            "noise_std": float(self.noise_std),
        }
        return obs, reward, terminated, truncated, info
