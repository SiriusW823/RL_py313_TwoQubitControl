import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------------------------------------
# A2C Actor-Critic Network
# ---------------------------------------------------------
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
        )
        self.actor = nn.Linear(64, act_dim)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        h = self.shared(x)
        logits = self.actor(h)
        value = self.critic(h)
        probs = torch.softmax(logits, dim=-1)
        return probs, value


# ---------------------------------------------------------
# A2C Training Function (NO Gym Needed)
# ---------------------------------------------------------
def train_a2c(env, num_episodes=300, gamma=0.99, lr=1e-3):
    obs = env.reset()

    obs_dim = len(obs)          # <= 8
    act_dim = 6                 # <= fixed 6 unitary actions

    model = ActorCritic(obs_dim, act_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    episode_rewards = []

    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0

        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            probs, value = model(obs_tensor)

            dist = torch.distributions.Categorical(probs)
            action = dist.sample().item()

            next_obs, reward, done, _ = env.step(action)
            total_reward += reward

            # TD target
            next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)
            _, next_value = model(next_obs_tensor)

            td_target = reward + (0 if done else gamma * next_value.item())
            td_error = td_target - value.item()

            # loss = -logπ(a|s)*TD + TD^2 (critic)
            loss = -dist.log_prob(torch.tensor(action)) * td_error + td_error**2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            obs = next_obs

        episode_rewards.append(total_reward)

        if ep % 10 == 0:
            print(f"[A2C] Episode {ep}/{num_episodes}, reward={total_reward:.3f}")

    return model, episode_rewards
