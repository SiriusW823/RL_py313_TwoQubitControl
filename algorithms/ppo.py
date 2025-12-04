import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim)
        )

    def forward(self, x):
        logits = self.net(x)
        probs = torch.softmax(logits, dim=-1)
        return probs


class Critic(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


def train_ppo(env, num_episodes=300, gamma=0.99, lr=3e-4, clip_eps=0.2, update_steps=10):
    obs = env.reset()
    obs_dim = len(obs)
    act_dim = 6

    actor = Actor(obs_dim, act_dim)
    critic = Critic(obs_dim)

    opt_actor = optim.Adam(actor.parameters(), lr=lr)
    opt_critic = optim.Adam(critic.parameters(), lr=lr)

    episode_rewards = []

    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0

        states, actions, rewards, values, logprobs = [], [], [], [], []

        # Collect episode
        while not done:
            s = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            probs = actor(s)
            dist = torch.distributions.Categorical(probs)

            action = dist.sample().item()
            value = critic(s).item()

            next_obs, reward, done, _ = env.step(action)

            states.append(obs)
            actions.append(action)
            rewards.append(reward)
            values.append(value)
            logprobs.append(dist.log_prob(torch.tensor(action)).item())

            obs = next_obs
            total_reward += reward

        # Compute returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.append(G)
        returns.reverse()
        returns = torch.tensor(returns, dtype=torch.float32)

        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions)
        old_logprobs = torch.tensor(logprobs)

        # Advantage
        values_t = critic(states).squeeze()
        advantages = returns - values_t.detach()

        # PPO update
        for _ in range(update_steps):
            probs = actor(states)
            dist = torch.distributions.Categorical(probs)
            new_logprobs = dist.log_prob(actions)

            ratio = torch.exp(new_logprobs - old_logprobs)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = (returns - critic(states).squeeze()).pow(2).mean()

            opt_actor.zero_grad()
            actor_loss.backward()
            opt_actor.step()

            opt_critic.zero_grad()
            critic_loss.backward()
            opt_critic.step()

        episode_rewards.append(total_reward)

        if ep % 10 == 0:
            print(f"[PPO] Episode {ep}/{num_episodes}, reward={total_reward:.3f}")

    return actor, episode_rewards
