"""
x_ml.x_ac
Actor-Critic (离散动作)

- Actor: 输出离散动作策略 π(a|s)，用策略梯度更新
- Critic: 估计状态价值 V(s)，用 TD(0) 目标计算 TD error 近似 advantage


"""

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import gymnasium as gym
from pathlib import Path


class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 96),
            nn.ReLU(),
            nn.Linear(96, 96),
            nn.ReLU(),
            nn.Linear(96, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 96),
            nn.ReLU(),
            nn.Linear(96, 96),
            nn.ReLU(),
            nn.Linear(96, 1),
        )

    def forward(self, x):
        return self.net(x)


class ActorCritic:
    def __init__(
        self,
        env_id="LunarLander-v3",
        actor_net=Actor,
        critic_net=Critic,
        actor_lr=5e-3,
        critic_lr=5e-3,
        num_episodes=100000,
        gamma=0.99,
        eval_episode=50,
        eval_interval=2000,
        critic_criterion=None,
        device="cuda",
        seed=None,
        save_path="./model/best.pth",
    ):
        self.device = device

        # env / eval_env：不要写在函数默认参数里 gym.make(...)（避免导入时副作用）
        self.env = gym.make(env_id)
        self.eval_env = gym.make(env_id)

        if seed is not None:
            self.env.reset(seed=seed)
            self.eval_env.reset(seed=seed)

        # 自动推断维度
        self.state_dim = int(self.env.observation_space.shape[0]) # type:ignore
        self.action_dim = int(self.env.action_space.n) #type:ignore

        # 保存路径
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

        # 网络与优化器
        self.actor = actor_net(self.state_dim, self.action_dim).to(self.device)
        self.critic = critic_net(self.state_dim).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.critic_criterion = critic_criterion if critic_criterion is not None else nn.MSELoss()

        # 超参与日志
        self.num_episodes = num_episodes
        self.gamma = gamma
        self.return_reward = []
        self.max_reward = float("-inf")

        self.eval_episode = eval_episode
        self.eval_interval = eval_interval

    def take_action(self, state, deterministic=False):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.actor(state)  # [1, action_dim]
            if deterministic:
                # 评估/展示时用贪心更稳定
                action = torch.argmax(logits, dim=1)
                return int(action.item())
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            return int(action.item())

    def update(self, state, action, reward, next_state, terminated):
        """
        （输入 T 步数据）
        - state: [T, state_dim]
        - action: [T]
        - reward: [T]
        - next_state: [T, state_dim]
        - done: [T]  (1 表示终止/截断)
        """
        state = torch.tensor(state, dtype=torch.float).to(self.device)               # [T, state_dim]
        action = torch.tensor(action, dtype=torch.long).to(self.device)             # [T]
        reward = torch.tensor(reward, dtype=torch.float).unsqueeze(1).to(self.device)      # [T, 1]
        next_state = torch.tensor(next_state, dtype=torch.float).to(self.device)    # [T, state_dim]
        terminated = torch.tensor(terminated, dtype=torch.float).unsqueeze(1).to(self.device)   # [T, 1]

        # Critic: TD target = r + gamma * V(s') * (1-done)
        v_s = self.critic(state)  # [T, 1]
        with torch.no_grad():
            v_next = self.critic(next_state)  # [T, 1]
            td_target = reward + self.gamma * v_next * (1 - terminated)  # [T, 1]

        critic_loss = self.critic_criterion(v_s, td_target)

        # Actor: advantage ≈ TD error = (td_target - V(s))
        # detach 的意义：Actor 更新时，把 Critic 当成“常量基线”，不让梯度流进 Critic
        td_error = (td_target - v_s).detach().squeeze(1)  # [T]

        logits = self.actor(state)  # [T, action_dim]
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(action)  # [T]

        actor_loss = (-log_probs * td_error).mean()

        # 先更 Actor，再更 Critic（顺序不关键，但保持清晰）
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def train(self):
        with tqdm(total=self.num_episodes) as pbar:
            for i in range(self.num_episodes):
                self.actor.train()
                self.critic.train()

                state, info = self.env.reset()
                done = False
                total_reward = 0.0

                states = []
                actions = []
                rewards = []
                next_states = []
                terminateds = []

                while not done:
                    action = self.take_action(state)
                    next_state, reward, terminated, truncated, info = self.env.step(action)

                    done_flag = bool(terminated or truncated)
                    total_reward += float(reward)

                    states.append(state)
                    actions.append(action)
                    rewards.append(reward)
                    next_states.append(next_state)
                    terminateds.append(terminated)

                    state = next_state
                    done = done_flag

                # 保持你的结构：每个 episode 收集完再 update 一次
                self.update(
                    np.array(states),
                    np.array(actions),
                    np.array(rewards),
                    np.array(next_states),
                    np.array(terminateds),
                )

                self.return_reward.append(total_reward)

                if (i + 1) % self.eval_interval == 0:
                    self.eval(pbar)

                pbar.update(1)
                pbar.set_postfix(train_return=total_reward, best_eval=self.max_reward)

    def eval(self, pbar=None):
        """
        测试并保存最佳模型（用 deterministic=True 更稳定）
        """
        self.actor.eval()
        self.critic.eval()

        reward_list = []
        for _ in range(self.eval_episode):
            state, info = self.eval_env.reset()
            done = False
            total_reward = 0.0

            while not done:
                action = self.take_action(state, deterministic=True)
                next_state, reward, terminated, truncated, info = self.eval_env.step(action)
                total_reward += float(reward)
                state = next_state
                done = bool(terminated or truncated)

            reward_list.append(total_reward)

        avg_reward = float(np.mean(reward_list))

        if avg_reward >= self.max_reward:
            self.max_reward = avg_reward
            torch.save(
                {
                    "actor": self.actor.state_dict(),
                    "critic": self.critic.state_dict(),
                    "state_dim": self.state_dim,
                    "action_dim": self.action_dim,
                },
                self.save_path,
            )

        if pbar is not None:
            pbar.set_postfix(eval_avg_reward=avg_reward, best_eval=self.max_reward)

        return avg_reward

    def inference(self, env_id="LunarLander-v3", render_mode="human"):
        """
        载入 best 模型进行可视化试玩
        """
        ckpt = torch.load(self.save_path, map_location="cpu")
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])

        play_env = gym.make(env_id, render_mode=render_mode)

        self.actor.eval()
        while True:
            state, info = play_env.reset()
            done = False
            while not done:
                action = self.take_action(state, deterministic=True)
                next_state, reward, terminated, truncated, info = play_env.step(action)
                state = next_state
                done = bool(terminated or truncated)


if __name__ == "__main__":
    ac = ActorCritic(
        env_id="LunarLander-v3",
        actor_lr=5e-3,
        critic_lr=5e-3,
        num_episodes=20000,
        gamma=0.99,
        eval_episode=20,
        eval_interval=500,
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=0,
        save_path="./model/best.pth",
    )
    ac.train()
    # ac.inference()
