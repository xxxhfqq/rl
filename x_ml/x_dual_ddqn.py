
"""
只需要创建DDDQN对象, 然后使用train 和 inference 这两个函数即可, 创建DQN对象时, 需要设置超参数
要修改参数, 需要在创建对象时自己手动加入, 更改默认值
"""
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import random
from tqdm import tqdm
import gymnasium as gym
from pathlib import Path


class Q_net(nn.Module):
    def __init__(self, input_dim=4, output_dim=2):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, 96)
        self.layerV = nn.Linear(96, 1)
        self.layerA = nn.Linear(96, output_dim)

        self.activation = nn.ReLU()
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)

        v = self.layerV(x)
        a = self.layerA(x)
        
        q = v + a - a.mean(dim=1, keepdim=True)
        return q

def cal_decay(num_episode, epsilon_start, epsilon_end):
    return (epsilon_end / epsilon_start) ** (1 / num_episode)
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, terminated, truncated):
        self.buffer.append([state, action, reward, next_state, terminated, truncated])
    
    def sample(self, batch_size):
        """
        返回的是state, action, reward, next_state, terminated, truncated, 各个都是列表
        """
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, terminated, truncated = zip(*batch)

        return np.array(state), action, reward, np.array(next_state), terminated, truncated
    def __len__(self):
        return len(self.buffer)
    
class DDDQN():
    """
    动作离散可数, 使用经验重放和semi-gradient, 每个step都更新网络
    """
    def __init__(self,net=Q_net, lr=1e-4, num_episodes=600, gamma=0.98,  buffer_size=10000, minimal_size=200, batch_size=128,
                            epsilon_start=0.9, epsilon_end=0.05, criterion = nn.MSELoss, env=gym.make("CartPole-v1"),
                            eval_step=50, eval_episode=50, eval_env=gym.make("CartPole-v1"), target_update_step=100, device="cuda"
                            ):
        self.target_update_step = target_update_step
        self.device = device
        self.save_path =  Path("./model/best.pth")
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.net = net
        self.q_net = self.net().to(self.device)
        self.target_net = self.net().to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.criterion = criterion()
        self.batch_size = batch_size

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.num_episodes = num_episodes
        self.epsilon_decay = cal_decay(self.num_episodes, self.epsilon, self.epsilon_end)
        

        self.buffer = ReplayBuffer(buffer_size)

        self.gamma = gamma

        self.minimal_size = minimal_size

        self.env = env
        self.eval_env = eval_env
        self.action_dim = self.env.action_space.n # type: ignore
        self.count = 0

        self.return_reward = []

        self.eval_step = eval_step
        self.eval_episode = eval_episode
        self.max_reward = float("-inf")

    def take_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor(np.array(state), dtype=torch.float).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action = self.q_net(state).argmax(dim=1).item()
        return action

    def update(self):
        state, action, reward, next_state, terminated, truncated = self.buffer.sample(self.batch_size)
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        action = torch.tensor(action, dtype=torch.long).unsqueeze(1).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float).unsqueeze(1).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float).to(self.device)
        done = [item1 or item2 for item1, item2 in zip(terminated, truncated)]
        done = torch.tensor(done, dtype=torch.float).unsqueeze(1).to(self.device)

        q1 = self.q_net(state).gather(1, action)
        with torch.no_grad():
            best_action = self.q_net(next_state).argmax(1).unsqueeze(1)
            q2 = self.target_net(next_state).gather(1, best_action)
            q2 = reward + (self.gamma * q2) * (1 - done)

        self.q_net.train()
        loss = self.criterion(q1, q2)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.q_net.eval()
        

    def train(self):
        with tqdm(total=self.num_episodes) as pbar:
            for i in range(self.num_episodes):
                done = False
                state, info = self.env.reset()
                total_reward = 0
                step_count = 0
                while not done:
                    step_count += 1

                    action = self.take_action(state)
                    next_state, reward, terminated, truncated, info = self.env.step(action)
                    self.buffer.add(state, action, reward, next_state, terminated, truncated)
                    state = next_state
                    done = terminated or truncated

                    if len(self.buffer) > self.minimal_size:
                        self.update()
                    total_reward += reward # type: ignore
                
                    if (self.count + 1) % self.target_update_step == 0:
                        self.target_net.load_state_dict(self.q_net.state_dict())
                    self.count = (1 + self.count) % self.target_update_step
                self.return_reward.append(total_reward)
                if (i + 1) % self.eval_step == 0:
                    self.eval(pbar)
                pbar.update()
                pbar.set_postfix(return_reward=total_reward, max_eval_reward=self.max_reward, epsilon=self.epsilon, step_count=step_count)

                self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


    def eval(self, pbar):
        """
        测试最佳模型, 随机玩几把看看效果
        """
        old_epsilon = self.epsilon
        self.epsilon = 0
        reward_list = []
        for i in range(self.eval_episode):
            total_reward = 0
            state, info = self.eval_env.reset()
            done = False
            while not done:
                action = self.take_action(state)
                next_state, reward, terminated, truncated, info = self.eval_env.step(action)
                state = next_state
                total_reward += reward # type: ignore
                done = terminated or truncated
            reward_list.append(total_reward)
        reward = 0
        for item in reward_list:
            reward += item
        reward = reward / len(reward_list)

        if reward >= self.max_reward:
            self.max_reward = reward
            torch.save(self.q_net.state_dict(), self.save_path)
        pbar.set_postfix(reward=reward)
        self.epsilon = old_epsilon

    def inference(self, env=gym.make("CartPole-v1", render_mode="human")):
        q_net = self.net().to(self.device)
        ckpt  = torch.load(self.save_path, map_location='cpu')
        q_net.load_state_dict(ckpt)

        while True:
            done = False
            state , info = env.reset()
            while not done:
                state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action = q_net(state).argmax(1).item()
                next_state, reward, terminated, truncated, info = env.step(action)
                state = next_state
                done = terminated or truncated