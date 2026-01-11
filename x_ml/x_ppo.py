"""
x_ml.x_ppo
PPO (Proximal Policy Optimization, 近端策略优化) - 离散动作版本

核心点：
- Rollout 采样一段长度 T（rollout_steps），并行 num_envs 个环境
- 用 GAE (Generalized Advantage Estimation, 广义优势估计) 算 advantage
- 用 clipped objective 限制策略更新：ratio = exp(new_logp - old_logp)
- 多轮 update_epochs + minibatch SGD
"""

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from tqdm import tqdm
from pathlib import Path


def layer_init(layer, std=1.0, bias_const=0.0):
    # PPO 常见初始化：orthogonal + 可控 std
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCriticNet(nn.Module):
    """
    共享骨干 + actor head + critic head
    离散动作：actor 输出 logits（不做 softmax），Categorical(logits=...) 内部会处理数值稳定性
    """
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.base = nn.Sequential(
            layer_init(nn.Linear(input_dim, 128), std=np.sqrt(2)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128), std=np.sqrt(2)),
            nn.Tanh(),
        )
        self.actor = layer_init(nn.Linear(128, action_dim), std=0.01)
        self.critic = layer_init(nn.Linear(128, 1), std=1.0)

    def forward(self, x):
        h = self.base(x)
        logits = self.actor(h)
        value = self.critic(h)
        return logits, value


class PPO:
    def __init__(
        self,
        env_id="LunarLander-v3",
        net=ActorCriticNet,
        device="cuda",
        seed=87569,

        # 采样设置
        num_envs=8,
        rollout_steps=128,
        total_timesteps=200_000_000,

        # PPO 超参（离散 LunarLander 推荐从这套开始）
        learning_rate=5e-3,
        anneal_lr=True,
        gamma=0.99,
        gae_lambda=0.95,
        update_epochs=4,
        minibatch_size=256,
        clip_coef=0.2,
        clip_vloss=True,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.02,

        # 评估/保存
        eval_episode=50,
        eval_interval_updates=1000,
        save_path="./model/ppo_best.pth",
    ):
        self.env_id = env_id
        self.device = device
        self.seed = seed

        self.num_envs = num_envs
        self.rollout_steps = rollout_steps
        self.total_timesteps = total_timesteps

        self.lr = learning_rate
        self.anneal_lr = anneal_lr

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.update_epochs = update_epochs # rollout 收集到的资料被用于更新网络, 更新的epoch数
        self.minibatch_size = minibatch_size
        self.clip_coef = clip_coef
        self.clip_vloss = clip_vloss
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl

        self.eval_episode = eval_episode
        self.eval_interval_updates = eval_interval_updates

        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

        # ====== env and eval_env ======
        # PPO 最佳实践：训练用 vector env，并行采样
        def make_env(rank):
            def thunk():
                e = gym.make(env_id)
                e.reset(seed=seed + rank)
                return e
            return thunk

        self.envs = gym.vector.SyncVectorEnv([make_env(i) for i in range(num_envs)], autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,)
        self.eval_env = gym.make(env_id)
        self.eval_env.reset(seed=seed + 9999)


        obs_shape = self.envs.single_observation_space.shape
        act_space = self.envs.single_action_space
        self.state_dim = int(obs_shape[0]) # type: ignore
        self.action_dim = int(act_space.n) # type: ignore

        # ====== net / optim ======
        self.net_class = net
        self.net = self.net_class(self.state_dim, self.action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, eps=1e-5)

        # ====== logs ======
        self.global_step = 0
        self.update_step = 0
        self.best_eval = float("-inf")

        # 记录 episode 回报（vector env 会异步结束，所以用 per-env 累加）
        self.return_reward = []
        self._running_ep_return = np.zeros(self.num_envs, dtype=np.float32)

    def take_action(self, obs, deterministic=False):
        """
        return: action, logprob, value.squeeze(-1), dist.entropy()

        obs: torch.Tensor [num_envs, state_dim] 或 [1, state_dim]
        个人理解：
        dist最后一个维度用于采样，其余维度作为shape,B * D， D前的维度就是sample()和entropy()的维度
        log_prob中， action的维度要和shape保持一致，或者广播一致， 输出和输入维度一样
        """
        with torch.no_grad():
            logits, value = self.net(obs)
            dist = torch.distributions.Categorical(logits=logits)
            if deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                action = dist.sample()
            logprob = dist.log_prob(action)
        return action, logprob, value.squeeze(-1), dist.entropy()

    def rollout(self, obs):
        """
        SAME_STEP autoreset 版本 rollout：
        - episode 结束判定：episode_done = terminated OR truncated（用于统计回合回报、清零）
        - GAE 自举判定：next_nonterminal = 1 - terminated（truncated 允许自举）
        - SAME_STEP 下 env.step 返回的 next_obs 对于 done env 是 reset 后的 obs；
        真正的 “final next obs” 在 infos["final_obs"] 里（用于算 V(s_{t+1}) 做自举）
        """
        T = self.rollout_steps
        N = self.num_envs
        D = self.state_dim

        state = torch.zeros((T, N, D), device=self.device)
        actions_buf = torch.zeros((T, N), device=self.device, dtype=torch.long)
        # 记录的数据为, 在当前step , 状态为state下, 采取action所激发的数据, 包括advantage
        logprobs_buf = torch.zeros((T, N), device=self.device)
        rewards_buf = torch.zeros((T, N), device=self.device)

        # 这里的 dones_buf 我们存 “terminated mask”
        # （因为 terminated 才表示真正的终止：不允许 bootstrap）
        dones_buf = torch.zeros((T, N), device=self.device)

        values_buf = torch.zeros((T, N), device=self.device)

        # 关键：每一步都存一个 “用于 delta 的 next_value”
        # next_values_buf[t, i] = V(actual_next_obs_{t,i})
        next_values_buf = torch.zeros((T, N), device=self.device)

        for t in range(T):
            state[t] = obs # tensor两个数组之间允许逐元素赋值

            action, logprob, value, _ = self.take_action(obs, deterministic=False)
            actions_buf[t] = action
            logprobs_buf[t] = logprob
            values_buf[t] = value

            next_obs, reward, terminated, truncated, infos = self.envs.step(action.cpu().numpy())

            rewards_buf[t] = torch.tensor(reward, device=self.device, dtype=torch.float32)

            terminated_mask = np.array(terminated, dtype=np.float32)
            dones_buf[t] = torch.tensor(terminated_mask, device=self.device, dtype=torch.float32)

            # ====== 采样的return统计 用 terminated OR truncated ======
            episode_done = np.logical_or(terminated, truncated)
            self._running_ep_return += reward.astype(np.float32)
            for i in range(N):
                if episode_done[i]:
                    self.return_reward.append(float(self._running_ep_return[i]))
                    self._running_ep_return[i] = 0.0

            # ====== 关键：为 GAE 准备 “actual_next_obs” ======
            # SAME_STEP 下，如果某个 env 结束了,next_obs[i] 已经是 reset 后的 obs，
            # 但我们要 bootstrap 的是 “final_obs”（真实的最后一步后继状态/终止前状态）
            actual_next_obs = next_obs
            if isinstance(infos, dict) and ("final_obs" in infos):
                final_obs = infos["final_obs"]              # shape=(num_envs,), dtype=object
                has_final = infos.get("_final_obs", episode_done)  # shape=(num_envs,), dtype=bool，布尔掩码数组，指示final_obs数组每个对应元素是否有效
                # 经过final_bos字段的判断， _final_obs字段是一定有的
                if np.any(has_final):
                    idx = np.where(has_final)[0] # where返回的一定是tuple, 每个元素是arr, 维度为1时取第一个
                    # print(idx.shape)
                    # 把 object 数组里的每个元素 stack 成二维 (k, obs_dim)
                    final_stack = np.stack([final_obs[i] for i in idx]).astype(np.float32)

                    actual_next_obs = np.array(next_obs, copy=True)
                    actual_next_obs[idx] = final_stack

            # 用 actual_next_obs 算 V(s_{t+1})，用于 truncation 自举
            with torch.no_grad():
                _, nv = self.net(torch.tensor(actual_next_obs, device=self.device, dtype=torch.float32))
                next_values_buf[t] = nv.squeeze(-1)

            # 下一步 rollout 继续用 env 返回的 next_obs（SAME_STEP 下已是 reset 后 obs）
            obs = torch.tensor(next_obs, device=self.device, dtype=torch.float32)
            self.global_step += N

        # ====== GAE：advantage / returns ======
        advantages = torch.zeros_like(rewards_buf, device=self.device)
        lastgaelam = torch.zeros((N,), device=self.device)

        for t in reversed(range(T)):
            next_nonterminal = 1.0 - dones_buf[t]  # 1 - terminated（truncated 仍为 1 => 会自举）
            delta = rewards_buf[t] + self.gamma * next_values_buf[t] * next_nonterminal - values_buf[t] # delta 是单步优势函数, 在这里截断自举
            lastgaelam = delta + self.gamma * self.gae_lambda * next_nonterminal * lastgaelam # lastgaelam是上一步的GAE-advantage, 也可以被截断
            advantages[t] = lastgaelam

        returns = advantages + values_buf # λ-return

        batch = {
            "obs": state,
            "actions": actions_buf,
            "logprobs": logprobs_buf,
            "values": values_buf,
            "advantages": advantages,
            "returns": returns,
        }
        return batch, obs



    def update(self, batch, total_updates):
        """
        PPO 更新：多轮 epoch + minibatch
        """
        # flatten: [T, N] -> [T*N]
        obs = batch["obs"].reshape(-1, self.state_dim)
        actions = batch["actions"].reshape(-1)
        old_logprobs = batch["logprobs"].reshape(-1)
        old_values = batch["values"].reshape(-1)
        advantages = batch["advantages"].reshape(-1)
        returns = batch["returns"].reshape(-1)

        # advantage normalize（能明显稳定训练）
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        batch_size = obs.shape[0]
        inds = np.arange(batch_size)

        # 学习率线性退火（anneal）
        if self.anneal_lr:
            frac = 1.0 - (self.update_step / float(total_updates))
            lr_now = frac * self.lr
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr_now

        for epoch in range(self.update_epochs):
            np.random.shuffle(inds)

            for start in range(0, batch_size, self.minibatch_size):
                mb_inds = inds[start:start + self.minibatch_size]

                logits, value = self.net(obs[mb_inds])
                value = value.squeeze(-1)

                dist = torch.distributions.Categorical(logits=logits)
                new_logprobs = dist.log_prob(actions[mb_inds])
                entropy = dist.entropy().mean()

                log_ratio = new_logprobs - old_logprobs[mb_inds]
                ratio = log_ratio.exp()

                mb_adv = advantages[mb_inds]

                # ====== Policy loss (clipped) ======
                pg_loss1 = ratio * mb_adv
                pg_loss2 = torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef) * mb_adv
                policy_loss = -torch.min(pg_loss1, pg_loss2).mean()

                # ====== Value loss (optional clipped) ======
                if self.clip_vloss:
                    v_clipped = old_values[mb_inds] + torch.clamp(
                        value - old_values[mb_inds], -self.clip_coef, self.clip_coef
                    )
                    v_loss_unclipped = (value - returns[mb_inds]) ** 2
                    v_loss_clipped = (v_clipped - returns[mb_inds]) ** 2
                    value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    value_loss = 0.5 * ((value - returns[mb_inds]) ** 2).mean()

                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # ====== Early stop by KL (Kullback-Leibler divergence, KL 散度) ======
                # approx_kl ≈ mean(old_logp - new_logp)
                approx_kl = (old_logprobs[mb_inds] - new_logprobs).mean().item()
                if self.target_kl is not None and approx_kl > self.target_kl:
                    return

    def evaluate(self):
        """
        用 deterministic=True（argmax）评估更稳定
        """
        self.net.eval()
        rewards = []
        for _ in range(self.eval_episode):
            obs, info = self.eval_env.reset()
            done = False
            total_r = 0.0
            while not done:
                obs_t = torch.tensor(obs, device=self.device, dtype=torch.float32).unsqueeze(0)
                action, _, _, _ = self.take_action(obs_t, deterministic=True)
                obs, r, terminated, truncated, info = self.eval_env.step(int(action.item()))
                total_r += float(r)
                done = bool(terminated or truncated)
            rewards.append(total_r)
        self.net.train()
        return float(np.mean(rewards))

    def train(self):
        # reset vector env
        obs, info = self.envs.reset(seed=self.seed)
        obs = torch.tensor(obs, device=self.device, dtype=torch.float32)

        total_updates = self.total_timesteps // (self.num_envs * self.rollout_steps)

        self.net.train()
        with tqdm(total=total_updates) as pbar:
            for update in range(total_updates):
                self.update_step = update

                batch, obs = self.rollout(obs)
                self.update(batch, total_updates)

                # eval & save
                if (update + 1) % self.eval_interval_updates == 0:
                    avg_r = self.evaluate()
                    if avg_r >= self.best_eval:
                        self.best_eval = avg_r
                        torch.save(
                            {
                                "net": self.net.state_dict(),
                                "state_dim": self.state_dim,
                                "action_dim": self.action_dim,
                                "env_id": self.env_id,
                            },
                            self.save_path,
                        )
                    pbar.set_postfix(eval_avg=avg_r, best_eval=self.best_eval, global_step=self.global_step)

                pbar.update(1)

    def inference(self, env_id=None, render_mode="human"):
        """
        载入 best 模型进行可视化试玩
        """
        ckpt = torch.load(self.save_path, map_location="cpu")
        self.net.load_state_dict(ckpt["net"])
        self.net.to(self.device)
        self.net.eval()

        env_id = env_id if env_id is not None else ckpt.get("env_id", self.env_id)
        play_env = gym.make(env_id, render_mode=render_mode)

        while True:
            obs, info = play_env.reset()
            done = False
            while not done:
                obs_t = torch.tensor(obs, device=self.device, dtype=torch.float32).unsqueeze(0)
                action, _, _, _ = self.take_action(obs_t, deterministic=True)
                obs, r, terminated, truncated, info = play_env.step(int(action.item()))
                done = bool(terminated or truncated)
    def load_best_model(self):
        ckpt = torch.load(self.save_path, map_location="cpu")
        self.action_dim = ckpt['action_dim']
        self.state_dim = ckpt['state_dim']
        
        self.net = self.net_class(self.state_dim, self.action_dim).to(self.device)
        self.net.load_state_dict(ckpt['net'])


if __name__ == "__main__":
    ppo = PPO(
        env_id="LunarLander-v3",
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=0,

        num_envs=8,
        rollout_steps=128,
        total_timesteps=1_000_000,

        learning_rate=2.5e-4,
        gamma=0.99,
        gae_lambda=0.95,
        update_epochs=4,
        minibatch_size=256,
        clip_coef=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.02,

        eval_episode=20,
        eval_interval_updates=50,
        save_path="./model/ppo_best.pth",
    )
    ppo.train()
    # ppo.inference()

"""
            elif self.autoreset_mode == AutoresetMode.SAME_STEP:
        (
            self._env_obs[i],
            self._rewards[i],
            self._terminations[i],
            self._truncations[i],
            env_info,
        ) = self.envs[i].step(action)

        if self._terminations[i] or self._truncations[i]:
            infos = self._add_info(
                infos,
                {"final_obs": self._env_obs[i], "final_info": env_info},
                i,
            )

            self._env_obs[i], env_info = self.envs[i].reset()
    else:
        raise ValueError(f"Unexpected autoreset mode, {self.autoreset_mode}")

    infos = self._add_info(infos, env_info, i)

# Concatenate the observations
self._observations = concatenate(
    self.single_observation_space, self._env_obs, self._observations
)
self._autoreset_envs = np.logical_or(self._terminations, self._truncations)

return (
    deepcopy(self._observations) if self.copy else self._observations,
    np.copy(self._rewards),
    np.copy(self._terminations),
    np.copy(self._truncations),
    infos,
)
"""