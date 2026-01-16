"""
单文件版: 本地分钟 CSV -> 技术指标 -> 自定义 Gymnasium 交易环境(支持5分钟 + A股T+1) -> PPO 训练/评估(best) -> 2025 回测 -> 推理(infer)

你可以先按 main() 从上到下读一遍, 再回过头看各个函数: 
- 数据:parse_dt / load_ohlcv
- 特征:add_indicators
- 切分:split_range / split_train_eval
- 环境:AStockT1Env(核心:T+1、整手、手续费、reward)
- 强化学习:train_best / backtest / infer_last
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from stockstats import StockDataFrame
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


INDICATORS = ["macd", "rsi_30", "cci_30", "dx_30", "close_30_sma", "close_60_sma"]
OHLCV_COLS = ["open", "high", "low", "close", "volume"]


def parse_dt(df: pd.DataFrame) -> pd.Series:
    """
    从本地 CSV 的 time/date 字段解析出 pandas datetime。

    支持: 
    - time=YYYYmmddHHMMSSfff(例如 20250101103000000)
    - date=YYYY-mm-dd + time=HHMMSS(或末 6 位)
    - fallback: 直接让 pandas 推断
    """
    if "time" not in df.columns:
        raise KeyError("CSV 必须包含 time 列")
    t = df["time"].astype(str)

    dt = pd.to_datetime(t, format="%Y%m%d%H%M%S%f", errors="coerce")
    if dt.isna().all() and "date" in df.columns:
        dt = pd.to_datetime(
            df["date"].astype(str) + " " + t.str[-6:],
            format="%Y-%m-%d %H%M%S",
            errors="coerce",
        )
    if dt.isna().all():
        dt = pd.to_datetime(t, errors="coerce")
    if dt.isna().all():
        raise RuntimeError("无法解析时间, 请检查 CSV time/date 格式")
    return dt


def load_ohlcv(csv_path: str) -> pd.DataFrame:
    """
    加载本地分钟数据 CSV, 并返回最小 OHLCV + datetime 的 DataFrame。

    期望 CSV 至少包含: 
    - time, open, high, low, close, volume
    """
    df = pd.read_csv(csv_path)
    df = df.assign(datetime=parse_dt(df))
    need = ["datetime", *OHLCV_COLS]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"CSV 缺少必要列: {missing}")
    return df[need].sort_values("datetime").reset_index(drop=True)


def add_indicators(df: pd.DataFrame, indicators: list[str]) -> pd.DataFrame:
    """
    用 stockstats 生成技术指标列。

    注意: stockstats 会把列名转成小写并做一定的内部处理, 因此这里统一小写化。
    """
    tmp = df.copy()
    tmp.columns = [c.lower() for c in tmp.columns]
    ss = StockDataFrame.retype(tmp)
    for ind in indicators:
        _ = ss[ind]
    out = pd.DataFrame(ss).reset_index()
    if "datetime" not in out.columns and "index" in out.columns:
        out = out.rename(columns={"index": "datetime"})
    out["datetime"] = pd.to_datetime(out["datetime"])
    return out


def prepare_feature_df(df: pd.DataFrame, tic: str) -> pd.DataFrame:
    """
    生成交易环境使用的最小 DataFrame: 
    - date: 时间戳(5分钟级别也可以)
    - close: 用于结算/交易的价格
    - indicators: 作为观测特征
    """
    out = df.rename(columns={"datetime": "date"}).copy() if "date" not in df.columns else df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["tic"] = tic
    out = out.sort_values(["date"]).reset_index(drop=True)
    # 统一 NaN -> 0, 避免观测出现 NaN
    return out.fillna(0)


def split_range(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """
    按日期范围切片(闭区间), 并重新构造 day 索引。
    """
    s, e = pd.to_datetime(start), pd.to_datetime(end)
    return df[(df["date"] >= s) & (df["date"] <= e)].copy().reset_index(drop=True)


def split_train_eval(df_train_full: pd.DataFrame, eval_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    从训练全集按“日期顺序”切出验证集(eval)。

    - 训练: 较早的日期
    - 验证: 训练末尾的一段日期(比例 eval_ratio)

    用途: 训练时 EvalCallback 会在 eval_env 上评估并保存 best_model.zip
    """
    dates = pd.to_datetime(df_train_full["date"]).sort_values().unique()
    cut = min(max(1, int(len(dates) * (1.0 - eval_ratio))), len(dates) - 1)
    last_train_date = dates[cut - 1]
    train_df = df_train_full[df_train_full["date"] <= last_train_date].copy()
    eval_df = df_train_full[df_train_full["date"] > last_train_date].copy()
    return train_df.reset_index(drop=True), eval_df.reset_index(drop=True)


def make_paths(data_csv: str, out_dir: str = "trained_model"):
    """
    输出路径约定: 
    - best_model.zip: EvalCallback 自动保存的最佳模型
    - ppo_xxx.zip: 训练结束时保存的最终模型
    """
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    stem = Path(data_csv).stem
    return out, out / "best_model.zip", out / f"ppo_{stem}.zip"


class AStockT1Env(gym.Env):
    """

    1) 时间建模更贴近真实成交: 
       - 在 bar(t) close 做决策
       - 在 bar(t+1) open 成交(因此 step 需要 i+1)
       - reward = equity_close[t+1] - equity_close[t]

    2) 观测使用 window 序列 + 归一化(更利于训练): 
       - window_size * features(open/high/low/close 会除以 last_close: volume 做 log1p)
       - 账户状态追加: pos_frac / sellable_frac / cash_frac

    A 股规则(本项目重点): 
    - T+1: 当天买入的仓位当天不能卖出(只能卖“可卖仓位”)
    - 100股整手: 买卖数量按 lot_size 取整
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        indicators: list[str],
        *,
        window_size: int = 240,
        initial_cash: float = 1_000_000.0,
        fee_rate: float = 0.0003,
        min_fee: float = 5.0,
        slippage_pct: float = 0.0,
        max_trade: int = 10_000,
        lot_size: int = 100,
        t_plus_one: bool = True,
        reward_scaling: float = 1.0,
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.indicators = indicators

        self.window_size = int(window_size)
        self.initial_cash = float(initial_cash)
        self.fee_rate = float(fee_rate)
        self.min_fee = float(min_fee)
        self.slippage_pct = float(slippage_pct)
        self.max_trade = int(max_trade)
        self.lot_size = int(lot_size)
        self.t_plus_one = bool(t_plus_one)
        self.reward_scaling = float(reward_scaling)

        # 必需列(用于 next-open 成交 & close 结算)
        for c in ("open", "close", "date"):
            if c not in self.df.columns:
                raise ValueError(f"df 缺少列: {c}")

        self.open_ = self.df["open"].astype(float).to_numpy()
        self.close_ = self.df["close"].astype(float).to_numpy()
        self.dates = pd.to_datetime(self.df["date"]).to_numpy()

        # feature: OHLCV + indicators(和 b.py 一样, 价格做归一化、volume做log1p)
        self.feature_cols = [c for c in OHLCV_COLS if c in self.df.columns] + list(self.indicators)
        self.n_features = len(self.feature_cols)
        self.features = self.df[self.feature_cols].to_numpy(dtype=np.float32)

        # 动作: 目标仓位比例(更稳定, 比“直接买卖股数”更容易学)
        # 为了简化(A股默认不做空), 这里用 [0, 1]
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        obs_dim = self.window_size * self.n_features + 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self._reset_state()

    def _reset_state(self):
        if len(self.df) < self.window_size + 2:
            raise ValueError("数据太短, 至少需要 window_size + 2 行(因为成交在 next open)")

        self.i = self.window_size - 1  # 当前决策点(在 close_i 决策)
        self.cash = float(self.initial_cash)
        self.shares = 0  # int shares held
        self.locked_today = 0  # shares bought today (T+1 locked)

        # 当前“交易日”key(用于跨日解锁: 按自然日, 不是交易所日历)
        self._current_trade_date = self._date_key(self.dates[self.i])

    @staticmethod
    def _date_key(dt64) -> str:
        # dt64 may be numpy datetime64; convert to date string key
        return str(pd.to_datetime(dt64).date())

    def _maybe_roll_day_for_index(self, idx: int):
        cur = self._date_key(self.dates[idx])
        if cur != self._current_trade_date:
            self.locked_today = 0
            self._current_trade_date = cur

    def _sellable(self) -> int:
        return int(self.shares - self.locked_today) if self.t_plus_one else int(self.shares)

    def _get_obs(self) -> np.ndarray:
        # window features
        start = self.i - self.window_size + 1
        window = self.features[start : self.i + 1].copy()  # (window_size, n_features)

        last_close = float(self.close_[self.i])
        if last_close > 0:
            for k, col in enumerate(self.feature_cols):
                if col in ("open", "high", "low", "close"):
                    window[:, k] = window[:, k] / np.float32(last_close)
                elif col == "volume":
                    window[:, k] = np.log1p(window[:, k])

        flat = window.reshape(-1).astype(np.float32)

        equity = self.cash + self.shares * last_close
        pos_value = self.shares * last_close
        sellable_value = self._sellable() * last_close

        if equity > 1e-8:
            pos_frac = pos_value / equity
            sellable_frac = sellable_value / equity
            cash_frac = self.cash / equity
        else:
            pos_frac, sellable_frac, cash_frac = 0.0, 0.0, 1.0

        tail = np.array([pos_frac, sellable_frac, cash_frac], dtype=np.float32)
        return np.concatenate([flat, tail], axis=0)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_obs(), {}

    def step(self, action):
        i = self.i
        next_i = i + 1

        # 必须有 next_i 才能成交
        terminated = i >= len(self.df) - 2
        if terminated:
            return self._get_obs(), 0.0, True, False, self._info(i)

        # 下一根 bar 开盘前: 如果跨日, 先解锁(这样 next open 的卖出/换仓符合 T+1)
        self._maybe_roll_day_for_index(next_i)

        price_i_close = float(self.close_[i])
        equity_i = self.cash + self.shares * price_i_close

        # 动作: 目标仓位比例(0..1)
        target_pos = float(np.clip(action[0], 0.0, 1.0))

        exec_price = float(self.open_[next_i])
        if exec_price <= 0:
            # 不可成交: 直接推进到 next_i
            self.i = next_i
            equity_next = self.cash + self.shares * float(self.close_[next_i])
            reward = (equity_next - equity_i) * self.reward_scaling
            return self._get_obs(), float(reward), False, False, self._info(self.i)

        # 目标持仓价值 -> 目标股数(按整数股、整手)
        target_value = equity_i * target_pos
        target_shares = int(target_value // exec_price)
        target_shares = (target_shares // self.lot_size) * self.lot_size

        delta = target_shares - int(self.shares)

        # 卖出(受 T+1 可卖约束)
        if delta < 0:
            sell_shares = min(-delta, self._sellable())
            sell_shares = (sell_shares // self.lot_size) * self.lot_size
            if sell_shares > 0:
                price = exec_price * (1.0 - self.slippage_pct)
                trade_value = price * sell_shares
                fee = max(trade_value * self.fee_rate, self.min_fee)
                proceeds = max(0.0, trade_value - fee)
                self.cash += proceeds
                self.shares -= sell_shares

        # 买入(受现金约束)
        elif delta > 0:
            buy_shares = min(delta, self.max_trade)
            buy_shares = (buy_shares // self.lot_size) * self.lot_size
            if buy_shares > 0:
                price = exec_price * (1.0 + self.slippage_pct)
                if price > 0:
                    # 由于存在“最低 5 元手续费”, 可买股数不能简单除法, 需要分段估算: 
                    # - 若 trade_value*fee_rate >= min_fee: 总成本 = trade_value*(1+fee_rate)
                    # - 否则: 总成本 = trade_value + min_fee
                    shares_by_rate = int(self.cash // (price * (1.0 + self.fee_rate)))
                    shares_by_min = int((self.cash - self.min_fee) // price) if self.cash > self.min_fee else 0
                    affordable = max(shares_by_rate, shares_by_min)
                    affordable = (affordable // self.lot_size) * self.lot_size
                    buy_shares = min(buy_shares, affordable)

                    if buy_shares > 0:
                        trade_value = price * buy_shares
                        fee = max(trade_value * self.fee_rate, self.min_fee)
                        total_cost = trade_value + fee
                        # 防止浮点误差导致现金变负
                        if total_cost <= self.cash + 1e-8:
                            self.cash -= total_cost
                            self.shares += buy_shares
                            if self.t_plus_one:
                                self.locked_today += buy_shares

        # 用 next bar close 结算 reward(b.py 的做法)
        price_next_close = float(self.close_[next_i])
        equity_next = self.cash + self.shares * price_next_close
        reward = (equity_next - equity_i) * self.reward_scaling

        self.i = next_i
        return self._get_obs(), float(reward), False, False, self._info(self.i)

    def _info(self, i: int) -> dict:
        price = float(self.close_[i])
        equity = self.cash + self.shares * price
        return {
            "date": str(pd.to_datetime(self.dates[i]).to_pydatetime()),
            "cash": float(self.cash),
            "shares": int(self.shares),
            "sellable": int(self._sellable()),
            "price_close": price,
            "equity": float(equity),
        }


def make_env(df: pd.DataFrame, env_kwargs: dict):
    """构造自定义 A 股交易环境(T+1/整手)。"""
    return AStockT1Env(df=df, **env_kwargs)


def train_best(df_train: pd.DataFrame, df_eval: pd.DataFrame, env_kwargs: dict, device: str, total_timesteps: int, eval_freq: int, out_dir: Path, last_model: Path):
    """
    训练 PPO, 并在训练过程中用 EvalCallback: 
    - 定期在 eval_env 上评估(deterministic=True)
    - 自动把评估最好的模型保存到 out_dir/best_model.zip

    device: 
    - "auto"/"cpu"/"cuda"/"cuda:0" 等, 直接透传给 SB3/PyTorch
    """
    # 用 Monitor 包装, 避免 SB3 在评估时提示 “Evaluation environment is not wrapped with Monitor”
    train_env = DummyVecEnv([lambda: Monitor(make_env(df_train, env_kwargs))])
    eval_env = DummyVecEnv([lambda: Monitor(make_env(df_eval, env_kwargs))])
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log="./tensorboard/",
        policy_kwargs={"net_arch": [128, 128]},
        device=device,
    )
    cb = EvalCallback(
        eval_env,
        best_model_save_path=str(out_dir),
        log_path=str(out_dir / "eval_logs"),
        eval_freq=max(1, int(eval_freq)),
        deterministic=True,
        render=False,
    )
    model.learn(total_timesteps=total_timesteps, callback=cb)
    model.save(str(last_model))
    return model


def backtest(model: PPO, df_test: pd.DataFrame, env_kwargs: dict):
    """在测试集上跑完一遍, 返回账户净值序列和动作序列。"""
    env = make_env(df_test, env_kwargs)
    obs, _ = env.reset()
    account = []
    actions = []
    dates = []
    for _ in range(len(df_test) - 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        # 兼容不同 info 字段命名: 优先 equity(当前环境), 否则 fallback 到 total_asset
        if "equity" in info:
            account.append(float(info["equity"]))
        else:
            account.append(float(info["total_asset"]))
        actions.append(float(action[0]))
        dates.append(info["date"])
        if terminated or truncated:
            break
    df_account = pd.DataFrame({"date": dates, "account_value": account})
    df_actions = pd.DataFrame({"date": dates, "action": actions})
    return df_account, df_actions


def infer_last(model: PPO, df: pd.DataFrame, env_kwargs: dict) -> dict:
    """
    推理: 用模型在给定 df 上从头跑到尾, 输出“最后一步”的动作。

    返回: 
    - raw_action: PPO 输出的原始动作(通常在 [-1, 1])
    - shares_action: 对应 env 内部交易股数 int(action*hmax)
    - last_date: 最后一步对应的日期

    说明: 这只是“离线推理”, 实盘需要你把最新行情不断 append 进 df 并维护账户状态。
    """
    env = make_env(df, env_kwargs)
    obs, _ = env.reset()
    last_action = None
    last_info = {}
    for _ in range(len(df) - 1):
        action, _ = model.predict(obs, deterministic=True)
        last_action = action
        obs, reward, terminated, truncated, info = env.step(action)
        last_info = info
        if terminated or truncated:
            break

    a = float(last_action[0]) # type: ignore
    raw_shares = int(a * env_kwargs["max_trade"])
    shares = (abs(raw_shares) // env_kwargs["lot_size"]) * env_kwargs["lot_size"]
    shares = shares if raw_shares >= 0 else -shares
    return {"raw_action": a, "shares_action": shares, "last_date": last_info.get("date")}


# ==============================
# 主流程(CLI entrypoint)
# ==============================
def main():
    """
    CLI 入口。

    mode:
    - train_test:训练(含 eval 保存 best)+ 用 best(若存在) 在 2025 回测
    - test_only: 只加载模型并回测 2025(优先 best_model.zip)
    - infer: 只加载模型并输出最后一步动作(方便你接实盘做“下一步下单”)
    """
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_csv",
        type=str,
        default="sh.601328.csv",
        help="本地分钟数据 CSV 路径, 例如 sh.601328.csv",
    )
    p.add_argument(
        "--train_end",
        type=str,
        default="2024-12-31",
        help="训练集结束日期(含)",
    )
    p.add_argument(
        "--test_start",
        type=str,
        default="2025-01-01",
        help="测试集开始日期(含)",
    )
    p.add_argument(
        "--test_end",
        type=str,
        default="2025-12-31",
        help="测试集结束日期(含)",
    )
    p.add_argument(
        "--total_timesteps",
        type=int,
        default=1_000_000,
        help="PPO 总训练步数",
    )
    p.add_argument(
        "--tic",
        type=str,
        default="",
        help="标的代码(默认从 CSV 文件名自动推断)",
    )
    p.add_argument(
        "--mode",
        type=str,
        default="train_test",
        choices=["train_test", "test_only", "infer"],
        help="train_test=训练+测试回测: test_only=只加载模型做测试回测: infer=推理输出最后一步动作",
    )
    p.add_argument(
        "--model_path",
        type=str,
        default="",
        help="test_only / infer 时使用的模型路径(默认优先 trained_model/best_model.zip)",
    )
    p.add_argument(
        "--eval_ratio",
        type=float,
        default=0.1,
        help="从训练集末尾切多少比例做验证集(用于保存 best model)",
    )
    p.add_argument(
        "--eval_freq",
        type=int,
        default=20_000,
        help="训练过程中每隔多少 step 做一次验证评估(影响 best_model 保存)",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="训练/推理设备: auto/cpu/cuda/cuda:0 ...(Stable-Baselines3/PyTorch)",
    )
    p.add_argument(
        "--max_trade",
        type=int,
        default=10_000,
        help="每步最大买/卖股数(动作[-1,1]会映射到 [-max_trade, max_trade])",
    )
    p.add_argument(
        "--lot_size",
        type=int,
        default=100,
        help="A股整手股数(默认100)",
    )
    p.add_argument(
        "--t_plus_one",
        type=int,
        default=1,
        choices=[0, 1],
        help="是否启用T+1(1=启用, 0=禁用)",
    )
    p.add_argument(
        "--fee_rate",
        type=float,
        default=0.0003,
        help="手续费比例(默认万分之三=0.0003, 买入/卖出都按成交金额计)",
    )
    p.add_argument(
        "--min_fee",
        type=float,
        default=5.0,
        help="最低手续费(元,默认5)",
    )
    args = p.parse_args()

    print("[INFO] 加载本地分钟数据:", args.data_csv)
    df = load_ohlcv(args.data_csv)
    print("[INFO] 数据行数:", len(df))

    tic = args.tic.strip() or Path(args.data_csv).stem
    df = add_indicators(df, INDICATORS)
    df = prepare_feature_df(df, tic)

    print("[INFO] 划分训练/测试集 ...")
    train_start = df["date"].min().strftime("%Y-%m-%d")
    df_train_full = split_range(df, train_start, args.train_end)
    df_test = split_range(df, args.test_start, args.test_end)
    print("[DATA] 训练全集:", df_train_full.shape, "测试集(2025):", df_test.shape)

    env_kwargs = {
        "indicators": INDICATORS,
        "window_size": 480,
        "initial_cash": 1_000_000.0,
        "fee_rate": args.fee_rate,
        "min_fee": args.min_fee,
        "slippage_pct": 0.0,
        "max_trade": args.max_trade,
        "lot_size": args.lot_size,
        "t_plus_one": bool(args.t_plus_one),
        "reward_scaling": 1.0,
    }

    out_dir, best_model_path, last_model_path = make_paths(args.data_csv, out_dir="trained_model")

    # ========= infer =========
    if args.mode == "infer":
        model_path = str(best_model_path) if best_model_path.exists() else (args.model_path.strip() or str(last_model_path))
        model = PPO.load(model_path, device=args.device)
        df_infer = df_test if len(df_test) > 0 else df_train_full
        result = infer_last(model, df_infer, env_kwargs)
        print("[INFER] model:", model_path)
        print("[INFER] device:", args.device)
        print("[INFER] last_date:", result["last_date"])
        print("[INFER] raw_action:", result["raw_action"])
        print("[INFER] shares_action (action*hmax,int):", result["shares_action"])
        return

    # ========= test_only =========
    if args.mode == "test_only":
        model_path = str(best_model_path) if best_model_path.exists() else (args.model_path.strip() or str(last_model_path))
        if not Path(model_path).exists():
            raise FileNotFoundError(f"未找到可用模型: {model_path}")
        model = PPO.load(model_path, device=args.device)
        print("[LOAD] model:", model_path)
        print("[LOAD] device:", args.device)
        print("[TEST] 测试回测 ...")
        df_account, df_actions = backtest(model, df_test, env_kwargs)
        df_account["date"] = pd.to_datetime(df_account["date"])
        df_account["daily_return"] = (
            df_account["account_value"].pct_change().fillna(0)
        )
        from pyfolio import timeseries

        perf = timeseries.perf_stats(
            returns=df_account.set_index("date")["daily_return"]
        )
        print("\n====== 测试集绩效 ======")
        print(perf)
        df_account.to_csv("account_value_test.csv", index=False)
        df_actions.to_csv("actions_test.csv", index=False)
        print("账户净值记录:", "account_value_test.csv")
        print("交易动作记录:", "actions_test.csv")
        return

    # ========= train_test =========
    df_train, df_eval = split_train_eval(df_train_full, eval_ratio=args.eval_ratio)
    print("[DATA] 训练子集:", df_train.shape, "验证集:", df_eval.shape)

    print("[TRAIN] PPO 开始训练 ...")
    model = train_best(
        df_train,
        df_eval,
        env_kwargs,
        args.device,
        args.total_timesteps,
        args.eval_freq,
        out_dir,
        last_model_path,
    )
    print("[OK] 最终模型已保存:", last_model_path)
    if best_model_path.exists():
        print("[OK] 最佳模型已保存:", best_model_path)

    print("[TEST] 测试回测 ...")
    model_to_test = (
        PPO.load(str(best_model_path), device=args.device) if best_model_path.exists() else model
    )
    df_account, df_actions = backtest(model_to_test, df_test, env_kwargs)

    df_account["date"] = pd.to_datetime(df_account["date"])
    df_account["daily_return"] = (
        df_account["account_value"].pct_change().fillna(0)
    )
    from pyfolio import timeseries

    perf = timeseries.perf_stats(
        returns=df_account.set_index("date")["daily_return"]
    )

    print("\n====== 测试集绩效 ======")
    print(perf)

    df_account.to_csv("account_value_test.csv", index=False)
    df_actions.to_csv("actions_test.csv", index=False)
    print("账户净值记录:", "account_value_test.csv")
    print("交易动作记录:", "actions_test.csv")


if __name__ == "__main__":
    main()
