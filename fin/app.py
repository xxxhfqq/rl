# -*- coding: utf-8 -*-
"""
从本地 CSV 加载分钟数据 + 技术指标 + PPO 训练 + 2025 测试回测
"""

import sys
import argparse
import warnings

import pandas as pd
from pathlib import Path

from stockstats import StockDataFrame
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# 避免 FinRL alpaca_trade_api 导入错误
sys.modules["alpaca_trade_api"] = type(sys)("dummy_alpaca")
warnings.filterwarnings("ignore")

from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from pyfolio import timeseries


# ==============================
# 1) 从本地 CSV 加载分钟数据
# ==============================
def load_local_minute_csv(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    if "time" not in df.columns:
        raise KeyError("CSV 必须包含 time 列")

    df["time_str"] = df["time"].astype(str)

    # 解析时间
    try:
        df["datetime"] = pd.to_datetime(
            df["time_str"],
            format="%Y%m%d%H%M%S%f",
            errors="raise",
        )
    except Exception:
        if "date" in df.columns:
            df["datetime"] = pd.to_datetime(
                df["date"].astype(str) + " " + df["time_str"].str[-6:],
                format="%Y-%m-%d %H%M%S",
                errors="coerce",
            )
        else:
            df["datetime"] = pd.to_datetime(df["time_str"], errors="coerce")

    if df["datetime"].isna().all():
        raise RuntimeError("无法解析时间，请检查 CSV time/date 格式")

    need_cols = ["datetime", "open", "high", "low", "close", "volume"]
    for col in need_cols:
        if col not in df.columns:
            raise KeyError(f"CSV 缺少必要列 {col}")

    df = df.sort_values("datetime").reset_index(drop=True)
    return df[need_cols]


# ==============================
# 2) 技术指标生成
# ==============================
def add_technical_indicators(df: pd.DataFrame, indicators: list[str]) -> pd.DataFrame:
    if "datetime" not in df.columns:
        raise KeyError("add_technical_indicators 需要 datetime 列")

    _orig_date = pd.to_datetime(df["datetime"]).reset_index(drop=True)

    df_tmp = df.copy()
    df_tmp.columns = [c.lower() for c in df_tmp.columns]
    ss = StockDataFrame.retype(df_tmp)
    for ind in indicators:
        _ = ss[ind]
    out = pd.DataFrame(ss).reset_index()

    if "datetime" not in out.columns and "index" in out.columns:
        out = out.rename(columns={"index": "datetime"})
    if "datetime" not in out.columns:
        out["datetime"] = _orig_date

    out["datetime"] = pd.to_datetime(out["datetime"])
    return out.reset_index(drop=True)


# ==============================
# 3) 对齐 FinRL StockTradingEnv 所需格式
# ==============================
def align_finrl_stocktrading_df(df: pd.DataFrame, tic: str) -> pd.DataFrame:
    """
    FinRL StockTradingEnv 关键要求：
    - 必须有 `date` 列（env 内部读取 self.data.date）
    - 必须有 `close` 列（env 内部读取 self.data.close）
    - index 必须是从 0..N-1 的 “day” 整数索引（env 内部 df.loc[self.day, :]）
    - 单股票：每个 day 只有一行；多股票：每个 day 有多个 tic 行（同一 index 重复）
    """
    if "datetime" not in df.columns and "date" not in df.columns:
        raise KeyError("数据必须包含 datetime 或 date 列以对齐 FinRL")

    out = df.copy()
    if "date" not in out.columns:
        out = out.rename(columns={"datetime": "date"})
    out["date"] = pd.to_datetime(out["date"])

    if "tic" not in out.columns:
        out["tic"] = tic
    else:
        out["tic"] = out["tic"].astype(str)

    # FinRL env 依赖 close / date / tech_indicator_list
    if "close" not in out.columns:
        raise KeyError("对齐 FinRL 失败：缺少 close 列")

    out = out.sort_values(["date", "tic"]).reset_index(drop=True)

    # 单股票：index 需要是 0..N-1
    # 多股票：同一 date 的多行必须拥有相同 day index
    out.index = pd.Index(pd.Series(out["date"]).factorize()[0])
    return out


# ==============================
# 4) 按时间切片
# ==============================
def data_split(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    s = pd.to_datetime(start_date)
    e = pd.to_datetime(end_date)
    if "date" not in df.columns:
        raise KeyError("data_split 需要 date 列")
    mask = (df["date"] >= s) & (df["date"] <= e)
    df2 = df.loc[mask].copy().reset_index(drop=True)
    df2.index = pd.Index(pd.Series(df2["date"]).factorize()[0])
    return df2


# ==============================
# 5) 检查 NaN
# ==============================
def check_for_nan(df, fillna=False):
    if fillna:
        df = df.fillna(0)
        print("[INFO] 填充 NaN 值为 0")
    else:
        if df.isnull().sum().sum() > 0:
            print("[INFO] 数据包含 NaN")
            print(df.isnull().sum())
            df = df.dropna()
            print("[INFO] 删除 NaN 行")
    return df


# ==============================
# 6) 主流程
# ==============================
def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_csv",
        type=str,
        default="601328_5min_last5years.csv",
        help="本地分钟数据 CSV 路径，例如 601328_5min_last5years.csv",
    )
    p.add_argument(
        "--train_end",
        type=str,
        default="2024-12-31",
        help="训练集结束日期（含）",
    )
    p.add_argument(
        "--test_start",
        type=str,
        default="2025-01-01",
        help="测试集开始日期（含）",
    )
    p.add_argument(
        "--test_end",
        type=str,
        default="2025-12-31",
        help="测试集结束日期（含）",
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
        help="标的代码（默认从 CSV 文件名自动推断）",
    )
    args = p.parse_args()

    print("[INFO] 加载本地分钟数据:", args.data_csv)
    raw_df = load_local_minute_csv(args.data_csv)
    print("[INFO] 数据行数:", len(raw_df))

    indicators = [
        "macd",
        "rsi_30",
        "cci_30",
        "dx_30",
        "close_30_sma",
        "close_60_sma",
    ]

    df_ind = add_technical_indicators(raw_df, indicators)
    tic = args.tic.strip() or Path(args.data_csv).stem
    df_ind["tic"] = tic

    df_env = align_finrl_stocktrading_df(df_ind, tic=tic)
    df_env = check_for_nan(df_env, fillna=True)

    print("[INFO] 划分训练/测试集 ...")
    train_start = df_env["date"].min().strftime("%Y-%m-%d")
    df_train = data_split(df_env, train_start, args.train_end)
    df_test = data_split(df_env, args.test_start, args.test_end)
    print("[DATA] 训练集:", df_train.shape, "测试集:", df_test.shape)

    stock_dim = 1
    state_space = stock_dim * (len(indicators) + 2) + 1

    env_kwargs = {
        "stock_dim": stock_dim,
        "hmax": 100,
        "initial_amount": 1_000_000,
        "num_stock_shares": [0] * stock_dim,
        "buy_cost_pct": [0.001] * stock_dim,
        "sell_cost_pct": [0.001] * stock_dim,
        "state_space": state_space,
        "action_space": stock_dim,
        "tech_indicator_list": indicators,
        "reward_scaling": 1e-4,
    }

    train_env = DummyVecEnv([lambda: StockTradingEnv(df=df_train, **env_kwargs)])

    print("[TRAIN] PPO 开始训练 ...")
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log="./tensorboard/",
        policy_kwargs={"net_arch": [128, 128]},
    )
    model.learn(total_timesteps=args.total_timesteps)

    out_dir = Path("trained_model")
    out_dir.mkdir(exist_ok=True)
    save_path = out_dir / f"ppo_{Path(args.data_csv).stem}.zip"
    model.save(str(save_path))
    print("[OK] 模型已保存:", save_path)

    print("[TEST] 测试回测 ...")
    e_trade_env = StockTradingEnv(df=df_test, **env_kwargs)
    df_account, df_actions = DRLAgent.DRL_prediction(
        model=model, environment=e_trade_env
    )

    df_account["date"] = pd.to_datetime(df_account["date"])
    df_account["daily_return"] = (
        df_account["account_value"].pct_change().fillna(0)
    )
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
