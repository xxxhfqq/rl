# -*- coding: utf-8 -*-
"""
FinRL + PPO（Proximal Policy Optimization，近端策略优化） + Tushare（TuShare）

目标：用 pip 安装版 FinRL，按“正确的 StockTradingEnv 参数格式”跑通闭环：
1) Tushare 拉取行情（pro.daily）
2) pandas 清洗/对齐交易日/缺失填充
3) stockstats 计算技术指标（用 FinRL 的 INDICATORS 列表）
4) 构建 StockTradingEnv（注意：必须按它的 __init__ 传参，不要传 initial_buy）
5) Stable-Baselines3 训练 PPO
6) 用 FinRL 的 DRLAgent.DRL_prediction 做测试
7) pyfolio 输出 perf_stats
8) 保存模型与 csv

运行（Windows PowerShell）：
  $env:TUSHARE_TOKEN="你的token"
  python app.py --total_timesteps 20000
"""

import os
import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# =============================
# 0) 股票列表：你后续只改这里=
# =============================
TICKER_LIST = [
    "601328.SH",  # 中国交通银行
    # "510300.SH",  # 沪深300ETF（可能权限不足）
]

# =============================
# 1) FinRL 模块
# =============================
from finrl import config
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent

# =============================
# 2) PPO：Stable-Baselines3
# =============================
from stable_baselines3 import PPO

# =============================
# 3) 技术指标：stockstats
# =============================
from stockstats import StockDataFrame

# =============================
# 4) 回测：pyfolio
# =============================
from pyfolio import timeseries


def check_and_make_directories(dirs):
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_start", type=str, default="2019-01-01")
    p.add_argument("--train_end", type=str, default="2020-12-31")
    p.add_argument("--trade_start", type=str, default="2021-01-01")
    p.add_argument("--trade_end", type=str, default="2021-12-31")
    p.add_argument("--total_timesteps", type=int, default=20_000)
    p.add_argument("--model_name", type=str, default="ppo_two_tickers")
    return p.parse_args()


def ymd_to_yyyymmdd(s):
    return s.replace("-", "")


def ensure_date_tic_columns(df):
    """
    把 Tushare 常见字段统一成 env 需要的字段：
    date（datetime）, tic（含 .SH/.SZ）, open/high/low/close/volume
    """
    x = df.copy()

    if "date" not in x.columns and "trade_date" in x.columns:
        x = x.rename(columns={"trade_date": "date"})
    if "tic" not in x.columns and "ts_code" in x.columns:
        x = x.rename(columns={"ts_code": "tic"})
    if "volume" not in x.columns and "vol" in x.columns:
        x = x.rename(columns={"vol": "volume"})

    need = ["date", "tic", "open", "high", "low", "close", "volume"]
    miss = [c for c in need if c not in x.columns]
    if miss:
        raise KeyError(f"数据缺少必要列：{miss}，当前列={list(x.columns)}")

    x["date"] = pd.to_datetime(x["date"], format="%Y%m%d", errors="coerce") if x["date"].dtype != "datetime64[ns]" else x["date"]
    return x


def download_market_data_tushare(token, ticker_list, start_date, end_date):
    """
    只用 pro.daily（最基础、最常见、最容易有权限的接口）。
    """
    import tushare as ts

    ts.set_token(token)
    pro = ts.pro_api()

    s = ymd_to_yyyymmdd(start_date)
    e = ymd_to_yyyymmdd(end_date)

    all_rows = []
    for ts_code in tqdm(ticker_list, desc="Tushare 拉取行情", total=len(ticker_list)):
        df = pro.daily(ts_code=ts_code, start_date=s, end_date=e)
        if df is None or len(df) == 0:
            raise RuntimeError(f"{ts_code} 没拉到数据（可能停牌/代码不对/无权限）")
        all_rows.append(df)

    raw = pd.concat(all_rows, ignore_index=True)
    raw = ensure_date_tic_columns(raw)

    raw = raw.sort_values(["date", "tic"]).reset_index(drop=True)
    return raw[["date", "tic", "open", "high", "low", "close", "volume"]]


def align_and_fill(df):
    """
    对齐交易日：让每个 date 每个 tic 都有一行，然后前向/后向填充。
    """
    df = ensure_date_tic_columns(df)
    df = df.sort_values(["date", "tic"]).reset_index(drop=True)

    all_dates = pd.Index(sorted(df["date"].unique()))
    out = []

    for tic, g in df.groupby("tic", sort=False):
        gg = g.set_index("date").reindex(all_dates)
        gg["tic"] = tic
        for col in ["open", "high", "low", "close", "volume"]:
            gg[col] = gg[col].ffill().bfill()
        gg = gg.reset_index().rename(columns={"index": "date"})
        out.append(gg)

    df2 = pd.concat(out, ignore_index=True).sort_values(["date", "tic"]).reset_index(drop=True)
    return df2


def add_technical_indicators(df, indicators):
    """
    用 stockstats 生成指标。
    stockstats 会把 date 当索引，所以最后要 reset_index() 把 date 还原为列。
    """
    df = ensure_date_tic_columns(df)
    out_list = []

    for tic, g in tqdm(df.groupby("tic", sort=False), desc="计算技术指标", total=df["tic"].nunique()):
        g = g.sort_values("date").copy()

        base = g[["date", "open", "high", "low", "close", "volume"]].copy()
        base.columns = [c.lower() for c in base.columns]
        ss = StockDataFrame.retype(base)

        for ind in indicators:
            _ = ss[ind]

        ss_df = pd.DataFrame(ss).reset_index()
        if "date" not in ss_df.columns:
            ss_df = ss_df.rename(columns={"index": "date"})
        ss_df["tic"] = tic

        out_list.append(ss_df)

    out = pd.concat(out_list, ignore_index=True).sort_values(["date", "tic"]).reset_index(drop=True)

    # 指标前期 NaN：填充
    out[indicators] = (
        out.groupby("tic")[indicators]
           .apply(lambda x: x.ffill().bfill())
           .reset_index(level=0, drop=True)
    )
    return out


def to_finrl_env_index(df):
    """
    StockTradingEnv 内部会用 self.df.loc[self.day, :] 取某一天的所有股票行，
    所以 index 必须是“天编号”，同一天共享同一个 index。
    """
    x = df.copy()
    x["date"] = pd.to_datetime(x["date"]).dt.strftime("%Y-%m-%d")
    x = x.sort_values(["date", "tic"]).reset_index(drop=True)
    x.index = pd.Index(pd.Series(x["date"]).factorize()[0])
    return x


def data_split(df, start_date, end_date):
    x = df.copy()
    dt = pd.to_datetime(x["date"])
    s = pd.to_datetime(start_date)
    e = pd.to_datetime(end_date)
    x = x[(dt >= s) & (dt <= e)].copy()
    x = x.sort_values(["date", "tic"]).reset_index(drop=True)
    x.index = pd.Index(pd.Series(x["date"]).factorize()[0])
    return x


def train_ppo(train_df, model_name, total_timesteps, indicators):
    """
    这里是你这次报错的关键修复点：
    按 StockTradingEnv.__init__ 的“正确签名”传参：
    - 必须提供 num_stock_shares（list，长度=stock_dim）
    - buy_cost_pct / sell_cost_pct 也应是 list（长度=stock_dim）
    - 不要传 initial_buy / hundred_each_trade（这个版本没有）
    """
    stock_dim = len(train_df.tic.unique())
    state_space = stock_dim * (len(indicators) + 2) + 1

    # ✅ 按正确签名构造 env_kwargs（这些名字来自 env_stocktrading.py 的 __init__）
    env_kwargs = {
        "stock_dim": stock_dim,
        "hmax": 1000,
        "initial_amount": 1_000_000,
        "num_stock_shares": [0] * stock_dim,
        "buy_cost_pct": [6.87e-5] * stock_dim,
        "sell_cost_pct": [1.0687e-3] * stock_dim,
        "reward_scaling": 1e-4,
        "state_space": state_space,
        "action_space": stock_dim,
        "tech_indicator_list": indicators,
        "print_verbosity": 1,
    }

    e_train_gym = StockTradingEnv(df=train_df, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()

    model = PPO(
        "MlpPolicy",
        env_train,
        verbose=1,
        n_steps=2048,
        batch_size=256,
        learning_rate=3e-4,
        ent_coef=0.01,
        clip_range=0.2,
        n_epochs=10,
        tensorboard_log=config.TENSORBOARD_LOG_DIR,
    )

    print("[训练] 开始 PPO 训练 ...")
    try:
        model.learn(total_timesteps=total_timesteps, progress_bar=True)
    except TypeError:
        model.learn(total_timesteps=total_timesteps)

    model_dir = Path(config.TRAINED_MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{model_name}.zip"
    model.save(str(model_path))

    return model, env_kwargs, str(model_path)


def trade_and_backtest(df_all, trained_model, env_kwargs, trade_start, trade_end):
    trade_df = data_split(df_all, trade_start, trade_end)
    e_trade_gym = StockTradingEnv(df=trade_df, **env_kwargs)

    df_account_value, df_actions = DRLAgent.DRL_prediction(
        model=trained_model,
        environment=e_trade_gym,
    )

    results_dir = Path(config.RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    account_csv = str(results_dir / "account_value.csv")
    actions_csv = str(results_dir / "actions.csv")
    df_account_value.to_csv(account_csv, index=False)
    df_actions.to_csv(actions_csv, index=False)

    df_account_value["date"] = pd.to_datetime(df_account_value["date"])
    df_account_value = df_account_value.sort_values("date")
    df_account_value["daily_return"] = df_account_value["account_value"].pct_change().fillna(0.0)
    daily_return = df_account_value.set_index("date")["daily_return"]

    perf_stats_strategy = timeseries.perf_stats(returns=daily_return)

    return account_csv, actions_csv, perf_stats_strategy


def main():
    args = parse_args()

    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("缺少 Tushare Token：请设置环境变量 TUSHARE_TOKEN")

    check_and_make_directories([
        config.DATA_SAVE_DIR,
        config.TRAINED_MODEL_DIR,
        config.TENSORBOARD_LOG_DIR,
        config.RESULTS_DIR,
    ])

    print(f"[股票列表] 当前 TICKER_LIST={TICKER_LIST}（你后续只改脚本顶部列表即可）")

    # 训练+测试总区间：train_start -> trade_end
    raw_df = download_market_data_tushare(
        token=token,
        ticker_list=TICKER_LIST,
        start_date=args.train_start,
        end_date=args.trade_end,
    )

    df_all = align_and_fill(raw_df)

    indicators = list(getattr(config, "INDICATORS", []))
    if not indicators:
        # 兜底（少数版本 config.INDICATORS 为空）
        indicators = ["macd", "rsi_30", "cci_30", "dx_30", "close_30_sma", "close_60_sma"]

    df_all = add_technical_indicators(df_all, indicators)
    df_all = to_finrl_env_index(df_all)

    train_df = data_split(df_all, args.train_start, args.train_end)
    print(f"[训练集] shape={train_df.shape}，股票数={len(train_df.tic.unique())}")

    trained_model, env_kwargs, model_path = train_ppo(
        train_df=train_df,
        model_name=args.model_name,
        total_timesteps=args.total_timesteps,
        indicators=indicators,
    )
    print(f"[模型] 已保存：{model_path}")

    account_csv, actions_csv, perf_strategy = trade_and_backtest(
        df_all=df_all,
        trained_model=trained_model,
        env_kwargs=env_kwargs,
        trade_start=args.trade_start,
        trade_end=args.trade_end,
    )

    print("\n==================== 回测结果（策略） ====================")
    print(perf_strategy)

    print("\n==================== 输出文件 ====================")
    print(f"账户净值: {account_csv}")
    print(f"交易动作: {actions_csv}")
    print(f"模型文件: {model_path}")


if __name__ == "__main__":
    main()
