"""
diag_dual_ma_jim.py —— 双均线状态机增强版轻量冒烟测试（合成数据，无需 DB）

验证：
  1) 基础版(DualMAStrategyPlugin) 与 状态机版(DualMAJimPlugin) 都能跑通不崩溃
  2) 状态机版交易数 <= 基础版（信号漏斗效应：震荡市过滤掉交叉）
  3) 回报可计算、daily_values 正常

用法（Windows PowerShell）：
  cd C:/Users/99395/WorkBuddy/multi_factor_selection
  ./venv_ml/Scripts/python.exe diag_dual_ma_jim.py
"""
import numpy as np
import pandas as pd
from backtest.dual_ma_plugin import DualMAStrategyPlugin
from backtest.dual_ma_jim_plugin import DualMAJimPlugin

rng = np.random.default_rng(42)


def make_df(n=260):
    """构造含 上涨 → 震荡 → 下跌 → 上涨 四段的价格路径，演练 UP/UNCLEAR/DOWN 状态。"""
    dates = pd.date_range("2020-01-01", periods=n, freq="B").strftime("%Y%m%d")
    seg = n // 4
    price = np.empty(n)
    p = 10.0
    for i in range(n):
        if i < seg:            # 上涨段
            drift = 0.004
        elif i < 2 * seg:       # 震荡段（无方向）
            drift = 0.0
        elif i < 3 * seg:       # 下跌段
            drift = -0.004
        else:                   # 再上涨段
            drift = 0.004
        p = p * (1 + drift) * (1 + rng.normal(0, 0.012))
        price[i] = max(p, 1.0)
    price = price.astype(float)
    df = pd.DataFrame({
        "trade_date": dates,
        "open": price * (1 + rng.normal(0, 0.003, n)),
        "high": price * (1 + np.abs(rng.normal(0, 0.008, n))),
        "low": price * (1 - np.abs(rng.normal(0, 0.008, n))),
        "close": price,
        "volume": rng.integers(1e5, 1e6, n).astype(float),
    })
    # 复权价 = 收盘价（合成数据无分红，前复权等价）
    df["adj_open"] = df["open"]
    df["adj_close"] = df["close"]
    return df


def run_one(cls, cfg, df):
    strat = cls(100_000.0, cfg)
    res = strat.run(df)
    return res


def main():
    df = make_df()
    # 复制 dual_ma 配置（与 config.py 一致；state_machine 由各自类决定）
    base_cfg = {
        "ma_short": 10, "ma_long": 30, "position_pct": 0.5,
        "take_profit": 0.30, "stop_loss": 0.10, "enable_tp_sl": True,
        "atr_stop_loss": True, "atr_period": 14, "atr_mult": 3.0, "trail_mult": 3.0,
        "use_kelly": True, "kelly_win_rate": 0.40, "kelly_win_loss_ratio": 2.5,
        "kelly_fraction": 1.0, "kelly_max_position": 0.20, "kelly_min_position": 0.05,
        "kelly_safety_discount": 1.0, "state_machine": False,
        "slope_lookback": 10, "slope_thresh": 0.0,
    }
    jim_cfg = dict(base_cfg)
    jim_cfg["state_machine"] = True

    r_base = run_one(DualMAStrategyPlugin, base_cfg, df.copy())
    r_jim = run_one(DualMAJimPlugin, jim_cfg, df.copy())

    n_base = len(r_base["trades"])
    n_jim = len(r_jim["trades"])
    print(f"[基础版 dual_ma]    收益={r_base['returns']:+.2f}%  交易={n_base}笔  daily={len(r_base['daily_values'])}")
    print(f"[状态机 dual_ma_jim] 收益={r_jim['returns']:+.2f}%  交易={n_jim}笔  daily={len(r_jim['daily_values'])}")
    print(f"[漏斗检查] 状态机交易数({n_jim}) <= 基础版({n_base}) ? {'OK' if n_jim <= n_base else 'FAIL(应更少)'}")
    print(f"[健全检查] 两段都有 daily_values 且长度一致 ? "
          f"{'OK' if len(r_base['daily_values']) == len(r_jim['daily_values']) > 0 else 'FAIL'}")


if __name__ == "__main__":
    main()
