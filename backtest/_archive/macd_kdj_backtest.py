# -*- coding: utf-8 -*-
"""
macd_kdj_backtest —— 已退休策略归档（请勿复活 / DO NOT RESURRECT）

═══════════════════════════════════════════════════════════════════════
退休原因（与 backtest/_archive/macd_kdj_plugin.py 同源陷阱）
───────────────────────────────────────────────────────────────────────
本函数把「MACD 金叉」「KDJ 金叉(J<30 且 K 上穿 D)」「J>80」当作**买卖
触发信号**。这正是「跟着Jim学量化」第 4 期《KDJ三条线到底在算什么》批评
的误区：

  • 金叉只是「已经发生的价格变化进入公式」，不是对未来结果的提前确认；
  • J 超过 100 或低于 0 只代表 K 与 D 的差被公式放大，≠ 价格突破任何百分比；
  • KDJ 周期一改，金叉出现的时间就变 → 信号对参数高度敏感；
  • 两条禁令：不要只挑贴合的图、不要看完结果再改参数（即样本偏差 / 过拟合）。

平台更早的 MACD 专题审计（plan_macd_divergence_study.md，2026-07-31）已
实证：金叉死叉 / 背离在 A 股**无正增量 edge**。合规替代 = run_macd_regime.py
（趋势门控 + 底背离 + 突破 + 非中枢，金叉绝不单独当按钮）。

处置方式：与 macd_kdj_plugin.py 一致 → 移入 _archive，从 run_backtest.py
策略注册表 + config.py 中删除，用户菜单不再可选。
═══════════════════════════════════════════════════════════════════════
"""

import pandas as pd


# ── 依赖的自包含副本（原定义于 run_backtest.py，避免归档件反向依赖活模块）──
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def max_drawdown_with_dates(values, dates=None, start_idx=0):
    """计算最大回撤，并返回发生区间（峰值日 → 谷值日）。"""
    if not values or len(values) < 2:
        return 0.0, None, None
    if start_idx < 0 or start_idx >= len(values) - 1:
        start_idx = 0
    peak = values[start_idx]
    peak_idx = start_idx
    max_dd = 0.0
    trough_idx = start_idx
    peak_idx_at_trough = start_idx
    for i in range(start_idx, len(values)):
        val = values[i]
        if val > peak:
            peak = val
            peak_idx = i
        dd = (peak - val) / peak * 100
        if dd > max_dd:
            max_dd = dd
            trough_idx = i
            peak_idx_at_trough = peak_idx
    peak_date = dates[peak_idx_at_trough] if dates else None
    trough_date = dates[trough_idx] if dates else None
    return max_dd, peak_date, trough_date


def backtest_macd_kdj(df, capital, cfg, start_idx=0):
    """MACD金叉+KDJ超卖买入"""
    fast, slow, sig = cfg.get("fast", 12), cfg.get("slow", 26), cfg.get("signal", 9)
    kp = cfg.get("kdj_period", 9)
    tp, sl = cfg.get("take_profit", 0.50), cfg.get("stop_loss", 0.15)
    close = df["close"].values
    s = df["close"]
    n = len(close)
    if n < 35:
        return None, 0, 0.0

    dif = ema(s, fast) - ema(s, slow)
    dea = ema(dif, sig)
    hist = (dif - dea).values

    low_n, high_n = s.rolling(kp).min(), s.rolling(kp).max()
    rsv = (s - low_n) / (high_n - low_n + 1e-10) * 100
    k = ema(rsv, 3)
    d = ema(k, 3)
    j_vals = (3*k - 2*d).values
    k_vals, d_vals = k.values, d.values

    cash, pos, cost = capital, 0, 0.0
    trade_records = []
    portfolio_values = []
    dates = []

    for i in range(n):
        if i < start_idx:
            portfolio_values.append(capital)
            dates.append(df.iloc[i]["trade_date"])
            continue
        p = close[i]
        if pos == 0 and i >= 34:
            macd_gold = (hist[i-1] <= 0 < hist[i]) or (dif.iloc[i] > dea.iloc[i] and dif.iloc[i-1] <= dea.iloc[i-1])
            kdj_low = j_vals[i] < 30 and k_vals[i] > d_vals[i]
            if macd_gold or kdj_low:
                amt = cash * 0.5
                pos = int(amt / p / 100) * 100
                if pos > 0:
                    cash -= pos * p * 1.0002
                    cost = p
                    trade_records.append({"action": "BUY", "price": p, "shares": pos})
        elif pos > 0 and i >= 34:
            macd_dead = (hist[i-1] >= 0 > hist[i]) or (dif.iloc[i] < dea.iloc[i] and dif.iloc[i-1] >= dea.iloc[i-1])
            kdj_high = j_vals[i] > 80
            if macd_dead or kdj_high or p > cost * (1+tp) or p < cost * (1-sl):
                cash += pos * p * 0.9988
                trade_records.append({"action": "SELL", "price": p, "shares": pos})
                pos, cost = 0, 0.0

        portfolio_values.append(cash + pos * p)
        dates.append(df.iloc[i]["trade_date"])

    final = portfolio_values[-1] if portfolio_values else capital
    ret = (final / capital - 1) * 100
    max_dd, pk, tr = max_drawdown_with_dates(portfolio_values, dates, start_idx)

    return ret, trade_records, max_dd, (pk, tr)
