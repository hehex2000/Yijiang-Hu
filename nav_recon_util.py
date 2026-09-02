# -*- coding: utf-8 -*-
"""从 trades CSV 重建日频 NAV 的共享工具（供 ①限价vs市价 / ②活跃税 两个工具复用）。

核心思路：
- 平台回测默认把每笔成交当「市价单」处理（穿越价差 + 平方根冲击），对左侧/逆向/
  价值类策略系统性高估成本——它们本应挂限价单（流动性提供者），几乎不吃冲击、甚至
  赚价差。本工具用同一份 trades 流水，分别在两种成交假设下重建组合净值，隔离「成交
  成本假设」这一单变量。
- 两模型强制使用同一初始资金（market 模型估算），使收益差纯粹来自成本差。
- 复用 run_monthly_rebalance 真实成本函数（佣金/印花税/滑点），不另立炉灶。

标记模式：
  slippage_frac_market(action, amount, ts_code, trade_date)
      -> 平台默认 flat 0.1%；若导入时已设 MFS_SQRT_IMPACT=1，则走平方根冲击。
  slippage_frac_limit(action, amount, ts_code, trade_date, maker_slip=0.0)
      -> 被动限价：不吃 taker 冲击；maker_slip 默认 0（保守：仅消除 taker 成本），
         也可传负值（如 -0.0005）表示吃到 half-spread 的反向收益。
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_monthly_rebalance import (
    get_conn, COMMISSION_RATE, COMMISSION_MIN, SLIPPAGE_RATE,
    stamp_duty_rate, sqrt_impact_slippage, USE_SQRT_IMPACT,
)


def _norm_action(a):
    a = str(a).strip().upper()
    if a in ("BUY", "B", "OPEN"):
        return "BUY"
    if a in ("SELL", "S", "CLOSE"):
        return "SELL"
    return a


def slippage_frac_market(action, amount, ts_code, trade_date, **_):
    """市价/taker 滑点占比：默认 flat，开启 sqrt 冲击则按流动性感知。"""
    if USE_SQRT_IMPACT and ts_code:
        frac, _, _ = sqrt_impact_slippage(amount, ts_code, trade_date)
        if frac is not None:
            return frac
    return SLIPPAGE_RATE


def slippage_frac_limit(action, amount, ts_code, trade_date, maker_slip=0.0, **_):
    """限价/maker 滑点占比：被动成交不吃 taker 冲击。"""
    return maker_slip


def cost_of(action, price, shares, trade_date, ts_code, slip_func, **kw):
    """单笔交易成本（佣金 + 滑点 + 印花税[仅卖出]）。"""
    amt = price * shares
    commission = max(amt * COMMISSION_RATE, COMMISSION_MIN)
    slip = amt * slip_func(action, amt, ts_code, trade_date, **kw)
    duty = amt * stamp_duty_rate(trade_date) if action == "SELL" else 0.0
    return commission + slip + duty


def fetch_close(code, start, end):
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT trade_date, close FROM daily WHERE ts_code=? "
        "AND trade_date>=? AND trade_date<=? ORDER BY trade_date",
        conn, params=(code, int(start), int(end)))
    conn.close()
    if len(df) == 0:
        return None
    df["trade_date"] = df["trade_date"].astype(int)
    return df.set_index("trade_date")["close"]


def fetch_closes_bulk(codes, start, end):
    """单次查询取回多标的日线收盘，返回 {ts_code: Series(trade_date->close)}。

    避免对大 Universe（上千只）逐标的开连接导致卡死。
    """
    if not codes:
        return {}
    conn = get_conn()
    try:
        q = ("SELECT ts_code, trade_date, close FROM daily "
             "WHERE ts_code IN ({}) AND trade_date>=? AND trade_date<=? "
             "ORDER BY ts_code, trade_date").format(
            ",".join("?" for _ in codes))
        df = pd.read_sql_query(q, conn, params=list(codes) + [int(start), int(end)])
    finally:
        conn.close()
    out = {}
    if len(df) == 0:
        return out
    df["trade_date"] = df["trade_date"].astype(int)
    for c, g in df.groupby("ts_code"):
        out[c] = g.set_index("trade_date")["close"]
    return out


def compute_init_cap(trades):
    """用 flat 0.1% 滑点估算所需初始资金（保证现金流非负，确定性）。"""
    cash = 0.0
    min_cash = 0.0
    for _, r in trades.iterrows():
        a = _norm_action(r["action"])
        amt = r["price"] * r["shares"]
        c = cost_of(a, r["price"], r["shares"], int(r["date"]), r["code"],
                    slippage_frac_market)
        if a == "BUY":
            cash -= (amt + c)
        else:
            cash += (amt - c)
        min_cash = min(min_cash, cash)
    return -min_cash if min_cash < 0 else 0.0


def reconstruct(trades, slip_func, init_cap=None, maker_slip=0.0, **kw):
    """重建日频 NAV 与绩效指标。

    返回 dict：nav(Series), total_return, annualized, max_dd, mean_nav,
               total_cost, years, active_tax_yr, n_trades
    slip_func 可加 maker_slip kw（limit 模型）。
    """
    trades = trades.copy()
    trades["date"] = trades["date"].astype(int)
    trades = trades.sort_values("date").reset_index(drop=True)
    d0 = int(trades["date"].min())
    d1 = int(trades["date"].max())

    if init_cap is None:
        init_cap = compute_init_cap(trades)

    # 预取所有标的日线收盘（复权口径用 raw close，两模型一致即可）
    codes = trades["code"].unique().tolist()
    closes = {}
    for c in codes:
        s = fetch_close(c, d0, d1)
        if s is not None:
            closes[c] = s
    if not closes:
        return None

    all_dates = sorted(set().union(*[set(s.index) for s in closes.values()]))
    # 逐标的对齐并 ffill（前向）+ bfill（补前导缺口，避免首期持仓无价→NaN 净值）
    close_mat = pd.DataFrame(
        {c: s.reindex(all_dates).ffill().bfill() for c, s in closes.items()}
    ).dropna(axis=1, how="all")

    # 按日推进：先处理当日成交，再按当日收盘价估值
    trade_by_date = {d: g for d, g in trades.groupby("date")}
    cash = init_cap
    hold = {}
    total_cost = 0.0
    nav_vals = []
    for dt in all_dates:
        if dt in trade_by_date:
            for _, r in trade_by_date[dt].iterrows():
                a = _norm_action(r["action"])
                amt = r["price"] * r["shares"]
                c = cost_of(a, r["price"], r["shares"], int(dt), r["code"],
                            slip_func, maker_slip=maker_slip, **kw)
                total_cost += c
                qty = abs(r["shares"])
                if a == "BUY":
                    cash -= (amt + c)
                    hold[r["code"]] = hold.get(r["code"], 0) + qty
                else:
                    cash += (amt - c)
                    hold[r["code"]] = hold.get(r["code"], 0) - qty
        row = close_mat.loc[dt]
        nav = cash + sum(hold.get(c, 0) * row[c] for c in hold)
        nav_vals.append(nav)

    nav = pd.Series(nav_vals, index=all_dates, dtype=float)
    first, last = nav.iloc[0], nav.iloc[-1]
    # all_dates 为 YYYYMMDD 整数，须转真实日期再算年数（否则会当天数差算出上百"年"）
    _dt = pd.to_datetime([str(int(d)) for d in all_dates], format="%Y%m%d")
    years = (_dt[-1] - _dt[0]).days / 365.25
    tot = last / first - 1 if first else 0.0
    ann = (last / first) ** (1 / years) - 1 if (first and years > 0) else 0.0
    peak = nav.cummax()
    mdd = (nav / peak - 1).min()
    mean_nav = nav.mean()

    # 总成交金额（单边）
    total_traded = float(trades["price"].astype(float)
                         .mul(trades["shares"].astype(float)).abs().sum())
    # 活跃税（年化换手成本率）：分母用 init_cap 作股本代理，口径无关、不被估值价扭曲
    active_tax_yr = (total_cost / (init_cap * years)
                     if (init_cap and years > 0) else 0.0)
    # 单边换手摩擦成本率（口径无关，跨策略可比）：总成本 / 总成交金额
    round_trip_cost = (total_cost / total_traded) if total_traded > 0 else 0.0

    return dict(
        nav=nav, total_return=tot, annualized=ann, max_dd=mdd,
        mean_nav=mean_nav, total_cost=total_cost, total_traded=total_traded,
        years=years, active_tax_yr=active_tax_yr,
        round_trip_cost=round_trip_cost, n_trades=len(trades),
        init_cap=init_cap,
    )


def load_trades(path):
    df = pd.read_csv(path)
    df = df.rename(columns={c: c.strip() for c in df.columns})
    need = ["date", "action", "code", "price", "shares"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"{path} 缺列: {miss}；现有列={list(df.columns)}")
    return df
