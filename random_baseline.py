"""随机选股基线：复用 run_weekly_highdiv_vol 的全部引擎，仅把"四因子选股"
替换为"从同一候选池(幸存者+流动性过滤)纯随机挑 top_n"。
目的：隔离 选股alpha vs 股票池/时期 的贡献。
"""
import sqlite3, random, sys, os
from datetime import datetime
import pandas as pd
import numpy as np
import run_weekly_highdiv_vol as M

# 确保 IPO_MIN_DAYS 存在
IPO_MIN_DAYS = getattr(M, "IPO_MIN_DAYS", 60)


def random_selection(rebalance_date, top_n=10, div_pct=25, turn_pct=85,
                     debt_pct=55, size_pct=50, verbose=False, factor_lag=0,
                     _seed=0):
    """与 select_highdiv_vol_stocks 同候选池，但纯随机挑 top_n。"""
    basic = M._load_stock_basic()
    conn = M.get_conn()
    rows = pd.read_sql_query(
        "SELECT DISTINCT ts_code FROM daily WHERE trade_date = ?", conn,
        params=(rebalance_date,))
    trading = set(rows["ts_code"].tolist())
    eligible = set()
    for c in trading:
        info = basic.get(c)
        if info is None:
            continue
        if info["excluded"]:
            continue
        ld = info["list_date"]
        if ld and rebalance_date < ld:
            continue
        if ld:
            try:
                d_ld = datetime.strptime(ld, "%Y%m%d")
                d_rb = datetime.strptime(rebalance_date, "%Y%m%d")
                if (d_rb - d_ld).days < IPO_MIN_DAYS:
                    continue
            except Exception:
                pass
        eligible.add(c)
    kept = M.prefilter_by_liquidity(conn, eligible, rebalance_date)
    conn.close()
    if len(kept) == 0:
        return {}
    if len(kept) <= top_n:
        sel = list(kept)
    else:
        rng = random.Random(f"rb-{rebalance_date}-{_seed}")
        sel = rng.sample(list(kept), top_n)
    return {c: {"name": M.get_stock_name(c)} for c in sel}


def run_random(seed, top_n=10):
    # monkeypatch
    def _patched(rb, top_n=top_n, div_pct=25, turn_pct=85, debt_pct=55,
                 size_pct=50, verbose=False, factor_lag=0):
        return random_selection(rb, top_n, div_pct, turn_pct, debt_pct,
                                size_pct, verbose, factor_lag, _seed=seed)
    M.select_highdiv_vol_stocks = _patched
    metrics = M.run_weekly_backtest(
        "20210104", "20260710", top_n=top_n,
        rbuy_blacklist=0, factor_lag=0, zero_fee=False)
    return metrics


if __name__ == "__main__":
    print("========== 随机选股基线 (同一幸存者池+流动性过滤, 周度调仓, top_n=10) ==========")
    print("对比目标: 四因子版 lag0 = 总+859% / 年化+56% / 回撤-6% / 夏普3.22 / 胜率74%\n")
    for seed in [1, 2, 3]:
        m = run_random(seed)
        print(f"[seed={seed}] 总收益={m['total_return']:.1f}%  年化={m['annual_return']:.1f}%  "
              f"最大回撤={m['max_drawdown']:.1f}%  夏普={m['sharpe']:.2f}  胜率={m['win_rate']:.1f}%  "
              f"交易={m['trades']}")
