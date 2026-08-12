# -*- coding: utf-8 -*-
"""
突破赢家选股 · 参数敏感性扫描 + 等权中证800基准
========================================================
目的：
  1) 给定「月调仓·等额纯持有·无加仓无止损」的固定持有机制，扫描突破赢家选股
     在 (L, VOL_MULT) × (突破回顾年 pre_years, 选股数 top_n) 网格上的表现，
     看 +76% 是否稳健、是否靠某一组参数撑着。
  2) 新增「等权中证800」月度再平衡基准（零成本·小数份额，反映被动指数真实收益），回答
     「突破赢家到底有没有 alpha」。

口径（与 A/B 对比一致）：
  股票池 zz800 时点成分股；区间 2014-01 ~ 2026-06；初始 100 万；
  次日开盘成交；费用=佣金0.025%/边(最低5元)+印花(2023-08-28起0.05%)+滑点0.1%/边。

容错：
  - base-feat（与 L 无关的滚动特征）缓存到 pickle，信号重建只跑 build_signal。
  - 每个 (L,VOL) 信号缓存到独立 pickle（复用上一轮 signal_cache.pkl 作 (60,1.5)）。
  - 结果增量写 RESULTS_CSV，重跑自动跳过已完成组合（防后台被杀丢结果）。
"""
import sqlite3, os, json, argparse
import numpy as np
import pandas as pd

from run_selection_ab_compare import (
    ord_of, select_breakout, build_price_lookup, asof_price,
    slot_net_ret, metrics, get_index_constituents, get_trade_dates,
    get_monthly_5th_trading_days, INIT, STOCK_POOL, START, END,
    SL, CR, stamp_rate,
)
from backtest_main_rise import load_data, add_base_features, build_signal
from run_monthly_rebalance import _apply_backadjust

RES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "data", "results", "selection_ab")
BASE_FEAT_CACHE = os.path.join(RES_DIR, "base_feat_cache_adj.pkl")
OLD_SIGNAL_CACHE = os.path.join(RES_DIR, "signal_cache_adj.pkl")
SIG_CACHE_TPL = os.path.join(RES_DIR, "signal_cache_L{L}_V{V}_adj.pkl")
MEM_CACHE = os.path.join(RES_DIR, "members_cache.json")
RESULT_CSV = os.path.join(RES_DIR, "results_sensitivity.csv")
BENCH_CSV = os.path.join(RES_DIR, "bench_equalweight_zz800.csv")

LOAD_START = "20080101"   # 信号历史需回溯到 2008，支撑 L=120 与 pre_years=5

# 参数网格
SIGNAL_GRID = [(60, 1.5), (40, 1.5), (90, 1.5), (60, 2.0)]
SEL_GRID = [(2, 10), (2, 15), (2, 20), (2, 30),
            (3, 10), (3, 15), (3, 20), (3, 30),
            (5, 10), (5, 15), (5, 20), (5, 30)]


def get_conn():
    return sqlite3.connect("D:/tu-shareData/astock_daily.db")


def load_base_feat():
    if os.path.exists(BASE_FEAT_CACHE):
        print(f"[load] base-feat 从缓存读取 {BASE_FEAT_CACHE}", flush=True)
        return pd.read_pickle(BASE_FEAT_CACHE)
    print(f"[load] 构建 base-feat（重活，一次性）：{LOAD_START}~{END} ...", flush=True)
    df, _ = load_data(LOAD_START, END)
    print(f"  日线行数={len(df):,}", flush=True)
    # 后复权：close/pre_close 改 close*adj_factor（与平台口径一致），避免除权除息日虚假突破
    df = _apply_backadjust(df, END)
    df = add_base_features(df)
    df.to_pickle(BASE_FEAT_CACHE)
    print(f"[load] base-feat 已缓存", flush=True)
    return df


def load_idx():
    conn = get_conn()
    rows = pd.read_sql_query(
        "SELECT trade_date, close FROM index_daily WHERE ts_code='000300.SH' "
        "AND trade_date BETWEEN ? AND ? ORDER BY trade_date",
        conn, params=(LOAD_START, END))
    conn.close()
    rows["trade_date"] = rows["trade_date"].astype(int)
    return rows.set_index("trade_date")["close"].astype("float32")


def get_signal(L, VOL):
    path = SIG_CACHE_TPL.format(L=L, V=VOL)
    if os.path.exists(path):
        print(f"[signal] 复用缓存 L={L} V={VOL}", flush=True)
        return pd.read_pickle(path)
    # 复用上一轮 (60,1.5) 的 signal_cache.pkl
    if (L, VOL) == (60, 1.5) and os.path.exists(OLD_SIGNAL_CACHE):
        print(f"[signal] 复用旧 signal_cache.pkl (60,1.5)", flush=True)
        return pd.read_pickle(OLD_SIGNAL_CACHE)
    print(f"[signal] 构建 L={L} V={VOL} ...", flush=True)
    df = load_base_feat()
    idx = load_idx()
    df_sig = build_signal(df, idx, L, VOL)
    df_pos = df_sig[df_sig["signal"] > 0][["ts_code", "trade_date", "signal"]].copy()
    df_pos.to_pickle(path)
    print(f"  突破信号(>0)行数={len(df_pos):,}", flush=True)
    return df_pos


def load_members(rebalances):
    if os.path.exists(MEM_CACHE):
        with open(MEM_CACHE) as f:
            return json.load(f)
    mem = []
    for (r, _, _) in rebalances:
        m = get_index_constituents(STOCK_POOL, trade_date=str(r))
        mem.append(list(m))
    with open(MEM_CACHE, "w") as f:
        json.dump(mem, f)
    return mem


def simulate_slots(rebalances, selections, px, top_n):
    """每槽资金 = INIT/top_n，保证满仓可比；月均收益 -> 净值。"""
    nav = INIT
    nav_series = [nav]
    monthly = []
    slot_notional = INIT / top_n
    for i, (r, buy_d, sell_d) in enumerate(rebalances):
        codes = selections[i]
        rets = []
        for code in codes:
            bp = asof_price(px, code, buy_d, "open") or asof_price(px, code, buy_d, "close")
            sp = asof_price(px, code, sell_d, "open") or asof_price(px, code, sell_d, "close")
            rr = slot_net_ret(bp, sp, buy_d, sell_d, notional=slot_notional)
            rets.append(rr if rr is not None else 0.0)
        pr = float(np.mean(rets)) if rets else 0.0
        nav *= (1.0 + pr)
        nav_series.append(nav)
        monthly.append(pr)
    return nav_series, monthly


def bench_simulate(rebalances, members_list, px):
    """等权中证800基准（诚实版）：每期对时点成分股等权、小数份额、零交易成本。
    目的=被动指数真实收益，用于回答'突破赢家是否有 alpha'。
    采用小数份额(无整数股现金拖累)、零成本(指数基金有效交易成本≈0)，
    主动策略带真实费用去比这个零成本被动基准，能赢才是真 alpha。"""
    nav = INIT
    nav_series = [nav]
    monthly = []
    for i, (r, buy_d, sell_d) in enumerate(rebalances):
        codes = members_list[i]
        rets = []
        for code in codes:
            bp = asof_price(px, code, buy_d, "open") or asof_price(px, code, buy_d, "close")
            sp = asof_price(px, code, sell_d, "open") or asof_price(px, code, sell_d, "close")
            if bp is not None and sp is not None and bp > 0 and sp > 0:
                rets.append(sp / bp - 1.0)        # 零成本等权指数单票月收益
        pr = float(np.mean(rets)) if rets else 0.0
        nav *= (1.0 + pr)
        nav_series.append(nav)
        monthly.append(pr)
    return nav_series, monthly


def main():
    os.makedirs(RES_DIR, exist_ok=True)
    # 1) 日程（确定性）
    td_all = get_trade_dates(START, END)
    r_dates = get_monthly_5th_trading_days(td_all)
    td_set = set(td_all)
    rebalances = []
    for r in r_dates:
        nxt = [d for d in td_all if d > r]
        buy_d = nxt[0] if nxt else None
        if buy_d is None:
            continue
        rebalances.append((r, buy_d, None))
    last_td = td_all[-1]
    for i in range(len(rebalances)):
        sell_d = rebalances[i + 1][1] if i + 1 < len(rebalances) else last_td
        rebalances[i] = (rebalances[i][0], rebalances[i][1], sell_d)
    print(f"[schedule] 调仓次数={len(rebalances)} 首={rebalances[0][0]} 末={rebalances[-1][0]}", flush=True)

    # 2) 时点成分股（供基准 + 价格查找）
    members_list = load_members(rebalances)

    # 3) 价格查找表：所有成分股并集
    all_codes = set()
    for m in members_list:
        all_codes.update(m)
    buy_dates = [b for (_, b, _) in rebalances]
    sell_dates = [s for (_, _, s) in rebalances]
    min_d, max_d = min(buy_dates + sell_dates), max(buy_dates + sell_dates)
    print(f"[price] 候选 {len(all_codes):,} 只, 区间 {min_d}~{max_d} ...", flush=True)
    px = build_price_lookup(list(all_codes), min_d, max_d)

    # 4) 等权中证800基准
    nav_b, m_b = bench_simulate(rebalances, members_list, px)
    tb, anb, ddb, shb, wb = metrics(nav_b, m_b, rebalances[0][0], last_td)
    print(f"\n[基准] 等权中证800(月度再平衡,零成本): 总收益={tb*100:.2f}% 年化={anb*100:.2f}% "
          f"回撤={ddb*100:.2f}% 夏普={shb:.2f} 胜率={wb*100:.1f}%", flush=True)
    pd.DataFrame([{"基准": "等权中证800(月度再平衡)",
                   "总收益": tb, "年化": anb, "最大回撤": ddb, "夏普": shb, "胜率": wb}]).to_csv(BENCH_CSV, index=False)
    print(f"[csv] 基准已写入 {BENCH_CSV}", flush=True)

    # 5) 参数扫描（增量续跑）
    done = {}
    if os.path.exists(RESULT_CSV):
        ed = pd.read_csv(RESULT_CSV)
        for _, row in ed.iterrows():
            done[(int(row["L"]), float(row["VOL_MULT"]), int(row["pre_years"]), int(row["top_n"]))] = True
    all_rows = []
    if os.path.exists(RESULT_CSV):
        all_rows = ed.to_dict("records")

    combos = [(L, VOL, py, tn) for (L, VOL) in SIGNAL_GRID for (py, tn) in SEL_GRID]
    print(f"[scan] 共 {len(combos)} 组合，已完成 {len(done)}", flush=True)

    for (L, VOL, py, tn) in combos:
        key = (L, VOL, py, tn)
        if key in done:
            continue
        df_pos = get_signal(L, VOL)
        sels = []
        for i, (r, _, _) in enumerate(rebalances):
            sels.append(select_breakout(r, df_pos, set(members_list[i]), pre_years=py, top_n=tn))
        nav, m = simulate_slots(rebalances, sels, px, tn)
        t, an, dd, sh, w = metrics(nav, m, rebalances[0][0], last_td)
        row = {"L": L, "VOL_MULT": VOL, "pre_years": py, "top_n": tn,
               "总收益": t, "年化": an, "最大回撤": dd, "夏普": sh, "胜率": w,
               "超额vs等权zz800": t - tb}
        all_rows.append(row)
        pd.DataFrame(all_rows).to_csv(RESULT_CSV, index=False)   # 增量落盘（防被杀丢结果）
        print(f"  L={L} V={VOL} py={py} n={tn}: 总={t*100:7.2f}% 年化={an*100:6.2f}% "
              f"回撤={dd*100:7.2f}% 夏普={sh:.2f} 胜率={w*100:4.1f}% 超额={ (t-tb)*100:7.2f}%", flush=True)

    # 6) 汇总
    df = pd.DataFrame(all_rows)
    best = df.loc[df["总收益"].idxmax()]
    worst = df.loc[df["总收益"].idxmin()]
    beats = (df["超额vs等权zz800"] > 0).sum()
    print("\n==================== 突破赢家 参数敏感性扫描 ====================")
    print(f"区间 {START}~{END}  股票池 zz800 时点成分股  初始 {INIT/1e4:.0f}万  月调仓等额纯持有")
    print(f"等权中证800基准(月度再平衡,零成本): 总收益={tb*100:.2f}% 年化={anb*100:.2f}% "
          f"回撤={ddb*100:.2f}% 夏普={shb:.2f} 胜率={wb*100:.1f}%")
    print(f"扫描 {len(df)} 组合: 最优 总收益={best['总收益']*100:.2f}%(L={best['L']} V={best['VOL_MULT']} "
          f"py={best['pre_years']} n={best['top_n']})；最差 总收益={worst['总收益']*100:.2f}%")
    print(f"跑赢等权中证800的组合数: {beats}/{len(df)}")
    print("DONE")


if __name__ == "__main__":
    main()
