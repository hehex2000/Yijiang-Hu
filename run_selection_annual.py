# -*- coding: utf-8 -*-
"""
突破赢家选股 · 年度换股策略（对比月度换股 + 基准）
========================================================
目的：把"突破赢家选法"的持有机制从月调仓改成"年度换股"（每年调一次、持有约1年），
看更低的换手是否能改善净收益、以及相对被动指数有无 alpha。

口径（与 A/B、sensitivity 一致）：
  股票池 zz800 时点成分股；区间 2014-01 ~ 2026-06；初始 100 万；
  次日开盘成交；费用=佣金0.025%/边(最低5元)+印花(2023-08-28起0.05%)+滑点0.1%/边。
  选股 = 过去 pre_years 年突破信号(build_signal 5要件)次数排序前 top_n。

年度日程：每年 1 月第5交易日为信号日 → 次日开盘买入 → 持有至次年同信号日次日开盘卖出
          （末次持有卖在 END）。

基准：
  B1 = 等权中证800（年度再平衡、零成本小数份额）—— 公平 active-vs-passive 测试
  B2 = 市值加权中证800 买入持有（参考，已知 +119.54%）

容错：base-feat / 信号 / 成分股 均缓存（复用 sensitivity 已建缓存），被杀可续跑。
"""
import sqlite3, os, json, argparse, math
import numpy as np
import pandas as pd
from datetime import date

from run_selection_ab_compare import (
    ord_of, select_breakout, build_price_lookup, asof_price,
    slot_net_ret, get_index_constituents, get_trade_dates,
    get_monthly_5th_trading_days, INIT, STOCK_POOL, START, END,
    SL, CR, stamp_rate,
)
from backtest_main_rise import load_data, add_base_features, build_signal
from run_monthly_rebalance import _apply_backadjust

RES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "data", "results", "selection_ab")
BASE_FEAT_CACHE = os.path.join(RES_DIR, "base_feat_cache_adj.pkl")
SIG_CACHE_TPL = os.path.join(RES_DIR, "signal_cache_L{L}_V{V}_adj.pkl")
MEM_CACHE = os.path.join(RES_DIR, "members_cache_annual.json")
RESULT_CSV = os.path.join(RES_DIR, "results_annual.csv")
BENCH_CSV = os.path.join(RES_DIR, "bench_annual.csv")

LOAD_START = "20080101"   # 支撑 pre_years=5 回溯到 2009

# 参数网格（聚焦：默认 / 月度最优 / 长回望大名单洗掉edge）
SIGNAL_GRID = [(60, 1.5), (60, 2.0)]
SEL_GRID = [(2, 10), (3, 15), (5, 30)]


def get_conn():
    return sqlite3.connect("D:/tu-shareData/astock_daily.db")


def load_base_feat():
    if os.path.exists(BASE_FEAT_CACHE):
        print(f"[load] base-feat 从缓存读取", flush=True)
        return pd.read_pickle(BASE_FEAT_CACHE)
    print(f"[load] 构建 base-feat（重活）{LOAD_START}~{END} ...", flush=True)
    df, _ = load_data(LOAD_START, END)
    # 后复权：close/pre_close 改 close*adj_factor（与平台口径一致），避免除权除息日虚假突破
    df = _apply_backadjust(df, END)
    df = add_base_features(df)
    df.to_pickle(BASE_FEAT_CACHE)
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


def get_annual_rebalance_dates(td_all):
    """每年 1 月第5交易日作为信号日。"""
    monthly = get_monthly_5th_trading_days(td_all)
    annual = [r for r in monthly if int(r) // 100 % 100 == 1]
    return annual


def simulate_slots(rebalances, selections, px, top_n):
    """每槽资金 = INIT/top_n，保证满仓可比；年收益 -> 净值。"""
    nav = INIT
    nav_series = [nav]
    yearly = []
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
        yearly.append(pr)
    return nav_series, yearly


def bench_simulate(rebalances, members_list, px):
    """等权中证800基准（零成本小数份额），年度再平衡。"""
    nav = INIT
    nav_series = [nav]
    yearly = []
    for i, (r, buy_d, sell_d) in enumerate(rebalances):
        codes = members_list[i]
        rets = []
        for code in codes:
            bp = asof_price(px, code, buy_d, "open") or asof_price(px, code, buy_d, "close")
            sp = asof_price(px, code, sell_d, "open") or asof_price(px, code, sell_d, "close")
            if bp is not None and sp is not None and bp > 0 and sp > 0:
                rets.append(sp / bp - 1.0)
        pr = float(np.mean(rets)) if rets else 0.0
        nav *= (1.0 + pr)
        nav_series.append(nav)
        yearly.append(pr)
    return nav_series, yearly


def metrics(nav_series, period_rets, start_td, end_td, periods_per_year=1):
    """修正版：年化按实际年数；Sharpe 按实际周期频率(年度×√1, 月度×√12)。"""
    nav_arr = np.array(nav_series, dtype=float)
    total = nav_arr[-1] / nav_arr[0] - 1.0
    years = (ord_of(end_td) - ord_of(start_td)) / 365.25
    ann = (nav_arr[-1] / nav_arr[0]) ** (1.0 / years) - 1.0 if years > 0 else 0.0
    runmax = np.maximum.accumulate(nav_arr)
    maxdd = ((nav_arr - runmax) / runmax).min()
    m = np.array(period_rets, dtype=float)
    sharpe = (m.mean() / m.std() * math.sqrt(periods_per_year)) if m.std() > 1e-12 else 0.0
    win = (m > 0).mean()
    return total, ann, maxdd, sharpe, win


def bench_capweight_buyhold():
    """市值加权中证800 买入持有（参考基准）。"""
    conn = get_conn()
    rows = pd.read_sql_query(
        "SELECT trade_date, close FROM index_daily WHERE ts_code='000906.SH' "
        "AND trade_date BETWEEN ? AND ? ORDER BY trade_date",
        conn, params=(START, END))
    conn.close()
    if len(rows) < 2:
        return 0.0
    return float(rows["close"].iloc[-1] / rows["close"].iloc[0] - 1.0)


def main():
    os.makedirs(RES_DIR, exist_ok=True)
    # 1) 年度日程
    td_all = get_trade_dates(START, END)
    annual_r = get_annual_rebalance_dates(td_all)
    td_set = set(td_all)
    rebalances = []
    for r in annual_r:
        nxt = [d for d in td_all if d > r]
        buy_d = nxt[0] if nxt else None
        if buy_d is None:
            continue
        rebalances.append((r, buy_d, None))
    last_td = td_all[-1]
    for i in range(len(rebalances)):
        sell_d = rebalances[i + 1][1] if i + 1 < len(rebalances) else last_td
        rebalances[i] = (rebalances[i][0], rebalances[i][1], sell_d)
    print(f"[schedule] 年度调仓次数={len(rebalances)} 首={rebalances[0][0]} 末={rebalances[-1][0]} "
          f"末次卖={last_td}", flush=True)

    # 2) 成分股
    members_list = load_members(rebalances)

    # 2.5) 丢弃领先空成分股期（zz800 快照最早 20140430，2014 信号日无成分股→空仓×1.0，
    #       剔除以诚实标注区间；不影响乘积，只让区间从 2015 起）
    paired = [(rb, m) for rb, m in zip(rebalances, members_list) if len(m) > 0]
    if len(paired) < len(rebalances):
        dropped = len(rebalances) - len(paired)
        print(f"[trim] 丢弃前 {dropped} 期(成分股为空, 早于 zz800 快照起始 20140430)", flush=True)
        rebalances = [rb for rb, _ in paired]
        members_list = [m for _, m in paired]
        for i in range(len(rebalances)):
            rebalances[i] = (rebalances[i][0], rebalances[i][1],
                             rebalances[i + 1][1] if i + 1 < len(rebalances) else last_td)
    print(f"[schedule-trimmed] 有效年度调仓次数={len(rebalances)} 首={rebalances[0][0]}", flush=True)

    # 3) 价格表
    all_codes = set()
    for m in members_list:
        all_codes.update(m)
    buy_dates = [b for (_, b, _) in rebalances]
    sell_dates = [s for (_, _, s) in rebalances]
    min_d, max_d = min(buy_dates + sell_dates), max(buy_dates + sell_dates)
    print(f"[price] 候选 {len(all_codes):,} 只, 区间 {min_d}~{max_d} ...", flush=True)
    px = build_price_lookup(list(all_codes), min_d, max_d)

    # 4) 基准
    nav_b, m_b = bench_simulate(rebalances, members_list, px)
    tb, anb, ddb, shb, wb = metrics(nav_b, m_b, rebalances[0][0], last_td, periods_per_year=1)
    tbh = bench_capweight_buyhold()
    print(f"\n[基准B1] 等权中证800(年度再平衡,零成本): 总={tb*100:.2f}% 年化={anb*100:.2f}% "
          f"回撤={ddb*100:.2f}% 夏普={shb:.2f} 胜率={wb*100:.1f}%", flush=True)
    print(f"[基准B2] 市值加权中证800(买入持有): 总={tbh*100:.2f}%", flush=True)
    pd.DataFrame([
        {"基准": "等权中证800(年度再平衡,零成本)", "总收益": tb, "年化": anb,
         "最大回撤": ddb, "夏普": shb, "胜率": wb},
        {"基准": "市值加权中证800(买入持有)", "总收益": tbh, "年化": (1+tbh)**(1/((ord_of(last_td)-ord_of(START))/365.25))-1,
         "最大回撤": None, "夏普": None, "胜率": None},
    ]).to_csv(BENCH_CSV, index=False)

    # 5) 参数扫描（增量续跑）
    done = {}
    if os.path.exists(RESULT_CSV):
        ed = pd.read_csv(RESULT_CSV)
        for _, row in ed.iterrows():
            done[(int(row["L"]), float(row["VOL_MULT"]), int(row["pre_years"]), int(row["top_n"]))] = True
    all_rows = ed.to_dict("records") if os.path.exists(RESULT_CSV) else []

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
        t, an, dd, sh, w = metrics(nav, m, rebalances[0][0], last_td, periods_per_year=1)
        row = {"L": L, "VOL_MULT": VOL, "pre_years": py, "top_n": tn,
               "总收益": t, "年化": an, "最大回撤": dd, "夏普": sh, "胜率": w,
               "超额vs等权基准": t - tb, "超额vs市值加权": t - tbh}
        all_rows.append(row)
        pd.DataFrame(all_rows).to_csv(RESULT_CSV, index=False)
        print(f"  L={L} V={VOL} py={py} n={tn}: 总={t*100:7.2f}% 年化={an*100:6.2f}% "
              f"回撤={dd*100:7.2f}% 夏普={sh:.2f} 胜率={w*100:4.1f}% "
              f"超额B1={(t-tb)*100:7.2f}% 超额B2={(t-tbh)*100:7.2f}%", flush=True)

    # 6) 汇总
    df = pd.DataFrame(all_rows)
    best = df.loc[df["总收益"].idxmax()]
    worst = df.loc[df["总收益"].idxmin()]
    beats_b1 = (df["超额vs等权基准"] > 0).sum()
    beats_b2 = (df["超额vs市值加权"] > 0).sum()
    print("\n==================== 突破赢家 年度换股 参数扫描 ====================")
    print(f"区间 {START}~{END}  股票池 zz800 时点成分股  初始 {INIT/1e4:.0f}万  年度换股等额纯持有")
    print(f"基准B1 等权中证800(年度再平衡,零成本): 总={tb*100:.2f}% 年化={anb*100:.2f}% "
          f"回撤={ddb*100:.2f}% 夏普={shb:.2f}")
    print(f"基准B2 市值加权中证800(买入持有): 总={tbh*100:.2f}%")
    print(f"扫描 {len(df)} 组合: 最优 总={best['总收益']*100:.2f}%(L={best['L']} V={best['VOL_MULT']} "
          f"py={best['pre_years']} n={best['top_n']})；最差 总={worst['总收益']*100:.2f}%")
    print(f"跑赢等权基准B1的组合数: {beats_b1}/{len(df)}；跑赢市值加权B2: {beats_b2}/{len(df)}")
    print("DONE")


if __name__ == "__main__":
    main()
