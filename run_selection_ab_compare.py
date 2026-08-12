# -*- coding: utf-8 -*-
"""
A/B 选股对比：突破赢家 vs 标准动量因子（同为纯持有、同等额、同股票池）
=================================================================
目的：只换「选股规则」，持有机制完全相同，干净对比两套选股谁选得好。
  A = 突破赢家：过去3年突破信号(build_signal 5要件合成)次数排序，取前15
  B = 标准动量因子：过去12个月收益率排名、跳最近1月，zz800取前15
两者都在同一股票池(zz800 时点成分股)上选，月调仓、等额纯持有、无加仓无止损。
区间 2014-01-01 ~ 2026-06-30。基准 zz800 / hs300 买入持有。

容错：信号构建(重活)缓存到 pickle；选股增量落盘到 json；被杀后可自动续跑。
"""
import sqlite3, os, json, argparse, math
import numpy as np
import pandas as pd
from datetime import date

from backtest_main_rise import (
    load_data, add_base_features, build_signal,
    COMMISSION_RATE as CR, COMMISSION_MIN as CMIN, SLIPPAGE_RATE as SL, stamp_rate,
)
from run_monthly_rebalance import (
    get_conn, get_trade_dates, get_monthly_5th_trading_days, get_index_constituents,
    _apply_backadjust,
)

STOCK_POOL = "000906.SH"   # zz800 时点成分股
BENCH_ZZ800 = "000906.SH"
BENCH_HS300 = "000300.SH"
START = "20140101"
END   = "20260630"
TOP_N = 15
PRE_YEARS = 3
L = 60
VOL_MULT = 1.5
INIT = 1_000_000.0

RES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "data", "results", "selection_ab")
SIGNAL_CACHE = os.path.join(RES_DIR, "signal_cache_adj.pkl")
# 选股缓存带 _adj 后缀，与后复权信号缓存(signal_cache_adj.pkl)绑定，
# 避免「后复权信号已重建、却误用旧 raw 选股缓存」导致结果仍是 raw 口径。
SEL_CACHE = os.path.join(RES_DIR, "selections_cache_adj.json")
RESULT_CSV = os.path.join(RES_DIR, "results_ab.csv")


def ord_of(td):
    td = int(td)
    return date(td // 10000, (td % 10000) // 100, td % 100).toordinal()


def select_breakout(r_i, df_sig_pos, members, pre_years=PRE_YEARS, top_n=TOP_N):
    """过去 pre_years 年突破 signal 次数排序，取前 top_n（限 zz800 成员）。"""
    y = int(str(r_i)[:4])
    start_i = int(f"{y - pre_years}0101")
    end_i = int(r_i)
    win = df_sig_pos[(df_sig_pos["trade_date"] >= start_i) & (df_sig_pos["trade_date"] < end_i)]
    if win.empty:
        return []
    cnt = win.groupby("ts_code")["signal"].sum()
    cnt = cnt[cnt > 0]
    cnt = cnt[cnt.index.isin(members)]
    cnt = cnt.sort_values(ascending=False)
    return cnt.head(top_n).index.tolist()


def select_momentum(r_i, top_n=TOP_N):
    """复用平台标准动量因子：12月收益排名、跳1月、zz800。"""
    from run_monthly_rebalance import select_momentum_stocks
    df = select_momentum_stocks(str(r_i), lookback_months=12, skip_recent_months=1,
                                top_n=top_n, index_code=STOCK_POOL)
    return df["ts_code"].tolist() if not df.empty else []


def build_price_lookup(codes, min_d, max_d):
    conn = get_conn()
    placeholders = ",".join("?" * len(codes))
    q = (f"SELECT ts_code, trade_date, open, close FROM daily "
         f"WHERE ts_code IN ({placeholders}) AND trade_date BETWEEN ? AND ?")
    px = pd.read_sql_query(q, conn, params=list(codes) + [min_d, max_d])
    conn.close()
    out = {}
    for code, g in px.groupby("ts_code"):
        g = g.sort_values("trade_date").set_index("trade_date")
        out[code] = g
    return out


def asof_price(px, code, td, col):
    if code not in px:
        return None
    s = px[code][col]
    if td in s.index:
        return float(s.loc[td])
    sub = s.loc[:td]
    if len(sub) == 0:
        return None
    return float(sub.iloc[-1])


def slot_net_ret(buy_p, sell_p, buy_td, sell_td, notional=INIT / TOP_N):
    if buy_p is None or sell_p is None or buy_p <= 0 or sell_p <= 0:
        return None
    sell_td = int(sell_td)   # DB trade_date 为 TEXT，stamp_rate 需 int 比较
    buy_fill = buy_p * (1 + SL)
    comm_b = max(notional * CR, CMIN)
    shares = int((notional - comm_b) / buy_fill)
    if shares <= 0:
        return None
    sell_fill = sell_p * (1 - SL)
    gross = shares * sell_fill
    comm_s = max(gross * CR, CMIN)
    stamp = gross * stamp_rate(sell_td)
    return (gross - comm_b - comm_s - stamp) / notional - 1.0   # 净收益率（非增长倍数）


def simulate(rebalances, selections, px):
    nav = INIT
    nav_series = [nav]
    monthly = []
    for i, (r_i, buy_d, sell_d) in enumerate(rebalances):
        codes = selections[i]
        slot_rets = []
        for code in codes:
            bp = asof_price(px, code, buy_d, "open") or asof_price(px, code, buy_d, "close")
            sp = asof_price(px, code, sell_d, "open") or asof_price(px, code, sell_d, "close")
            r = slot_net_ret(bp, sp, buy_d, sell_d)
            slot_rets.append(r if r is not None else 0.0)
        if not slot_rets:
            period_ret = 0.0
        else:
            period_ret = float(np.mean(slot_rets))
        nav *= (1.0 + period_ret)
        nav_series.append(nav)
        monthly.append(period_ret)
    return nav_series, monthly


def metrics(nav_series, monthly, start_td, end_td):
    nav_arr = np.array(nav_series)
    total = nav_arr[-1] / nav_arr[0] - 1.0
    years = (ord_of(end_td) - ord_of(start_td)) / 365.25
    ann = (nav_arr[-1] / nav_arr[0]) ** (1.0 / years) - 1.0 if years > 0 else 0.0
    runmax = np.maximum.accumulate(nav_arr)
    maxdd = ((nav_arr - runmax) / runmax).min()
    m = np.array(monthly)
    sharpe = (m.mean() / m.std() * math.sqrt(12)) if m.std() > 1e-12 else 0.0
    win = (m > 0).mean()
    return total, ann, maxdd, sharpe, win


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="忽略缓存重算信号")
    args = ap.parse_args()
    os.makedirs(RES_DIR, exist_ok=True)

    # 1) 突破信号（重活只做一次，缓存到 pickle）
    if os.path.exists(SIGNAL_CACHE) and not args.rebuild:
        df_sig_pos = pd.read_pickle(SIGNAL_CACHE)
        print(f"[load] 从缓存加载突破信号：{len(df_sig_pos):,} 行", flush=True)
    else:
        load_start = "20100101"
        print(f"[load] {load_start}~{END} 全市场日线+信号构建 ...", flush=True)
        df, idx = load_data(load_start, END)
        print(f"  日线行数={len(df):,} 股票数={df['ts_code'].nunique():,}", flush=True)
        # 后复权：close/pre_close 改 close*adj_factor（open 保持 raw 作执行价），
        # 与平台动量/突破选股口径一致，避免除权除息日虚假突破。
        print(f"  [1/3] 后复权(close*adj_factor) ...", flush=True)
        df = _apply_backadjust(df, END)
        print(f"  [2/3] 构建 base features（均线/量比/动量等，重活）...", flush=True)
        df = add_base_features(df)
        print(f"  [3/3] 合成突破信号(build_signal) ...", flush=True)
        df_sig = build_signal(df, idx, L, VOL_MULT)
        df_sig_pos = df_sig[df_sig["signal"] > 0][["ts_code", "trade_date", "signal"]]
        df_sig_pos.to_pickle(SIGNAL_CACHE)
        print(f"  突破信号(>0)总行数={len(df_sig_pos):,}", flush=True)
        del df, idx, df_sig

    # 2) 月调仓日程（确定性，可重算）
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

    # 3) 两套选股（增量落盘 + 自动续跑）
    sel_A, sel_B = [], []
    if os.path.exists(SEL_CACHE):
        with open(SEL_CACHE) as f:
            c = json.load(f)
        sel_A, sel_B = c.get("A", []), c.get("B", [])
        print(f"[resume] 已缓存 {len(sel_A)} 期选股，续跑剩余 {len(rebalances)-len(sel_A)} 期 ...", flush=True)
    start_k = len(sel_A)
    for k, (r, _, _) in enumerate(rebalances):
        if k < start_k:
            continue
        members = set(get_index_constituents(STOCK_POOL, trade_date=str(r)))
        a = select_breakout(r, df_sig_pos, members)
        b = select_momentum(r)
        sel_A.append(a)
        sel_B.append(b)
        if (k + 1) % 6 == 0 or (k + 1) == len(rebalances):
            with open(SEL_CACHE, "w") as f:
                json.dump({"A": sel_A, "B": sel_B}, f)
        if (k + 1) % 12 == 0:
            print(f"  [{k+1:>3}] r={r} 突破赢家={len(a)} 标准动量={len(b)}", flush=True)
    print(f"[select] 完成 {len(sel_A)} 期选股", flush=True)

    # 4) 价格查找表
    codes_union = set()
    for s in sel_A + sel_B:
        codes_union.update(s)
    buy_dates = [b for (_, b, _) in rebalances]
    sell_dates = [s for (_, _, s) in rebalances]
    min_d, max_d = min(buy_dates + sell_dates), max(buy_dates + sell_dates)
    print(f"[price] 候选 {len(codes_union)} 只, 区间 {min_d}~{max_d} ...", flush=True)
    px = build_price_lookup(list(codes_union), min_d, max_d)

    # 5) 模拟
    nav_A, m_A = simulate(rebalances, sel_A, px)
    nav_B, m_B = simulate(rebalances, sel_B, px)

    # 6) 基准
    def bench_ret(code):
        conn = get_conn()
        rows = pd.read_sql_query(
            "SELECT trade_date, close FROM index_daily WHERE ts_code=? "
            "AND trade_date BETWEEN ? AND ? ORDER BY trade_date",
            conn, params=(code, START, END))
        conn.close()
        if len(rows) < 2:
            return 0.0
        return rows["close"].iloc[-1] / rows["close"].iloc[0] - 1.0
    rb_zz = bench_ret(BENCH_ZZ800)
    rb_hs = bench_ret(BENCH_HS300)

    # 7) 指标
    tA, anA, ddA, shA, wA = metrics(nav_A, m_A, rebalances[0][0], last_td)
    tB, anB, ddB, shB, wB = metrics(nav_B, m_B, rebalances[0][0], last_td)
    overlaps = []
    for a, b in zip(sel_A, sel_B):
        if a and b:
            overlaps.append(len(set(a) & set(b)) / TOP_N)
    avg_overlap = float(np.mean(overlaps)) if overlaps else 0.0

    # 8) 输出 + 落盘 CSV（即便会话日志丢失也能从文件取数）
    print("\n==================== A/B 选股对比（等额纯持有·月调仓·无加仓无止损） ====================")
    print(f"区间 {START}~{END}  股票池 zz800 时点成分股  每层 {TOP_N} 只  初始 {INIT/1e4:.0f}万")
    print(f"{'方案':<14}{'总收益':>10}{'年化':>9}{'最大回撤':>10}{'夏普':>7}{'胜率':>8}{'超额vs zz800':>14}")
    print(f"{'A 突破赢家':<14}{tA*100:>9.2f}%{anA*100:>8.2f}%{ddA*100:>9.2f}%{shA:>7.2f}{wA*100:>7.1f}%{(tA-rb_zz)*100:>13.2f}%")
    print(f"{'B 标准动量':<14}{tB*100:>9.2f}%{anB*100:>8.2f}%{ddB*100:>9.2f}%{shB:>7.2f}{wB*100:>7.1f}%{(tB-rb_zz)*100:>13.2f}%")
    print(f"{'基准 zz800':<14}{rb_zz*100:>9.2f}%{'—':>9}{'—':>10}{'—':>7}{'—':>8}{'0.00%':>14}")
    print(f"{'基准 hs300':<14}{rb_hs*100:>9.2f}%")
    print(f"\n两套选股月均重叠率={avg_overlap*100:.1f}%（越高说明两套其实在选同一批票）")
    print("DONE")

    out = pd.DataFrame([
        {"方案": "A 突破赢家", "总收益": tA, "年化": anA, "最大回撤": ddA, "夏普": shA, "胜率": wA, "超额vs_zz800": tA - rb_zz},
        {"方案": "B 标准动量", "总收益": tB, "年化": anB, "最大回撤": ddB, "夏普": shB, "胜率": wB, "超额vs_zz800": tB - rb_zz},
        {"方案": "基准 zz800", "总收益": rb_zz, "年化": None, "最大回撤": None, "夏普": None, "胜率": None, "超额vs_zz800": 0.0},
        {"方案": "基准 hs300", "总收益": rb_hs, "年化": None, "最大回撤": None, "夏普": None, "胜率": None, "超额vs_zz800": rb_hs - rb_zz},
    ])
    out.to_csv(RESULT_CSV, index=False)
    print(f"[csv] 已写入 {RESULT_CSV}", flush=True)


if __name__ == "__main__":
    main()
