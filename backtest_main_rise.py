# -*- coding: utf-8 -*-
"""
主升浪战法（跟着Jim学量化 BV1a5ub6ZEUW）· 量化验证回测
=================================================
把视频的「四个条件」拆成可逐项检查的量化规则，用自有 A股数据验证其是否真有正期望。
注意：Jim 是白名单作者（方法论干净、无喊单引流），但「策略本身赚不赚钱」必须用数据验——
本脚本即做真值校验（skill §0 第5步）。

视频四条件 → 量化代理（合理定义，非逐字复刻；Jim 未给固定参数）：
  ① 方向向上   : close > MA20 > MA60 且 MA20/MA60 均上行(斜率>0)
  ② 相对强度领先: 个股 L日收益 > 沪深300 L日收益(跑赢大盘) 且 行业内 L日动量排名前20%
  ③ 放量突破   : 收盘创 L日新高(收线确认,防针状假突破) 且 量 > VOL_MULT × MA20量
  ④ 退出       : 跌破 MA20(趋势参考位失守) 或 跌破 入场价×(1−K×入场日波动)(按波动留空间)
                 → 次日开盘执行；封顶 MAX_HOLD 交易日
仓位：等权、最多 MAX_POS 个并发（对应「仓位提前设边界,避免信号集中」）。

费用模型与平台一致：佣金0.00025/边(最低5元)+印花(2023-08-28起0.0005,此前0.001)+滑点0.001/边。
"""
import sqlite3, os, argparse, math
from collections import defaultdict
from datetime import date
import numpy as np
import pandas as pd

def td_to_ord(td):
    td = int(td)
    return date(td // 10000, (td % 10000) // 100, td % 100).toordinal()
def years_between(td1, td2):
    return (td_to_ord(td2) - td_to_ord(td1)) / 365.25

DB = "D:/tu-shareData/astock_daily.db"
BENCH = "000300.SH"  # 沪深300 作为「大盘」相对强度基准

# ---------- 费用模型 ----------
COMMISSION_RATE = 0.00025
COMMISSION_MIN = 5.0
SLIPPAGE_RATE = 0.001
STAMP_OLD = 0.001
STAMP_NEW = 0.0005
STAMP_CUT = 20230828

def stamp_rate(td):
    return STAMP_NEW if td >= STAMP_CUT else STAMP_OLD

def trade_net_ret(notional, buy_price, sell_price, buy_td, sell_td):
    buy_fill = buy_price * (1 + SLIPPAGE_RATE)
    comm_b = max(notional * COMMISSION_RATE, COMMISSION_MIN)
    shares = int((notional - comm_b) / buy_fill)
    if shares <= 0:
        return None
    sell_fill = sell_price * (1 - SLIPPAGE_RATE)
    gross = shares * sell_fill
    comm_s = max(gross * COMMISSION_RATE, COMMISSION_MIN)
    stamp = gross * stamp_rate(sell_td)
    proceeds = gross - comm_s - stamp
    return proceeds / notional - 1.0

# ---------- 默认参数 ----------
START = "20150101"
END   = "20260630"
CIRC_MV_MIN = 2_000_000     # 流通市值 >= 20亿（daily_basic 单位千元）
AMOUNT_MIN  = 30_000        # 成交额 >= 3000万（daily.amount 单位千元）
MIN_AGE_DAYS = 60
L_DEF = 60
VOL_MULT_DEF = 1.5
K_DEF = 2.0
MAX_HOLD_DEF = 120
MAX_POS_DEF = 10
INIT_CAPITAL = 1_000_000.0

def load_data(start, end):
    con = sqlite3.connect(DB)
    d = pd.read_sql_query(
        "SELECT ts_code, trade_date, open, high, low, close, pre_close, vol, amount "
        "FROM daily WHERE trade_date BETWEEN ? AND ?", con, params=(start, end))
    db_ = pd.read_sql_query("SELECT ts_code, trade_date, circ_mv, volume_ratio FROM daily_basic", con)
    sb = pd.read_sql_query("SELECT ts_code, name, list_date, industry FROM stock_basic", con)
    idx = pd.read_sql_query(
        "SELECT trade_date, close AS idx_close FROM index_daily WHERE ts_code=? AND trade_date BETWEEN ? AND ? "
        "ORDER BY trade_date", con, params=(BENCH, start, end))
    con.close()
    for c in ["open","high","low","close","pre_close","vol","amount"]:
        d[c] = d[c].astype("float32")
    db_["circ_mv"] = db_["circ_mv"].astype("float32")
    db_["volume_ratio"] = db_["volume_ratio"].astype("float32")
    d["trade_date"] = d["trade_date"].astype(int)
    db_["trade_date"] = db_["trade_date"].astype(int)   # 注意:daily_basic.trade_date 是 TEXT,必须转 int 才能 merge_asof
    idx["trade_date"] = idx["trade_date"].astype(int)
    d = d.sort_values("trade_date")
    db_ = db_.sort_values("trade_date")
    df = pd.merge_asof(d, db_, on="trade_date", by="ts_code", direction="backward")
    sb["is_st"] = sb["name"].str.contains("ST", na=False).astype(bool)
    sb["list_date"] = pd.to_datetime(sb["list_date"], errors="coerce")
    df = df.merge(sb[["ts_code","name","is_st","list_date","industry"]], on="ts_code", how="left")
    df["is_st"] = df["is_st"].fillna(False).astype(bool)
    idx = idx.set_index("trade_date")["idx_close"].astype("float32")
    df = df.sort_values(["ts_code","trade_date"]).reset_index(drop=True)
    return df, idx

def add_base_features(df):
    """与 L 无关的基础特征（方向/量比/相对强度大盘侧在 build_signal 内按 L 计算）。"""
    df = df.copy()
    g = df.groupby("ts_code", sort=False)
    df["MA20"] = g["close"].transform(lambda s: s.rolling(20).mean())
    df["MA60"] = g["close"].transform(lambda s: s.rolling(60).mean())
    df["MA20_up"] = (df["MA20"] > df.groupby("ts_code")["MA20"].shift(5))
    df["MA60_up"] = (df["MA60"] > df.groupby("ts_code")["MA60"].shift(5))
    df["dir_up"] = (df["close"] > df["MA20"]) & (df["MA20"] > df["MA60"]) & df["MA20_up"] & df["MA60_up"]
    df["vol_ma20"] = g["vol"].transform(lambda s: s.rolling(20).mean())
    df["vol_ratio_raw"] = df["vol"] / df["vol_ma20"]
    df["ret"] = df["close"] / df["pre_close"] - 1.0
    df["is_limit_up_close"] = (df["high"] == df["close"]) & (df["close"] >= df["pre_close"]*1.095)
    ref = pd.to_datetime(df["trade_date"].astype(str), format="%Y%m%d")
    df["age_cal"] = (ref - df["list_date"]).dt.days
    # 次日开盘（真实）
    df["next_open"] = g["open"].shift(-1)
    # 股票池资格
    df["eligible"] = (
        (~df["is_st"]) &
        (df["age_cal"] >= MIN_AGE_DAYS*1.5) &
        (df["circ_mv"] >= CIRC_MV_MIN) &
        (df["amount"] >= AMOUNT_MIN) &
        (~df["is_limit_up_close"]) &
        (df["next_open"].notna())
    )
    return df

def build_signal(df, idx, L, VOL_MULT):
    """按 L / VOL_MULT 生成信号布尔列与入场波动列（vol_L）。"""
    df = df.copy()
    g = df.groupby("ts_code", sort=False)
    # 放量突破：收盘创 L 日新高（排除当日）+ 量放大
    cons_high = g["close"].transform(lambda s: s.rolling(L).max()).shift(1)
    df["breakout"] = (df["close"] > cons_high) & (df["vol_ratio_raw"] > VOL_MULT)
    # 相对强度：个股 L 日收益 vs 沪深300 L 日收益
    mom_L = g["close"].transform(lambda s: s / s.shift(L) - 1.0)
    idx_ret_L = (idx / idx.shift(L) - 1.0)
    df["idx_ret_L"] = df["trade_date"].map(idx_ret_L.to_dict())
    df["rs_lead"] = mom_L > df["idx_ret_L"]
    # 同类排名：行业内 L 日动量前 20%
    df["mom_L_val"] = mom_L
    df["ind_rank"] = df.groupby(["trade_date","industry"])["mom_L_val"].rank(pct=True)
    df["rs_ind"] = df["ind_rank"] > 0.80   # 同类排名靠前 = 行业动量前20%
    # 入场波动（用于退出止损宽度）
    df["vol_L"] = g["ret"].transform(lambda s: s.rolling(L).std())
    # 综合信号
    df["signal"] = df["eligible"] & df["dir_up"] & df["rs_lead"] & df["breakout"] & df["rs_ind"]
    return df

def run_backtest(df, K, MAX_HOLD):
    """前向扫描退出。返回 trades 列表。"""
    df = df.sort_values(["ts_code","trade_date"]).reset_index(drop=True)
    OPEN = df["open"].values.astype(float)
    CLOSE = df["close"].values.astype(float)
    MA20 = df["MA20"].values.astype(float)
    VOLV = df["vol_L"].values.astype(float)
    DATES = df["trade_date"].values.astype(int)
    SIG = df["signal"].values.astype(bool)
    TS = df["ts_code"].values
    # 每个股票的 [start,end) 行区间（df 已按 [ts_code,trade_date] 排序，连续块）
    bounds = {}
    start = 0
    prev = TS[0]
    for i in range(1, len(TS)+1):
        if i == len(TS) or TS[i] != prev:
            bounds[prev] = (start, i)
            if i < len(TS):
                prev = TS[i]; start = i
    trades = []
    n_sig = 0
    for ts, (s, e) in bounds.items():
        o = OPEN[s:e]; c = CLOSE[s:e]; m = MA20[s:e]; v = VOLV[s:e]; d = DATES[s:e]
        sig = SIG[s:e]
        idxs = np.where(sig)[0]
        n_sig += len(idxs)
        for li in idxs:
            if li + 1 >= len(o):
                continue
            entry_open = o[li + 1]
            if not np.isfinite(entry_open):
                continue
            stop = entry_open * (1.0 - K * v[li + 1]) if np.isfinite(v[li+1]) else entry_open*0.85
            exited = False
            end = min(li + 1 + MAX_HOLD, len(o) - 1)
            exit_row = end
            for k in range(li + 1, end + 1):
                if c[k] < m[k] or c[k] < stop:
                    if k + 1 < len(o):
                        ex_open = o[k + 1]; ex_date = int(d[k + 1])
                    else:
                        ex_open = c[k]; ex_date = int(d[k])
                    reason = "ma20" if c[k] < m[k] else "stop"
                    exit_row = k + 1
                    exited = True
                    break
            if not exited:
                ex_open = o[end]; ex_date = int(d[end]); reason = "maxhold"
            hold = exit_row - (li + 1)   # 交易日数
            trades.append((ts, int(d[li + 1]), float(entry_open), ex_date, float(ex_open), reason, int(hold)))
    return trades, n_sig

def summarize_trades(trades):
    nets = []
    holds = []
    reasons = defaultdict(int)
    for (ts, ed, eo, xd, xo, reason, hold) in trades:
        nr = trade_net_ret(100000.0, eo, xo, ed, xd)
        if nr is not None:
            nets.append(nr)
            holds.append(hold)
            reasons[reason] += 1
    nets = np.array(nets)
    holds = np.array(holds)
    n = len(nets)
    if n == 0:
        return None
    return {
        "n": n,
        "mean_net_%": round(nets.mean()*100, 4),
        "median_net_%": round(np.median(nets)*100, 4),
        "win_%": round((nets > 0).mean()*100, 2),
        "avg_hold_days": round(holds.mean(), 1),
        "reason_ma20_%": round(reasons["ma20"]/n*100, 1),
        "reason_stop_%": round(reasons["stop"]/n*100, 1),
        "reason_maxhold_%": round(reasons["maxhold"]/n*100, 1),
    }

def portfolio_equity(trades, df, INIT, MAX_POS):
    """等权、最多 MAX_POS 并发的日频组合净值（含买/卖滑点、佣金、印花）。"""
    bounds = {}
    start = 0; prev = df["ts_code"].values[0]
    TS = df["ts_code"].values
    for i in range(1, len(TS)+1):
        if i == len(TS) or TS[i] != prev:
            bounds[prev] = (start, i)
            if i < len(TS):
                prev = TS[i]; start = i
    OPEN = df["open"].values.astype(float)
    CLOSE = df["close"].values.astype(float)
    DATES = df["trade_date"].values.astype(int)
    # 每股票 date->index 用于收盘价标记
    stock_dates = {ts: DATES[s:e] for ts,(s,e) in bounds.items()}
    stock_close = {ts: CLOSE[s:e] for ts,(s,e) in bounds.items()}
    by_entry = defaultdict(list); by_exit = defaultdict(list)
    for t in trades:
        by_entry[t[1]].append(t); by_exit[t[3]].append(t)
    trade_dates = np.unique(DATES)
    trade_dates.sort()
    def sidx(dts, d):
        j = np.searchsorted(dts, d, side="right") - 1
        return j if j >= 0 else 0
    cash = INIT
    positions = []  # {ts, entry_date, entry_open, shares}
    eq_series = []
    for d in trade_dates:
        # 退出
        for t in by_exit.get(d, []):
            for p in positions:
                if p["ts"] == t[0] and p["entry_date"] == t[1] and abs(p["entry_open"]-t[2]) < 1e-6:
                    ex_open = t[4]; sell_fill = ex_open*(1-SLIPPAGE_RATE)
                    gross = p["shares"]*sell_fill
                    comm = max(gross*COMMISSION_RATE, COMMISSION_MIN)
                    stamp = gross*stamp_rate(d)
                    cash += gross - comm - stamp
                    positions.remove(p)
                    break
        # 当前净值（用于等权分配）
        eq_cur = cash
        for p in positions:
            arr = stock_close[p["ts"]]; dts = stock_dates[p["ts"]]
            j = sidx(dts, d)
            eq_cur += p["shares"]*arr[j]*(1-SLIPPAGE_RATE)
        # 入场
        for t in by_entry.get(d, []):
            if len(positions) >= MAX_POS:
                continue
            alloc = eq_cur / MAX_POS
            if alloc <= 0:
                continue
            entry_open = t[2]; buy_fill = entry_open*(1+SLIPPAGE_RATE)
            comm_b = max(alloc*COMMISSION_RATE, COMMISSION_MIN)
            shares = int((alloc - comm_b)/buy_fill)
            if shares <= 0:
                continue
            cash -= shares*buy_fill + comm_b
            positions.append({"ts": t[0], "entry_date": t[1], "entry_open": entry_open, "shares": shares})
        # 标记
        eq = cash
        for p in positions:
            arr = stock_close[p["ts"]]; dts = stock_dates[p["ts"]]
            j = sidx(dts, d)
            eq += p["shares"]*arr[j]*(1-SLIPPAGE_RATE)
        eq_series.append((int(d), eq))
    return eq_series

def max_drawdown(eq_series):
    dates = [x[0] for x in eq_series]; eqs = np.array([x[1] for x in eq_series])
    runmax = np.maximum.accumulate(eqs)
    dd = (eqs - runmax)/runmax
    return dd.min()*100, dd

def benchmark_annual(idx, start, end):
    s = idx.loc[idx.index >= int(start)]
    s = s.loc[s.index <= int(end)]
    if len(s) < 2:
        return None
    ret = s.iloc[-1]/s.iloc[0] - 1
    years = years_between(s.index[0], s.index[-1])
    ann = (1+ret)**(1/max(years,0.5)) - 1
    return ann*100, ret*100

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=START)
    ap.add_argument("--end", default=END)
    ap.add_argument("--L", type=int, default=L_DEF)
    ap.add_argument("--vol-mult", type=float, default=VOL_MULT_DEF)
    ap.add_argument("--K", type=float, default=K_DEF)
    ap.add_argument("--max-hold", type=int, default=MAX_HOLD_DEF)
    ap.add_argument("--max-pos", type=int, default=MAX_POS_DEF)
    ap.add_argument("--sensitivity", action="store_true", help="跑参数敏感性网格")
    ap.add_argument("--out", default="data/results/main_rise")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"[load] {args.start}~{args.end} ...")
    df, idx = load_data(args.start, args.end)
    print(f"  日线行数={len(df):,} 股票数={df['ts_code'].nunique():,}")
    df = add_base_features(df)

    # ---------- 主配置 ----------
    print(f"[signal] L={args.L} VOL_MULT={args.vol_mult} ...")
    df_sig = build_signal(df, idx, args.L, args.vol_mult)
    n_sig_rows = int(df_sig["signal"].sum())
    print(f"  信号数={n_sig_rows:,}")
    trades, n_sig = run_backtest(df_sig, args.K, args.max_hold)
    print(f"[backtest] 实际成交笔数={len(trades):,}")
    summ = summarize_trades(trades)
    if summ is None:
        print("[warn] 无成交（窗口过短或过滤过严），跳过组合与敏感性。")
        pd.DataFrame([{"config":"none","note":"no trades"}]).to_csv(
            os.path.join(args.out, "main_rise_main.csv"), index=False, encoding="utf-8-sig")
        return
    print(f"  每笔净: 均值={summ['mean_net_%']:+.4f}% 胜率={summ['win_%']:.1f}% 均持={summ['avg_hold_days']}天 "
          f"退出(ma20/stop/maxhold)={summ['reason_ma20_%']}/{summ['reason_stop_%']}/{summ['reason_maxhold_%']}%")

    # 组合净值
    eq = portfolio_equity(trades, df_sig, INIT_CAPITAL, args.max_pos)
    mdd, _ = max_drawdown(eq)
    eq0 = eq[0][1]; eq1 = eq[-1][1]
    years = years_between(eq[0][0], eq[-1][0])
    ann = (eq1/eq0)**(1/max(years,0.5)) - 1
    bench_ann, bench_ret = benchmark_annual(idx, args.start, args.end)
    print(f"[portfolio] 总收益={(eq1/eq0-1)*100:+.1f}% 年化={ann*100:+.2f}% 最大回撤={mdd:.1f}% "
          f"沪深300年化={bench_ann:+.2f}% (价格收益,未计股息)")

    main_row = {
        "config": f"L{args.L}_V{args.vol_mult}_K{args.K}_H{args.max_hold}_P{args.max_pos}",
        "n_trades": summ["n"],
        "mean_net_%": summ["mean_net_%"],
        "win_%": summ["win_%"],
        "avg_hold": summ["avg_hold_days"],
        "port_total_%": round((eq1/eq0-1)*100, 2),
        "port_ann_%": round(ann*100, 2),
        "port_maxdd_%": round(mdd, 2),
        "bench_ann_%": round(bench_ann, 2) if bench_ann else None,
        "exit_ma20_%": summ["reason_ma20_%"],
        "exit_stop_%": summ["reason_stop_%"],
        "exit_maxhold_%": summ["reason_maxhold_%"],
    }
    pd.DataFrame([main_row]).to_csv(os.path.join(args.out, "main_rise_main.csv"), index=False, encoding="utf-8-sig")
    # 交易流水（透明）
    pd.DataFrame(trades, columns=["ts_code","entry_date","entry_open","exit_date","exit_open","reason","hold_days"]).to_csv(
        os.path.join(args.out, "main_rise_trades.csv"), index=False, encoding="utf-8-sig")
    print(f"  已保存 main_rise_main.csv / main_rise_trades.csv")

    # ---------- 参数敏感性网格（视频自证疑虑:换参数结果大变?）----------
    if args.sensitivity:
        print("[sensitivity] 网格扫描 ...")
        rows = []
        Ls = [40, 60, 90]
        Vs = [1.3, 1.5, 2.0]
        Ks = [1.5, 2.0, 2.5]
        for L in Ls:
            for V in Vs:
                ds = build_signal(df, idx, L, V)
                tr, _ = run_backtest(ds, 2.0, args.max_hold)
                sm = summarize_trades(tr)
                rows.append({"L":L,"VOL_MULT":V,"K":2.0,"n":sm["n"],"mean_net_%":sm["mean_net_%"],
                             "win_%":sm["win_%"],"avg_hold":sm["avg_hold_days"]})
        for K in Ks:
            ds = build_signal(df, idx, 60, 1.5)
            tr, _ = run_backtest(ds, K, args.max_hold)
            sm = summarize_trades(tr)
            rows.append({"L":60,"VOL_MULT":1.5,"K":K,"n":sm["n"],"mean_net_%":sm["mean_net_%"],
                         "win_%":sm["win_%"],"avg_hold":sm["avg_hold_days"]})
        sdf = pd.DataFrame(rows)
        sdf.to_csv(os.path.join(args.out, "main_rise_sensitivity.csv"), index=False, encoding="utf-8-sig")
        print(sdf.to_string(index=False))
        print("  已保存 main_rise_sensitivity.csv")

    print("DONE")

if __name__ == "__main__":
    main()
