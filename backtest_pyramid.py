# -*- coding: utf-8 -*-
"""
金字塔加仓 vs 补仓陷阱 —— 北宸的交易笔记《加仓的学问》量化验证
================================================================
视频: BV1jSuM65Ehj (UP: 北宸的交易笔记)
核心论点(风控科普, 非买卖战法):
  - 越跌越补(马丁格尔)= 死亡螺旋; 只在盈利时顺势加仓的"正金字塔"才是正确的仓位纪律。
  - 三条铁律: 亏损不加仓 / 加仓必上移止损 / 单笔亏损锁死1-2%。

本脚本要验证的, 是把"加仓方案"这个**仓位管理维度**从"入场信号"里单独拎出来测:
  用同一组"右侧突破"入场信号(Livermore/主升浪同族), 在同一批信号上跑 4 种资金方案,
  比较它们的期望值(每笔均净收益)、胜率、以及最致命的左尾(单笔最惨)。

四种方案(每方案总预算 = 同一笔 notional C, 资本公平):
  SINGLE    : 一次性建仓(基准)
  PYRAMID   : 正金字塔 40/30/20/10, 仅在盈利(+5%)时递减加仓
  INVERTED  : 倒金字塔 10/20/30/40, 在盈利(+5%)时递增加仓(视频说这是错误版)
  MARTINGALE: 补仓/越跌越补 40/30/20/10, 在亏损(-5%)时加仓

退出: 所有方案统一用跟踪止损(从持仓最高价回撤 TRAIL 离场), 封顶 MAX_HOLD 交易日。
费用模型与平台一致: 佣金0.00025/边(最低5元)+印花(2023-08-28起0.0005)+滑点0.001/边。

诚信边界: 视频是风控教科书(B+), 不主张 UP 主是骗子; 本脚本证伪的是"金字塔本身能造出正期望",
而不是证伪"金字塔是更好的风控纪律"——后者的风险维度结论由数据说话。
"""
import os, argparse
import numpy as np
import pandas as pd

# 复用主升浪引擎的数据加载 + 突破信号 + 费用模型(同一族入场信号, 与 §5.11/§5.12 一致)
from backtest_main_rise import (
    load_data, add_base_features, build_signal,
    trade_net_ret, INIT_CAPITAL,
    COMMISSION_RATE, COMMISSION_MIN, SLIPPAGE_RATE, stamp_rate,
)

DB_OUT = "data/results/pyramid"
START = "20150101"
END   = "20260630"

# ---------- 加仓方案定义 ----------
UP_STEP   = 0.05     # 盈利加仓触发: 收盘较上一档入场价 +5%
DOWN_STEP = 0.05     # 补仓触发: 收盘较上一档入场价 -5%
TRAIL     = 0.10     # 跟踪止损: 从持仓最高价回撤 10% 离场(纪律型方案用)
HARD_DOWN = 0.45     # 补仓方案硬止损: 跌破入场价 55% 才离场(=死扛, 无移动止损)
MAX_HOLD  = 60       # 封顶持仓交易日
L_DEF, VOL_MULT_DEF = 60, 1.5

# stop 字段: 'trail'=纪律型移动止损(金字塔/倒金字塔/基准); 'hard'=死扛深止损(补仓)
SCHEMES = {
    "SINGLE":    dict(sizes=[1.00],                 mode="single", stop="trail", label="一次性建仓(基准, 移动止损)"),
    "PYRAMID":   dict(sizes=[0.40,0.30,0.20,0.10], mode="up",     stop="trail", label="正金字塔40-30-20-10(盈利加+移动止损)"),
    "INVERTED":  dict(sizes=[0.10,0.20,0.30,0.40], mode="up",     stop="trail", label="倒金字塔10-20-30-40(盈利加+移动止损)"),
    "MARTINGALE":dict(sizes=[0.40,0.30,0.20,0.10], mode="down",   stop="hard",  label="补仓/越跌越补40-30-20-10(亏损加+死扛)"),
}


def simulate_one(o, c, h, l, dts, li, scheme, C):
    """对单只股票的单个信号, 模拟一种加仓方案, 返回该笔净收益(或 None)。
    时间约定: T-1 收盘判定触发/止损, T 开盘执行(无未来函数)。
    止损纪律: 纪律型('trail')用移动止损; 补仓型('hard')用死扛深止损(对应视频"无止损死扛")。"""
    n = len(o)
    k0 = li + 1                       # 入场执行日(T+1 开盘)
    if k0 >= n - 1:
        return None
    entry_open = o[k0]
    if not np.isfinite(entry_open) or entry_open <= 0:
        return None
    sizes = scheme["sizes"]; mode = scheme["mode"]; stop = scheme["stop"]
    deployed = [C * sizes[0]]
    tr_entry = [entry_open]
    last_ref = entry_open
    n_tr = 1
    rolling_high = h[k0]
    end_k = min(k0 + MAX_HOLD, n - 1)
    exit_open = None; exit_k = None; reason = None
    for k in range(k0 + 1, end_k + 1):
        prev_c = c[k - 1]
        prev_h = h[k - 1]
        if np.isfinite(prev_h):
            rolling_high = max(rolling_high, prev_h)
        # 1) 止损判定
        if stop == "trail":
            stop_level = max(tr_entry[0], rolling_high * (1.0 - TRAIL))
            if np.isfinite(prev_c) and prev_c < stop_level:
                exit_open = o[k]; exit_k = k; reason = "trail"
                break
        else:  # hard: 死扛, 仅跌破 HARD_DOWN 才离场
            if np.isfinite(prev_c) and prev_c < entry_open * HARD_DOWN:
                exit_open = o[k]; exit_k = k; reason = "hard"
                break
        # 2) 再加仓(仅在非一次性方案且未满档)
        if mode != "single" and n_tr < len(sizes) and np.isfinite(prev_c):
            if mode == "up" and prev_c >= last_ref * (1.0 + UP_STEP):
                ap = o[k]
                if np.isfinite(ap) and ap > 0:
                    deployed.append(C * sizes[n_tr]); tr_entry.append(ap); last_ref = ap; n_tr += 1
            elif mode == "down" and prev_c <= last_ref * (1.0 - DOWN_STEP):
                ap = o[k]
                if np.isfinite(ap) and ap > 0:
                    deployed.append(C * sizes[n_tr]); tr_entry.append(ap); last_ref = ap; n_tr += 1
    if exit_open is None:
        exit_open = o[end_k]; exit_k = end_k; reason = "maxhold"
    # 计算净收益(逐档买入 + 一次性卖出)
    total_shares = 0.0; total_invested = 0.0
    for cap, px in zip(deployed, tr_entry):
        buy_fill = px * (1.0 + SLIPPAGE_RATE)
        comm_b = max(cap * COMMISSION_RATE, COMMISSION_MIN)
        sh = int((cap - comm_b) / buy_fill)
        if sh <= 0:
            continue
        total_shares += sh
        total_invested += sh * buy_fill + comm_b
    if total_shares <= 0 or total_invested <= 0:
        return None
    sell_fill = exit_open * (1.0 - SLIPPAGE_RATE)
    gross = total_shares * sell_fill
    comm_s = max(gross * COMMISSION_RATE, COMMISSION_MIN)
    stamp = gross * stamp_rate(int(dts[exit_k]))
    proceeds = gross - comm_s - stamp
    net = proceeds / total_invested - 1.0
    return dict(net=net, n_tr=n_tr, reason=reason, entry_date=int(dts[k0]), exit_date=int(dts[exit_k]))


def run_schemes(df_sig):
    """对全部信号跑 4 种方案, 返回 {scheme: [net, ...]}, {scheme: [(entry_date,net), ...]}。"""
    df = df_sig.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    OPEN = df["open"].values.astype(float)
    CLOSE = df["close"].values.astype(float)
    HIGH = df["high"].values.astype(float)
    LOW = df["low"].values.astype(float)
    DATES = df["trade_date"].values.astype(int)
    SIG = df["signal"].values.astype(bool)
    TS = df["ts_code"].values
    # 每只股票连续块区间
    bounds = {}; start = 0; prev = TS[0]
    for i in range(1, len(TS) + 1):
        if i == len(TS) or TS[i] != prev:
            bounds[prev] = (start, i)
            if i < len(TS):
                prev = TS[i]; start = i
    out = {name: [] for name in SCHEMES}
    out_meta = {name: [] for name in SCHEMES}
    C = 100000.0
    for ts, (s, e) in bounds.items():
        o = OPEN[s:e]; c = CLOSE[s:e]; h = HIGH[s:e]; l = LOW[s:e]; dts = DATES[s:e]
        sig = SIG[s:e]
        idxs = np.where(sig)[0]
        for li in idxs:
            for name, sc in SCHEMES.items():
                r = simulate_one(o, c, h, l, dts, li, sc, C)
                if r is not None:
                    out[name].append(r["net"])
                    out_meta[name].append((r["entry_date"], r["net"]))
    return out, out_meta


def summarize(name, nets):
    a = np.array(nets, dtype=float)
    n = len(a)
    if n == 0:
        return None
    p5 = np.percentile(a, 5); p1 = np.percentile(a, 1)
    worst1 = np.sort(a)[:max(1,n//100)]          # 最惨 1% 的平均
    return dict(
        scheme=name, n=n,
        mean_net_pct=round(a.mean()*100, 4),
        median_net_pct=round(np.median(a)*100, 4),
        win_pct=round((a > 0).mean()*100, 2),
        p5_pct=round(p5*100, 3),
        p1_pct=round(p1*100, 3),
        worst_pct=round(a.min()*100, 3),
        worst1pct_mean_pct=round(worst1.mean()*100, 3),
        pct_lt_20_pct=round((a < -0.20).mean()*100, 2),
        pct_lt_30_pct=round((a < -0.30).mean()*100, 2),
        best_pct=round(a.max()*100, 3),
    )


def serial_equity(meta, init=INIT_CAPITAL):
    """理想序列净值: 同一批信号按入场日排序, 逐笔等权复合(忽略并发/现金约束, 四方案同待遇, 仅比相对)。"""
    meta_sorted = sorted(meta, key=lambda x: x[0])
    eq = [init]; eq_dates = [meta_sorted[0][0]]
    cur = init
    for _, r in meta_sorted:
        cur *= (1.0 + r)
        eq.append(cur)
    eq = np.array(eq)
    total = eq[-1]/eq[0] - 1
    runmax = np.maximum.accumulate(eq)
    mdd = (eq/runmax - 1).min()*100
    n = len(eq_dates)
    years = (int(meta_sorted[-1][0])//10000 - int(meta_sorted[0][0])//10000) + 1
    ann = (eq[-1]/eq[0])**(1/max(years,1)) - 1
    return dict(total_pct=round(total*100,2), ann_pct=round(ann*100,2), maxdd_pct=round(mdd,2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=START)
    ap.add_argument("--end", default=END)
    ap.add_argument("--L", type=int, default=L_DEF)
    ap.add_argument("--vol-mult", type=float, default=VOL_MULT_DEF)
    ap.add_argument("--out", default=DB_OUT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print(f"[load] {args.start}~{args.end} ...")
    df, idx = load_data(args.start, args.end)
    print(f"  日线行数={len(df):,} 股票数={df['ts_code'].nunique():,}")
    df = add_base_features(df)
    print(f"[signal] L={args.L} VOL_MULT={args.vol_mult} (右侧突破, 同族 Livermore/主升浪) ...")
    df_sig = build_signal(df, idx, args.L, args.vol_mult)
    n_sig_rows = int(df_sig["signal"].sum())
    print(f"  突破信号行数={n_sig_rows:,}")

    out, out_meta = run_schemes(df_sig)
    print(f"[backtest] 各方案有效笔数: " + ", ".join(f"{k}={len(v):,}" for k,v in out.items()))

    rows = []
    seq_rows = []
    for name, sc in SCHEMES.items():
        nets = out[name]
        sm = summarize(name, nets)
        if sm is None:
            continue
        sm["label"] = sc["label"]
        rows.append(sm)
        seq = serial_equity(out_meta[name])
        seq_rows.append(dict(scheme=name, **seq))
        print(f"  {name:10s} 均净={sm['mean_net_pct']:+.4f}% 胜率={sm['win_pct']:.1f}% "
              f"最差单笔={sm['worst_pct']:+.2f}% P1={sm['p1_pct']:+.2f}% | 序列总收益={seq['total_pct']:+.1f}% 最大回撤={seq['maxdd_pct']:.1f}%")

    pdf = pd.DataFrame(rows)[["scheme","label","n","mean_net_pct","median_net_pct","win_pct",
                               "p5_pct","p1_pct","worst_pct","worst1pct_mean_pct",
                               "pct_lt_20_pct","pct_lt_30_pct","best_pct"]]
    sdf = pd.DataFrame(seq_rows)
    pdf.to_csv(os.path.join(args.out, "pyramid_compare.csv"), index=False, encoding="utf-8-sig")
    sdf.to_csv(os.path.join(args.out, "pyramid_serial.csv"), index=False, encoding="utf-8-sig")
    # 逐笔净收益(供分布分析)
    allnets = []
    for name, ns in out.items():
        for v in ns:
            allnets.append((name, round(v*100, 4)))
    pd.DataFrame(allnets, columns=["scheme","net_pct"]).to_csv(
        os.path.join(args.out, "pyramid_nets.csv"), index=False, encoding="utf-8-sig")
    print(f"\n已保存 pyramid_compare.csv / pyramid_serial.csv / pyramid_nets.csv")
    print("DONE")


if __name__ == "__main__":
    main()
