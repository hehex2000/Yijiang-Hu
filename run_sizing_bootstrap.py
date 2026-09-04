# -*- coding: utf-8 -*-
"""
P0 · 仓位管理 Bootstrap 蒙特卡洛（零回测成本，复用 533 笔 round trip）

方法（对齐 docs/position_sizing_experiment_plan.md §3.1）：
  - 输入：mean_reversion 基线 533 笔平仓对 (net_pct 已含费, hold 天数)
  - 重采样：stationary/moving block bootstrap，块长 L=20，保留 2022 熊市类局部聚集
  - 对照：iid bootstrap（证明忽略聚类会低估尾部）
  - 仓位机制：组合级共享资金池，每笔占用 权益×f，并发上限 floor(1/f)
             信号到达按历史到达率(71笔/年)映射持有天数→交易序号；容量满则跳过
  - 每档 5000 路径，seed 固定

输出：data/results/position_sizing/sizing_bootstrap.csv
"""
import sqlite3
import numpy as np
import pandas as pd
import config
import os

CSV_IN = "data/results/mean_reversion_trade_expectancy/round_trips.csv"
OUT_DIR = "data/results/position_sizing"
OUT_CSV = os.path.join(OUT_DIR, "sizing_bootstrap.csv")
SEED = 20260904
N_PATHS = 5000
BLOCK_L = 20          # 块长（笔）
TRADING_DAYS_YR = 243
ARR_PER_YR = 71.0     # 组合级到达率：533笔/40只/7.5年 ≈ 1.78/只/年 → 组合 ~71/年
HORIZON_YEARS = 25.0
M = int(round(ARR_PER_YR * HORIZON_YEARS))   # 每条路径总交易笔数 ≈ 1775
GAP_DAYS = TRADING_DAYS_YR / ARR_PER_YR       # 平均相邻到达间隔(交易日) ≈ 3.42

# 8 档仓位（来自计划 §3.1）
FRACS = [1.00, 0.667, 0.393, 0.30, 0.20, 0.107, 0.078, 0.05]
LABELS = {
    1.00: "FULL", 0.667: "RISK_2PCT_NOM", 0.393: "KELLY_FULL",
    0.30: "RISK_2PCT_MEAN", 0.20: "FIXED20", 0.107: "RISK_2PCT_REAL",
    0.078: "RISK_2PCT_P99", 0.05: "QUARTER_KELLY",
}


def load_trades():
    df = pd.read_csv(CSV_IN, encoding="utf-8-sig")
    r = df["net_pct"].to_numpy(dtype=float) / 100.0   # 转小数
    h = df["hold"].to_numpy(dtype=float)
    return r, h


def block_bootstrap(r, h, n_paths, M, rng):
    """moving block bootstrap（固定块长 L），保留局部聚类。返回 (n_paths, M)。"""
    n = len(r)
    n_blocks = int(np.ceil(M / BLOCK_L))
    starts = rng.integers(0, n, size=(n_paths, n_blocks))
    offs = np.arange(BLOCK_L)[None, None, :]
    idx = (starts[:, :, None] + offs) % n          # (n_paths, n_blocks, L)
    idx = idx.reshape(n_paths, n_blocks * BLOCK_L)[:, :M]
    return r[idx], h[idx]


def iid_bootstrap(r, h, n_paths, M, rng):
    n = len(r)
    idx = rng.integers(0, n, size=(n_paths, M))
    return r[idx], h[idx]


def simulate_correct(net_mat, hold_mat, f, n_paths, M):
    cap = max(1, int(np.floor(1.0 / f + 1e-9)))
    hold_idx = np.ceil(hold_mat / GAP_DAYS).astype(np.int64)
    hold_idx = np.clip(hold_idx, 1, None)
    terminals = np.empty(n_paths)
    mdds = np.empty(n_paths)
    realized = np.empty(n_paths)
    min_eqs = np.empty(n_paths)
    for p in range(n_paths):
        cash = 1.0
        deployed = 0.0
        peak = 1.0          # running peak（逐时点）
        max_dd = 0.0        # 最大回撤 = min_t(equity_t / running_peak_t - 1)
        min_eq = 1.0        # 全局最低权益（占初始 1.0 的比例，回答"满仓归零"稻草人）
        cnt = 0
        # 持仓列表：每条 [exit_idx, deployed, net]（未按 exit 排序，故每步全扫描）
        q = []
        nm = net_mat[p]
        hi = hold_idx[p]
        for i in range(M):
            # 1) 关闭所有到期头寸（cap≤20，全扫描开销可忽略）
            if q:
                surv = []
                for item in q:
                    if item[0] <= i:
                        d = item[1]
                        cash += d * (1.0 + item[2])
                        deployed -= d
                        tot = cash + deployed
                        if tot > peak:
                            peak = tot
                        if tot < min_eq:
                            min_eq = tot
                        dd = tot / peak - 1.0
                        if dd < max_dd:
                            max_dd = dd
                    else:
                        surv.append(item)
                q = surv
            # 2) 开盘（容量未满则开 1 笔，占用权益×f）
            if len(q) < cap:
                d = (cash + deployed) * f
                cash -= d
                deployed += d
                q.append([i + hi[i], d, nm[i]])
                cnt += 1
        # 收尾关闭剩余
        for item in q:
            d = item[1]
            cash += d * (1.0 + item[2])
            deployed -= d
            tot = cash + deployed
            if tot > peak:
                peak = tot
            if tot < min_eq:
                min_eq = tot
            dd = tot / peak - 1.0
            if dd < max_dd:
                max_dd = dd
        terminals[p] = cash + deployed
        mdds[p] = max_dd
        realized[p] = cnt
        min_eqs[p] = min_eq
    return terminals, mdds, realized, min_eqs


def summarize(terminals, mdds, realized, min_eqs, method, f):
    cap = max(1, int(np.floor(1.0 / f + 1e-9)))
    yrs = M / ARR_PER_YR
    t = terminals
    logt = np.log(np.clip(t, 1e-9, None))
    geo = logt.mean() / M
    cagr_med = np.median(t) ** (1.0 / yrs) - 1.0
    return {
        "method": method,
        "label": LABELS.get(f, f"{f:.3f}"),
        "f": f,
        "cap": cap,
        "trades_realized_mean": realized.mean(),
        "terminal_p5": np.percentile(t, 5),
        "terminal_p50": np.percentile(t, 50),
        "terminal_p95": np.percentile(t, 95),
        "terminal_mean": t.mean(),
        "ruin_50pct": (t < 0.5).mean(),
        "ruin_20pct": (t < 0.2).mean(),
        "mdd_p50": np.percentile(mdds, 50),
        "mdd_p95": np.percentile(mdds, 95),
        "worst_eq_p5": np.percentile(min_eqs, 5),
        "worst_eq_p50": np.percentile(min_eqs, 50),
        "geo_growth_per_trade": geo,
        "cagr_p50": cagr_med,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    r, h = load_trades()
    print(f"[数据] {len(r)} 笔；均值净收益 {r.mean()*100:.3f}% 胜率 {(r>0).mean()*100:.2f}%")
    print(f"[设置] N_PATHS={N_PATHS} M={M}(≈{HORIZON_YEARS}年) 块长={BLOCK_L} GAP={GAP_DAYS:.2f}日/笔")
    rows = []
    for method, fn in [("block", block_bootstrap), ("iid", iid_bootstrap)]:
        rng = np.random.default_rng(SEED + (0 if method == "block" else 1))
        nm, hm = fn(r, h, N_PATHS, M, rng)
        for f in FRACS:
            term, mdd, real, meq = simulate_correct(nm, hm, f, N_PATHS, M)
            s = summarize(term, mdd, real, meq, method, f)
            rows.append(s)
            print(f"  [{method}] f={f:.3f} {s['label']:>14s} 终值中位={s['terminal_p50']:.3f}x "
                  f"破产50%={s['ruin_50pct']*100:.2f}% 破产20%={s['ruin_20pct']*100:.2f}% "
                  f"MDD_P95={s['mdd_p95']*100:.1f}% 最差权益P5={s['worst_eq_p5']*100:.1f}% "
                  f"实现笔数={s['trades_realized_mean']:.0f}")
    out = pd.DataFrame(rows)
    # 调整列顺序
    cols = ["method", "label", "f", "cap", "trades_realized_mean", "terminal_p5",
            "terminal_p50", "terminal_p95", "terminal_mean", "ruin_50pct",
            "ruin_20pct", "mdd_p50", "mdd_p95", "worst_eq_p5", "worst_eq_p50",
            "geo_growth_per_trade", "cagr_p50"]
    out = out[cols]
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n已保存: {OUT_CSV}")
    # 控制台对照表（block 方法）
    print("\n=== block 方法 · 终值中位(x) / 破产概率 / MDD ===")
    blk = out[out.method == "block"]
    print(blk[["label", "f", "cap", "trades_realized_mean", "terminal_p50",
               "ruin_50pct", "ruin_20pct", "mdd_p95", "worst_eq_p5",
               "cagr_p50"]].to_string(index=False))


if __name__ == "__main__":
    main()
