# -*- coding: utf-8 -*-
"""P1① 翻倍抽本 vs 持有 vs 移动止盈 —— 「锁定利润」与「让利润奔跑」的内在矛盾。

来源：B站 BV1G7466MEPN（悦悦笔记）五法之①「翻倍抽本」。
P0 已证：翻倍抽本 ≡ 涨 100% 减仓 50%（再涨 25% 减仓 20%），成本价不进入任何收益式。
本脚本在 P0 之上补两件事（P1① 的增量价值）：
    1. 引入「移动止盈」作为**诚实的**替代方案（视频只给了抽本，没给对照）
    2. 度量**最大回撤**（视频吹"零成本奔跑"，但回避了回撤——情绪管理才是它真正的价值）

三路径在**同一价格轨迹**上并行模拟（共享 px，零新增数据）：
    HOLD   纯持有 1 股到终点
    TRIM   涨 100% 卖 50%；再涨 25% 卖 20%（视频原规则）
    TRAIL  从峰值回撤 trail% 清仓（默认 20%，移动止盈）

口径（与 P0 一致）：
    - hfq 后复权为主；raw 不复权为对照（双口径必查）
    - 成本 RT_COST=0.3% 仅卖出侧计
    - 取数复用 run_daily20_macd.load_closes

用法：
    python analyze_negative_cost_p1a.py --probe        # 快跑验证
    python analyze_negative_cost_p1a.py --step 20       # 全样本
    python analyze_negative_cost_p1a.py --step 20 --compare   # 追加 raw 对照
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from run_daily20_macd import load_closes

RT_COST = 0.003
HORIZON = 250
STEP = 20
OUT_DIR = os.path.join("data", "results", "negative_cost")


def sim_three(px, k, cost=RT_COST, mult1=2.0, sell1=0.50,
              mult2=2.5, sell2=0.20, trail=0.20):
    """同一价格轨迹上并行模拟 HOLD / TRIM / TRAIL，返回终点财富与最大回撤。

    回撤用流式峰谷法（不存整条轨迹，省内存；280k 事件 × 250 步可承受）。
    返回 dict：三条路径终点财富（相对初始投入 k 的净值）与各自最大回撤(负比例)。
    """
    t_shares = 1.0
    t_cash = 0.0
    t_stage = 0
    tr_shares = 1.0
    tr_cash = 0.0
    tr_peak = px[0] if len(px) and np.isfinite(px[0]) else k
    tr_sold = False

    h_peak = h_w = -1e18
    h_mdd = 0.0
    t_peak = t_w = -1e18
    t_mdd = 0.0
    tr_peakv = tr_w = -1e18
    tr_mdd = 0.0

    for p in px:
        if not np.isfinite(p) or p <= 0:
            continue
        # ── HOLD ──
        h_w = p
        if h_w > h_peak:
            h_peak = h_w
        d = (h_w - h_peak) / h_peak
        if d < h_mdd:
            h_mdd = d
        # ── TRIM（视频规则）──
        if t_stage == 0 and p >= mult1 * k:
            m = t_shares * sell1
            t_cash += m * p * (1 - cost)
            t_shares -= m
            t_stage = 1
        elif t_stage == 1 and p >= mult2 * k:
            m = t_shares * sell2
            t_cash += m * p * (1 - cost)
            t_shares -= m
            t_stage = 2
        t_w = t_shares * p + t_cash
        if t_w > t_peak:
            t_peak = t_w
        d = (t_w - t_peak) / t_peak
        if d < t_mdd:
            t_mdd = d
        # ── TRAIL（移动止盈）──
        if not tr_sold:
            if p > tr_peak:
                tr_peak = p
            elif p <= tr_peak * (1 - trail):
                m = tr_shares
                tr_cash += m * p * (1 - cost)
                tr_shares = 0.0
                tr_sold = True
        tr_w = tr_shares * p + tr_cash
        if tr_w > tr_peakv:
            tr_peakv = tr_w
        d = (tr_w - tr_peakv) / tr_peakv
        if d < tr_mdd:
            tr_mdd = d

    p_end = px[-1] if np.isfinite(px[-1]) else np.nan
    return {
        "hold": p_end,
        "trim": t_shares * p_end + t_cash,
        "trail": tr_shares * p_end + tr_cash,
        "dd_hold": h_mdd,
        "dd_trim": t_mdd,
        "dd_trail": tr_mdd,
    }


def run_events(closes, codes, horizon=HORIZON, step=STEP, probe=False,
               trail=0.20):
    rows = []
    n_codes = 40 if probe else len(codes)
    for ci, ts in enumerate(codes[:n_codes]):
        s = closes[ts]
        if s is None:
            continue
        s = s.dropna()
        px = s.values
        dates = s.index.values
        T = len(px)
        if T < horizon + 5:
            continue
        for i in range(0, T - horizon - 1, step):
            k = px[i]
            if not np.isfinite(k) or k <= 0:
                continue
            seg = px[i + 1: i + 1 + horizon]
            if len(seg) < horizon or not np.isfinite(seg[-1]):
                continue
            r = sim_three(seg, k, trail=trail)
            if not (np.isfinite(r["hold"]) and np.isfinite(r["trim"]) and np.isfinite(r["trail"])):
                continue
            rows.append({
                "ts_code": ts,
                "k": k,
                "ret_hold": r["hold"] / k - 1.0,
                "ret_trim": r["trim"] / k - 1.0,
                "ret_trail": r["trail"] / k - 1.0,
                "dd_hold": r["dd_hold"],
                "dd_trim": r["dd_trim"],
                "dd_trail": r["dd_trail"],
                # 标记 TRIM 是否触发过减仓（用于自证组）
                "trim_triggered": (r["trim"] != r["hold"]) or True,  # 占位，下方按收益差重判
            })
        if probe and ci >= 39:
            break
    return pd.DataFrame(rows)


def _winsor(x, p=0.01):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return x
    return np.clip(x, *np.quantile(x, [p, 1 - p]))


def _fmt(df, cols, nd=2):
    d = df[cols].copy()
    for c in cols:
        if c in ("分组", "路径", "口径"):   # 字符串标签列，跳过数值格式化
            continue
        d[c] = pd.to_numeric(d[c], errors="coerce").map(
            lambda v: ("%.*f" % (nd, v)) if np.isfinite(v) else "nan")
    return d.to_string(index=False)


def bucket_table(ev, title=""):
    buckets = [(None, 0.0, "后续下跌 (≤0)"),
               (0.0, 1.0, "小涨 (0~100%)"),
               (1.0, 3.0, "大涨 (100~300%)"),
               (3.0, None, "暴涨 (>300%)")]
    rows = []
    for lo, hi, lab in buckets:
        m = (ev["ret_hold"] > lo) if lo is not None else (ev["ret_hold"] <= 0)
        if hi is not None:
            m = m & (ev["ret_hold"] <= hi)
        sub = ev[m]
        if len(sub) < 5:
            continue
        d_t = sub["ret_trim"] - sub["ret_hold"]
        d_r = sub["ret_trail"] - sub["ret_hold"]
        rows.append({
            "分组": lab,
            "样本数": len(sub),
            "持有(中位%)": np.median(sub["ret_hold"]) * 100,
            "抽本-持有(中位%)": np.median(d_t) * 100,
            "移动止盈-持有(中位%)": np.median(d_r) * 100,
            "抽本未跑输持有(%)": (d_t >= 0).mean() * 100,
            "移动止盈未跑输持有(%)": (d_r >= 0).mean() * 100,
        })
    if rows:
        if title:
            print(title)
        print(_fmt(pd.DataFrame(rows),
                   ["分组", "样本数", "持有(中位%)", "抽本-持有(中位%)",
                    "移动止盈-持有(中位%)", "抽本未跑输持有(%)", "移动止盈未跑输持有(%)"]))
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=STEP)
    ap.add_argument("--horizon", type=int, default=HORIZON)
    ap.add_argument("--trail", type=float, default=0.20, help="移动止盈回撤阈值")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()

    t_all = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 72)
    print("P1① 翻倍抽本 vs 持有 vs 移动止盈（trail=%.0f%%）" % (args.trail * 100))
    print("=" * 72)

    print("\n[load] zz800 历史成分收盘（hfq）...", flush=True)
    t0 = time.time()
    codes, closes = load_closes(hfq=True)
    print("       %d 只，耗时 %.1fs" % (len(codes), time.time() - t0), flush=True)

    t0 = time.time()
    df = run_events(closes, codes, horizon=args.horizon, step=args.step,
                    probe=args.probe, trail=args.trail)
    print("       事件数 %d，耗时 %.1fs" % (len(df), time.time() - t0), flush=True)
    if df.empty:
        print("无有效事件")
        return

    # 自证：TRIM 在「从未翻倍」组必须 ≡ HOLD（收益差与回撤差均 0）
    print("\n── 自证：同一轨迹上 TRIM 与 HOLD 在减仓未发生时应完全一致 ──")
    never = df[(df["ret_trim"] - df["ret_hold"]).abs() < 1e-9]
    print("  抽本从未触发（收益差≈0）的样本 %d 条（占 %.1f%%）"
          % (len(never), 100 * len(never) / len(df)))
    if len(never):
        dd_diff = (never["dd_trim"] - never["dd_hold"]).abs().max()
        print("  这些样本上 回撤差 最大绝对值 = %.2e → %s"
              % (dd_diff, "通过" if dd_diff < 1e-12 else "异常！"))

    # 总体：三路径终值与回撤
    print("\n── 总体：三路径终值收益与最大回撤 ──")
    summ = []
    for name, col, dd in [("HOLD 纯持有", "ret_hold", "dd_hold"),
                          ("TRIM 翻倍抽本", "ret_trim", "dd_trim"),
                          ("TRAIL 移动止盈", "ret_trail", "dd_trail")]:
        r = df[col]
        ddcol = df[dd]
        summ.append({
            "路径": name,
            "收益_中位(%)": np.median(r) * 100,
            "收益_缩尾均值(%)": _winsor(r.values).mean() * 100,
            "跑赢HOLD比例(%)": (r > df["ret_hold"]).mean() * 100,
            "最大回撤_中位(%)": np.median(ddcol) * 100,
            "最大回撤_最差(%)": ddcol.min() * 100,
        })
    print(_fmt(pd.DataFrame(summ),
               ["路径", "收益_中位(%)", "收益_缩尾均值(%)", "跑赢HOLD比例(%)",
                "最大回撤_中位(%)", "最大回撤_最差(%)"]))
    print("\n  → 若 TRIM/TRAIL 的「收益_中位」低于 HOLD 而「最大回撤」更小，")
    print("    即证它们是【回撤管理工具】，不是【收益增强工具】。")

    # 核心：按后续走势分桶，看三路径相对表现
    print("\n── 核心：按后续走势分桶（同一轨迹三路径对照）──")
    bucket_table(df, "【全部事件】")

    # 回撤维度单独论证「情绪管理价值」
    print("\n── 回撤维度：TRIM/TRAIL 是否真降低了最大回撤 ──")
    print("  （回撤为负值；「幅度更低」= 更接近 0 = dd_trim/dd_trail > dd_hold）")
    dd_t = df["dd_trim"] - df["dd_hold"]          # >0 表示 TRIM 回撤幅度更低（更好）
    dd_r = df["dd_trail"] - df["dd_hold"]
    print("  TRIM  回撤幅度更低（中位）%.2f pp；占比 %.1f%% 事件回撤更小"
          % (np.median(dd_t) * 100, (dd_t > 0).mean() * 100))
    print("  TRAIL 回撤幅度更低（中位）%.2f pp；占比 %.1f%% 事件回撤更小"
          % (np.median(dd_r) * 100, (dd_r > 0).mean() * 100))

    # 内在矛盾量化：在牛市（ret_hold 大）里，TRIM 牺牲了多少上涨？
    print("\n── 「锁定利润 vs 让利润奔跑」内在矛盾 ──")
    print("  理论：减仓/止盈在高位锁利 → 后续越涨越吃亏；下跌时占便宜。")
    bull = df[df["ret_hold"] > 1.0]  # 大涨以上
    if len(bull) >= 20:
        print("  大涨组（持有收益>100%%，n=%d）：" % len(bull))
        print("    抽本相对持有 中位 %.2f%%，未跑输比例 %.1f%%"
              % ((bull["ret_trim"] - bull["ret_hold"]).median() * 100,
                 (bull["ret_trim"] >= bull["ret_hold"]).mean() * 100))
        print("    移动止盈相对持有 中位 %.2f%%，未跑输比例 %.1f%%"
              % ((bull["ret_trail"] - bull["ret_hold"]).median() * 100,
                 (bull["ret_trail"] >= bull["ret_hold"]).mean() * 100))

    # 双口径对照
    if args.compare:
        print("\n── raw / hfq 双口径对照 ──")
        _, closes_r = load_closes(hfq=False)
        df_r = run_events(closes_r, codes, horizon=args.horizon, step=args.step,
                          probe=args.probe, trail=args.trail)
        sr = []
        for lab, d_ in [("hfq（主）", df), ("raw（对照）", df_r)]:
            r = d_["ret_trim"]
            sr.append({
                "口径": lab,
                "事件数": len(d_),
                "抽本收益_中位(%)": np.median(r) * 100,
                "抽本收益_缩尾均值(%)": _winsor(r.values).mean() * 100,
                "抽本跑赢HOLD(%)": (r > d_["ret_hold"]).mean() * 100,
                "抽本最大回撤_中位(%)": np.median(d_["dd_trim"]) * 100,
            })
        print(_fmt(pd.DataFrame(sr),
                   ["口径", "事件数", "抽本收益_中位(%)", "抽本收益_缩尾均值(%)",
                    "抽本跑赢HOLD(%)", "抽本最大回撤_中位(%)"]))

    stamp = time.strftime("%Y%m%d")
    out = os.path.join(OUT_DIR, "p1a_events_%s%s.csv" % (stamp, "_probe" if args.probe else ""))
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print("\n逐事件明细已写入：%s" % out)
    print("总耗时 %.1fs" % (time.time() - t_all))


if __name__ == "__main__":
    main()
