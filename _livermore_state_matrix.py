# -*- coding: utf-8 -*-
"""
② 利弗莫尔策略 状态转移矩阵 + churn 成本量化
=============================================
测度论视频待办②：把 livermore 策略的"进出场状态"建成 PMF→迁移矩阵，
定量暴露 churn（频繁进出）成本——这是该策略的核心亏损源。

方法：
  1. 跑 run_livermore_v2.run_window（box 压缩验证版，对齐 alpha 研究窗口 2018-2025）。
  2. 从 trade_records(entry/exit 日期) 重建每日持仓数时间线 → 日级状态 {OUT, IN}。
  3. 算 2x2 转移矩阵 + 持仓数 PMF + 平均持有天数 + 换手次数 + 胜率。
  4. 用"关费用"(calc_fee 置 0) 重跑一次，两次总收益差 = churn 成本(pp)。
     —— 决策逻辑不依赖费用，故两次交易序列完全一致，差值即纯交易成本拖累。

不修改 run_livermore_v2.py 源码：状态由 trade_records 重建，费用开关用 monkeypatch。
"""
import sys
import os
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_livermore_v2 as lv
import run_monthly_rebalance as mr

START = "20180101"
END = "20251231"

# box 压缩验证版（与 alpha 研究 / ⑤ 同口径）
CFG = dict(
    lookback=20, sector_top_pct=0.30, exit_pct=0.03, ma_period=20,
    stop_loss=0.05, max_hold=5, market_exit=True, confirm_days=2,
    trailing_stop=0.12, fail_exit_days=2, sector_mode="momentum",
    vol_size=False, vol_skip=False, vol_lookback=20,
    box_len=30, box_width=0.10, risk_per=0.02, vol_win=5,
)
LAYERS = dict(market=True, sector=True, exit=True)


def run_once(fee_on=True):
    """跑一次 run_window；fee_on=False 时关闭所有交易成本。"""
    orig = lv.calc_fee
    if not fee_on:
        lv.calc_fee = lambda *a, **k: 0.0
    try:
        r = lv.run_window(START, END, CFG, LAYERS)
    finally:
        lv.calc_fee = orig
    return r


def daily_position_counts(r):
    """从 trade_records 重建每日(收盘)持仓数时间线。
    每条记录: entry_date +1, exit_date -1；按日累加得到每日持仓数。
    开仓当日先卖后买，净变化正确；窗口末仍持仓者无 exit 记录→计数保持正(正确)。"""
    nav = r["nav"]
    dates = [d for d, _ in nav]
    trades = r.get("trade_records", [])
    delta = defaultdict(int)
    for t in trades:
        delta[t["entry_date"]] += 1
        delta[t["exit_date"]] -= 1
    counts = []
    cur = 0
    for d in dates:
        cur += delta.get(d, 0)
        counts.append(max(cur, 0))   # 防御性 clip（理论上不会为负）
    return dates, counts


def transition_matrix(dates, counts):
    """日级 2 态 {OUT(0), IN(>=1)} 转移矩阵，返回 (matrix, row_labels, counts_dict)。"""
    states = ["OUT" if c == 0 else "IN" for c in counts]
    labels = ["OUT", "IN"]
    mat = np.zeros((2, 2), dtype=float)
    for a, b in zip(states[:-1], states[1:]):
        mat[labels.index(a), labels.index(b)] += 1
    row_sums = mat.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        prob = np.where(row_sums > 0, mat / row_sums, 0.0)
    return prob, labels, mat.astype(int)


def position_pmf(counts, max_hold=5):
    """持仓数分布(PMF)。"""
    from collections import Counter
    c = Counter(counts)
    keys = list(range(max_hold + 1))
    total = len(counts)
    return {k: c.get(k, 0) / total for k in keys}, c


def churn_stats(r):
    """换手 / 持有 / 胜率统计。"""
    trades = r.get("trade_records", [])
    n = len(trades)
    if n == 0:
        return dict(n_trades=0)
    holds = [t["hold_days"] for t in trades]
    rets = [t["ret"] for t in trades]
    wins = [x for x in rets if x > 0]
    losses = [x for x in rets if x <= 0]
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    exits = Counter(t["exit_type"] for t in trades)
    return dict(
        n_trades=n,
        win_rate=len(wins) / n,
        avg_hold=np.mean(holds),
        avg_win=avg_win, avg_loss=avg_loss,
        profit_factor=(abs(avg_win * len(wins) / (avg_loss * len(losses)))
                       if (losses and avg_loss != 0) else float("inf")),
        exits=dict(exits),
    )


def main():
    print(f"[1/3] 跑 livermore 回测(开费用) {START}-{END} ...")
    r_on = run_once(fee_on=True)
    if r_on is None:
        print("  [跳过] 回测返回 None（交易日不足或无成分数据）")
        return
    dates, counts = daily_position_counts(r_on)
    prob, labels, raw = transition_matrix(dates, counts)
    pmf, _ = position_pmf(counts, CFG["max_hold"])
    cs = churn_stats(r_on)
    m_on = r_on["m"]

    print(f"[2/3] 跑 livermore 回测(关费用, 量化 churn) ...")
    r_off = run_once(fee_on=False)
    m_off = r_off["m"] if r_off else None

    churn_pp = (m_off["total"] - m_on["total"]) * 100 if m_off else float("nan")
    churn_of_nav = churn_pp  # pp
    # churn 占策略总收益比
    churn_share = (churn_pp / (m_on["total"] * 100)) if m_on["total"] != 0 else float("nan")

    # ── 打印 ──
    print("\n" + "=" * 64)
    print("利弗莫尔策略 状态转移矩阵（日级 {OUT, IN}）")
    print(f"窗口 {START}~{END} | 配置 box_len=30/box_width=0.10")
    print("=" * 64)
    print(f"  交易日数: {len(dates)} | 总交易笔数: {cs.get('n_trades')} | 总收益(开费): {m_on['total']*100:.2f}%")
    print(f"  平均持有天数: {cs.get('avg_hold',0):.1f} | 胜率: {cs.get('win_rate',0)*100:.1f}%")
    print(f"  盈利/亏损 均值: {cs.get('avg_win',0)*100:.2f}% / {cs.get('avg_loss',0)*100:.2f}%")
    print("\n  转移矩阵(行=昨日态, 列=今日态, 数值=概率):")
    hdr = "        " + "  ".join(f"{l:>6}" for l in labels)
    print(hdr)
    for i, l in enumerate(labels):
        row = "  ".join(f"{prob[i,j]*100:5.1f}%" for j in range(len(labels)))
        print(f"  {l:>4}  {row}")
    print(f"\n  原始转移计数:\n  OUT→OUT={raw[0,0]}  OUT→IN={raw[0,1]}  IN→OUT={raw[1,0]}  IN→IN={raw[1,1]}")
    p_in_out = prob[1, 0] * 100
    p_out_in = prob[0, 1] * 100
    print(f"  出场率 P(IN→OUT)={p_in_out:.1f}%  → 平均持仓 {1/(p_in_out/100):.1f} 日" if p_in_out > 0 else "  (无 IN→OUT)")
    print(f"  入场率 P(OUT→IN)={p_out_in:.1f}%  → 平均空仓 {1/(p_out_in/100):.1f} 日" if p_out_in > 0 else "  (无 OUT→IN)")
    print("\n  持仓数 PMF:")
    for k in sorted(pmf):
        bar = "#" * int(pmf[k] * 60)
        print(f"    {k} 只: {pmf[k]*100:5.1f}% {bar}")

    print("\n  ── churn 成本量化（开费用 vs 关费用）──")
    print(f"  总收益(开费): {m_on['total']*100:.2f}%")
    print(f"  总收益(关费): {m_off['total']*100:.2f}%  (假设决策不受费影响, 交易序列一致)")
    print(f"  churn 成本(拖累): {churn_pp:+.2f} pp")
    print(f"  占策略总收益比: {churn_share*100:+.1f}%")
    if cs.get("n_trades"):
        # 单边成本率（按 max_hold 等权估算单笔名义）
        notional = 1_000_000.0 / CFG["max_hold"]
        one_way = mr.COMMISSION_RATE + mr.STAMP_DUTY_RATE + mr.SLIPPAGE_RATE
        est_one_trade_cost = notional * one_way * 2  # 买+卖
        est_total = est_one_trade_cost * cs["n_trades"]
        print(f"  单边成本率(估): {one_way*100:.3f}% | 单笔名义≈{notional/10000:.0f}万 | 估算 churn≈{est_total/10000:.1f}万")

    # ── 落盘 CSV ──
    out_dir = "data/results/livermore"
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "livermore_state_matrix.csv")
    rows = []
    rows.append(["metric", "value"])
    rows.append(["window", f"{START}~{END}"])
    rows.append(["n_days", len(dates)])
    rows.append(["n_trades", cs.get("n_trades")])
    rows.append(["total_ret_fee_on_pct", round(m_on["total"] * 100, 2)])
    rows.append(["total_ret_fee_off_pct", round(m_off["total"] * 100, 2) if m_off else "NA"])
    rows.append(["churn_cost_pp", round(churn_pp, 2)])
    rows.append(["churn_share_of_ret_pct", round(churn_share * 100, 1) if m_on["total"] else "NA"])
    rows.append(["win_rate_pct", round(cs.get("win_rate", 0) * 100, 1)])
    rows.append(["avg_hold_days", round(cs.get("avg_hold", 0), 1)])
    rows.append(["P_OUT_TO_IN_pct", round(p_out_in, 1)])
    rows.append(["P_IN_TO_OUT_pct", round(p_in_out, 1)])
    rows.append(["avg_cash_days", round(1 / (p_out_in / 100), 1) if p_out_in > 0 else "NA"])
    rows.append(["avg_hold_span_days", round(1 / (p_in_out / 100), 1) if p_in_out > 0 else "NA"])
    for k in sorted(pmf):
        rows.append([f"pos_pmf_{k}", round(pmf[k] * 100, 2)])
    for i, l in enumerate(labels):
        for j, l2 in enumerate(labels):
            rows.append([f"trans_{l}_{l2}_pct", round(prob[i, j] * 100, 2)])
    with open(out, "w", encoding="utf-8", newline="") as f:
        for row in rows:
            f.write(",".join(str(x) for x in row) + "\n")
    print(f"\n  已保存: {out}")

    # ── 判定 ──
    # churn_pp = (关费总收益 - 开费总收益)*100，恒 >=0；越大=费用拖累越重。
    gross_ret = (m_off["total"] * 100) if m_off else float("nan")
    fee_share_of_gross = (churn_pp / gross_ret) if gross_ret else float("nan")
    print("\n--- 判定：churn 是否核心亏损源 ---")
    if churn_pp >= 3:
        verdict = (f"✅ churn 成本 {churn_pp:+.1f}pp（吞噬毛收益约 {fee_share_of_gross*100:.0f}%、"
                   f"≈净收益的 {churn_share*100:.0f}%），平均持仓仅 {cs.get('avg_hold',0):.1f} 日、"
                   f"{cs.get('n_trades')} 笔换手 → 频繁进出是核心亏损源")
    elif churn_pp >= 0:
        verdict = f"⚠️ churn 成本 {churn_pp:+.1f}pp（轻微），非主导亏损源"
    else:
        verdict = f"❌ 关费后收益反而更低（异常，疑似决策受费影响或数值问题）"
    print(f"  {verdict}")


if __name__ == "__main__":
    main()
