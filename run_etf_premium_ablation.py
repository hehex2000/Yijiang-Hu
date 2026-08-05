# -*- coding: utf-8 -*-
"""
ETF 折溢价过滤 —— 控制变量对比回测
====================================
回答一个问题：把折溢价过滤接进 ETF 轮动，到底是帮忙还是添乱？

对比 4 组（其余参数完全一致）：
  A off      基准，不过滤
  B uniform  统一硬阈值 5%
  C strict   跨境更严（3%/5%）—— 即 etf_premium_filter 的原始设定
  D qdii     仅对申赎受限品种(501018/513100)用 8%

跨 3 个区间 × 2 种 top_n，避免单一样本期的偶然性。
"""
import sys, os, io, csv, contextlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_etf_rotation as rot

MODES = ["off", "uniform", "qdii", "rolling"]
PERIODS = [
    ("20180101", "20260803", "全区间 2018-2026"),
    ("20180101", "20211231", "前段 2018-2021"),
    ("20220101", "20260803", "后段 2022-2026"),
]
TOP_NS = [2, 3]
OUT_DIR = os.path.join("data", "results", "etf_premium_ablation")


def run_one(start, end, mode, top_n):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = rot.run_etf_rotation(
            start_date=start, end_date=end, method="dual",
            roc_period=20, ma_period=60, top_n=top_n,
            verbose=False, premium_filter=mode)
    return r


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    print("=" * 100)
    print("  ETF 折溢价过滤 · 控制变量对比")
    print("=" * 100)

    for start, end, label in PERIODS:
        for top_n in TOP_NS:
            print(f"\n【{label} | top_n={top_n}】")
            print(f"  {'模式':<10}{'总收益':>10}{'年化':>9}{'最大回撤':>10}"
                  f"{'夏普':>8}{'交易':>7}{'拦截':>7}  vs基准")
            print("  " + "-" * 78)
            base_ret = None
            for mode in MODES:
                r = run_one(start, end, mode, top_n)
                if not r:
                    print(f"  {mode:<10} 回测失败")
                    continue
                if mode == "off":
                    base_ret = r["total_return"]
                delta = (r["total_return"] - base_ret) if base_ret is not None else 0.0
                nblk = len(r.get("premium_blocked") or [])
                mark = "" if mode == "off" else f"{delta:+.2f}pp"
                print(f"  {mode:<10}{r['total_return']:>9.2f}%{r['annual_return']:>8.2f}%"
                      f"{r['max_drawdown']:>9.2f}%{r['sharpe']:>8.2f}{r['trades']:>7}"
                      f"{nblk:>7}  {mark}")
                rows.append({
                    "period": label, "start": start, "end": end, "top_n": top_n,
                    "mode": mode, "total_return": round(r["total_return"], 4),
                    "annual_return": round(r["annual_return"], 4),
                    "max_drawdown": round(r["max_drawdown"], 4),
                    "sharpe": round(r["sharpe"], 4), "trades": r["trades"],
                    "blocked": nblk, "delta_vs_off": round(delta, 4),
                })

    # ── 汇总：各模式相对基准的平均增益 ──
    print("\n" + "=" * 100)
    print("  汇总：各模式相对基准(off)的平均增益")
    print("=" * 100)
    print(f"  {'模式':<10}{'场景数':>8}{'平均Δ收益':>12}{'胜出场景':>10}{'平均拦截次数':>14}")
    print("  " + "-" * 60)
    for mode in MODES:
        sub = [r for r in rows if r["mode"] == mode]
        if not sub:
            continue
        avg = sum(r["delta_vs_off"] for r in sub) / len(sub)
        wins = sum(1 for r in sub if r["delta_vs_off"] > 0)
        avgblk = sum(r["blocked"] for r in sub) / len(sub)
        print(f"  {mode:<10}{len(sub):>8}{avg:>11.2f}pp{wins:>7}/{len(sub)}{avgblk:>14.1f}")

    # ── 落盘 ──
    csv_path = os.path.join(OUT_DIR, "premium_ablation_compare.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  明细已保存：{csv_path}")

    # ── 拦截明细（全区间 top_n=2，各模式）──
    print("\n" + "=" * 100)
    print("  拦截明细（全区间 2018-2026, top_n=2）")
    print("=" * 100)
    for mode in ["uniform", "qdii", "rolling"]:
        r = run_one("20180101", "20260803", mode, 2)
        blk = r.get("premium_blocked") or []
        print(f"\n  【{mode}】共拦截 {len(blk)} 次")
        for d, code, name, prem in blk:
            print(f"    {d}  {name}({code})  溢价 {prem:+.2%}")


if __name__ == "__main__":
    main()
