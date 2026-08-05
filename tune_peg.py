"""
PEG 最优版(质量+动量) 调参对比驾驶舱  ——  进程内顺序版
====================================================
直接 import run_peg 并在同一进程里顺序调用 run_backtest()，
避免子进程被环境 detach 导致并发抢库/卡死。

变体网格（在已知最优版 ⑦ 基础上）：
  - 动量窗口：6 / 12 / 24 月
  - 质量阈值：ROE>=8债<=70 / ROE>=10债<=60 / ROE>=12债<=50
  - VaR(95%) 风控兜底：最优版上叠加 (cap 2.5% / 1.5%)；以及“最优因子+风控”

用法：
  venv_ml/Scripts/python.exe tune_peg.py
"""
import os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_peg as RP

START, END = "20140101", "20260715"

# (标签, 参数dict) —— 仅传 run_backtest 支持的叠加层参数
VARIANTS = [
    ("⑦基准 动量12·质量8/70",          dict(momentum_months=12, min_roe=8,  max_debt=70)),
    ("动量6·质量8/70",                 dict(momentum_months=6,  min_roe=8,  max_debt=70)),
    ("动量24·质量8/70",                dict(momentum_months=24, min_roe=8,  max_debt=70)),
    ("动量12·质量10/60",               dict(momentum_months=12, min_roe=10, max_debt=60)),
    ("动量12·质量12/50",               dict(momentum_months=12, min_roe=12, max_debt=50)),
    ("动量6·质量10/60",                dict(momentum_months=6,  min_roe=10, max_debt=60)),
    ("动量24·质量10/60",               dict(momentum_months=24, min_roe=10, max_debt=60)),
    ("⑦+VaR95(2.5%)",                 dict(momentum_months=12, min_roe=8,  max_debt=70, var_guard=True,  var_cap=0.025)),
    ("⑦+VaR95(1.5%)",                 dict(momentum_months=12, min_roe=8,  max_debt=70, var_guard=True,  var_cap=0.015)),
    ("最优因子+VaR95(2.5%)",           dict(momentum_months=24, min_roe=10, max_debt=60, var_guard=True,  var_cap=0.025)),
]


def main():
    RP._CAPITAL = 1_000_000
    rows = []
    print(f"PEG 调参对比：{len(VARIANTS)} 个变体，区间 {START}~{END}\n")
    for label, params in VARIANTS:
        RP._STATS = {"eligible": 0, "g1": 0, "g2": 0, "g3": 0, "selected": 0, "rebal": 0}
        t0 = time.time()
        res = RP.run_backtest(
            start_date=START, end_date=END, top_n=30, verbose=False,
            stock_pool="all", freq="annual", stab_years=3,
            interrupt_start=None, interrupt_months=0, interrupt_pct=0.0,
            **params,
        )
        el = round(time.time() - t0, 1)
        bench = res.get("bench", {})
        bench_ret = bench.get("000985.SH")
        excess = (res["total_return"] - bench_ret) if bench_ret is not None else None
        rows.append({
            "label": label, "total": res["total_return"], "annual": res["annual_return"],
            "mdd": res["max_drawdown"], "sharpe": res["sharpe"],
            "var_daily": res.get("var95_daily", 0) * 100,
            "cash_days": res.get("var_off_days", 0),
            "excess": excess, "elapsed": el,
        })
        print(f">>> {label}  -> 总收益 {res['total_return']:+.2f}%  年化 {res['annual_return']:+.2f}%  "
              f"最大回撤 {res['max_drawdown']:+.2f}%  夏普 {res['sharpe']:+.2f}  "
              f"VaR日度 {res.get('var95_daily',0)*100:.2f}%  现金 {res.get('var_off_days',0)}天  "
              f"(用时 {el}s)")

    print("\n" + "=" * 100)
    print("对比汇总（全历史 2014-2026，30只/年调/护栏③=3年）")
    print("=" * 100)
    print(f"{'变体':<24}{'总收益%':>9}{'年化%':>8}{'最大回撤%':>11}{'夏普':>7}{'VaR日度%':>10}{'现金天':>8}{'超额全指':>10}{'用时s':>7}")
    for r in rows:
        ex = f"{r['excess']:+.1f}" if r["excess"] is not None else "-"
        print(f"{r['label']:<24}{r['total']:>9.2f}{r['annual']:>8.2f}{r['mdd']:>11.2f}"
              f"{r['sharpe']:>7.2f}{r['var_daily']:>10.2f}{r['cash_days']:>8}{ex:>10}{r['elapsed']:>7.1f}")

    import csv
    od = os.path.join(HERE, "data", "results", "peg")
    os.makedirs(od, exist_ok=True)
    out_csv = os.path.join(od, "tuning_summary.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["变体", "总收益%", "年化%", "最大回撤%", "夏普", "VaR日度%", "现金天", "超额全指", "用时s"])
        for r in rows:
            w.writerow([r["label"], f"{r['total']:.2f}", f"{r['annual']:.2f}", f"{r['mdd']:.2f}",
                        f"{r['sharpe']:.2f}", f"{r['var_daily']:.2f}", r["cash_days"],
                        f"{r['excess']:.1f}" if r["excess"] is not None else "", r["elapsed"]])
    print(f"\n汇总已保存：{out_csv}")


if __name__ == "__main__":
    main()
