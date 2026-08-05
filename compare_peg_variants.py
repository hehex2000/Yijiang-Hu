"""
PEG 策略改良对比驾驶舱
========================
一次跑多个 PEG 变体，汇总对比能否救回"九年阴跌"。
所有变体共用：全历史 20140101~20260715 / 30只 / 年度调仓 / 护栏③=3年。

注意：run_peg.run_backtest 的默认值已改为「质量+动量+VaR」，因此本表每个变体
都显式写出全部开关，避免「传 {} 却偷偷带上最优配置」的语义漂移。
"""
import os, sys, time
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_peg as rp

# 每个变体显式声明开关（覆盖 run_peg 的默认「质量+动量+VaR」），保证对比语义清晰
VARIANTS = [
    ("① 基线(纯PEG+三护栏)",
        {"min_roe": 0, "max_debt": 100, "momentum_months": 0, "var_guard": False}),
    ("② 沪深300大票",
        {"stock_pool": "hs300", "min_roe": 0, "max_debt": 100, "momentum_months": 0, "var_guard": False}),
    ("③ 质量叠加(ROE>=8,负债<=70)",
        {"min_roe": 8, "max_debt": 70, "momentum_months": 0, "var_guard": False}),
    ("④ 动量叠加(12月>0)",
        {"min_roe": 0, "max_debt": 100, "momentum_months": 12, "var_guard": False}),
    ("⑤ 流动性加权",
        {"min_roe": 0, "max_debt": 100, "momentum_months": 0, "var_guard": False, "weight": "liquidity"}),
    ("⑥ 沪深300+质量",
        {"stock_pool": "hs300", "min_roe": 8, "max_debt": 70, "momentum_months": 0, "var_guard": False}),
    ("⑦ 质量+动量(无VaR)",
        {"min_roe": 8, "max_debt": 70, "momentum_months": 12, "var_guard": False}),
    ("⑧ 质量+动量+VaR(2.5%)",
        {"min_roe": 8, "max_debt": 70, "momentum_months": 12, "var_guard": True, "var_cap": 0.025}),
]

COMMON = dict(start_date="20140101", end_date="20260715",
              top_n=30, freq="annual", stab_years=3, verbose=False)


def _reset_stats():
    rp._STATS = {"eligible": 0, "g1": 0, "g2": 0, "g3": 0, "selected": 0, "rebal": 0}


def main():
    print(f"PEG 改良对比：{len(VARIANTS)} 个变体，区间 {COMMON['start_date']}~{COMMON['end_date']}\n")
    rows = []
    t0 = time.time()
    for name, extra in VARIANTS:
        _reset_stats()
        print(f">>> 运行 {name}  {extra}")
        t1 = time.time()
        res = rp.run_backtest(**COMMON, **extra)
        sec = time.time() - t1
        bench_csi = res["bench"].get("000985.SH")
        bench_hs = res["bench"].get("000300.SH")
        rows.append({
            "变体": name,
            "总收益%": round(res["total_return"], 2),
            "年化%": round(res["annual_return"], 2),
            "最大回撤%": round(res["max_drawdown"], 2),
            "夏普": round(res["sharpe"], 2),
            "胜率%": round(res["win_rate"], 1),
            "超额_中证全指": round(res["total_return"] - bench_csi, 2) if bench_csi is not None else None,
            "超额_沪深300": round(res["total_return"] - bench_hs, 2) if bench_hs is not None else None,
            "用时s": round(sec, 1),
        })
        print(f"    -> 总收益 {res['total_return']:+.2f}%  年化 {res['annual_return']:+.2f}%  "
              f"最大回撤 {res['max_drawdown']:.2f}%  夏普 {res['sharpe']:.2f}  (用时 {sec:.0f}s)\n")

    df = pd.DataFrame(rows)
    print("=" * 100)
    print("对比汇总（全历史 2014-2026，30只/年调/护栏③=3年）")
    print("=" * 100)
    # 终端对齐打印
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.to_string(index=False))
    out = "data/results/peg/comparison_summary.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n汇总已保存：{out}")
    print(f"总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
