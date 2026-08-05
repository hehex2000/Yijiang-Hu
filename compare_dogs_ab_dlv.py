# -*- coding: utf-8 -*-
"""
三方对比：
  A) 狗股(基线, 无约束)      data/results/dogs_annual/backtest_20140101_20260720.csv
  B) 狗股(单行业<=2)          data/results/dogs_annual/backtest_20140101_20260720_indcap2.csv
  C) 红利低波(月度)           data/results/monthly_rebalance/backtest_20140505_20260720.csv

统一对齐到共同窗口(C 的起点 20140505 起)，按 raw 口径(狗股用 value_raw)对比：
逐年收益、全程收益、年化、最大回撤、夏普、逐年胜负。
狗股 CSV 另给 hfq(value) 口径供参考。
"""
import pandas as pd
import numpy as np

def load(path, col):
    df = pd.read_csv(path, dtype={"date": str})
    s = pd.Series(df[col].values, index=df["date"].values, name=col)
    return s.sort_index()

def metrics(nav):
    ret = nav.iloc[-1] / nav.iloc[0] - 1
    n_days = len(nav)
    ann = (nav.iloc[-1] / nav.iloc[0]) ** (252.0 / max(n_days - 1, 1)) - 1
    peak = nav.cummax()
    mdd = (nav / peak - 1).min()
    d = nav.pct_change().dropna()
    sharpe = d.mean() / d.std() * np.sqrt(252) if d.std() > 0 else 0.0
    return ret, ann, mdd, sharpe

def yearly(nav):
    y = pd.Series(nav.values, index=pd.Index([d[:4] for d in nav.index]))
    out = {}
    for yr in sorted(set(y.index)):
        sub = nav[[d.startswith(yr) for d in nav.index]]
        # 用上一年末做基点(首年用首日)
        prev = nav[nav.index < sub.index[0]]
        base = prev.iloc[-1] if len(prev) else sub.iloc[0]
        out[yr] = sub.iloc[-1] / base - 1
    return out

def main():
    A_raw = load("data/results/dogs_annual/backtest_20140101_20260720.csv", "value_raw")
    A_hfq = load("data/results/dogs_annual/backtest_20140101_20260720.csv", "value")
    B_raw = load("data/results/dogs_annual/backtest_20140101_20260720_indcap2.csv", "value_raw")
    B_hfq = load("data/results/dogs_annual/backtest_20140101_20260720_indcap2.csv", "value")
    C = load("data/results/monthly_rebalance/backtest_20140505_20260720.csv", "value")

    # 共同窗口
    common = sorted(set(A_raw.index) & set(B_raw.index) & set(C.index))
    print(f"共同窗口: {common[0]} → {common[-1]} ({len(common)} 交易日)\n")
    series = {
        "狗股·基线(raw)": A_raw[common],
        "狗股·行业≤2(raw)": B_raw[common],
        "狗股·基线(hfq)": A_hfq[common],
        "狗股·行业≤2(hfq)": B_hfq[common],
        "红利低波·月度": C[common],
    }

    rows = []
    for name, nav in series.items():
        ret, ann, mdd, sharpe = metrics(nav)
        rows.append({"策略": name, "总收益": f"{ret*100:+.2f}%", "年化": f"{ann*100:+.2f}%",
                     "最大回撤": f"{mdd*100:.2f}%", "夏普": f"{sharpe:.3f}"})
    print("===== 全程指标 (共同窗口) =====")
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n===== 逐年收益 =====")
    ys = {name: yearly(nav) for name, nav in series.items()}
    years = sorted(ys["狗股·基线(raw)"].keys())
    tab = []
    for yr in years:
        tab.append({
            "年份": yr,
            "狗股基线(raw)": f"{ys['狗股·基线(raw)'][yr]*100:+.2f}%",
            "狗股行业≤2(raw)": f"{ys['狗股·行业≤2(raw)'][yr]*100:+.2f}%",
            "Δ(约束-基线)": f"{(ys['狗股·行业≤2(raw)'][yr]-ys['狗股·基线(raw)'][yr])*100:+.2f}pp",
            "红利低波": f"{ys['红利低波·月度'][yr]*100:+.2f}%",
        })
    print(pd.DataFrame(tab).to_string(index=False))

    wins = sum(1 for yr in years if ys['狗股·行业≤2(raw)'][yr] > ys['狗股·基线(raw)'][yr])
    print(f"\n行业≤2 逐年跑赢基线: {wins}/{len(years)} 年")
    winsC_raw = sum(1 for yr in years if ys['红利低波·月度'][yr] > ys['狗股·基线(raw)'][yr])
    winsC_hfq = sum(1 for yr in years if ys['红利低波·月度'][yr] > ys['狗股·基线(hfq)'][yr])
    print(f"红利低波 逐年跑赢 狗股基线(raw): {winsC_raw}/{len(years)} 年")
    print(f"红利低波 逐年跑赢 狗股基线(hfq): {winsC_hfq}/{len(years)} 年")
    print("\n[口径说明] 狗股 raw=原始价(不含分红,偏低) / hfq=后复权(分红当日再投,偏高),真实介于两者之间;")
    print("[口径说明] 红利低波月度回测的净值为单轨(原始价+分红未单独建模),与狗股 raw 口径最接近。")

if __name__ == "__main__":
    main()
