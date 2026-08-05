# -*- coding: utf-8 -*-
"""
狗股年度调仓：基线(无约束) vs 单行业<=2 的纯年度对比。
窗口取两个 CSV 的完整共同区间 (20140102 -> 20260720)，不做任何对齐裁剪。
口径：raw=原始价(不含分红,偏低) / hfq=后复权(分红再投,偏高)，真实介于两者之间。
"""
import pandas as pd
import numpy as np

def load(path, col):
    df = pd.read_csv(path, dtype={"date": str})
    return pd.Series(df[col].values, index=df["date"].values, name=col).sort_index()

def metrics(nav):
    ret = nav.iloc[-1] / nav.iloc[0] - 1
    n_days = len(nav)
    ann = (nav.iloc[-1] / nav.iloc[0]) ** (252.0 / max(n_days - 1, 1)) - 1
    peak = nav.cummax()
    mdd = (nav / peak - 1).min()
    d = nav.pct_change().dropna()
    sharpe = d.mean() / d.std() * np.sqrt(252) if d.std() > 0 else 0.0
    # 找出最大回撤发生的区间(峰->谷)
    peak_idx = nav.cummax()
    draw = nav / peak_idx - 1
    trough_i = draw.idxmin()
    # 该谷对应的峰值位置
    peak_val_before = peak_idx[:trough_i].iloc[-1]
    peak_date = peak_idx[:trough_i].index[-1]
    return ret, ann, mdd, sharpe, peak_date, trough_i

def yearly(nav):
    out = {}
    for yr in sorted(set(d[:4] for d in nav.index)):
        sub = nav[[d.startswith(yr) for d in nav.index]]
        prev = nav[nav.index < sub.index[0]]
        base = prev.iloc[-1] if len(prev) else sub.iloc[0]
        out[yr] = sub.iloc[-1] / base - 1
    return out

def main():
    base_raw = load("data/results/dogs_annual/backtest_20140101_20260720.csv", "value_raw")
    base_hfq = load("data/results/dogs_annual/backtest_20140101_20260720.csv", "value")
    cap2_raw = load("data/results/dogs_annual/backtest_20140101_20260720_indcap2.csv", "value_raw")
    cap2_hfq = load("data/results/dogs_annual/backtest_20140101_20260720_indcap2.csv", "value")

    common = sorted(set(base_raw.index) & set(cap2_raw.index))
    print(f"年度对比窗口: {common[0]} -> {common[-1]} ({len(common)} 交易日)\n")

    rows = []
    for name, nav in [("狗股·基线(raw)", base_raw[common]),
                      ("狗股·基线(hfq)", base_hfq[common]),
                      ("狗股·行业<=2(raw)", cap2_raw[common]),
                      ("狗股·行业<=2(hfq)", cap2_hfq[common])]:
        ret, ann, mdd, sharpe, pk, tr = metrics(nav)
        rows.append({"策略": name, "总收益": f"{ret*100:+.2f}%", "年化": f"{ann*100:+.2f}%",
                     "最大回撤": f"{mdd*100:.2f}%", "夏普": f"{sharpe:.3f}",
                     "回撤区间": f"{pk}~{tr}"})
    print("===== 全程指标 (完整年度窗口) =====")
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n===== 逐年收益 (raw 口径, 约束 - 基线) =====")
    yb, yc = yearly(base_raw[common]), yearly(cap2_raw[common])
    years = sorted(yb.keys())
    tab = []
    for yr in years:
        tab.append({"年份": yr,
                    "基线(raw)": f"{yb[yr]*100:+.2f}%",
                    "行业<=2(raw)": f"{yc[yr]*100:+.2f}%",
                    "Δ": f"{(yc[yr]-yb[yr])*100:+.2f}pp"})
    print(pd.DataFrame(tab).to_string(index=False))

    wins = sum(1 for yr in years if yc[yr] > yb[yr])
    print(f"\n行业<=2 逐年跑赢基线: {wins}/{len(years)} 年")
    print("\n[口径说明] raw=原始价(不含分红,偏低); hfq=后复权(分红再投,偏高); 真实收益介于两者之间。")
    print("[重要] 单行业<=2 约束只限制『同一行业持股数』，不限制市场β/仓位，因此对系统性回撤几乎无改善——")
    print("        回撤主要来自满仓做多+年度调仓无年内风控，约束解决的是『单行业黑天鹅』而非『大盘下跌』。")

if __name__ == "__main__":
    main()
