# -*- coding: utf-8 -*-
"""
狗股年度调仓：年内风控 A/B 多方对比。
  base  : 无风控 (backtest_20140101_20260720.csv)
  trail : 移动止损20% (_trail20)
  dd    : 组合回撤减仓25% (_dd25)
  combo : 移动止损20% + 回撤减仓25% (_trail20_dd25)
窗口取各 CSV 完整共同区间。raw=原始价(不含分红,偏低) / hfq=后复权(分红再投,偏高)。
"""
import pandas as pd
import numpy as np

FILES = {
    "无风控(base)": "data/results/dogs_annual/backtest_20140101_20260720.csv",
    "移动止损20%": "data/results/dogs_annual/backtest_20140101_20260720_trail20.csv",
    "回撤减仓25%": "data/results/dogs_annual/backtest_20140101_20260720_dd25.csv",
    "止损20%+减仓25%": "data/results/dogs_annual/backtest_20140101_20260720_trail20_dd25.csv",
}

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
    return ret, ann, mdd, sharpe

def yearly(nav):
    out = {}
    for yr in sorted(set(d[:4] for d in nav.index)):
        sub = nav[[d.startswith(yr) for d in nav.index]]
        prev = nav[nav.index < sub.index[0]]
        base = prev.iloc[-1] if len(prev) else sub.iloc[0]
        out[yr] = sub.iloc[-1] / base - 1
    return out

def main():
    raw = {k: load(v, "value_raw") for k, v in FILES.items()}
    hfq = {k: load(v, "value") for k, v in FILES.items()}
    common = sorted(set.intersection(*[set(s.index) for s in raw.values()]))
    print(f"对比窗口: {common[0]} -> {common[-1]} ({len(common)} 交易日)\n")

    print("===== 全程指标 (raw 口径) =====")
    rows = []
    for k in FILES:
        ret, ann, mdd, sharpe = metrics(raw[k][common])
        rh, ah, mdh, sh = metrics(hfq[k][common])
        rows.append({"策略": k, "总收益(raw)": f"{ret*100:+.2f}%", "年化(raw)": f"{ann*100:+.2f}%",
                     "最大回撤(raw)": f"{mdd*100:.2f}%", "夏普(raw)": f"{sharpe:.3f}",
                     "总收益(hfq)": f"{rh*100:+.2f}%", "最大回撤(hfq)": f"{mdh*100:.2f}%"})
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n===== 逐年收益 (raw 口径) =====")
    ys = {k: yearly(raw[k][common]) for k in FILES}
    years = sorted(ys["无风控(base)"].keys())
    tab = []
    for yr in years:
        tab.append({"年份": yr,
                    "base": f"{ys['无风控(base)'][yr]*100:+.2f}%",
                    "止损20%": f"{ys['移动止损20%'][yr]*100:+.2f}%",
                    "减仓25%": f"{ys['回撤减仓25%'][yr]*100:+.2f}%",
                    "组合": f"{ys['止损20%+减仓25%'][yr]*100:+.2f}%"})
    print(pd.DataFrame(tab).to_string(index=False))

    print("\n[口径说明] raw=原始价(不含分红,偏低); hfq=后复权(分红再投,偏高); 真实收益介于两者之间。")
    print("[说明] 回撤减仓/移动止损在『触发日收盘』清仓，次日才体现现金；最大回撤口径已含触发日当日谷值。")

if __name__ == "__main__":
    main()
