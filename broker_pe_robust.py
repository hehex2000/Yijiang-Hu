# -*- coding: utf-8 -*-
"""
券商 PE/PB 因子 —— 稳健性诊断（Gate 4 前置 / 剥出身）

Gate 1 显示 PB 的 IC 高达 -0.68、t_NW -3.4，但分层里 Q2 的 std 有 43%，
意味着结果可能被单一极端事件（2014-2015 券商暴涨暴跌）主导。
本脚本回答一个问题：**这个信号是不是只靠一两次历史事件撑起来的？**

检验项目
--------
1) 逐年 IC / 逐年 Q0-Q4 多空利差  → 看稳定性与集中度
2) 剔除 2014-06~2015-12（券商史诗级行情）后重测 → 看是否塌缩
3) 非重叠调仓的多空净值曲线 → 看真实路径（不给重叠样本刷指标的机会）
4) 事件表：信号处于极值（PB 分位 <10 / >90）的具体时段及后续表现

用法
----
  python broker_pe_robust.py
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

OUT_DIR = os.path.join("data", "results", "broker_pe")
SRC = os.path.join(OUT_DIR, "sector_daily.csv")
EXCLUDE = ("20140601", "20151231")     # 券商史诗级行情窗口


def spearman(x, y, min_n=20):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < min_n:
        return np.nan
    return stats.spearmanr(np.asarray(x)[m], np.asarray(y)[m]).correlation


def load():
    sec = pd.read_csv(SRC, dtype={"trade_date": str})
    sec["yr"] = sec["trade_date"].str[:4].astype(int)
    return sec


# ------------------------------------------------------------ 1) 逐年
def yearly(sec, factor="pb", window=750, H=250):
    col = f"{factor}_pct{window}"
    ret = f"fwd_exc_{H}"
    print("=" * 88)
    print(f"【1】逐年诊断  {factor.upper()}分位({window}日) → 未来{H}日超额")
    print("=" * 88)
    print(f"{'年份':<6}{'n':>6}{'IC':>9}{'Q0(低估值)':>13}{'Q4(高估值)':>13}"
          f"{'Q0-Q4利差':>12}  备注")
    print("-" * 88)
    rows = []
    for yr, g in sec.groupby("yr"):
        g = g[[col, ret]].dropna()
        if len(g) < 60:
            continue
        ic = spearman(g[col].values, g[ret].values)
        gq = g.copy()
        gq["q"] = pd.qcut(gq[col], 5, labels=False, duplicates="drop")
        m = gq.groupby("q")[ret].mean()
        q0 = m.get(0, np.nan)
        q4 = m.get(4, np.nan)
        spread = q0 - q4
        note = ""
        if 2014 <= yr <= 2015:
            note = "← 券商史诗级行情"
        rows.append(dict(yr=yr, n=len(g), ic=ic, q0=q0, q4=q4, spread=spread))
        print(f"{yr:<6}{len(g):>6}{ic:>9.3f}{q0*100:>12.2f}%{q4*100:>12.2f}%"
              f"{spread*100:>11.2f}%  {note}")
    print("-" * 88)
    d = pd.DataFrame(rows)
    if len(d):
        pos = (d.spread > 0).mean() * 100
        print(f"  Q0-Q4 利差为正的年份: {(d.spread>0).sum()}/{len(d)} ({pos:.0f}%)")
        # 集中度：最大 2 年贡献
        s = d.spread.sort_values(ascending=False)
        top2 = s.head(2).sum()
        tot = d.spread.sum()
        print(f"  最好的 2 年贡献了总利差的 {top2/tot*100:.0f}%"
              f"{'   ⚠️ 高度集中' if top2/tot > 0.5 else ''}")
    return d


# ------------------------------------------------------------ 2) 剔除事件
def exclude_test(sec, factor="pb", window=750):
    col = f"{factor}_pct{window}"
    print("\n" + "=" * 88)
    print(f"【2】剔除券商史诗级行情 {EXCLUDE[0]}~{EXCLUDE[1]} 后重测")
    print("=" * 88)
    print(f"{'H':>6}{'全样本IC':>11}{'剔除后IC':>11}{'变化':>10}{'剔除后t(非重叠)':>18}")
    print("-" * 88)
    mask = (sec["trade_date"] >= EXCLUDE[0]) & (sec["trade_date"] <= EXCLUDE[1])
    print(f"  剔除 {mask.sum()} 个交易日（以及其前后 H 日的 forward 窗口）")
    for H in (60, 120, 250):
        ret = f"fwd_exc_{H}"
        full = sec[[col, ret]].dropna()
        # 剔除：起点在窗口内，或终点在窗口内
        d = sec.copy()
        d["end_date"] = d["trade_date"].shift(-(H + 1))
        keep = ~((d["trade_date"] >= EXCLUDE[0]) & (d["trade_date"] <= EXCLUDE[1])) & \
               ~((d["end_date"] >= EXCLUDE[0]) & (d["end_date"] <= EXCLUDE[1]))
        sub = d[keep][[col, ret]].dropna()
        ic_full = spearman(full[col].values, full[ret].values)
        ic_sub = spearman(sub[col].values, sub[ret].values)
        # 非重叠 t
        ssub = sub.iloc[::H]
        t = np.nan
        if len(ssub) >= 8:
            r = stats.pearsonr(ssub[col].values, ssub[ret].values)[0]
            t = r * np.sqrt((len(ssub) - 2) / max(1e-12, 1 - r ** 2))
        print(f"{H:>6}{ic_full:>11.3f}{ic_sub:>11.3f}"
              f"{(ic_sub-ic_full):>10.3f}{t:>18.2f}")
    print("-" * 88)
    print("  判定：若剔除后 |IC| 塌缩到 <0.15，说明信号主要来自那一次行情，不可依赖。")


# ------------------------------------------------------------ 3) 非重叠净值
def nonoverlap_equity(sec, factor="pb", window=750, H=250):
    """非重叠调仓：每 H 日按分位建仓，Q0 多头 / Q4 空头（等权各半）。"""
    col = f"{factor}_pct{window}"
    ret = f"fwd_exc_{H}"
    print("\n" + "=" * 88)
    print(f"【3】非重叠调仓多空净值（每 {H} 日调一次，Q0 多 / Q4 空）")
    print("=" * 88)
    d = sec[[col, ret, "trade_date"]].dropna()
    d = d.iloc[::H].copy()             # 非重叠
    if len(d) < 10:
        print("  样本不足")
        return
    d["q"] = pd.qcut(d[col], 5, labels=False, duplicates="drop")
    long_ = (d["q"] == 0).astype(float)
    short_ = (d["q"] == 4).astype(float)
    # 多空各半仓
    d["ls"] = 0.5 * long_ * d[ret] - 0.5 * short_ * d[ret]
    # 覆盖度不足时按空仓处理
    d["nav"] = (1 + d["ls"]).cumprod()
    n = len(d)
    tot = d["nav"].iloc[-1] - 1
    ann = (1 + tot) ** (250 / (H * n)) - 1 if tot > -1 else np.nan
    dd = (d["nav"] / d["nav"].cummax() - 1).min()
    win = (d["ls"] > 0).mean() * 100
    t = d["ls"].mean() / (d["ls"].std() / np.sqrt(n)) if d["ls"].std() > 0 else np.nan
    print(f"  调仓次数 {n}  |  总收益 {tot*100:+.2f}%  |  年化 {ann*100:+.2f}%")
    print(f"  最大回撤 {dd*100:.2f}%  |  单期胜率 {win:.1f}%  |  t = {t:.2f}")
    print(f"  起 {d.trade_date.iloc[0]}  止 {d.trade_date.iloc[-1]}")
    print("\n  路径：")
    for i in range(0, len(d), max(1, len(d) // 12)):
        r = d.iloc[i]
        print(f"    {r.trade_date}  分位{r[col]:5.1f}  本期{r[ret]*100:+7.2f}%  "
              f"仓位{'多' if r['q']==0 else ('空' if r['q']==4 else '空仓'):<3}"
              f"  净值{r['nav']:.3f}")
    print("\n  判定：非重叠调仓下年化与 t 若大幅弱于 Gate 1，则说明重叠样本刷高了指标。")


# ------------------------------------------------------------ 4) 极值事件表
def extreme_events(sec, factor="pb", window=750, H=250):
    col = f"{factor}_pct{window}"
    ret = f"fwd_exc_{H}"
    print("\n" + "=" * 88)
    print(f"【4】极值事件表：{factor.upper()}分位 <10 或 >90 的时段及其后续{H}日超额")
    print("=" * 88)
    d = sec[[col, ret, "trade_date"]].dropna().reset_index(drop=True)
    e = d[(d[col] < 10) | (d[col] > 90)]
    if len(e) == 0:
        print("  无极值时段")
        return
    seg, prev = [], None
    for i in e.index:
        if prev is None or i - prev > 40:
            seg.append([i, i])
        else:
            seg[-1][1] = i
        prev = i
    print(f"{'起始':<10}{'结束':<10}{'天数':>6}{'均分位':>9}{'板块后续%':>12}{'基准后续%':>12}{'超额%':>10}")
    print("-" * 88)
    for a, b in seg:
        sub = d.loc[a:b]
        fwd = sec.loc[sec.trade_date.isin(sub.trade_date), [f"fwd_sec_{H}", f"fwd_bch_{H}"]].mean()
        print(f"{sub.trade_date.iloc[0]:<10}{sub.trade_date.iloc[-1]:<10}{len(sub):>6}"
              f"{sub[col].mean():>9.1f}{fwd[f'fwd_sec_{H}']*100:>12.2f}"
              f"{fwd[f'fwd_bch_{H}']*100:>12.2f}"
              f"{(fwd[f'fwd_sec_{H}']-fwd[f'fwd_bch_{H}'])*100:>10.2f}")
    print("-" * 88)


def main():
    sec = load()
    print(f"[数据] {len(sec)} 行 | {sec.trade_date.min()} ~ {sec.trade_date.max()}\n")
    for factor, window in (("pb", 750), ("pe", 750)):
        yearly(sec, factor, window, 250)
        exclude_test(sec, factor, window)
        nonoverlap_equity(sec, factor, window, 250)
        extreme_events(sec, factor, window, 250)
        print("\n" + "#" * 88 + "\n")


if __name__ == "__main__":
    main()
