#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""退出时点事件研究 —— 检验「该不该卖」是否取决于当前浮盈。

来源：B站 BV1KBbe6jEPV《随机最优停止：最优平仓阈值与等待的负价值》。
视频主张：**最优平仓阈值只取决于 mu/sigma/r，与你已赚多少无关**
（"赚 20% 不是平仓理由"）。

本脚本用大样本事件研究直接检验这个命题，不跑完整策略回测，
从而绕开换手成本/复权口径两类噪音（见 backtest_optimal_stop.py 的教训）：
换手成本会吃掉退出规则的真实增量，raw 口径会把现金分红误计为下跌。

────────────────────────────────────────────────────────────
核心设计：两类卖出事件，都记录「卖出时的浮盈 g = P_sell / K」

  A. threshold（阈值止盈）：买入后等价格涨到 K*mult 就卖
     → g 恒 >= mult。只能看到高浮盈区间。
  B. fixed_hold（固定持有期卖出，H 日后无条件卖）
     → g 任意分布（含亏损）。**这才是"浮盈 5% vs 40%"对比的载体。**

对每个卖出事件，测「卖出后还涨不涨」：
  - fwd_h   : 该票从卖出日到 T+h 的前向收益（h ∈ 20/60/120 交易日）
  - xs_h    : 同一卖出日、池内全部股票等权前向收益（横截面对照，剔除市场时序）
  - excess_h: fwd_h − xs_h     < 0 → 卖出后跑输市场 → 卖对了（止盈有价值）
  - mfe_h   : 卖出后 h 日内的最大有利偏移 max(P)/P_sell − 1（"卖飞"程度）

命题检验（关键看 fixed_hold 事件按 g 分桶）：
  · 若各浮盈桶的 excess 无显著差异、且 g 与 excess 无相关
      → 浮盈水平不携带"该不该卖"的信息 → **支持视频**
  · 若高浮盈桶 excess 显著更负（涨多了之后更容易跑输）
      → 浮盈水平携带信息 → **推翻视频**，止盈有实证依据

无前视保证：
  · 买入日 i 只用 P[i] 定 K，向后扫描触发，前向收益严格取卖出日之后；
  · 横截面对照同样用当日之后的收益，不含未来；
  · 同一票的事件之间强制冷却 cooldown 日（默认 60），控制重叠。

用法：
  python exit_event_study.py --rule hold --hold-days 60 --price-mode hfq
  python exit_event_study.py --rule threshold --mult 1.30 --price-mode hfq
  python exit_event_study.py --rule both --compare
"""
import argparse
import os

import numpy as np
import pandas as pd

from run_daily20_macd import load_closes

TRADING_DAYS = 252
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'results')

HORIZONS = (20, 60, 120)

# fixed_hold 事件的浮盈分桶边界（g = P_sell / K）
GAIN_BINS = [-np.inf, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20, 1.30, 1.50, np.inf]
GAIN_LABELS = ['<-10%', '-10~-5%', '-5~0%', '0~5%', '5~10%',
               '10~20%', '20~30%', '30~50%', '>50%']


# ───────────────────────── 前向收益矩阵 ─────────────────────────
def build_fwd(closes, horizons=HORIZONS):
    """返回 (fwd_dict, xs_mean, xs_med)。

    fwd[h]   : DataFrame，值 = P[t+h]/P[t] − 1（严格向后，无前视）
    xs_mean  : Series，横截面算术均值 = 池内等权组合的真实收益。
               经济意义上的机会成本基准（卖掉个股、把钱买成等权组合）。
    xs_med   : Series，横截面中位数。
               统计意义上的位置基准 —— 个股收益右偏，"算术均值 > 个股中位数"
               是分散化红利的机械结果，不能直接当作"卖出错了"，
               必须再用中位数基准做同位置比较。

    两个基准有分工：excess=fwd−xs_mean 回答"该不该卖"，
                  excess_med=fwd−xs_med 回答"这只票在同批股票里排第几"。
    """
    fwd, xs_mean, xs_med = {}, {}, {}
    for h in horizons:
        f = closes.shift(-h) / closes - 1.0
        f[~np.isfinite(f)] = np.nan
        fwd[h] = f
        xs_mean[h] = f.mean(axis=1)     # 忽略 NaN
        xs_med[h] = f.median(axis=1)
    return fwd, xs_mean, xs_med


# ───────────────────────── 事件生成 ─────────────────────────
def gen_events(closes, fwd, xs_mean, xs_med, rule, mult=1.30, hold_days=60,
               max_hold=250, cooldown=60, start=None, end=None):
    """按状态机扫描每只票，生成卖出事件表。

    状态机：买入(K=P[i]) → 等卖出信号 → 卖出日 j → 冷却 cooldown → 再买入。
    未触发（超过 max_hold 或数据结束）则视为持有到期，前进 max_hold 后重新买入。
    """
    dates = list(closes.index)
    if start is not None:
        dates = [d for d in dates if d >= start]
    if end is not None:
        dates = [d for d in dates if d <= end]
    pos0 = list(closes.index).index(dates[0])
    pos1 = list(closes.index).index(dates[-1])

    rows = []
    n = closes.shape[0]
    fwd_v = {h: fwd[h].values for h in HORIZONS}
    xs_v = {h: xs_mean[h].values for h in HORIZONS}
    xm_v = {h: xs_med[h].values for h in HORIZONS}

    for c in closes.columns:
        p = closes[c].values
        i = pos0
        while i < pos1:
            if not np.isfinite(p[i]) or p[i] <= 0:
                i += 1
                continue
            K = p[i]

            # ── 找卖出日 j ──
            if rule == 'threshold':
                j = None
                lim = min(n, i + 1 + max_hold)
                for k in range(i + 1, lim):
                    if np.isfinite(p[k]) and p[k] >= K * mult:
                        j = k
                        break
                if j is None:
                    i += max_hold          # 持有到期未触发，重新建仓
                    continue
            else:                           # fixed_hold
                j = i + int(hold_days)
                if j >= n or not np.isfinite(p[j]) or p[j] <= 0:
                    i += cooldown
                    continue

            ps = p[j]
            g = ps / K
            rec = {'ts_code': c, 'buy_date': closes.index[i],
                   'sell_date': closes.index[j], 'K': K, 'P_sell': ps,
                   'gain': g, 'hold_days': j - i}

            for h in HORIZONS:
                f = fwd_v[h][j, closes.columns.get_loc(c)]
                x = xs_v[h][j]
                xm = xm_v[h][j]
                rec[f'fwd{h}'] = f
                rec[f'xs{h}'] = x
                rec[f'excess{h}'] = (f - x) if (np.isfinite(f) and np.isfinite(x)) else np.nan
                rec[f'excess_med{h}'] = ((f - xm)
                                         if (np.isfinite(f) and np.isfinite(xm)) else np.nan)
                # 最大有利偏移（卖出后 h 日内最高价 / 卖出价 − 1）
                seg = p[j + 1: min(n, j + 1 + h)]
                seg = seg[np.isfinite(seg)]
                rec[f'mfe{h}'] = (seg.max() / ps - 1.0) if len(seg) else np.nan

            rows.append(rec)
            i = j + cooldown

    df = pd.DataFrame(rows)
    return df


# ───────────────────────── 统计 ─────────────────────────
def _winsor(x, p=0.01):
    """双边缩尾均值：肥尾的收益分布下，比原始均值稳健得多。"""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return np.nan
    lo, hi = np.quantile(x, p), np.quantile(x, 1 - p)
    return float(np.clip(x, lo, hi).mean())


def _tstat(x):
    """原始均值 t 值（保留，用于看显著性是否被肥尾撑起）。"""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan, len(x)
    sd = x.std(ddof=1)
    if sd <= 0:
        return np.nan, len(x)
    return x.mean() / (sd / np.sqrt(len(x))), len(x)


def _tstat_w(x):
    """缩尾均值 t 值：用缩尾后的样本算标准误，剔除肥尾对显著性的虚增。"""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return np.nan
    lo, hi = np.quantile(x, 0.01), np.quantile(x, 0.99)
    xw = np.clip(x, lo, hi)
    sd = xw.std(ddof=1)
    if sd <= 0:
        return np.nan
    return xw.mean() / (sd / np.sqrt(len(xw)))


def summarize_by_gain(df, h=60, label=''):
    """按卖出时浮盈 g 分桶，统计卖出后 h 日的超额收益。"""
    if not len(df):
        return pd.DataFrame()
    out = []
    cats = pd.cut(df['gain'], bins=GAIN_BINS, labels=GAIN_LABELS)
    col = f'excess{h}'
    mfe = f'mfe{h}'
    for lab in GAIN_LABELS:
        sub = df[cats == lab]
        if not len(sub):
            continue
        e = sub[col].dropna()
        em = sub[f'excess_med{h}'].dropna()
        m = sub[mfe].dropna()
        t, n = _tstat(e)
        tw = _tstat_w(e)
        out.append({
            '浮盈桶': lab,
            '事件数': len(sub),
            '有效样本': n,
            '占比%': round(100 * len(sub) / len(df), 1),
            f'超额{h}均值%': round(100 * e.mean(), 2) if n else np.nan,
            f'超额{h}缩尾%': round(100 * _winsor(e), 2) if n else np.nan,
            't值': round(t, 2) if np.isfinite(t) else np.nan,
            '缩尾t值': round(tw, 2) if np.isfinite(tw) else np.nan,
            '跑赢等权组合%': round(100 * (e > 0).mean(), 1) if n else np.nan,
            f'中位超额{h}%': round(100 * em.median(), 2) if len(em) else np.nan,
            '跑赢中位股%': round(100 * (em > 0).mean(), 1) if len(em) else np.nan,
            f'卖飞中位%': round(100 * m.median(), 2) if len(m) else np.nan,
            '平均持有天数': round(sub['hold_days'].mean(), 0),
        })
    res = pd.DataFrame(out)
    if label:
        print(f"\n── {label}（按卖出时浮盈分桶，h={h} 日）──")
        print(res.to_string(index=False))
    return res


def overall(df, h=60, label=''):
    """全样本汇总：卖出后 h 日的绝对/相对表现。"""
    e = df[f'excess{h}'].dropna()
    em = df[f'excess_med{h}'].dropna()
    f = df[f'fwd{h}'].dropna()
    x = df[f'xs{h}'].dropna()
    m = df[f'mfe{h}'].dropna()
    t, n = _tstat(e)
    tw = _tstat_w(e)
    d = {
        'h(日)': h,
        '规则': label,
        '事件数': len(df),
        '前向%': round(100 * f.mean(), 2) if len(f) else np.nan,
        '等权组合%': round(100 * x.mean(), 2) if len(x) else np.nan,
        '超额%': round(100 * e.mean(), 2) if n else np.nan,
        '超额缩尾%': round(100 * _winsor(e), 2) if n else np.nan,
        't值': round(t, 2) if np.isfinite(t) else np.nan,
        '缩尾t值': round(tw, 2) if np.isfinite(tw) else np.nan,
        '跑赢等权组合%': round(100 * (e > 0).mean(), 1) if n else np.nan,
        '中位超额%': round(100 * em.median(), 2) if len(em) else np.nan,
        '跑赢中位股%': round(100 * (em > 0).mean(), 1) if len(em) else np.nan,
        '卖飞中位%': round(100 * m.median(), 2) if len(m) else np.nan,
        '平均持有天数': round(df['hold_days'].mean(), 0),
    }
    return d


def corr_gain_excess(df, h=60):
    """浮盈 g 与卖出后超额收益的相关性（Spearman，抗异常值）。"""
    sub = df[['gain', f'excess{h}']].dropna()
    if len(sub) < 10:
        return np.nan, np.nan, len(sub)
    r = sub['gain'].rank().corr(sub[f'excess{h}'].rank())
    return r, None, len(sub)


# ───────────────────────── 主流程 ─────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rule', default='both',
                    choices=['threshold', 'hold', 'both'])
    ap.add_argument('--mult', type=float, default=1.30,
                    help='threshold 规则的止盈倍数（1.20 / 1.30 / 1.50）')
    ap.add_argument('--hold-days', type=int, default=60,
                    help='hold 规则的固定持有交易日数')
    ap.add_argument('--max-hold', type=int, default=250,
                    help='threshold 规则的最长等待交易日（超时视为未触发）')
    ap.add_argument('--cooldown', type=int, default=60,
                    help='同一票两次事件之间的冷却交易日')
    ap.add_argument('--price-mode', default='hfq', choices=['raw', 'hfq'])
    ap.add_argument('--start', default='20200101')
    ap.add_argument('--end', default='20251231')
    ap.add_argument('--compare', action='store_true',
                    help='额外跑 raw 口径做对照，量化复权口径的影响')
    args = ap.parse_args()

    hfq = (args.price_mode == 'hfq')
    codes, closes = load_closes(hfq=hfq)
    closes = closes.astype(float)
    # 事件扫描需要 start 之前的预热数据（前向收益/持有期），保留全样本，
    # 仅在生成事件时限制买入日区间。
    print(f"[数据] 收盘矩阵 {closes.shape}  口径={args.price_mode}  "
          f"买入日区间 {args.start}~{args.end}")

    fwd, xs_mean, xs_med = build_fwd(closes)
    print(f"[数据] 前向收益矩阵 h={HORIZONS} 已构建（等权组合 + 横截面中位 双基准）")

    rules = []
    if args.rule in ('threshold', 'both'):
        rules.append(('threshold', f'阈值{args.mult:.2f}×卖出'))
    if args.rule in ('hold', 'both'):
        rules.append(('hold', f'持有{args.hold_days}日卖出'))

    frames = {}
    for rule, label in rules:
        print(f"\n[扫描] {label} ...")
        df = gen_events(closes, fwd, xs_mean, xs_med, rule,
                        mult=args.mult, hold_days=args.hold_days,
                        max_hold=args.max_hold, cooldown=args.cooldown,
                        start=int(args.start), end=int(args.end))
        frames[rule] = df
        print(f"  事件数 {len(df)}")

    print("\n" + "=" * 78)
    print("一、全样本：卖出之后，股票是继续涨还是跌？")
    print("=" * 78)
    overall_rows = []
    for h in HORIZONS:
        for rule, label in rules:
            df = frames.get(rule)
            if df is None or not len(df):
                continue
            overall_rows.append(overall(df, h=h, label=label))
    print(pd.DataFrame(overall_rows).to_string(index=False))

    print("\n" + "=" * 78)
    print("二、命题检验：浮盈分桶 → 卖出后表现是否有差异（h=60 日）")
    print("=" * 78)
    print("  判据：若各桶超额无差异、g 与超额无相关 → 支持视频「赚多少不是平仓理由」")
    print("        若高浮盈桶超额显著更负            → 推翻视频，止盈有实证依据")

    for rule, label in rules:
        df = frames.get(rule)
        if df is None or not len(df):
            continue
        res = summarize_by_gain(df, h=60, label=label)
        r, _, n = corr_gain_excess(df, h=60)
        print(f"  → Spearman(gain, excess60) = {r:+.4f}  (n={n})"
              f"  注：秩相关对肥尾稳健，是分桶结论的主判据")
        if rule == 'hold' and len(res):
            lo = res[res['浮盈桶'].isin(['<-10%', '-10~-5%', '-5~0%', '0~5%'])]
            hi = res[res['浮盈桶'].isin(['30~50%', '>50%'])]
            if len(lo) and len(hi):
                for key in (f'超额60均值%', f'超额60缩尾%',
                            '跑赢等权组合%', '中位超额60%', '跑赢中位股%'):
                    print(f"  → {key:<14} 低浮盈(<5%) {lo[key].mean():+7.2f} vs "
                          f"高浮盈(>30%) {hi[key].mean():+7.2f}  "
                          f"差 {hi[key].mean() - lo[key].mean():+.2f}")

    if args.compare and 'hold' in frames:
        print("\n" + "=" * 78)
        print("三、复权口径敏感性（raw vs hfq）")
        print("=" * 78)
        _, closes_alt = load_closes(hfq=not hfq)
        closes_alt = closes_alt.astype(float)
        fwd_a, xs_a, xm_a = build_fwd(closes_alt)
        df_a = gen_events(closes_alt, fwd_a, xs_a, xm_a, 'hold',
                          hold_days=args.hold_days, cooldown=args.cooldown,
                          start=int(args.start), end=int(args.end))
        alt_name = 'raw' if hfq else 'hfq'
        rows = [overall(frames['hold'], h=60, label=f'{args.price_mode}|hold'),
                overall(df_a, h=60, label=f'{alt_name}|hold')]
        print(pd.DataFrame(rows).to_string(index=False))
        summarize_by_gain(df_a, h=60, label=f'{alt_name} 口径 hold 规则')

    os.makedirs(OUT_DIR, exist_ok=True)
    for rule, df in frames.items():
        if not len(df):
            continue
        extra = f"{args.mult:.2f}" if rule == 'threshold' else f"{args.hold_days}d"
        out = os.path.join(
            OUT_DIR,
            f"exit_event_{rule}_{extra}_{args.start}_{args.end}_{args.price_mode}.csv")
        df.to_csv(out, index=False)
        print(f"\n[输出] {out}")


if __name__ == '__main__':
    main()
