#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""最优停止阈值 vs 固定止盈 —— A股退出规则 A/B 实验（M1 内存化快速引擎）。

来源：B站 BV1KBbe6jEPV《随机最优停止：最优平仓阈值与等待的负价值》（评级 B-）。
实验目的：把视频的"最优平仓阈值 S* = beta/(beta-1) * K"规则化，在 A 股
红利低波骨架上做证伪/证实，重点回答三问：
  ① 用 A 股真实参数估出的阈值是多少？还能不能触发？
  ② 阈值规则是否优于拍脑袋的固定 20%/30% 止盈？
  ③ sigma / mu / r 的估计误差会把阈值晃到什么程度（实盘可用性）？

统一实验骨架（只换退出规则，隔离单一变量）：
  调仓日重选红利低波 topN（zz800 池）· 满仓 · 无择时 · 等权。

调仓模式（--rebal-mode，默认 overlap）：
  full    全卖全买：换手最高、费用最重，与实盘偏离最大（初版口径）
  overlap 保留重叠持仓：只卖掉出篮子的 + 把留在篮子里的 trim/补到等权
          （与 run_daily20_macd._rebalance_to 同语义），换手大幅下降
  drift   保留重叠持仓且不做权重再平衡：留在篮子里的完全不动，只卖出篮子的、
          买入新进的  → 换手下界
  三种模式下 K 均为「加权平均买入成本」：加仓时按股数加权更新，
  减仓不改动 K（符合平均成本法，与最优停止理论里「成本价 K」一致）。

退出规则：
  hold     不止盈，持有到下次月度重选
  fixed20  价格 >= 1.20 * K 卖出
  fixed30  价格 >= 1.30 * K 卖出
  optimal  价格 >= beta/(beta-1) * K 卖出
            beta 为特征方程 0.5*s^2*beta*(beta-1) + mu*beta = r 的较大正根
            beta = ( -(mu - 0.5*s^2) + sqrt((mu - 0.5*s^2)^2 + 2*s^2*r) ) / s^2

无前视保证：
  · sigma / mu 一律用「截至 t 的过去 N 日对数收益」滚动估计，不含未来；
  · 退出信号 T 日收盘用当日 close 判定并以当日 close 成交（尾盘集合竞价近似）；
  · 选股用调仓日前一交易日 prev_td 的数据（select_div_low_vol 内部已保证）；
  · K = 该笔买入的实际成交价（成本口径与 NAV 口径一致）。

用法：
  python backtest_optimal_stop.py --mode theory
  python backtest_optimal_stop.py --mode backtest --start 20200101 --end 20251231
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

from run_daily20_macd import load_closes, build_vol_lookup, select_div_low_vol
from run_monthly_rebalance import calc_fee

TRADING_DAYS = 252
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'results')


# ───────────────────────── 数学内核 ─────────────────────────
def beta_root(sigma, mu, r):
    """特征方程 0.5σ²β(β−1) + μβ = r 的较大正根（数值稳定版）。

    展开：0.5σ²β² + (μ − 0.5σ²)β − r = 0
    取 + 号根：β = (−b + √(b² + 2σ²r)) / σ²，其中 b = μ − 0.5σ²。
    """
    s2 = np.maximum(np.asarray(sigma, dtype=float), 1e-8) ** 2
    b = np.asarray(mu, dtype=float) - 0.5 * s2
    disc = np.maximum(b * b + 2.0 * s2 * float(r), 0.0)
    return (-b + np.sqrt(disc)) / s2


def threshold_mult(beta):
    """阈值倍数 β/(β−1)。β ≤ 1 时理论上永不卖出 → 返回 inf。"""
    beta = np.asarray(beta, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        th = beta / (beta - 1.0)
    return np.where(np.isfinite(beta) & (beta > 1.0 + 1e-9), th, np.inf)


def mode_theory(args):
    """纯数学核验：σ / r / μ 网格下的 β 与阈值倍数。

    回答两个问题：
      1) 视频的 β=2.73 / 2.1 / 2.2（阈值 1.578 / 1.909 / 1.833）能否由
         同一组 (μ, r) 只变 σ 得到？（数字自洽性）
      2) r 的方向：μ 独立固定时 r↑ 阈值是升是降？（视频"推论②"条件性）
    """
    print("=" * 78)
    print("【数学核验 1】σ / r / μ 网格 → β 与阈值倍数 S*/K")
    print("=" * 78)
    rows = []
    for r in [0.02, 0.03, 0.05]:
        for mu in [0.0, 0.03, 0.05]:
            for sig in [0.15, 0.20, 0.25, 0.30, 0.40]:
                b = float(beta_root(sig, mu, r))
                th = float(threshold_mult(b))
                rows.append({'r': r, 'mu': mu, 'sigma': sig,
                             'beta': round(b, 3),
                             'S*/K': round(th, 3) if np.isfinite(th) else np.inf})
    df = pd.DataFrame(rows)
    piv = df.pivot_table(index=['r', 'mu'], columns='sigma', values='S*/K')
    print("\n阈值倍数 S*/K（行= r 与 μ，列= σ）")
    print(piv.round(3).to_string())

    print("\n" + "=" * 78)
    print("【数学核验 2】视频三组数字的来源反推")
    print("=" * 78)
    print("视频给出：β=2.73→阈值1.578（基准）；σ20%→30% 时 β=2.1→阈值1.909；")
    print("          贴现率变化 β=2.2→阈值1.833。")
    print("\n反推：给定 (μ, r)，只变 σ，能否同时命中 2.73 与 2.10？")
    for r in [0.02, 0.03, 0.05]:
        for mu in [0.0, 0.02, -0.02]:
            b15 = float(beta_root(0.15, mu, r))
            b20 = float(beta_root(0.20, mu, r))
            b30 = float(beta_root(0.30, mu, r))
            print(f"  r={r:.0%} μ={mu:+.0%} → β(σ15%)={b15:.3f} β(σ20%)={b20:.3f} "
                  f"β(σ30%)={b30:.3f}")

    print("\n" + "=" * 78)
    print("【数学核验 3】r 的方向（视频「推论②」是条件成立，非无条件）")
    print("=" * 78)
    print("μ 固定独立：")
    for mu in [0.0, 0.03]:
        line = []
        for r in [0.02, 0.03, 0.05, 0.08]:
            b = float(beta_root(0.25, mu, r))
            line.append(f"r={r:.0%}:β={b:.3f},S*/K={float(threshold_mult(b)):.3f}")
        print(f"  μ={mu:.0%}, σ=25% → " + " | ".join(line))
    print("\n风险中性 μ = r − δ（δ=股息率，μ 随 r 变）：")
    for delta in [0.03]:
        line = []
        for r in [0.02, 0.03, 0.05, 0.08]:
            b = float(beta_root(0.25, r - delta, r))
            line.append(f"r={r:.0%}:β={b:.3f},S*/K={float(threshold_mult(b)):.3f}")
        print(f"  δ={delta:.0%}, σ=25% → " + " | ".join(line))
    print("\n结论提示：μ 独立时 r↑→β↑→阈值↓（越没耐心越早兑现）；")
    print("          只有 μ=r−δ 风险中性框架下才是 r↑→阈值↑（视频的说法）。")


# ───────────────────────── 数据准备 ─────────────────────────
def rebal_date_set(trade_dates, freq='monthly'):
    """调仓日集合：monthly 每月首交易日 / quarterly 每季首 / annual 每年首。

    为什么要有这个：最优停止理论的持有期假设是**无限期**（永续美式期权），
    而策略实际持有期 = 调仓周期。月度调仓只持有 ~21 个交易日，价格几乎不可能
    涨到 2× 以上 → 阈值必然不触发。用 quarterly/annual 拉长持有期，才能区分
    「参数估不准」与「框架（持有期）错配」这两种失效原因。
    """
    s = pd.Series(list(trade_dates), dtype='int64')
    if freq == 'quarterly':
        ym = s // 100
        key = (ym // 100) * 10 + ((ym % 100 - 1) // 3 + 1)
    elif freq == 'annual':
        key = s // 10000
    else:
        key = s // 100
    return set(s.groupby(key.values).min().tolist())


def build_param_matrices(closes_full, trade_dates, window):
    """对数收益滚动 σ（年化）与 μ（年化）。index=trade_date 全历史，按需切片。"""
    px = closes_full.ffill()
    lr = np.log(px / px.shift(1))
    sigma = (lr.rolling(window, min_periods=max(5, window // 2)).std()
             * np.sqrt(TRADING_DAYS))
    mu = lr.rolling(window, min_periods=max(5, window // 2)).mean() * TRADING_DAYS
    return sigma, mu


# ───────────────────────── 回测主循环 ─────────────────────────
def _fit_buy(px, target, cash, td):
    """把买入股数缩放到现金装得下的最大值（含费），返回 (shares, fee)。"""
    if not np.isfinite(px) or px <= 0 or cash <= 0 or target <= 0:
        return 0.0, 0.0
    sh = target / px
    for _ in range(5):
        fee = calc_fee('buy', px, sh, trade_date=td)
        cost = sh * px + fee
        if cost <= cash:
            return sh, fee
        sh *= (cash / cost) * 0.999
        if sh * px < 1e-6:
            break
    return 0.0, 0.0


def simulate(trade_dates, closes_mat, col_idx, sigma_mat, mu_mat, basket_map,
             exit_rule, r, capital, mu_mode, div_yield, collect_thr=False,
             freq='monthly', rebal_mode='overlap', trim_keep=0.5):
    """单一退出规则的回测。返回 (nav, stats, thr_samples)。

    rebal_mode 见模块 docstring：full / overlap / drift。
    退出检查只在非调仓日进行（调仓日由再平衡主导），与 full 模式保持一致，
    保证换手口径是唯一被改动的变量。
    """
    n = len(trade_dates)
    cash = float(capital)
    pos = {}
    nav = np.zeros(n)
    ms = rebal_date_set(trade_dates, freq)

    n_buy = n_sell_rebal = n_sell_exit = n_trim = 0
    tot_fee = 0.0
    turnover = 0.0
    exit_pnl = []
    thr_samples = []
    max_gain = 0.0
    max_gain_chk = 0.0     # 仅在「判定有效」的 (日, 仓位) 上统计的最大浮盈
    n_sigma_nan = 0
    n_mult_bad = 0
    n_eval = 0
    n_ge = 0
    max_margin = -np.inf
    margin_at = None
    top_gain = []

    def _sell(px, sh, i_td):
        """卖出 sh 股：扣费入账并计入换手。"""
        nonlocal cash, tot_fee, turnover
        fee = calc_fee('sell', px, sh, trade_date=i_td)
        tot_fee += fee
        cash += sh * px - fee
        turnover += sh * px
        return fee

    def _add(px, sh, fee, c):
        """买入 sh 股：扣款并把 K 更新为加权平均成本。"""
        nonlocal cash, tot_fee, turnover, n_buy
        cash -= sh * px + fee
        tot_fee += fee
        turnover += sh * px
        old = pos.get(c)
        if old is None:
            pos[c] = {'sh': sh, 'k': px}
        else:
            new_sh = old['sh'] + sh
            pos[c] = {'sh': new_sh,
                      'k': (old['sh'] * old['k'] + sh * px) / new_sh}
        n_buy += 1

    def _mv():
        """当前持仓市值（跳过无效价）。"""
        return sum(d['sh'] * row[col_idx[c]] for c, d in pos.items()
                   if np.isfinite(row[col_idx[c]]))

    for i, td in enumerate(trade_dates):
        row = closes_mat[i]

        if td in ms:
            codes = [c for c in (basket_map.get(td) or []) if c in col_idx]

            if rebal_mode == 'full':
                # ── 全卖全买（初版口径，换手最重）──
                for c in list(pos):
                    px = row[col_idx[c]]
                    if not np.isfinite(px) or px <= 0:
                        continue
                    _sell(px, pos[c]['sh'], td)
                    n_sell_rebal += 1
                    del pos[c]
                if codes:
                    equity = float(cash)
                    w = 1.0 / len(codes)
                    for c in codes:
                        px = row[col_idx[c]]
                        if not np.isfinite(px) or px <= 0:
                            continue
                        sh, fee = _fit_buy(px, equity * w, cash, td)
                        if sh <= 0 or sh * px + fee > cash:
                            continue
                        _add(px, sh, fee, c)

            elif codes:
                # ── 保留重叠持仓：只卖出篮子的，其余 trim/补到目标 ──
                for c in list(pos):
                    if c in codes:
                        continue
                    px = row[col_idx[c]]
                    if not np.isfinite(px) or px <= 0:
                        continue
                    _sell(px, pos[c]['sh'], td)
                    n_sell_rebal += 1
                    del pos[c]

                w = 1.0 / len(codes)
                eq = float(cash) + _mv()

                # 等权再平衡：先把留在篮子里的超额部分 trim 掉（减仓不改 K）
                # 已触发过「高浮盈减仓」的票：保持半仓，不再补回等权，
                # 否则减仓效果会被下一次调仓立刻撤销、只剩白付的成本。
                if rebal_mode == 'overlap':
                    for c in list(pos):
                        if pos[c].get('trimmed'):
                            continue
                        px = row[col_idx[c]]
                        if not np.isfinite(px) or px <= 0:
                            continue
                        delta = eq * w - pos[c]['sh'] * px
                        if delta < -0.5:
                            sell_sh = min(pos[c]['sh'], -delta / px)
                            if sell_sh <= 0:
                                continue
                            _sell(px, sell_sh, td)
                            pos[c]['sh'] -= sell_sh
                            n_sell_rebal += 1
                            if pos[c]['sh'] <= 1e-9:
                                del pos[c]

                # 补买 / 新进
                for c in codes:
                    if pos.get(c, {}).get('trimmed'):
                        continue          # 减过仓的票不再补回
                    px = row[col_idx[c]]
                    if not np.isfinite(px) or px <= 0:
                        continue
                    delta = eq * w - pos.get(c, {'sh': 0.0})['sh'] * px
                    if delta <= 0.5:
                        continue
                    sh, fee = _fit_buy(px, delta, cash, td)
                    if sh <= 0 or sh * px + fee > cash:
                        continue
                    _add(px, sh, fee, c)

        elif exit_rule != 'hold' and pos:
            for c in list(pos):
                j = col_idx[c]
                px = row[j]
                if not np.isfinite(px) or px <= 0:
                    continue
                k = pos[c]['k']
                if exit_rule == 'fixed20':
                    mult = 1.20
                    thr = k * mult
                elif exit_rule == 'fixed30':
                    mult = 1.30
                    thr = k * mult
                elif exit_rule == 'trim20':
                    mult = 1.20
                    thr = k * mult
                elif exit_rule == 'trim30':
                    mult = 1.30
                    thr = k * mult
                elif exit_rule == 'optimal':
                    s = sigma_mat[i, j]
                    if not np.isfinite(s) or s <= 0:
                        n_sigma_nan += 1
                        continue
                    if mu_mode == 'zero':
                        mu_v = 0.0
                    elif mu_mode == 'realized':
                        m = mu_mat[i, j]
                        mu_v = m if np.isfinite(m) else 0.0
                    else:
                        mu_v = float(r) - float(div_yield)
                    mult = float(threshold_mult(beta_root(s, mu_v, r)))
                    if collect_thr:
                        thr_samples.append(mult)
                    if not np.isfinite(mult) or mult > 100:
                        n_mult_bad += 1
                        continue
                    thr = k * mult
                else:
                    continue

                n_eval += 1
                g = px / k if k > 0 else np.nan
                if np.isfinite(g):
                    if g > max_gain_chk:
                        max_gain_chk = g
                    # 决定性统计量：每次判定里「浮盈倍数 − 阈值倍数」的最大值。
                    # < 0 表示没有任何一次判定接近触发；数值即"最接近触发还差多少倍"。
                    margin = g - mult
                    if margin > max_margin:
                        max_margin = margin
                        margin_at = (td, c, g, mult)
                    top_gain.append((g, mult, td, c))
                    if len(top_gain) > 40000:
                        top_gain.sort(reverse=True)
                        del top_gain[10:]
                if px >= thr:
                    n_ge += 1
                    if exit_rule in ('trim20', 'trim30'):
                        # 高浮盈减仓：只卖出 (1-trim_keep) 比例，保留底仓。
                        # 标记 trimmed，避免同一票反复触发；K 保持不变。
                        if pos[c].get('trimmed'):
                            n_ge -= 1
                            continue
                        sell_sh = pos[c]['sh'] * (1.0 - trim_keep)
                        if sell_sh <= 0:
                            n_ge -= 1
                            continue
                        fee = calc_fee('sell', px, sell_sh, trade_date=td)
                        tot_fee += fee
                        cash += sell_sh * px - fee
                        turnover += sell_sh * px
                        exit_pnl.append(px / k - 1.0)
                        n_sell_exit += 1
                        n_trim += 1
                        pos[c]['sh'] -= sell_sh
                        pos[c]['trimmed'] = True
                        if pos[c]['sh'] <= 1e-9:
                            del pos[c]
                        continue
                    sh = pos[c]['sh']
                    fee = calc_fee('sell', px, sh, trade_date=td)
                    tot_fee += fee
                    cash += sh * px - fee
                    turnover += sh * px
                    exit_pnl.append(px / k - 1.0)
                    n_sell_exit += 1
                    del pos[c]

        eq = float(cash)
        for c, d in pos.items():
            j = col_idx[c]
            px = row[j]
            if np.isfinite(px):
                eq += d['sh'] * px
                # 自证：记录持仓期间出现过的最大 价格/成本 比。
                # 若最优阈值真在可达区间，max_gain 必然 >= 最小阈值；
                # 若 max_gain 远低于阈值，则"0 次触发"是真实的而非代码 bug。
                g = px / d['k'] if d['k'] > 0 else np.nan
                if np.isfinite(g) and g > max_gain:
                    max_gain = g
        nav[i] = eq

    yrs = n / TRADING_DAYS
    stats = {'n_buy': n_buy, 'n_sell_rebal': n_sell_rebal,
             'n_sell_exit': n_sell_exit, 'n_trim': n_trim, 'total_fee': tot_fee,
             'turnover': turnover,
             'turnover_ann': turnover / yrs / float(capital) if yrs > 0 else np.nan,
             'fee_ann': tot_fee / yrs / float(capital) if yrs > 0 else np.nan,
             'max_gain': max_gain,
             'max_gain_chk': max_gain_chk,
             'n_eval': n_eval, 'n_ge': n_ge,
             'n_sigma_nan': n_sigma_nan, 'n_mult_bad': n_mult_bad,
             'max_margin': max_margin if np.isfinite(max_margin) else np.nan,
             'margin_at': margin_at,
             'top_gain': sorted(top_gain, reverse=True)[:5],
             'exit_pnl_mean': float(np.mean(exit_pnl)) if exit_pnl else np.nan,
             'exit_pnl_n': len(exit_pnl)}
    return nav, stats, thr_samples


def metrics(nav):
    nav = np.asarray(nav, dtype=float)
    if len(nav) < 2 or nav[0] <= 0:
        return {}
    ret = nav[1:] / nav[:-1] - 1.0
    yrs = len(nav) / TRADING_DAYS
    peak = np.maximum.accumulate(nav)
    dd = nav / peak - 1.0
    return {
        'total': nav[-1] / nav[0] - 1.0,
        'cagr': (nav[-1] / nav[0]) ** (1.0 / yrs) - 1.0 if yrs > 0 else np.nan,
        'mdd': float(dd.min()),
        'sharpe': float(ret.mean() / ret.std() * np.sqrt(TRADING_DAYS))
                  if ret.std() > 0 else np.nan,
        'vol': float(ret.std() * np.sqrt(TRADING_DAYS)),
    }


# ───────────────────────── 回测模式 ─────────────────────────
def mode_backtest(args):
    # ⚠️ 口径分离（run_daily20_macd.load_closes 文档硬要求）：
    #   选股信号（build_vol_lookup → select_div_low_vol）**永远用 raw**，
    #   NAV 估值与成交才按 --price-mode 切换。两处共用同一矩阵会把
    #   「改复权口径」与「改选股」两个变量耦合，双跑结论不可归因。
    codes, closes_sig = load_closes(hfq=False)
    if args.price_mode == 'hfq':
        _, closes_px = load_closes(hfq=True)
        # hfq 与 raw 的票集应一致；按 raw 的列序对齐，避免列错位
        closes_px = closes_px.reindex(columns=closes_sig.columns)
        print(f"[load] 选股矩阵(raw) {closes_sig.shape} / "
              f"成交矩阵(hfq) {closes_px.shape}")
    else:
        closes_px = closes_sig
        print(f"[load] 收盘矩阵(raw) {closes_sig.shape}")

    all_dates = [int(d) for d in closes_sig.index]
    trade_dates = [d for d in all_dates if args.start <= d <= args.end]
    if not trade_dates:
        print(f"[error] 区间 {args.start}~{args.end} 无交易日")
        sys.exit(2)
    print(f"[区间] {trade_dates[0]} ~ {trade_dates[-1]}，共 {len(trade_dates)} 个交易日")

    # 选股永远走 raw 矩阵（隔离变量）
    vol_lookup = build_vol_lookup(closes_sig, window=120)
    ms = rebal_date_set(trade_dates, args.rebal_freq)

    basket_map = {}
    print(f"[选股] 预计算 {len(ms)} 个调仓日的红利低波 top{args.top_n} 篮子...")
    for i, td in enumerate(trade_dates):
        if td not in ms:
            continue
        prev_td = trade_dates[i - 1] if i > 0 else td
        basket_map[td] = select_div_low_vol(prev_td, args.top_n, vol_lookup)
    n_ok = sum(1 for v in basket_map.values() if v)
    print(f"[选股] 完成：{n_ok}/{len(ms)} 个调仓日有非空篮子")

    cols = list(closes_sig.columns)
    col_idx = {c: j for j, c in enumerate(cols)}
    # 成交/估值/σ-μ 一律走 closes_px（raw 或 hfq）
    closes_mat = closes_px.loc[trade_dates].values

    exits = [e.strip() for e in args.exits.split(',') if e.strip()]
    results = []
    thr_all = {}

    for sw in [int(x) for x in args.sigma_window.split(',')]:
        # σ/μ 是退出规则的定价输入，应与 NAV 同口径（GBM 描述的是含分红的总收益过程）
        sigma, mu = build_param_matrices(closes_px, trade_dates, sw)
        sigma_mat = sigma.loc[trade_dates].values
        mu_mat = mu.loc[trade_dates].values
        print(f"\n[参数] 调仓={args.rebal_freq}/{args.rebal_mode}/{args.price_mode} "
              f"σ窗口={sw}日 r={args.r} μ模式={args.mu_mode}")
        for ex in exits:
            nav, st, thr = simulate(
                trade_dates, closes_mat, col_idx, sigma_mat, mu_mat,
                basket_map, ex, args.r, args.capital, args.mu_mode,
                args.div_yield, collect_thr=(ex == 'optimal'),
                freq=args.rebal_freq, rebal_mode=args.rebal_mode,
                trim_keep=args.trim_keep)
            m = metrics(nav)
            if not m:
                continue
            row = {'exit': ex, 'sigma_win': sw, **m,
                   'trades': st['n_buy'] + st['n_sell_rebal'] + st['n_sell_exit'],
                   'exit_sells': st['n_sell_exit'],
                   'n_trim': st['n_trim'],
                   'fee': st['total_fee'],
                   'turn_ann': st['turnover_ann'],
                   'fee_ann': st['fee_ann'],
                   'exit_pnl': st['exit_pnl_mean']}
            results.append(row)
            if ex == 'optimal' and thr:
                thr_all[sw] = np.array(thr, dtype=float)
            print(f"  {ex:<9} 总收益 {m['total']*100:+7.2f}% | 年化 {m['cagr']*100:+6.2f}% "
                  f"| 回撤 {m['mdd']*100:7.2f}% | Sharpe {m['sharpe']:5.2f} "
                  f"| 年换手 {st['turnover_ann']*100:6.1f}% | 年费用 "
                  f"{st['fee_ann']*100:5.2f}% | 退出卖出 {st['n_sell_exit']:4d} 次"
                  + (f" | 最大浮盈(全/判定) {st['max_gain']:.2f}/{st['max_gain_chk']:.2f}× "
                     f"| 判定{st['n_eval']} 命中{st['n_ge']} σ缺失{st['n_sigma_nan']} "
                     f"阈值失效{st['n_mult_bad']} "
                     f"| 最接近触发 margin {st['max_margin']:+.2f}× "
                     f"({st['margin_at'][0]} {st['margin_at'][1]} "
                     f"浮盈{st['margin_at'][2]:.2f}× vs 阈值{st['margin_at'][3]:.2f}×)"
                     if ex == 'optimal' and st['margin_at'] else ""))

        if sw in thr_all:
            t = thr_all[sw]
            t = t[np.isfinite(t) & (t <= 100)]
            if len(t):
                print(f"  [阈值分布 σ窗口={sw}] 判定 {len(t)} 次："
                      f"中位 {np.median(t):.2f}× | 均值 {t.mean():.2f}× | "
                      f"p10 {np.percentile(t,10):.2f}× | p90 {np.percentile(t,90):.2f}× | "
                      f"最小 {t.min():.2f}×")

        if ex == 'optimal' and st['top_gain']:
            print("    浮盈最高的 5 个仓位判定（验证「涨得多的 = 高波动 → 阈值更高」自败机制）：")
            for g, mu_, td, c in st['top_gain']:
                print(f"      {td} {c}  浮盈 {g:.2f}×  当时阈值 {mu_:.2f}×  "
                      f"缺口 {g - mu_:+.2f}×")

    df = pd.DataFrame(results)
    print("\n" + "=" * 96)
    print("汇总")
    print("=" * 96)
    show = df.copy()
    for c in ['total', 'cagr', 'mdd', 'vol', 'turn_ann', 'fee_ann']:
        show[c] = (show[c] * 100).round(2)
    show = show.rename(columns={'total': '总收益%', 'cagr': '年化%', 'mdd': '最大回撤%',
                                'vol': '年化波动%', 'sharpe': 'Sharpe',
                                'trades': '交易数', 'exit_sells': '退出卖出',
                                'fee': '总费用', 'exit_pnl': '退出均收益',
                                'turn_ann': '年换手%', 'fee_ann': '年费用%',
                                'sigma_win': 'σ窗口', 'exit': '退出规则'})
    print(show.to_string(index=False))

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(
        OUT_DIR,
        f"optimal_stop_{args.start}_{args.end}_{args.rebal_freq}_"
        f"{args.rebal_mode}_{args.mu_mode}_{args.price_mode}.csv")
    df.to_csv(out, index=False, encoding='utf-8-sig')
    print(f"\n[输出] {out}")


def main():
    ap = argparse.ArgumentParser(description='最优停止阈值 vs 固定止盈 A/B')
    ap.add_argument('--mode', default='backtest', choices=['theory', 'backtest'])
    ap.add_argument('--start', type=int, default=20200101)
    ap.add_argument('--end', type=int, default=20251231)
    ap.add_argument('--top-n', type=int, default=20)
    ap.add_argument('--capital', type=float, default=1_000_000)
    ap.add_argument('--exits', default='hold,fixed20,fixed30,optimal')
    ap.add_argument('--trim-keep', type=float, default=0.5,
                    help='trim20/trim30 触发后保留的仓位比例（0.5=减半仓）')
    ap.add_argument('--sigma-window', default='250')
    ap.add_argument('--r', type=float, default=0.03)
    ap.add_argument('--mu-mode', default='zero', choices=['zero', 'realized', 'neutral'])
    ap.add_argument('--div-yield', type=float, default=0.035)
    ap.add_argument('--rebal-freq', default='monthly',
                    choices=['monthly', 'quarterly', 'annual'])
    ap.add_argument('--rebal-mode', default='overlap',
                    choices=['full', 'overlap', 'drift'])
    ap.add_argument('--price-mode', default='raw', choices=['raw', 'hfq'])
    args = ap.parse_args()

    if args.mode == 'theory':
        mode_theory(args)
    else:
        mode_backtest(args)


if __name__ == '__main__':
    main()
