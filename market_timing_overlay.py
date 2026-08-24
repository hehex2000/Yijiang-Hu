"""市场情绪广度振荡器 + 非对称择时闸门 (market timing overlay)

设计目标（2026-08-21，来自 B站工具演示视频 BV1etgF6nEPD 的屎里淘金·A项）：
  平台最大空白 = 所有策略满仓穿越、无大盘风控。本模块补一个「市场择时 overlay」：
  用全市场广度合成 0-100 振荡器，沸点(>=boil)清仓、冰点(<=ice)满仓，中间线性减仓。
  关键约束 = 非对称「只卖不买」：闸门只 ever 把仓位从策略自然水平往下压（cap ∈ [floor, 1]），
  绝不在恐慌冰点强制加仓（抄底接飞刀）。它补的是「下行保护」，不是「择时增强」。

接入方式：
  - 主策略回测里：先算基线月度收益 ret_ser（无闸门），再算振荡器 osc_ser，
    用 position_cap 得到每月 cap ∈ [0,1]，把该月收益缩放为 cap*ret、并记现金 sleeve 切换成本。
  - 阈值(boil/ice)必须经 walk_forward_thresholds 滚动选出，避免在全样本上 eyeball 挑阈过拟合。

注意内存：compute_breadth_oscillator 对 (date × stock) 矩阵做多个 rolling（含 rolling(252)），
    全市场 5812 只 × 3548 日约 1GB 瞬态。本机内存够；沙箱验证请用子采样矩阵（见文件末测试）。
"""

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# 1. 市场广度振荡器（0-100）
# ----------------------------------------------------------------------------
def compute_breadth_oscillator(close_p, ma=(20, 60, 200), hi_win=252, lo_win=252):
    """用全市场收盘价矩阵合成 0-100 市场情绪振荡器。

    成分（各归一化到 0-1，等权平均 ×100）：
      1-3. 站上 MA20 / MA60 / MA200 的个股占比
      4.  涨跌家数比 AD% = ((adv-dec)/(adv+dec)+1)/2
      5.  新高-新低 NH-NL = ((nh-nl)/(nh+nl)+1)/2（相对 trailing 252 日高低）

    close_p : DataFrame，index=trade_date(已排序)，columns=ts_code，values=close(原始价，非复权)
    返回    : Series(index=close_p.index, 0-100)  —— 早期无足够历史(如 MA200)为 NaN。
    """
    px = close_p.sort_index()
    comps = []

    # 站上各周期均线占比
    for w in ma:
        m = px.rolling(w, min_periods=max(5, w // 2)).mean()
        above = (px >= m)            # NaN(未上市) 比较得 False → 计入"不在上方"，早期略偏空，可接受
        comps.append(above.mean(axis=1))

    # 涨跌家数比（当日收益符号）
    ret = px.diff()
    adv = (ret > 0).sum(axis=1)
    dec = (ret < 0).sum(axis=1)
    ad = ((adv - dec) / (adv + dec + 1e-9) + 1) / 2
    comps.append(ad)

    # 新高-新低（相对 trailing 252 日高低，shift(1) 避免用当日本身）
    hi = px.rolling(hi_win, min_periods=60).max()
    lo = px.rolling(lo_win, min_periods=60).min()
    nh = (px >= hi.shift(1)).sum(axis=1)
    nl = (px <= lo.shift(1)).sum(axis=1)
    nhnl = ((nh - nl) / (nh + nl + 1e-9) + 1) / 2
    comps.append(nhnl)

    osc = pd.concat(comps, axis=1).mean(axis=1) * 100
    return osc


# ----------------------------------------------------------------------------
# 2. 非对称闸门：position cap ∈ [floor, 1]，monotonic decreasing in oscillator
# ----------------------------------------------------------------------------
def position_cap(osc_value, boil, ice, floor=0.0):
    """振荡器 → 目标仓位上限 cap。

    osc >= boil : 返回 floor（清仓，默认 0）
    osc <= ice  : 返回 1.0（满仓，=策略自然水平）
    ice<osc<boil: 线性从 1.0 降到 floor
    floor       : 清仓时的最低保留仓位（默认 0=全清；设 0.2 表示永不空仓、最多降到 2 成）

    非对称 = cap 永远 <= 1.0：闸门只减仓（卖），绝不在冰点强制加仓（接飞刀）。
    """
    if not (osc_value == osc_value):      # NaN → 无信号，默认满仓
        return 1.0
    if osc_value >= boil:
        return max(0.0, floor)
    if osc_value <= ice:
        return 1.0
    return max(0.0, floor, (boil - osc_value) / (boil - ice))


def build_gate_series(osc_series, boil, ice, floor=0.0):
    """osc_series(Series) → cap Series(同 index)。"""
    return osc_series.apply(lambda x: position_cap(x, boil, ice, floor))


# ----------------------------------------------------------------------------
# 3. 应用闸门到收益序列（含现金 sleeve 切换成本）
# ----------------------------------------------------------------------------
def _gated_nav(osc_arr, ret_arr, boil, ice, floor=0.0, cost=0.002):
    """给定振荡器数组与月度收益数组，返回 (nav, r_gated, caps)。

    模型：
      r_gated[t] = cap[t] * ret[t]  -  cost * |cap[t] - cap[t-1]|
      cap[t]*ret[t]        : 仅部署 cap 比例到策略，剩余(1-cap)现金(收益0)；
                              cap=0 时选择成本被乘零（空仓不换股，正确）。
      cost*|Δcap|          : 现金 sleeve 买卖的单向成本（1→0 清仓付 cost，0→1 回补付 cost）。
    首月 cap[t-1] 视为=cap[t]（无切换成本）。
    """
    caps = np.array([position_cap(o, boil, ice, floor) for o in osc_arr])
    gc = np.abs(np.diff(caps, prepend=caps[0])) * cost   # 首月 diff=0
    r = caps * np.asarray(ret_arr, dtype=float) - gc
    nav = np.cumprod(1.0 + r)
    return nav, r, caps


def _metrics(r, rf=0.025):
    r = np.asarray(r, dtype=float)
    if len(r) == 0:
        return 0.0, 0.0, 0.0
    nav = np.cumprod(1.0 + r)
    n = len(r)
    yrs = n / 12.0
    cagr = (nav[-1] / nav[0]) ** (1.0 / yrs) - 1 if yrs > 0 else 0.0
    peak = np.maximum.accumulate(nav)
    mdd = (nav / peak - 1).min()
    sr = r - rf / 12.0
    sharpe = (sr.mean() * 12.0) / (r.std() * np.sqrt(12.0)) if r.std() > 0 else 0.0
    return cagr, mdd, sharpe


# ----------------------------------------------------------------------------
# 4. Walk-forward 选阈（防过拟合）
# ----------------------------------------------------------------------------
def walk_forward_thresholds(osc_ser, ret_ser, train_months=36,
                            grid_boil=(70, 75, 80, 85, 90),
                            grid_ice=(10, 15, 20, 25, 30),
                            floor=0.0, cost=0.002, dd_limit=0.30, rf=0.025):
    """逐月滚动选阈：每个月用前 train_months 个月的 (osc, ret) 在网格上挑 (boil,ice)，
    用「最大化 cagr 但 maxDD 不得差于 -dd_limit」的目标，再把该阈值应用到当月（样本外累积）。

    返回 (boil, ice, oos_metrics) ：
      boil, ice   = 最后一个 fold 选定的「上线参数」（代表最新市场状态）
      oos_metrics = 全样本外累积窗口的指标 dict{cagr, mdd, sharpe}
    """
    df = pd.DataFrame({"osc": osc_ser, "ret": ret_ser}).dropna()
    df = df.sort_index()
    months = list(df.index)
    if len(months) <= train_months:
        # 数据不足：退化为全样本挑阈
        best_b, best_i, _ = _pick_best(df["osc"].values, df["ret"].values,
                                       grid_boil, grid_ice, floor, cost, dd_limit, rf)
        return best_b, best_i, _metrics(df["ret"].values, rf)

    oos_r = []
    last_b, last_i = grid_boil[-1], grid_ice[0]
    for i in range(train_months, len(months)):
        tr_osc = df["osc"].values[i - train_months:i]
        tr_ret = df["ret"].values[i - train_months:i]
        te_osc = df["osc"].values[i]
        te_ret = df["ret"].values[i]
        b, ice, _ = _pick_best(tr_osc, tr_ret, grid_boil, grid_ice, floor, cost, dd_limit, rf)
        last_b, last_i = b, ice
        _, r_g, _ = _gated_nav(np.array([te_osc]), np.array([te_ret]), b, ice, floor, cost)
        oos_r.append(r_g[0])
    return last_b, last_i, _metrics(oos_r, rf)


def _pick_best(tr_osc, tr_ret, grid_boil, grid_ice, floor, cost, dd_limit, rf):
    best_score, best_b, best_i = -1e9, grid_boil[-1], grid_ice[0]
    for b in grid_boil:
        for ice in grid_ice:
            _, _, caps = _gated_nav(tr_osc, tr_ret, b, ice, floor, cost)
            cagr, mdd, _ = _metrics(tr_ret * caps - np.abs(np.diff(caps, prepend=caps[0])) * cost, rf)
            # 目标：maxDD 不差于 -dd_limit 时最大化 cagr；违反时按 maxDD 打分（越接近0越好）
            score = cagr if mdd >= -dd_limit else mdd
            if score > best_score:
                best_score, best_b, best_i = score, b, ice
    return best_b, best_i, best_score
