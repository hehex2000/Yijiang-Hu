# -*- coding: utf-8 -*-
"""
risk_metrics.py — 尾部风险度量模块（测度论视角）

对应《概率根本不是你想的那样》视频的核心论点：
  概率不是平面百分比，而是事件空间上的"体积度量"。
  单点概率≈0，但区间面积（尾部体积）才是真正决定风险的东西。

本模块从 净值/日收益 序列计算：
  - VaR      (历史法 / 参数正态 / Cornish-Fisher 调整)  —— 分位左侧"面积"
  - CVaR/ES  (历史法)                                    —— 超过 VaR 的平均亏损（尾部"体积"）
  - 偏度 / 超额峰度                                       —— 分布是否左偏、是否肥尾
  - 尾部比率 (tail ratio)                                 —— 右尾/左尾 面积不对称
  - 最大回撤（净值口径，便于统一报告）

设计：纯 numpy/pandas，零第三方依赖（norm_ppf 用 Acklam 近似，不用 scipy）。
可直接被任意回测引擎调用：传入 nav（净值序列）或 returns（日收益序列）皆可。

集成示例（接 run_livermore_v2.run_window 的返回值）：
    from risk_metrics import risk_summary
    r = run_window(start, end, cfg, layers)
    nav = [v for _, v in r["nav"]]
    rep = risk_summary(nav=nav, label="利弗莫尔-全改进")
    print(rep["report"])
"""

import numpy as np
import pandas as pd

__all__ = [
    "to_simple_returns", "skewness", "excess_kurtosis", "max_drawdown_from_nav",
    "var_historical", "cvar_historical", "var_parametric_normal",
    "var_cornish_fisher", "tail_ratio", "norm_ppf", "risk_summary",
]


# ────────────────────────────────────────────────────────────────────────────
#  基础工具
# ────────────────────────────────────────────────────────────────────────────

def _as_float_array(x):
    """接受 list / ndarray / Series / 净值序列，返回干净的一维 float 数组（去 NaN/inf）。"""
    arr = np.asarray(pd.Series(x).astype(float).values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return arr


def to_simple_returns(nav):
    """由净值序列算日简单收益 r_t = nav_t/nav_{t-1} - 1。"""
    nav = _as_float_array(nav)
    if len(nav) < 2:
        return np.array([])
    ret = nav[1:] / nav[:-1] - 1.0
    return ret[np.isfinite(ret)]


def norm_ppf(p):
    """
    标准正态分位数（逆 CDF），Acklam 有理逼近，纯 numpy 实现（避免 scipy 依赖）。
    p in (0,1)。极端 p 误差 < 1e-9，足够金融用途。
    """
    p = np.asarray(p, dtype=float)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    x = np.empty_like(p)
    mask_low = p < plow
    mask_high = p > phigh
    mask_mid = ~(mask_low | mask_high)
    q = p - 0.5
    r = q * q
    x[mask_mid] = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
                  (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)
    if np.any(mask_low):
        q = np.sqrt(-2 * np.log(p[mask_low]))
        x[mask_low] = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                      ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
    if np.any(mask_high):
        q = np.sqrt(-2 * np.log(1 - p[mask_high]))
        x[mask_high] = -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                       ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
    return x


def skewness(x):
    """Fisher-Pearson 标准化偏度。负=左偏（亏损尾更肥），正=右偏（盈利尾更肥）。"""
    x = _as_float_array(x)
    n = len(x)
    if n < 3:
        return np.nan
    mu = x.mean()
    s = x.std(ddof=0)
    if s == 0:
        return 0.0
    return float((np.mean((x - mu) ** 3)) / (s ** 3))


def excess_kurtosis(x):
    """超额峰度（Fisher, 已减 3）。0=正态，正=肥尾（极端值更多）。"""
    x = _as_float_array(x)
    n = len(x)
    if n < 4:
        return np.nan
    mu = x.mean()
    s = x.std(ddof=0)
    if s == 0:
        return 0.0
    return float((np.mean((x - mu) ** 4)) / (s ** 4) - 3.0)


def max_drawdown_from_nav(nav):
    """净值口径最大回撤（负值，如 -0.55 表示 -55%）。"""
    nav = _as_float_array(nav)
    if len(nav) < 2:
        return 0.0
    peak = np.maximum.accumulate(nav)
    dd = nav / peak - 1.0
    return float(dd.min())


# ────────────────────────────────────────────────────────────────────────────
#  VaR / CVaR
# ────────────────────────────────────────────────────────────────────────────

def var_historical(returns, alpha=0.95):
    """历史法 VaR（单日）。返回正数=亏损幅度。取收益分布 (1-alpha) 左分位，VaR = -分位值。"""
    r = _as_float_array(returns)
    if len(r) == 0:
        return np.nan
    q = np.percentile(r, (1 - alpha) * 100.0)
    return float(-q)


def cvar_historical(returns, alpha=0.95):
    """历史法 CVaR / Expected Shortfall（单日）。返回正数=平均亏损幅度。
    超过 VaR 阈值那部分收益的均值（更负），取负。"""
    r = _as_float_array(returns)
    if len(r) == 0:
        return np.nan
    thr = np.percentile(r, (1 - alpha) * 100.0)
    tail = r[r <= thr]
    if len(tail) == 0:
        return float(-thr)
    return float(-tail.mean())


def var_parametric_normal(returns, alpha=0.95):
    """参数法 VaR，假设收益正态。VaR = -(mu + sigma * z_{1-alpha})。"""
    r = _as_float_array(returns)
    if len(r) < 2:
        return np.nan
    mu, sigma = r.mean(), r.std(ddof=1)
    z = norm_ppf(1 - alpha)          # 左尾分位，负值（如 95% → -1.645）
    return float(-(mu + sigma * z))


def var_cornish_fisher(returns, alpha=0.95):
    """Cornish-Fisher 调整 VaR：用偏度/超额峰度修正正态分位，对左偏/肥尾更敏感。
        z_cf = z + (z^2-1)S/6 + (z^3-3z)K/24 - (2z^3-5z)S^2/36
    VaR = -(mu + sigma * z_cf)"""
    r = _as_float_array(returns)
    if len(r) < 4:
        return np.nan
    mu, sigma = r.mean(), r.std(ddof=1)
    S = skewness(r)
    K = excess_kurtosis(r)
    if not np.isfinite(S) or not np.isfinite(K):
        return var_parametric_normal(r, alpha)
    z = norm_ppf(1 - alpha)
    z_cf = z + (z**2 - 1) * S / 6.0 + (z**3 - 3 * z) * K / 24.0 - (2 * z**3 - 5 * z) * S**2 / 36.0
    return float(-(mu + sigma * z_cf))


def tail_ratio(returns, alpha=0.95):
    """尾部比率 = |右尾分位| / |左尾分位|（绝对值）。>1 右尾更肥；<1 左尾更肥（警惕）。"""
    r = _as_float_array(returns)
    if len(r) == 0:
        return np.nan
    left = abs(np.percentile(r, (1 - alpha) * 100.0))
    right = abs(np.percentile(r, alpha * 100.0))
    if left == 0:
        return float(np.inf)
    return float(right / left)


# ────────────────────────────────────────────────────────────────────────────
#  汇总
# ────────────────────────────────────────────────────────────────────────────

def risk_summary(nav=None, returns=None, label="策略", periods=252, alphas=(0.95, 0.99)):
    """统一风险度量。传入 nav 或 returns 之一。返回 dict（含 'report' 字符串）。"""
    if returns is None:
        if nav is None:
            raise ValueError("必须传入 nav 或 returns 之一")
        returns = to_simple_returns(nav)
    r = _as_float_array(returns)
    n = len(r)

    out = dict(label=label, n_days=n)
    if n < 2:
        out["report"] = f"[风险度量] {label}: 样本不足({n}日)，跳过"
        return out

    mu_d, sd_d = r.mean(), r.std(ddof=1)
    out["daily_mean"] = float(mu_d)
    out["daily_vol"] = float(sd_d)
    out["ann_vol"] = float(sd_d * np.sqrt(periods))
    out["skew"] = float(skewness(r))
    out["excess_kurt"] = float(excess_kurtosis(r))
    out["worst_day"] = float(r.min())
    out["best_day"] = float(r.max())
    out["tail_ratio"] = float(tail_ratio(r, 0.95))
    if nav is not None:
        out["max_drawdown"] = float(max_drawdown_from_nav(nav))
    else:
        out["max_drawdown"] = float(max_drawdown_from_nav(np.cumprod(1 + r)))

    for a in alphas:
        out[f"var_{int(a*100)}_hist"] = float(var_historical(r, a))
        out[f"var_{int(a*100)}_norm"] = float(var_parametric_normal(r, a))
        out[f"var_{int(a*100)}_cf"] = float(var_cornish_fisher(r, a))
        out[f"var_{int(a*100)}_cvar"] = float(cvar_historical(r, a))
        out[f"var_{int(a*100)}_norm_ann"] = float(var_parametric_normal(r, a) * np.sqrt(periods))

    # ── 报告字符串（中文，匹配平台风格） ──
    lines = []
    lines.append(f"═══ 风险度量（测度论视角）: {label} ══")
    lines.append(f"  样本天数: {n}  日均值: {mu_d:+.4%}  日波动: {sd_d:.4%}  年化波动: {out['ann_vol']:.2%}")
    lines.append(f"  偏度: {out['skew']:+.3f}   超额峰度: {out['excess_kurt']:+.3f}   "
                 f"(偏度<0左偏 / 峰度>0肥尾)")
    lines.append(f"  尾部比率(95%): {out['tail_ratio']:.2f}   "
                 f"(>1右尾肥 / <1左尾肥，需警惕)")
    lines.append(f"  最差单日: {out['worst_day']:+.2%}   最大回撤: {out['max_drawdown']:+.2%}")
    for a in alphas:
        h = out[f"var_{int(a*100)}_hist"]
        cf = out[f"var_{int(a*100)}_cf"]
        norm = out[f"var_{int(a*100)}_norm"]
        cv = out[f"var_{int(a*100)}_cvar"]
        lines.append(
            f"  VaR{a*100:.0f}%: 历史 {h:.2%} | 正态 {norm:.2%} | CF调整 {cf:.2%}   "
            f"CVaR{a*100:.0f}%: {cv:.2%}")
        gap = cf - norm
        if gap > 1e-4:
            lines.append(f"           → CF-正态差 {gap:+.2%}  ⚠ 左偏/肥尾被正态低估")
        elif abs(gap) <= 1e-4:
            lines.append(f"           → CF-正态差 {gap:+.2%}  正态近似ok")
    lines.append(f"  [解读] CVaR>VaR 表示尾部平均亏损比分位更重；CF调整>正态 表示"
                 f"分布非正态已放大左尾风险（视频: 模型把尾部切太薄会低估极端风险）")
    out["report"] = "\n".join(lines)
    return out


# ────────────────────────────────────────────────────────────────────────────
#  自检
# ────────────────────────────────────────────────────────────────────────────

def _self_test():
    rng = np.random.default_rng(20260812)
    print("── 自检 1：正态收益（应 skew≈0, 峰度≈0, CF≈正态） ──")
    normal = rng.normal(0.0003, 0.01, 4000)
    s1 = risk_summary(returns=normal, label="正态合成")
    print(s1["report"])

    print("\n── 自检 2：左偏+肥尾（偶发暴跌，应 skew<0, 峰度>0, CF>正态, CVaR>VaR） ──")
    base = rng.normal(0.0004, 0.009, 4000)
    crash = rng.normal(-0.045, 0.02, 4000)
    mask = rng.random(4000) < 0.03          # 3% 概率暴跌日
    fat = np.where(mask, crash, base)
    s2 = risk_summary(returns=fat, label="左偏肥尾合成")
    print(s2["report"])

    print("\n── 断言检查 ──")
    ok = True
    if not (abs(s1["skew"]) < 0.1 and abs(s1["excess_kurt"]) < 0.3):
        print("✗ 正态自检偏度/峰度异常"); ok = False
    if not (s2["skew"] < -0.1):
        print("✗ 左偏合成偏度未<0"); ok = False
    if not (s2["excess_kurt"] > 0.5):
        print("✗ 左偏合成峰度未>0"); ok = False
    if not (s2["var_95_cf"] > s2["var_95_norm"]):
        print("✗ 左偏下 CF-VaR 未大于正态VaR"); ok = False
    if not (s2["var_95_cvar"] > s2["var_95_hist"]):
        print("✗ CVaR 未大于 VaR"); ok = False
    print("✓ 全部通过" if ok else "✗ 存在失败项")


if __name__ == "__main__":
    _self_test()
