# -*- coding: utf-8 -*-
"""
chan_lun_core.py — 缠论概念 → 可量化机器实现（核心检测库）
============================================================
设计原则（来自 plan_zhongshu_*.md 两份规划文档）:
  1. 无未来函数: 所有函数接收"截至 T-1 的序列", 返回"在 T-1 这一刻能知道的信号"。
     调用方负责把序列截断到 T-1, 信号在 T 开盘执行。本库函数本身是纯函数。
  2. 阈值是唯一旋钮(玄学), 所有阈值必须网格搜索, 不宣称某值"正确"。
  3. 结构检测只能作"确认层"/"风险降档", 不能当唯一信号(与利弗莫尔同源死结)。
  4. 不手工画线(不可标准回测); 全部规则化、透明、可复现。

缠论层 → 量化等价映射:
  分型   → Fractal (N 根两侧极值)
  笔     → Zigzag leg (相邻拐点价格变动 >= threshold)
  线段   → Swing structure (HH-HL / LH-LL 链)
  中枢   → 布林带宽滚动分位低位 (波动率收缩 / 盘整期)
  背驰   → MACD 背离 (价格极值与 MACD 极值反向)
  三类买卖点 → 由 笔/线段/中枢/背驰 派生的近似规则 (仅供研究)

依赖: numpy / pandas (不依赖 talib, 纯 numpy 实现 MACD)。
"""

import numpy as np
import pandas as pd

__all__ = [
    "fractals", "zigzag", "swing_trend", "bollinger_width",
    "bb_width_pct", "in_consolidation", "macd", "macd_divergence",
    "classify_buy_points", "ZIGZAG_DEFAULT", "CONSOL_DEFAULT",
]


# =====================================================================
# 1. 分型 Fractal — 平凡积木
# =====================================================================
def fractals(highs, lows, n=2):
    """返回 (top_idx[], bottom_idx[])。
    顶分型: high[i] 严格大于左右各 n 根; 底分型: low[i] 严格小于左右各 n 根。
    平台型(相等)只取最右, 避免重复极值 — 调用方用 zigzag 进一步过滤。
    """
    h = np.asarray(highs, dtype=float)
    l = np.asarray(lows, dtype=float)
    N = len(h)
    if N < 2 * n + 1:
        return [], []
    tops, bots = [], []
    for i in range(n, N - n):
        is_top = h[i] > max(h[i - n], h[i + n]) and all(
            h[i] >= h[j] for j in range(i - n, i + n + 1) if j != i)
        is_bot = l[i] < min(l[i - n], l[i + n]) and all(
            l[i] <= l[j] for j in range(i - n, i + n + 1) if j != i)
        if is_top:
            tops.append(i)
        if is_bot:
            bots.append(i)
    return tops, bots


# =====================================================================
# 2. 笔 Zigzag leg — threshold 是唯一旋钮
# =====================================================================
ZIGZAG_DEFAULT = dict(threshold=0.05, n=2)


def zigzag(highs, lows, threshold=0.05, n=2):
    """交替顶底序列, 相邻反向拐点价格变动 >= threshold 才保留。
    返回 pivots: list of {'i','kind','price'} (kind in 'high'/'low')。
    """
    highs = np.asarray(highs, dtype=float)
    lows = np.asarray(lows, dtype=float)
    tops, bots = fractals(highs, lows, n)
    allp = sorted([(i, "high", highs[i]) for i in tops] +
                  [(i, "low", lows[i]) for i in bots])
    pivots = []
    last = None  # (i, kind, price)
    for i, kind, px in allp:
        if last is None:
            pivots.append({"i": i, "kind": kind, "price": px})
            last = (i, kind, px)
            continue
        if kind == last[1]:
            # 同向: 保留更极值者 (合并平台)
            better = (kind == "high" and px > last[2]) or (kind == "low" and px < last[2])
            if better:
                pivots[-1] = {"i": i, "kind": kind, "price": px}
                last = (i, kind, px)
            continue
        move = abs(px - last[2]) / last[2] if last[2] != 0 else 0.0
        if move >= threshold:
            pivots.append({"i": i, "kind": kind, "price": px})
            last = (i, kind, px)
    return pivots


# =====================================================================
# 3. 线段 Swing structure — HH-HL / LH-LL 链
# =====================================================================
def swing_trend(pivots):
    """由笔链判定当前线段方向: 'up' / 'down' / 'neutral'。
    up   = 更高高点 + 更高低点 (higher-high & higher-low)
    down = 更低低点 + 更低高点 (lower-low  & lower-high)
    不足 2 个 high + 2 个 low 时返回 'neutral'。
    """
    if len(pivots) < 4:
        return "neutral"
    highs = [p for p in pivots if p["kind"] == "high"]
    lows = [p for p in pivots if p["kind"] == "low"]
    if len(highs) >= 2 and len(lows) >= 2:
        hh = highs[-1]["price"] > highs[-2]["price"]
        hl = lows[-1]["price"] > lows[-2]["price"]
        lh = highs[-1]["price"] < highs[-2]["price"]
        ll = lows[-1]["price"] < lows[-2]["price"]
        if hh and hl:
            return "up"
        if lh and ll:
            return "down"
    return "neutral"


# =====================================================================
# 4. 中枢 Consolidation — 布林带宽滚动分位低位
# =====================================================================
CONSOL_DEFAULT = dict(win=20, lookback=120, th=0.25)


def bollinger_width(closes, win=20):
    """20 日布林带宽 = std/mean (波动率代理)。"""
    s = pd.Series(np.asarray(closes, dtype=float))
    m = s.rolling(win).mean()
    sd = s.rolling(win).std()
    return (sd / m).values


def bb_width_pct(closes, win=20, lookback=120):
    """每根 bar 的带宽在过往 lookback 窗口内的滚动分位 (0~1)。
    分位越低 = 越处于盘整/中枢期。
    """
    w = np.asarray(bollinger_width(closes, win), dtype=float)
    n = len(w)
    out = np.full(n, np.nan)
    min_len = max(5, int(lookback * 0.5))
    for i in range(n):
        lo = max(0, i - lookback + 1)
        winv = w[lo:i + 1]
        winv = winv[~np.isnan(winv)]
        if len(winv) < min_len or np.isnan(w[i]):
            continue
        out[i] = float((winv < w[i]).mean())
    return out


def in_consolidation(closes, win=20, lookback=120, th=0.25):
    """bool 数组: True = 处于中枢期(带宽分位 < th)。"""
    pct = bb_width_pct(closes, win, lookback)
    return (pct < th) & ~np.isnan(pct)


# =====================================================================
# 5. 背驰 Divergence — 纯 numpy MACD
# =====================================================================
def macd(closes, fast=12, slow=26, signal=9):
    """返回 (dif, dea) 均为 np.array。"""
    s = pd.Series(np.asarray(closes, dtype=float))
    ema_f = s.ewm(span=fast, adjust=False).mean()
    ema_s = s.ewm(span=slow, adjust=False).mean()
    dif = (ema_f - ema_s).values
    dea = pd.Series(dif).ewm(span=signal, adjust=False).mean().values
    return dif, dea


def macd_divergence(highs, lows, closes, fast=12, slow=26, signal=9,
                    pivot_n=2, lookback=60, min_gap=10):
    """返回 (bull_idx set, bear_idx set)。
    bull (底背驰): 价格创新低 但 dif 未创新低 → 下跌动能衰竭。
    bear (顶背驰): 价格创新高 但 dif 未创新高 → 上涨动能衰竭。
    仅取 lookback 窗口内的最值拐点, 避免远端极值污染。
    """
    closes = np.asarray(closes, dtype=float)
    highs = np.asarray(highs, dtype=float)
    lows = np.asarray(lows, dtype=float)
    dif, _ = macd(closes, fast, slow, signal)
    tops, bots = fractals(highs, lows, pivot_n)
    bull, bear = set(), set()
    for a in range(1, len(tops)):
        i1, i2 = tops[a - 1], tops[a]
        if i2 - i1 < min_gap:
            continue
        if highs[i2] > highs[i1] and dif[i2] < dif[i1]:
            seg = highs[max(0, i2 - lookback):i2 + 1]
            if highs[i2] >= np.nanmax(seg):
                bear.add(i2)
    for a in range(1, len(bots)):
        i1, i2 = bots[a - 1], bots[a]
        if i2 - i1 < min_gap:
            continue
        if lows[i2] < lows[i1] and dif[i2] > dif[i1]:
            seg = lows[max(0, i2 - lookback):i2 + 1]
            if lows[i2] <= np.nanmin(seg):
                bull.add(i2)
    return bull, bear


# =====================================================================
# 6. 三类买卖点 — 由 笔/线段/中枢/背驰 派生的近似规则
# =====================================================================
def classify_buy_points(pivots, bull_div, bear_div, closes, in_consol,
                        hold_bars=20):
    """返回 list of (idx, type) with type in {'b1','b2','b3'}。
    近似定义(仅供研究, 不宣称缠论正统):
      b1 一买: 下跌线段末端 + 底背驰(bull_div)  → 趋势反转买点
      b2 二买: b1 之后反弹, 回踩不破 b1 低点    → 确认买点
      b3 三买: 中枢(in_consol)结束后向上突破, 回踩不重返中枢 → 中继买点
    无未来函数: 全部基于截至 idx 的已知信息。
    """
    closes = np.asarray(closes, dtype=float)
    pts = sorted([(p["i"], p["kind"], p["price"]) for p in pivots])
    buys = []
    # ---- b1: 底背驰 + 当前处于下跌线段(最后可见为 low) ----
    last_kind = pts[-1][1] if pts else None
    for bi in sorted(bull_div):
        if bi >= len(closes):
            continue
        if last_kind == "low":
            buys.append((bi, "b1"))
    # ---- b2 / b3 基于 b1 之后的低点行为 ----
    b1_list = [b for b in buys if b[1] == "b1"]
    for (b1i, _) in b1_list:
        # 找 b1 之后最近的 low pivot 价格作为"回踩低点"
        later_lows = [(i, p) for (i, k, p) in pts if i > b1i and k == "low"]
        if not later_lows:
            continue
        li, lp = later_lows[0]
        # b2: 回踩不破 b1 低点 → 二买
        if lp > closes[b1i]:
            buys.append((li, "b2"))
        # b3: 中枢结束后突破, 且回踩仍在中枢上沿之上 (简化: 此 bar 不在中枢内)
        if li < len(in_consol) and not in_consol[li]:
            # 且 b1 之前曾处于中枢 (提供"出中枢"语境)
            pre = in_consol[max(0, b1i - hold_bars):b1i]
            if pre.any():
                buys.append((li, "b3"))
    # 去重 (同 idx 保留 b1优先)
    seen = {}
    for i, t in buys:
        if i not in seen or t == "b1":
            seen[i] = t
    return sorted(seen.items())


# =====================================================================
# 7. 便捷封装: 一次性计算某只股票的全部缠论状态序列
# =====================================================================
def compute_states(highs, lows, closes,
                   zig_th=0.05, consol=None, div_lookback=60):
    """返回 dict, 每条均为 len(closes) 的同长序列/集合, 供回测逐 bar 取用。
    consol: 中枢参数 dict 或 None(用 CONSOL_DEFAULT)。
    """
    consol = consol or CONSOL_DEFAULT
    pivots = zigzag(highs, lows, threshold=zig_th)
    trend = swing_trend(pivots)
    consol_mask = in_consolidation(closes, **consol)
    bull, bear = macd_divergence(highs, lows, closes, lookback=div_lookback)
    buys = classify_buy_points(pivots, bull, bear, closes, consol_mask)
    buys_idx = set(i for i, _ in buys)
    return dict(
        pivots=pivots, trend=trend, consolidation=consol_mask,
        bull_div=bull, bear_div=bear, buy_points=buys_idx,
        buy_points_typed=buys,
    )
