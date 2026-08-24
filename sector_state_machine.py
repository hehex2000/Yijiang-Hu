# -*- coding: utf-8 -*-
"""
板块三态状态机（sector three-state machine）
============================================

核心思想：对每个标的（ETF / 板块 / 指数）的价格序列，逐期判定其处于三种
「趋势状态」之一，用于前置 gate 过滤候选——只进 **右侧趋势 + 趋势加速**，
剔除 **加速见底（左侧 / 下跌减速见底中）**。

为什么是「状态确认」而非「强度排序」（§5.12 已证板块强度=负贡献）：
  · 横截面动量强弱排序 → 容易买到「最强=刚见顶」的标的（动量均值回归）。
  · 本状态机不看「谁最强」，而是逐标的用 **价格结构**（MA 位置 + 长期趋势
    斜率 + 动量加速/减速）确认它是否处于「可交易的健康趋势态」。
  · gate 只做「排除左侧下落刀」，不动原有打分排序——排序仍由策略自身的
    RSRS+双动量决定。二者解耦，避免把状态机变成又一个强度排序器。

三态定义（基于升序收盘价 close_asc，旧→新）：
  ACCEL_BOTTOM  加速见底（左侧）：价格在长期趋势线(MA60)下方，或价格已站上
                MA60 但 MA60 斜率仍向下（死猫反弹）——结构上仍处于图表左侧，
                不抄底、不接下落刀。
  RIGHT_TREND   右侧趋势（右侧）：价格>MA60 且 MA60 斜率为正（长期上升趋势
                确认），动量平稳或温和减速——右侧基底，可安全持有。
  TREND_ACCEL   趋势加速（突破）：在右侧趋势基础上，短期动量相对长期动量
                加速（roc20 > roc60）→ 突破主升，可积极持有。
                （注：若 60 日涨幅已抛物线化(>overbought_thr)，降级为
                 RIGHT_TREND，避免追绝对顶——呼应 §5.12 不追最强。）

本模块为 **纯函数**（输入 numpy 数组，无 DB 依赖），便于合成数据单测与
跨策略复用。DB 取数 + 闸门封装放在调用方（run_etf_rotation_v6_merged.py）。
"""

import numpy as np

# ── 默认参数（可在调用处覆盖做 A/B 扫描）────────────────────
MA_SHORT = 20          # 短期均线
MA_LONG = 60           # 长期趋势线
SLOPE_PERIOD = 20      # 长期趋势斜率回看（MA60 现在 vs SLOPE_PERIOD 日前）
ACCEL_THR = 0.05       # 动量加速度阈值：roc20 - roc60 > 此值 → 趋势加速
OVERBOUGHT_THR = 0.80  # 60 日涨幅抛物线阈值：> 此值视为见顶，降级为右侧
HIST_LEN = 90          # 判定所需最少历史长度（MA_LONG + SLOPE_PERIOD + 缓冲）

# ── 状态常量 ──────────────────────────────────────────────
ACCEL_BOTTOM = "ACCEL_BOTTOM"   # 加速见底（左侧）
RIGHT_TREND = "RIGHT_TREND"     # 右侧趋势
TREND_ACCEL = "TREND_ACCEL"     # 趋势加速
UNKNOWN = "UNKNOWN"             # 数据不足，无法判定

STATE_CN = {
    ACCEL_BOTTOM: "加速见底(左侧)",
    RIGHT_TREND: "右侧趋势",
    TREND_ACCEL: "趋势加速",
    UNKNOWN: "未知(数据不足)",
}

# gate 允许进入的状态（右侧 + 趋势加速；UNKNOWN 保守保留不误杀）
GATE_PASS = (RIGHT_TREND, TREND_ACCEL, UNKNOWN)


def classify_state(close_asc, ma_short=MA_SHORT, ma_long=MA_LONG,
                   slope_period=SLOPE_PERIOD, accel_thr=ACCEL_THR,
                   overbought_thr=OVERBOUGHT_THR):
    """对单期（序列末端点）判定趋势状态。

    Args:
        close_asc: 升序收盘价 numpy array（旧→新），末位为当前期。
        ma_short / ma_long / slope_period / accel_thr / overbought_thr: 可调参数。

    Returns:
        (state_str, details_dict)
        state_str in {ACCEL_BOTTOM, RIGHT_TREND, TREND_ACCEL, UNKNOWN}
        details 含用于诊断的全部中间量。
    """
    n = len(close_asc)
    need = ma_long + slope_period
    if n < need:
        return UNKNOWN, {"reason": "history_too_short(%d<%d)" % (n, need)}

    price = float(close_asc[-1])
    ma_s = float(np.mean(close_asc[-ma_short:]))
    ma_l = float(np.mean(close_asc[-ma_long:]))
    # 长期趋势斜率（诊断用；MA60 滞后大，不直接用于决策）
    ma_l_prev = float(np.mean(close_asc[-(ma_long + slope_period):-slope_period]))
    ma_l_slope = (ma_l - ma_l_prev) / (ma_l_prev + 1e-12)

    # 动量：短期(roc20) vs 长期(roc60)
    roc_s = price / float(close_asc[-(1 + ma_short)]) - 1.0
    roc_l = price / float(close_asc[-(1 + ma_long)]) - 1.0
    accel = roc_s - roc_l

    above = price > ma_l
    # 趋势结构确认：短期均线站上长期均线（金叉结构，比 MA60 斜率滞后小得多，
    # 能及时确认"右侧"而非等到 MA60 斜率转正——后者会让刚启动的政策牛也被误判左侧）
    aligned = ma_s > ma_l

    details = {
        "price": price, "ma_short": ma_s, "ma_long": ma_l,
        "ma_long_slope": ma_l_slope, "roc_short": roc_s, "roc_long": roc_l,
        "accel": accel, "above_ma_long": above, "ma_aligned": aligned,
    }

    # ── 状态判定（优先级：先判左侧，再判右侧细分）──
    # 1) 价格在长期趋势线(MA60)下方 → 左侧（结构未反转，下落刀）
    if not above:
        return ACCEL_BOTTOM, details

    # 2) 价格站上 MA60 但短期均线仍在长期均线下方（MA20<MA60）→
    #    短弱于长，趋势结构未确认（死猫反弹/未金叉）→ 仍左侧
    if not aligned:
        return ACCEL_BOTTOM, details

    # 3) 上升趋势结构确认（金叉+价格在上）。按动量是否加速细分：
    if accel > accel_thr:
        # 趋势加速；但若已抛物线化（60 日涨幅过高）→ 降级为右侧，不追绝对顶
        if roc_l > overbought_thr:
            details["note"] = "parabolic_top_roc60=%.0f%%" % (roc_l * 100)
            return RIGHT_TREND, details
        return TREND_ACCEL, details

    # 4) 平稳/温和减速的上升 → 右侧趋势
    return RIGHT_TREND, details


def state_series(close_asc, ma_short=MA_SHORT, ma_long=MA_LONG,
                 slope_period=SLOPE_PERIOD, accel_thr=ACCEL_THR,
                 overbought_thr=OVERBOUGHT_THR):
    """对整条序列逐期判定，返回与 close_asc 等长的状态数组（前 need-1 期为 UNKNOWN）。

    用于可视化/调试：观察某标的如何在三态间切换。
    """
    n = len(close_asc)
    out = [UNKNOWN] * n
    need = ma_long + slope_period
    for i in range(need - 1, n):
        st, _ = classify_state(close_asc[:i + 1], ma_short=ma_short, ma_long=ma_long,
                               slope_period=slope_period, accel_thr=accel_thr,
                               overbought_thr=overbought_thr)
        out[i] = st
    return out


if __name__ == "__main__":
    # ── 合成数据自测（无 DB 依赖，纯逻辑验证）──
    print("=== 板块三态状态机 合成数据自测 ===")

    def mk(n, kind):
        x = np.arange(n, dtype=float)
        if kind == "uptrend_steady":       # 线性上升（%收益平稳减速）→ 右侧趋势
            return 100.0 + 0.3 * x
        if kind == "uptrend_accel":        # 60日前高位→20日前回踩→近期突破(且突破段长使MA60斜率转正)
            # 触发 accel>0 需 P60>P20：60日前=140(高), 20日前=90(回踩), 今=150(突破)
            p = np.empty(n, dtype=float)
            p[0:40] = 100.0
            p[40:60] = np.linspace(100.0, 140.0, 20)   # 60日前=140
            p[60:80] = np.linspace(140.0, 90.0, 20)    # 20日前=90
            p[80:n] = np.linspace(90.0, 150.0, n - 80) # 今=150
            return p
        if kind == "downtrend_decel":      # 持续下行（结构未反转，价格<MA60）→ 加速见底
            return 200.0 - 0.5 * x
        if kind == "deadcat":              # 长期下行中的反弹（MA60仍向下）→ 加速见底
            base = 200.0 - 0.5 * x
            bounce = np.where(x > 60, 30.0 * np.sin((x - 60) / 8.0), 0.0)
            return base + bounce
        if kind == "parabolic":            # 抛物线暴涨 → 降级右侧（不追顶）
            return 100.0 * np.exp(0.012 * x)
        raise ValueError(kind)

    cases = {
        "线性上升(应=右侧趋势)": "uptrend_steady",
        "加速上升(应=趋势加速)": "uptrend_accel",
        "下跌减速(应=加速见底)": "downtrend_decel",
        "死猫反弹(应=加速见底)": "deadcat",
        "抛物线暴涨(应=右侧·降级)": "parabolic",
    }
    expect = {
        "线性上升(应=右侧趋势)": RIGHT_TREND,
        "加速上升(应=趋势加速)": TREND_ACCEL,
        "下跌减速(应=加速见底)": ACCEL_BOTTOM,
        "死猫反弹(应=加速见底)": ACCEL_BOTTOM,
        "抛物线暴涨(应=右侧·降级)": RIGHT_TREND,
    }
    ok = 0
    for label, kind in cases.items():
        p = mk(120, kind)
        st, det = classify_state(p)
        flag = "OK " if st == expect[label] else "FAIL"
        if st == expect[label]:
            ok += 1
        print("  [%s] %s: 得=%s | above=%s slope=%+.4f accel=%+.4f roc60=%+.2f%% %s"
              % (flag, label, STATE_CN[st], det.get("above_ma_long"),
                 det.get("ma_long_slope", 0), det.get("accel", 0),
                 det.get("roc_long", 0) * 100, det.get("note", "")))
    print("\n自测结果：%d/%d 通过" % (ok, len(cases)))
