"""
factor_utils.py — 因子计算共享工具

汲取自 B站「跟着Jim学量化」换手率科普（2026-07-30）的方法论，沉淀为平台统一惯例：

  原则① 分母显式：换手率须用"流通股本"口径（本平台统一取 turnover_rate_f =
           自由流通股本换手率），不用总股本口径，也不跨股票直接比原始成交量。
  原则② 和它自己平时比：异动/异常类特征默认用「自身 trailing 历史分位」而非
           绝对阈值，消除规模偏差、跨股票可比。
  原则③ 异动须配对价格：换手放大必须同时记录价格动作（价变/价滞/价跌是三种
           不同记录），单维"放量"不可解读为方向信号。
  护栏   换手率（及一切量能异常）只作描述器 / 风险 overlay，不预测方向。

汲取自同系列「筹码峰怎么看」科普（2026-07-30）的护栏（本平台目前无任何
"筹码/成本分布"类市场结构估算特征，且按 Jim 论点不该新建——下列约定用于
防范未来误建）：

  筹码峰/成本分布本质 = 历史成交 × 一套分配假设 的估算图，软件看不到任何账户，
  峰值只是"拍出来的分配假设"，并非市场持仓的 X 光片。若要新增此类"市场结构估算"
  特征，必须显式声明且输出只标 estimate、绝不当信号/真相、非方向性：

    1. 分配方式      —— 新成交如何分配到各价位（等比例衰减 / 先进先出 / 时间加权…）
    2. 衰减速度      —— 原有分布按什么 half-life 衰减
    3. 价格分箱粒度  —— 价位如何离散化
    4. 换手率分母    —— 复用原则①的 turnover_rate_f（自由流通口径）
    5. 输入数据颗粒度—— 日线代表价 vs 分价成交明细，二者峰形不同

  命名纪律：本平台既有 `avg_cost` 是**我们自身持仓的加权成本**（真实账本，
  由实际买入成交累加），绝不可标注/误读为"市场筹码分布"。任何形态类结论须基于
  逐日演化，而非单日静态快照。

本模块只依赖 pandas / numpy，供 factor_calculator 与回测脚本共同复用。
"""

from typing import Optional, Sequence

import numpy as np
import pandas as pd


def own_history_pct(series: Sequence, window: int = 60) -> float:
    """自身历史分位（经验 CDF）：最新值在其自身 trailing-`window` 内的百分位 (0~1)。

    Jim 原则②「和它自己平时比」的落地实现。与跨股票截面分位不同，这里只和
    该标的自身历史比，天然消除规模/行业偏差，且对任意绝对值都可比。

    Args:
        series: 时序（如某股每日 turnover_rate_f）。
        window: 回望窗口（交易日）。取序列末尾 window 个样本计算分位；
                若 window >= len(series) 则使用全样本。

    Returns:
        分位 (0~1)：当前值在其自身历史中越高，分位越高（最大值≈1、最小值≈0）；
        采用严格小于 P(X < 当前) 的经验分位，避免低换手股因大量"等于自身最低"
        的平尾被错误抬高（若用 P(X<=当前) 会在低端产生平尾膨胀，误判为异常活跃）。
        NaN 若有效样本不足。
    """
    s = pd.Series(series, dtype="float64").dropna()
    if len(s) < 2:
        return float("nan")
    w = max(2, int(min(window, len(s))))
    trail = s.tail(w)
    latest = float(trail.iloc[-1])
    # 经验分位 = (严格小于 当前值 的样本数) / 样本总数（高端敏感、低端不膨胀）
    return float((trail < latest).mean())


def volume_price_regime(
    turnover_pct: float,
    price_chg: float,
    hi_thr: float = 0.80,
    flat_thr: float = 0.01,
) -> float:
    """量价 regime 分类（Jim 原则③：放量须配对价格记录）。

    把"换手率异常"与"当天价格动作"组合成离散 regime，避免单维误读：

        0 = 常态        （换手未达异常阈值 turnover_pct < hi_thr）
        1 = 放量 + 价变大（换手异常 且 价格明显上涨 > +flat_thr）
        2 = 放量 + 价滞  （换手异常 但 价格几乎没动 |chg| <= flat_thr）
        3 = 放量 + 价跌  （换手异常 且 价格明显下跌 < -flat_thr）

    Args:
        turnover_pct: 自身历史分位 (0~1)，如 own_history_pct 的输出。
        price_chg:    当天价格变动（小数，如 0.012 = +1.2%）。
        hi_thr:       判定"换手异常"的分位阈值（默认 0.80 = 自身 80% 分位以上）。
        flat_thr:     判定"价滞"的涨跌幅阈值（小数，默认 0.01 = ±1%）。

    Returns:
        regime 编码 (0~3)；任一输入为 NaN 时返回 NaN。
    """
    if turnover_pct is None or price_chg is None:
        return float("nan")
    if np.isnan(turnover_pct) or np.isnan(price_chg):
        return float("nan")
    if turnover_pct < hi_thr:
        return 0.0
    if price_chg > flat_thr:
        return 1.0
    if price_chg < -flat_thr:
        return 3.0
    return 2.0


def latest_price_change(hist_data: pd.DataFrame) -> Optional[float]:
    """从 hist_data 取最新一日价格变动（小数）。

    兼容列名：涨跌幅 / pct_chg（均为百分比，÷100）；或回退用最后两日 close 计算。
    无可用价格时返回 None。
    """
    for col in ("涨跌幅", "pct_chg"):
        if col in hist_data.columns:
            v = hist_data[col].values[-1]
            if v is not None and not pd.isna(v):
                return float(v) / 100.0
    if "close" in hist_data.columns and len(hist_data) >= 2:
        c = hist_data["close"].values
        if not pd.isna(c[-1]) and not pd.isna(c[-2]) and c[-2] != 0:
            return float(c[-1] / c[-2] - 1.0)
    return None
