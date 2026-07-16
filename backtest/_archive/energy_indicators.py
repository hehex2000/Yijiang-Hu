"""
能量型指标计算模块 - AR/BR/CR/VR 四个指标

AR：人气指标 — 以开盘价为基准，衡量盘中多空力量对比
BR：意愿指标 — 以前日收盘价为基准，衡量隔夜情绪和买入意愿
CR：能量指标 — 以典型价格(TP)为中轴，衡量趋势动量
VR：成交量指标 — 衡量上涨日与下跌日成交量对比，判断资金流向

核心用途：
- 作为独立策略的买卖信号来源
- 作为风控模块的辅助过滤因子
"""

import pandas as pd
import numpy as np
from loguru import logger


def calculate_ar(df: pd.DataFrame, n: int = 26) -> pd.Series:
    """
    计算 AR 指标（人气指标）

    AR = Σ(High - Open) / Σ(Open - Low) × 100

    以当天开盘价为基准：
    - 分子：最高价到开盘价的距离（买方动能）
    - 分母：开盘价到最低价的距离（卖方动能）

    AR 高 → 开盘后买方占优；AR 低 → 开盘后卖方占优

    Args:
        df: 包含 adj_open, adj_high, adj_low 的 DataFrame
        n: 周期（默认 26）

    Returns:
        AR 指标值的 Series，前 n-1 行为 NaN

    边界处理:
        - 分母为 0 时返回 NaN
        - 数据不足 n 行时返回全 NaN
    """
    if len(df) < n:
        logger.warning(f"calculate_ar: DataFrame 长度 {len(df)} < 周期 {n}，返回全 NaN")
        return pd.Series([np.nan] * len(df), index=df.index, name='ar')

    # 分子：周期内 (最高价 - 开盘价) 的和，代表多头动能
    ar_up = (df['adj_high'] - df['adj_open']).rolling(window=n).sum()

    # 分母：周期内 (开盘价 - 最低价) 的和，代表空头动能
    ar_down = (df['adj_open'] - df['adj_low']).rolling(window=n).sum()

    # 容错：分母为 0 或接近 0 时返回 NaN
    with np.errstate(divide='ignore', invalid='ignore'):
        ar = np.where(ar_down == 0, np.nan, (ar_up / ar_down) * 100)

    result = pd.Series(ar, index=df.index, name='ar')

    # NaN 统计（排除窗口期内的 NaN）
    nan_count = result.isna().sum()
    extra_nan = nan_count - min(n - 1, len(df))
    if extra_nan > 0:
        logger.debug(f"calculate_ar: 有 {extra_nan} 行因分母为0产生 NaN")

    return result


def calculate_br(df: pd.DataFrame, n: int = 26) -> pd.Series:
    """
    计算 BR 指标（意愿指标）

    BR = Σ max(0, High - 前收盘价) / Σ max(0, 前收盘价 - Low) × 100

    以昨日收盘价为基准：
    - 分子：高于昨日收盘价的向上动能（仅统计做多力量）
    - 分母：低于昨日收盘价的向下动能（仅统计做空力量）
    - max(0, ...) 确保只保留有效方向的动能

    BR > 100 → 市场更愿意以高于昨收的价格买入，买入意愿强
    BR < 100 → 市场抛售压力更大

    Args:
        df: 包含 adj_high, adj_low, adj_close 的 DataFrame
            前一日收盘价通过 adj_close.shift(1) 获取
        n: 周期（默认 26）

    Returns:
        BR 指标值的 Series，前 n 行为 NaN（需要前一日收盘价，多一行）

    边界处理:
        - 分母为 0 时返回 NaN
        - 前一日收盘价缺失（第一行）时该日贡献为 0
    """
    if len(df) < n + 1:
        logger.warning(f"calculate_br: DataFrame 长度 {len(df)} < 周期+1 {n + 1}，返回全 NaN")
        return pd.Series([np.nan] * len(df), index=df.index, name='br')

    prev_close = df['adj_close'].shift(1)

    # 分子：最高价高于昨收的部分（仅正向动能）
    br_up_raw = df['adj_high'] - prev_close
    br_up = np.maximum(0, br_up_raw.fillna(0))

    # 分母：昨收高于最低价的部分（仅负向动能绝对值）
    br_down_raw = prev_close - df['adj_low']
    br_down = np.maximum(0, br_down_raw.fillna(0))

    # 滚动求和
    br_up_sum = pd.Series(br_up, index=df.index).rolling(window=n).sum()
    br_down_sum = pd.Series(br_down, index=df.index).rolling(window=n).sum()

    # 容错
    with np.errstate(divide='ignore', invalid='ignore'):
        br = np.where(br_down_sum == 0, np.nan, (br_up_sum / br_down_sum) * 100)

    result = pd.Series(br, index=df.index, name='br')
    return result


def calculate_cr(df: pd.DataFrame, n: int = 26) -> pd.Series:
    """
    计算 CR 指标（能量指标）

    TP = (High + Low + Close) / 3

    CR = Σ max(0, High - 昨日TP) / Σ max(0, 昨日TP - Low) × 100

    以典型价格 TP 为中轴（价格重心）：
    - 分子：当日最高价高于昨日重心的部分
    - 分母：当日最低价低于昨日重心的部分

    CR > 100 → 价格相对重心的上行动能更强
    CR < 100 → 价格相对重心的下行动能更强

    CR 对趋势反转更敏感，因为重心偏移比单纯收盘价变化更早体现

    Args:
        df: 包含 adj_high, adj_low, adj_close 的 DataFrame
        n: 周期（默认 26）

    Returns:
        CR 指标值的 Series，前 n+1 行为 NaN（需要昨日TP）

    边界处理:
        - 分母为 0 时返回 NaN
        - 昨日 TP 缺失时该日贡献为 0
    """
    if len(df) < n + 1:
        logger.warning(f"calculate_cr: DataFrame 长度 {len(df)} < 周期+1 {n + 1}，返回全 NaN")
        return pd.Series([np.nan] * len(df), index=df.index, name='cr')

    # 计算典型价格 TP = (H + L + C) / 3
    tp = (df['adj_high'] + df['adj_low'] + df['adj_close']) / 3

    # 昨日 TP
    prev_tp = tp.shift(1)

    # 分子：当日最高价高于昨日TP的部分
    cr_up_raw = df['adj_high'] - prev_tp
    cr_up = np.maximum(0, cr_up_raw.fillna(0))

    # 分母：当日最低价低于昨日TP的部分
    cr_down_raw = prev_tp - df['adj_low']
    cr_down = np.maximum(0, cr_down_raw.fillna(0))

    # 滚动求和
    cr_up_sum = pd.Series(cr_up, index=df.index).rolling(window=n).sum()
    cr_down_sum = pd.Series(cr_down, index=df.index).rolling(window=n).sum()

    # 容错
    with np.errstate(divide='ignore', invalid='ignore'):
        cr = np.where(cr_down_sum == 0, np.nan, (cr_up_sum / cr_down_sum) * 100)

    result = pd.Series(cr, index=df.index, name='cr')
    return result


def calculate_vr(df: pd.DataFrame, n: int = 26) -> pd.Series:
    """
    计算 VR 指标（成交量指标 / 容量指标）

    VR = (UVS + 0.5 × FVS) / (DVS + 0.5 × FVS) × 100

    其中：
    - UVS：N 日内所有收盘价上涨交易日的成交量总和
    - DVS：N 日内所有收盘价下跌交易日的成交量总和
    - FVS：N 日内所有收盘价平盘交易日的成交量总和

    平盘日成交量一分为二，同时加到分子和分母，保持数据完整性

    VR > 100 → 上涨日成交量 > 下跌日成交量，资金偏流入
    VR < 100 → 下跌日成交量占优，资金偏弱
    VR 在底部率先回升 → 可能的高胜率左侧买入信号

    Args:
        df: 包含 adj_close, vol 的 DataFrame
            vol 的单位不影响结果（比值计算）
        n: 周期（默认 26）

    Returns:
        VR 指标值的 Series，前 n-1 行为 NaN

    边界处理:
        - 分母为 0 时返回 NaN
        - 成交量缺失时使用 0
    """
    if len(df) < n:
        logger.warning(f"calculate_vr: DataFrame 长度 {len(df)} < 周期 {n}，返回全 NaN")
        return pd.Series([np.nan] * len(df), index=df.index, name='vr')

    # 判断涨跌平
    price_diff = df['adj_close'].diff()
    is_up = price_diff > 0
    is_down = price_diff < 0
    is_flat = price_diff == 0  # 含第一行的 NaN（NaN == 0 → False）

    # 获取成交量（兼容 vol / volume 两种列名）
    vol_col = None
    for col in ['vol', 'volume']:
        if col in df.columns:
            vol_col = col
            break
    if vol_col is None:
        raise KeyError("calculate_vr: 缺少成交量列（需要 'vol' 或 'volume'）")
    
    vol = df[vol_col].fillna(0)

    # 滚动 N 日求和
    uvs = (vol * is_up).rolling(window=n).sum()    # 上涨日成交量之和
    dvs = (vol * is_down).rolling(window=n).sum()   # 下跌日成交量之和
    fvs = (vol * is_flat).rolling(window=n).sum()   # 平盘日成交量之和

    # VR = (UVS + 0.5 * FVS) / (DVS + 0.5 * FVS) × 100
    numerator = uvs + 0.5 * fvs
    denominator = dvs + 0.5 * fvs

    with np.errstate(divide='ignore', invalid='ignore'):
        vr = np.where(denominator == 0, np.nan, (numerator / denominator) * 100)

    result = pd.Series(vr, index=df.index, name='vr')
    return result


def calculate_all_energy_indicators(df: pd.DataFrame, n: int = 26) -> pd.DataFrame:
    """
    一次性计算所有 4 个能量指标并添加到 DataFrame

    Args:
        df: 包含 adj_open, adj_high, adj_low, adj_close, vol 的 DataFrame
        n: 公共周期（默认 26）

    Returns:
        添加了 ar, br, cr, vr 列的 DataFrame（原地修改 + 返回）
    """
    logger.info(f"正在计算能量指标（周期={n}）...")

    df = df.copy()

    df['ar'] = calculate_ar(df, n=n)
    df['br'] = calculate_br(df, n=n)
    df['cr'] = calculate_cr(df, n=n)
    df['vr'] = calculate_vr(df, n=n)

    # 统计有效数据
    valid_mask = df['ar'].notna() & df['br'].notna() & df['cr'].notna() & df['vr'].notna()
    logger.info(f"✓ 能量指标计算完成：{valid_mask.sum()}/{len(df)} 行有效数据")

    return df


def generate_energy_signals(df: pd.DataFrame, n: int = 26) -> pd.DataFrame:
    """
    根据能量指标生成买卖信号

    买入信号（多指标共振，全部满足才触发）：
        AR < 100  AND  BR < 100  AND  BR < AR（底背离）
        AND  CR < 100  AND  VR < 100

        含义：四个指标同时跌破基准线，且隔夜情绪(BR)比盘中人气(AR)更弱，
              说明悲观已充分释放，可能出现底部反转。

    卖出信号（任一指标过热即触发，满足任一即触发）：
        AR > 150  OR  BR > 150  OR  CR > 150  OR  VR > 150

        含义：任一指标进入非理性繁荣区，后续买盘可能枯竭，应止盈卖出。

    Args:
        df: 包含 ar, br, cr, vr 列的 DataFrame
        n: 周期（用于确认前 n+1 行信号为无效）

    Returns:
        添加了 buy_signal, sell_signal 列的 DataFrame
    """
    required_cols = ['ar', 'br', 'cr', 'vr']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.error(f"generate_energy_signals: 缺少必需列 {missing}")
        raise ValueError(f"DataFrame 缺少必需的列：{missing}。请先调用 calculate_all_energy_indicators()")

    df = df.copy()

    ar = df['ar']
    br = df['br']
    cr = df['cr']
    vr = df['vr']

    # 买入信号：多指标共振（AND 逻辑）
    buy_condition = (
        (ar < 100) &
        (br < 100) &
        (br < ar) &       # 底背离：隔夜意愿比盘中人气跌得更深
        (cr < 100) &
        (vr < 100)
    )

    # 卖出信号：任一指标过热（OR 逻辑）
    sell_condition = (
        (ar > 150) |
        (br > 150) |
        (cr > 150) |
        (vr > 150)
    )

    df['buy_signal'] = False
    df['sell_signal'] = False

    # 只在有效指标行设置信号（跳过 NaN 行）
    valid_mask = ar.notna() & br.notna() & cr.notna() & vr.notna()
    df.loc[valid_mask, 'buy_signal'] = buy_condition[valid_mask]
    df.loc[valid_mask, 'sell_signal'] = sell_condition[valid_mask]

    # 统计信号数量
    buy_count = df['buy_signal'].sum()
    sell_count = df['sell_signal'].sum()
    logger.info(f"✓ 能量信号生成：买入信号 {buy_count} 次，卖出信号 {sell_count} 次")

    return df
