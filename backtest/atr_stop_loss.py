"""
ATR 动态止损工具模块
=====================
TA-Lib 实现（C 语言底层，速度极快），可被任意策略插件调用。

ATR 计算逻辑（Wilder 平滑）:
    1. TR = max(H - L, |H - C_prev|, |L - C_prev|)
    2. 初始 ATR = 前 N 个 TR 的简单平均
    3. 后续 ATR = (ATR_prev × (N-1) + TR) / N

止损逻辑:
    1. 初始止损 = 入场价 - ATR × atr_mult（建仓日设定）
    2. 追踪止损（单向递增）：防线只升不降
       stop_price = max(旧stop, 最高价 - ATR × trail_mult)
"""

import numpy as np
import talib as ta


class ATRStopLoss:
    """
    ATR 动态止损工具

    Usage:
        sl = ATRStopLoss(atr_period=14, atr_mult=3.0, trail_mult=3.0)

        # 回测开始时计算整列 ATR
        atr_values = sl.calc_atr(high_arr, low_arr, close_arr)

        # 建仓日
        sl.on_entry(entry_price=10.5, atr_val=atr_values[i])

        # 每日循环
        sl.update(high_price=high[i], atr_val=atr_values[i])
        should_stop, stop_price, reason = sl.check_stop(close_price=close[i])
        if should_stop:
            # 执行卖出
            ...
    """

    def __init__(self, atr_period: int = 14, atr_mult: float = 3.0, trail_mult: float = 3.0):
        """
        Args:
            atr_period: ATR 计算周期（默认 14 日）
            atr_mult:   初始止损倍数（止损价 = 入场价 - ATR x atr_mult）
            trail_mult: 追踪止损倍数（追踪止损价 = 最高价 - ATR x trail_mult）
        """
        self.atr_period = atr_period
        self.atr_mult = atr_mult
        self.trail_mult = trail_mult

        # 运行时状态
        self.atr_values = None        # 整列 ATR 值 (numpy array)
        self.entry_price = 0.0        # 建仓价格
        self.stop_price = 0.0         # 当前止损价
        self.highest_price = 0.0      # 建仓后最高价（用于追踪止损）
        self.is_active = False        # 是否已激活

    # ── Public API ──────────────────────────────────────

    def calc_atr(self, high, low, close) -> np.ndarray:
        """
        计算整列 ATR 值（TA-Lib C 实现，速度极快）

        Args:
            high:  最高价数组 (numpy array 或可转换为 numpy 的类型)
            low:   最低价数组
            close: 收盘价数组

        Returns:
            numpy array，与输入同长度，前 atr_period-1 个值为 nan
        """
        high_arr = np.asarray(high, dtype=float)
        low_arr = np.asarray(low, dtype=float)
        close_arr = np.asarray(close, dtype=float)

        # TA-Lib ATR：C 实现，自动处理 Wilder 平滑
        atr = ta.ATR(high_arr, low_arr, close_arr, timeperiod=self.atr_period)
        atr = np.asarray(atr, dtype=float)

        # 前 atr_period-1 个值填充为 0（与 numpy 版行为一致）
        atr[:self.atr_period - 1] = 0.0

        self.atr_values = atr
        return atr

    def on_entry(self, entry_price: float, atr_val: float):
        """
        建仓时调用，设置初始止损价

        Args:
            entry_price: 建仓价格（开盘价或收盘价）
            atr_val:     建仓当日（或前一日）的 ATR 值
        """
        self.entry_price = entry_price
        self.stop_price = entry_price - self.atr_mult * atr_val
        self.highest_price = entry_price
        self.is_active = True

    def update(self, high_price: float, atr_val: float):
        """
        每日更新追踪止损价（防线只升不降）

        Args:
            high_price: 当日最高价
            atr_val:    当日的 ATR 值
        """
        if not self.is_active:
            return

        # 更新建仓后最高价
        if high_price > self.highest_price:
            self.highest_price = high_price

        # 计算新追踪止损价
        new_stop = max(
            self.stop_price,
            self.highest_price - self.trail_mult * atr_val
        )

        # 防线只升不降
        self.stop_price = max(self.stop_price, new_stop)

    def check_stop(self, close_price: float) -> tuple:
        """
        检查是否触发止损

        Args:
            close_price: 当日收盘价

        Returns:
            (should_stop: bool, stop_price: float, reason: str)
            - should_stop: True 表示应止损
            - stop_price:  当前止损价
            - reason:      原因描述
        """
        if not self.is_active:
            return False, self.stop_price, ""

        if close_price < self.stop_price:
            reason = f"ATR止损(收盘{close_price:.2f} < 防线{self.stop_price:.2f})"
            return True, self.stop_price, reason

        return False, self.stop_price, ""

    def reset(self):
        """重置状态（下次建仓前调用）"""
        self.entry_price = 0.0
        self.stop_price = 0.0
        self.highest_price = 0.0
        self.is_active = False

    # ── 静态工具方法 ────────────────────────────────────

    @staticmethod
    def calc_tr(high_price: float, low_price: float, prev_close: float) -> float:
        """计算单日 True Range"""
        hl = high_price - low_price
        hc = abs(high_price - prev_close)
        lc = abs(low_price - prev_close)
        return max(hl, hc, lc)

    @staticmethod
    def update_atr(prev_atr: float, today_tr: float, period: int = 14) -> float:
        """用 Wilders 平滑更新 ATR 值"""
        return (prev_atr * (period - 1) + today_tr) / period
