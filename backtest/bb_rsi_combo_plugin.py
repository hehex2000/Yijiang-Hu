# -*- coding: utf-8 -*-
"""
布林带+RSI组合择时策略（BB+RSI Combo）

基于UP主Jim视频《【干货】布林带不是买卖信号，读了原版书才知道》中的正确用法设计。

核心理念（来自John Bollinger原版书）:
- 布林带不是买卖信号！它只是告诉你价格在哪。
- 布林带的本质：通过标准差反映市场波动幅度（一把"会伸缩的尺子"）
- 带宽收窄 = 弹簧被压缩 = 能量蓄积 → 最有价值的信号

策略设计:
1. 带宽挤压（Squeeze）检测 — 布林带最有价值的信号形态
   - 带宽 = (上轨 - 下轨) / 中轨
   - 当前带宽接近历史最低水平 → 能量蓄积中
   - 不是触碰轨道，而是带宽的宽窄变化

2. RSI动量确认 — 与布林带组合过滤假信号
   - 布林带指示"价格在哪"（相对位置）
   - RSI指示"动能有多强"（动量强度）
   - 两者同时处于极端区域 → 信号可信度大幅提升

3. 三重共振入场
   - 条件1: 带宽处于挤压状态（能量蓄积）
   - 条件2: 价格接近布林带下轨（超卖区域）
   - 条件3: RSI < 30（动能彻底耗尽）

4. 双确认出场
   - 条件: 价格接近上轨 + RSI > 70（超买）
   - 或: 止损/止盈触发

5. 半凯利公式仓位管理
   - 每次投入 = 半凯利公式 × 不确定性折扣
   - 参考《凯利公式——一个教我们每次买票买多少的数学公式》
   - 实操信号：全凯利太激进，半凯利保留大部分增长率但降低波动

信号基于前一日数据（shift(1)），避免未来函数
买入/卖出均用次日开盘价执行
"""

import pandas as pd
import numpy as np
import talib as ta
from backtest.base_strategy import BaseStrategy
from backtest.atr_stop_loss import ATRStopLoss
from backtest.kelly_sizer import KellySizer
from loguru import logger


class BbRsiComboStrategyPlugin(BaseStrategy):
    """布林带+RSI组合择时策略（带宽挤压+半凯利）"""

    def __init__(self, capital: float, cfg: dict):
        super().__init__("BbRsiComboStrategyPlugin", capital, cfg)

        # ── 布林带参数 ──
        self.bb_period = cfg.get("bb_period", 20)
        self.bb_std = cfg.get("bb_std", 2.0)

        # ── RSI参数 ──
        self.rsi_period = cfg.get("rsi_period", 14)
        self.rsi_oversold = cfg.get("rsi_oversold", 30)
        self.rsi_overbought = cfg.get("rsi_overbought", 70)

        # ── 带宽挤压参数 ──
        self.squeeze_lookback = cfg.get("squeeze_lookback", 50)
        self.squeeze_threshold = cfg.get("squeeze_threshold", 1.05)

        # ── 止损止盈 ──
        self.stop_loss = cfg.get("stop_loss", 0.05)
        self.take_profit = cfg.get("take_profit", 0.30)

        # ── ATR 动态止损 ──
        self.use_atr_sl = cfg.get("atr_stop_loss", False)
        self.atr_sl = ATRStopLoss(
            atr_period=cfg.get("atr_period", 14),
            atr_mult=cfg.get("atr_mult", 2.0),
            trail_mult=cfg.get("trail_mult", 2.0),
        )

        # ── 凯利公式仓位 ──
        self.use_kelly = cfg.get("use_kelly", True)
        if self.use_kelly:
            self.kelly = KellySizer(
                estimated_win_rate=cfg.get("kelly_win_rate", 0.55),
                estimated_win_loss_ratio=cfg.get("kelly_win_loss_ratio", 1.5),
                kelly_fraction=cfg.get("kelly_fraction", 0.5),
                max_position_pct=cfg.get("kelly_max_position", 0.20),
                min_position_pct=cfg.get("kelly_min_position", 0.05),
                safety_discount=cfg.get("kelly_safety_discount", 0.8),
            )
        else:
            self.position_mode = cfg.get("position_mode", "half")

        logger.info(
            f"BbRsiComboStrategyPlugin initialized: "
            f"bb_period={self.bb_period}, bb_std={self.bb_std}, "
            f"rsi_oversold={self.rsi_oversold}, rsi_overbought={self.rsi_overbought}, "
            f"squeeze_lookback={self.squeeze_lookback}, squeeze_threshold={self.squeeze_threshold}, "
            f"kelly={'enabled' if self.use_kelly else 'disabled'}"
        )

    def _calculate_bollinger(self, close: pd.Series) -> pd.DataFrame:
        """计算布林带指标（使用 TA-Lib）"""
        upper, middle, lower = ta.BBANDS(
            close.values,
            timeperiod=self.bb_period,
            nbdevup=self.bb_std,
            nbdevdn=self.bb_std,
            matype=0  # 0 = SMA
        )
        return pd.DataFrame({
            'middle': middle, 'upper': upper, 'lower': lower
        })

    def _calculate_rsi(self, close: pd.Series) -> pd.Series:
        """计算RSI指标（使用 TA-Lib）"""
        return pd.Series(ta.RSI(close.values, timeperiod=self.rsi_period), index=close.index)

    def _detect_squeeze(self, data: pd.DataFrame) -> pd.Series:
        """
        检测布林带挤压（Squeeze）状态

        挤压 = 带宽处于历史最低水平附近，相当于弹簧被压到最紧。
        
        判断标准:
        - 当前带宽 / 历史最低带宽 < squeeze_threshold（默认1.05）
        - 即当前带宽在历史最低的5%以内

        John Bollinger原版: "Squeeze"是布林带最高质量的信号形态，
        预示着市场能量蓄积，即将有趋势性突破。
        """
        # 带宽 = (上轨 - 下轨) / 中轨
        bandwidth = (data['upper'] - data['lower']) / data['middle']
        data['bandwidth'] = bandwidth

        # 历史最低带宽（滚动窗口）
        min_bandwidth = data['bandwidth'].rolling(window=self.squeeze_lookback).min()
        data['min_bandwidth'] = min_bandwidth

        # 挤压判断: 当前带宽 < 历史最低 × 阈值
        # squeeze_threshold=1.05 意味着带宽在历史最低的5%以内视为挤压
        is_squeezed = (data['bandwidth'] / data['min_bandwidth']) < self.squeeze_threshold
        data['is_squeezed'] = is_squeezed

        return is_squeezed

    def run(self, df: pd.DataFrame, start_idx: int = 0) -> dict:
        """
        运行BB+RSI组合策略

        Returns:
            {"returns": float, "trades": list, "daily_values": list}
        """
        logger.info(f"Running BbRsiComboStrategyPlugin on {len(df)} days of data, start_idx={start_idx}...")
        self.trades = []
        self.daily_values = []
        self.position = 0
        self.cash = self.capital

        if len(df) == 0:
            return {"returns": 0.0, "trades": [], "daily_values": []}

        df = df.sort_values('trade_date').reset_index(drop=True)

        if start_idx < 0 or start_idx >= len(df):
            start_idx = 0

        # 列名映射（兼容有无复权列）
        data = df.copy()
        if "adj_close" not in data.columns:
            if "close" in data.columns:
                data["adj_close"] = data["close"]
            else:
                logger.error("缺少必要列: close / adj_close")
                return {"returns": 0.0, "trades": [], "daily_values": []}
        if "adj_open" not in data.columns:
            if "open" in data.columns:
                data["adj_open"] = data["open"]
            else:
                logger.error("缺少必要列: open / adj_open")
                return {"returns": 0.0, "trades": [], "daily_values": []}

        # === 计算指标 ===
        # 布林带
        bb = self._calculate_bollinger(data['adj_close'])
        data = pd.concat([data, bb], axis=1)

        # RSI
        data['rsi'] = self._calculate_rsi(data['adj_close'])

        # 带宽挤压检测
        self._detect_squeeze(data)

        # === ATR ===
        if self.use_atr_sl:
            high_col = 'high' if 'high' in data.columns else 'adj_close'
            low_col = 'low' if 'low' in data.columns else 'adj_close'
            atr_arr = self.atr_sl.calc_atr(data[high_col].values, data[low_col].values, data['adj_close'].values)
        else:
            atr_arr = np.zeros(len(data))

        # === 信号生成（基于前一日数据，避免未来函数）===
        prev_close = data['adj_close'].shift(1)
        prev_lower = data['lower'].shift(1)
        prev_upper = data['upper'].shift(1)
        prev_rsi = data['rsi'].shift(1)
        prev_squeezed = data['is_squeezed'].shift(1)

        # 入场条件（三重共振）:
        # 1. 价格接近下轨（在2%缓冲范围内）
        # 2. RSI < 超卖阈值
        # 3. 布林带处于挤压状态
        cond_price_low = (prev_close <= prev_lower * 1.02).fillna(False)
        cond_rsi_low = (prev_rsi < self.rsi_oversold).fillna(False)
        cond_squeezed = prev_squeezed.fillna(False)

        buy_signal = cond_price_low & cond_rsi_low & cond_squeezed

        # 出场条件（双确认）:
        # 价格接近上轨（在2%缓冲范围内）AND RSI > 超买阈值
        cond_price_high = (prev_close >= prev_upper * 0.98).fillna(False)
        cond_rsi_high = (prev_rsi > self.rsi_overbought).fillna(False)

        sell_signal = cond_price_high & cond_rsi_high

        data['buy_signal'] = buy_signal
        data['sell_signal'] = sell_signal

        # === 遍历数据执行交易 ===
        entry_reason = ""  # 记录最后一次入场理由（用于止损/止盈日志）

        for i in range(start_idx, len(data)):
            row = data.iloc[i]
            date = row['trade_date']
            open_price = float(row['adj_open'])
            close_price = float(row['adj_close'])

            # 跳过早期无指标数据
            if pd.isna(row['middle']) or pd.isna(row['rsi']):
                v = self.cash + self.position * close_price if self.position > 0 else self.cash
                self.daily_values.append({'date': date, 'portfolio_value': v})
                continue

            # 前一日收盘价（用于止损判断）
            prev_close_val = float(data.iloc[i - 1]['adj_close']) if i >= 1 else None

            # === 买入逻辑 ===
            if self.position == 0 and bool(data.iloc[i]['buy_signal']):
                # 计算仓位
                if self.use_kelly:
                    position_pct = self.kelly.get_position_pct()
                    kelly_info = self.kelly.get_info()
                    logger.debug(f"Kelly仓位: f*={kelly_info.get('f_star', 0):.4f}, "
                                 f"final={position_pct:.2%}")
                else:
                    position_pct = 0.95 if self.position_mode == 'full' else 0.50

                if position_pct <= 0:
                    # 凯利公式给出零仓位（期望值为负），跳过
                    v = self.cash
                    self.daily_values.append({'date': date, 'portfolio_value': v})
                    continue

                buy_amount = self.cash * position_pct
                shares = int(buy_amount / open_price / 100) * 100
                if shares > 0:
                    # ── 修复未来函数：日志中显示的是用于决策的前一日值 ──
                    rsi_val = float(prev_rsi.iloc[i]) if i < len(prev_rsi) else 0
                    # 带宽相关指标在 _detect_squeeze 中已计算，但用于决策的是 shift(1) 后的值
                    prev_bw = data['bandwidth'].shift(1).iloc[i] if i < len(data) else 0
                    prev_min_bw = data['min_bandwidth'].shift(1).iloc[i] if i < len(data) else 0
                    bw = float(prev_bw) if pd.notna(prev_bw) else 0
                    min_bw = float(prev_min_bw) if pd.notna(prev_min_bw) else 0
                    squeeze_ratio = bw / min_bw if min_bw > 0 else 999

                    entry_reason = (
                        f"BB+RSI挤压入场(RSI={rsi_val:.1f},"
                        f"带宽={bw:.3f},挤压比={squeeze_ratio:.3f},"
                        f"仓位={position_pct:.1%})"
                    )
                    success = self.buy(date, open_price, shares, reason=entry_reason)
                    if success and self.use_atr_sl:
                        self.atr_sl.on_entry(entry_price=open_price, atr_val=atr_arr[i])

            # === 卖出逻辑 ===
            elif self.position > 0:
                # ── ATR 追踪止损 ──
                if self.use_atr_sl:
                    high_val = float(data.iloc[i].get('high', close_price))
                    self.atr_sl.update(high_price=high_val, atr_val=atr_arr[i])
                    should_stop, stop_price, atr_reason = self.atr_sl.check_stop(close_price=close_price)
                    if should_stop:
                        self.sell(date, open_price, reason=atr_reason)
                        v = self.cash + self.position * close_price if self.position > 0 else self.cash
                        self.daily_values.append({'date': date, 'portfolio_value': v})
                        continue

                should_sell = False
                sell_reason = ""

                # 1. 出场信号（价格近上轨 + RSI > 70 双确认）
                if bool(data.iloc[i]['sell_signal']):
                    should_sell = True
                    rsi_val = float(prev_rsi.iloc[i]) if i < len(prev_rsi) else 0
                    sell_reason = f"BB+RSI超买出场(RSI={rsi_val:.1f})"

                # 2. 止盈（基于前一日收盘价）
                if not should_sell and self.avg_cost > 0 and prev_close_val is not None:
                    profit_pct = (prev_close_val - self.avg_cost) / self.avg_cost
                    if profit_pct >= self.take_profit:
                        should_sell = True
                        sell_reason = f"止盈({profit_pct:.1%})"

                # 3. 止损（ATR启用时跳过固定止损）
                if not should_sell and not self.use_atr_sl:
                    if self.avg_cost > 0 and prev_close_val is not None:
                        loss_pct = (prev_close_val - self.avg_cost) / self.avg_cost
                        if loss_pct <= -self.stop_loss:
                            should_sell = True
                            sell_reason = f"硬止损({loss_pct:.1%})"

                if should_sell and self.position > 0:
                    self.sell(date, open_price, reason=sell_reason)

            # 记录每日资产
            v = self.cash + self.position * close_price if self.position > 0 else self.cash
            self.daily_values.append({'date': date, 'portfolio_value': v})

        # 回测结束平仓
        if self.position > 0:
            last_date = data.iloc[-1]['trade_date']
            last_price = float(data.iloc[-1]['adj_close'])
            self.sell(last_date, last_price, reason="回测结束平仓")
            v_final = self.cash
            self.daily_values[-1]['portfolio_value'] = v_final

        ret = self.calc_returns()
        logger.info(f"BbRsiComboStrategyPlugin finished: returns={ret:.2f}%, trades={len(self.trades)}")
        return {"returns": ret, "trades": self.trades, "daily_values": self.daily_values}
