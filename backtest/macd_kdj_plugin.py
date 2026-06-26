#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MACD/RSI 组合策略插件（继承 BaseStrategy）
- 前一日MACD金叉+RSI<70 → 今日开盘买入
- 前一日MACD死叉或RSI>70 → 今日开盘卖出
- 止盈止损基于前一日收盘价判断

优化：使用 TA-Lib 计算 MACD 和 RSI（性能提升 10-100 倍）
"""
import pandas as pd
import numpy as np
import talib as ta  # ← 新增 TA-Lib
from backtest.base_strategy import BaseStrategy
from backtest.atr_stop_loss import ATRStopLoss
from loguru import logger


class MACDRSIStrategyPlugin(BaseStrategy):
    """MACD/RSI 组合策略（插件版）"""
    
    def __init__(self, capital: float, cfg: dict):
        super().__init__("MACDRSIStrategyPlugin", capital, cfg)
        self.macd_fast = cfg.get("macd_fast", 12)
        self.macd_slow = cfg.get("macd_slow", 26)
        self.macd_signal = cfg.get("macd_signal", 9)
        self.rsi_period = cfg.get("rsi_period", 14)
        self.rsi_overbought = cfg.get("rsi_overbought", 70)
        self.rsi_oversold = cfg.get("rsi_oversold", 30)
        self.take_profit = cfg.get("take_profit", 0.25)
        self.stop_loss = cfg.get("stop_loss", 0.10)
        self.position_mode = cfg.get("position_mode", "half")
        # ── ATR 动态止损 ──
        self.use_atr_sl = cfg.get("atr_stop_loss", False)
        self.atr_sl = ATRStopLoss(
            atr_period=cfg.get("atr_period", 14),
            atr_mult=cfg.get("atr_mult", 3.0),
            trail_mult=cfg.get("trail_mult", 3.0),
        )
        logger.info(f"MACDRSIStrategyPlugin initialized: macd=({self.macd_fast},{self.macd_slow},{self.macd_signal})")
    
    def _calculate_macd(self, close: pd.Series):
        """计算MACD指标（使用 TA-Lib 优化）"""
        # TA-Lib MACD 返回 (macd, macdsignal, macdhist) - 都是 numpy ndarray
        macd, macdsignal, macdhist = ta.MACD(
            close.values,
            fastperiod=self.macd_fast,
            slowperiod=self.macd_slow,
            signalperiod=self.macd_signal
        )
        # 转成 pandas Series（保留 index，以便使用 .shift() 等方法）
        return pd.Series(macd, index=close.index), pd.Series(macdsignal, index=close.index)
    
    def _calculate_rsi(self, close: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI指标（使用 TA-Lib 优化）"""
        return pd.Series(ta.RSI(close.values, timeperiod=period), index=close.index)
    
    def run(self, df: pd.DataFrame, start_idx: int = 0) -> dict:
        """
        运行MACD/RSI策略
        返回: {"returns": float, "trades": list}
        """
        logger.info(f"Running MACDRSIStrategyPlugin on {len(df)} days of data...")
        self.trades = []
        self.daily_values = []
        self.position = 0
        self.cash = self.capital
        
        if len(df) == 0:
            return {"returns": 0.0, "trades": [], "daily_values": []}
        
        df = df.sort_values('trade_date').reset_index(drop=True)
        
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
        
        # 计算指标
        dif, dea = self._calculate_macd(data['adj_close'])
        rsi = self._calculate_rsi(data['adj_close'], self.rsi_period)
        
        # 信号基于前一日数据（修复未来函数）
        prev_golden = (dif.shift(1) > dea.shift(1)) & (dif.shift(2) <= dea.shift(2))
        prev_death = (dif.shift(1) < dea.shift(1)) & (dif.shift(2) >= dea.shift(2))
        prev_rsi = rsi.shift(1)
        
        # ── ATR 计算 ──
        if self.use_atr_sl:
            high_col = 'high' if 'high' in data.columns else 'adj_close'
            low_col = 'low' if 'low' in data.columns else 'adj_close'
            atr_arr = self.atr_sl.calc_atr(data[high_col].values, data[low_col].values, data['adj_close'].values)
        else:
            atr_arr = np.zeros(len(data))
        
        for i in range(len(data)):
            row = data.iloc[i]
            date = row['trade_date']
            open_price = row['adj_open']
            close_price = row['adj_close']
            
            # 跳过早期数据
            if i < 2 or pd.isna(close_price):
                v = self.cash + self.position * close_price if self.position > 0 else self.cash
                v = self.cash + self.position * close_price if self.position > 0 else self.cash
                self.daily_values.append({'date': date, 'portfolio_value': v})
                continue
            
            prev_close = float(data.iloc[i - 1]['adj_close'])
            
            # 止盈止损 + ATR追踪止损
            if self.position > 0:
                # ── ATR 追踪止损 ──
                if self.use_atr_sl:
                    high_val = float(row.get('high', close_price))
                    self.atr_sl.update(high_price=high_val, atr_val=atr_arr[i])
                    should_stop, stop_price, atr_reason = self.atr_sl.check_stop(close_price=close_price)
                    if should_stop:
                        self.sell(date, open_price, reason=atr_reason)
                        continue

                current_return = (prev_close - self.avg_cost) / self.avg_cost if self.avg_cost > 0 else 0
                if current_return >= self.take_profit:
                    # 卖出
                    self.sell(date, open_price, reason=f"止盈({current_return:.1%})")
                    continue
                elif not self.use_atr_sl and current_return <= -self.stop_loss:
                    # 卖出
                    self.sell(date, open_price, reason=f"止损({current_return:.1%})")
                    continue
            
            # 买入信号（前一日MACD金叉 + RSI未超买）
            if self.position == 0 and self.cash > 0:
                in_golden = bool(prev_golden.iloc[i])
                rsi_ok = pd.notna(prev_rsi.iloc[i]) and prev_rsi.iloc[i] < self.rsi_overbought
                if in_golden and rsi_ok:
                    buy_amount = self.cash * (0.95 if self.position_mode == 'full' else 0.50)
                    shares = int(buy_amount / open_price / 100) * 100
                    if shares > 0:
                        success = self.buy(date, open_price, shares, reason="MACD金叉+RSI未超买")
                        # ATR初始化（仅当买入成功时）
                        if success and self.use_atr_sl:
                            self.atr_sl.on_entry(entry_price=open_price, atr_val=atr_arr[i])
            
            # 卖出信号（前一日MACD死叉 或 RSI超买）
            elif self.position > 0:
                in_death = bool(prev_death.iloc[i])
                rsi_over = pd.notna(prev_rsi.iloc[i]) and prev_rsi.iloc[i] > self.rsi_overbought
                if in_death or rsi_over:
                    self.sell(date, open_price, reason="MACD死叉或RSI超买")
            
            v = self.cash + self.position * close_price if self.position > 0 else self.cash
            self.daily_values.append({'date': date, 'portfolio_value': v})
        
        # 回测结束平仓
        if self.position > 0:
            last_price = float(data.iloc[-1]['adj_close'])
            self.sell(data.iloc[-1]['trade_date'], last_price, reason="回测结束平仓")
        
        ret = self.calc_returns()
        logger.info(f"MACDRSIStrategyPlugin finished: returns={ret:.2f}%, trades={len(self.trades)}")
        return {"returns": ret, "trades": self.trades, "daily_values": self.daily_values}
