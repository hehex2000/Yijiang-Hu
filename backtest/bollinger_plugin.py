#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
布林带均值回归策略插件（继承 BaseStrategy）
- 信号基于前一日收盘价 + 布林带上下轨判断
- 今日开盘价成交
- 止盈止损基于前一日收盘价判断
价格触及下轨 → 买入（超卖）
价格触及上轨 → 卖出（超买）
止盈50%，止损15%
"""
import pandas as pd
import numpy as np
from backtest.base_strategy import BaseStrategy
from loguru import logger


class BollingerStrategyPlugin(BaseStrategy):
    """布林带均值回归策略（插件版）"""
    
    def __init__(self, capital: float, cfg: dict):
        super().__init__("BollingerStrategyPlugin", capital, cfg)
        self.bb_period = cfg.get("bb_period", 20)
        self.bb_std = cfg.get("bb_std", 2.5)
        self.take_profit = cfg.get("take_profit", 0.50)
        self.stop_loss = cfg.get("stop_loss", 0.15)
        self.position_mode = cfg.get("position_mode", "half")
        logger.info(f"BollingerStrategyPlugin initialized: period={self.bb_period}, std={self.bb_std}")
    
    def _calculate_bollinger(self, close: pd.Series) -> pd.DataFrame:
        """计算布林带指标"""
        middle = close.rolling(window=self.bb_period).mean()
        std = close.rolling(window=self.bb_period).std()
        return pd.DataFrame({
            'middle': middle, 'upper': middle + std * self.bb_std,
            'lower': middle - std * self.bb_std
        })
    
    def _calculate_rsi(self, close: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta).where(delta < 0, 0).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def run(self, df: pd.DataFrame, start_idx: int = 0) -> dict:
        """
        运行布林带策略
        返回: {"returns": float, "trades": list}
        """
        logger.info(f"Running BollingerStrategyPlugin on {len(df)} days of data...")
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
        bb = self._calculate_bollinger(data['adj_close'])
        data = pd.concat([data, bb], axis=1)
        data['rsi'] = self._calculate_rsi(data['adj_close'])
        
        # 信号基于前一日数据（修复未来函数）
        prev_close = data['adj_close'].shift(1)
        prev_upper = data['upper'].shift(1)
        prev_lower = data['lower'].shift(1)
        prev_rsi = data['rsi'].shift(1)
        
        data['buy_signal'] = (prev_close <= prev_lower * 1.02) & (prev_rsi < 40)
        data['sell_signal_bb'] = prev_close >= prev_upper
        
        for i in range(len(data)):
            row = data.iloc[i]
            date = row['trade_date']
            open_price = row['adj_open']
            close_price = row['adj_close']
            
            # 跳过早期无指标数据
            if pd.isna(row['upper']) or pd.isna(row['lower']):
                v = self.cash + self.position * close_price if self.position > 0 else self.cash
                self.daily_values.append({'date': date, 'portfolio_value': v})
                continue
            
            # 前一日收盘价（用于止盈止损判断）
            prev_close_val = float(data.iloc[i - 1]['adj_close']) if i >= 1 else float('nan')
            
            # 买入逻辑
            if self.position == 0 and bool(data.iloc[i]['buy_signal']) and not pd.isna(row['rsi']):
                buy_amount = self.cash * (0.95 if self.position_mode == 'full' else 0.50)
                shares = int(buy_amount / open_price / 100) * 100
                if shares > 0:
                    self.buy(date, open_price, shares, reason="布林带下轨+RSI超卖")
            
            # 卖出逻辑
            elif self.position > 0:
                sell_signal = False
                sell_reason = ""
                
                # 1. 前一日触及上轨（超买）
                if bool(data.iloc[i]['sell_signal_bb']) and not pd.isna(row['rsi']):
                    sell_signal = True
                    sell_reason = "布林带上轨(昨日收盘)"
                
                # 2. 止盈（用前一日收盘价判断）
                if self.avg_cost > 0 and pd.notna(prev_close_val):
                    profit_pct = (prev_close_val - self.avg_cost) / self.avg_cost
                    if profit_pct >= self.take_profit:
                        sell_signal = True
                        sell_reason = f"止盈({profit_pct:.1%})"
                    elif profit_pct <= -self.stop_loss:
                        sell_signal = True
                        sell_reason = f"止损({profit_pct:.1%})"
                
                if sell_signal and self.position > 0:
                    self.sell(date, open_price, reason=sell_reason)
            
            v = self.cash + self.position * close_price if self.position > 0 else self.cash
            self.daily_values.append({'date': date, 'portfolio_value': v})
        
        # 回测结束平仓
        if self.position > 0:
            last_price = float(data.iloc[-1]['adj_close'])
            self.sell(data.iloc[-1]['trade_date'], last_price, reason="回测结束平仓")
        
        ret = self.calc_returns()
        logger.info(f"BollingerStrategyPlugin finished: returns={ret:.2f}%, trades={len(self.trades)}")
        return {"returns": ret, "trades": self.trades, "daily_values": self.daily_values}
