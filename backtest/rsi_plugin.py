#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RSI 超买超卖策略插件（继承 BaseStrategy）
- 前一日RSI < 超卖线 → 今日开盘买入
- 前一日RSI > 超买线 → 今日开盘卖出
- 止盈止损基于前一日收盘价判断
"""
import pandas as pd
import numpy as np
from backtest.base_strategy import BaseStrategy
from loguru import logger


class RSIStrategyPlugin(BaseStrategy):
    """RSI 超买超卖策略（插件版）"""
    
    def __init__(self, capital: float, cfg: dict):
        super().__init__("RSIStrategyPlugin", capital, cfg)
        self.rsi_period = cfg.get("rsi_period", 14)
        self.rsi_oversold = cfg.get("rsi_oversold", 40)
        self.rsi_overbought = cfg.get("rsi_overbought", 60)
        self.take_profit = cfg.get("take_profit", 0.50)
        self.stop_loss = cfg.get("stop_loss", 0.15)
        self.position_mode = cfg.get("position_mode", "half")
        logger.info(f"RSIStrategyPlugin initialized: period={self.rsi_period}, oversold={self.rsi_oversold}, overbought={self.rsi_overbought}")
    
    def _calculate_rsi(self, prices: pd.Series) -> pd.Series:
        """计算RSI指标"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(com=self.rsi_period - 1, min_periods=self.rsi_period).mean()
        avg_loss = loss.ewm(com=self.rsi_period - 1, min_periods=self.rsi_period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def run(self, df: pd.DataFrame, start_idx: int = 0) -> dict:
        """
        运行RSI策略
        返回: {"returns": float, "trades": list}
        """
        logger.info(f"Running RSIStrategyPlugin on {len(df)} days of data...")
        self.trades = []
        self.daily_values = []
        self.position = 0
        self.cash = self.capital
        
        if len(df) == 0:
            return {"returns": 0.0, "trades": []}
        
        df = df.sort_values('trade_date').reset_index(drop=True)
        
        # 列名映射（兼容有无复权列）
        data = df.copy()
        if "adj_close" not in data.columns:
            if "close" in data.columns:
                data["adj_close"] = data["close"]
            else:
                logger.error("缺少必要列: close / adj_close")
                return {"returns": 0.0, "trades": []}
        if "adj_open" not in data.columns:
            if "open" in data.columns:
                data["adj_open"] = data["open"]
            else:
                logger.error("缺少必要列: open / adj_open")
                return {"returns": 0.0, "trades": []}
        
        data['rsi'] = self._calculate_rsi(data['adj_close'])
        
        # 信号基于前一日RSI（修复未来函数）
        prev_rsi = data['rsi'].shift(1)
        data['buy_signal'] = prev_rsi < self.rsi_oversold
        data['sell_signal'] = prev_rsi > self.rsi_overbought
        
        for i in range(len(data)):
            row = data.iloc[i]
            date = row['trade_date']
            open_price = row['adj_open']
            close_price = row['adj_close']
            
            if pd.isna(row['rsi']):
                v = self.cash + self.position * close_price if self.position > 0 else self.cash
                self.daily_values.append({'date': date, 'portfolio_value': v})
                continue
            
            prev_close_val = float(data.iloc[i - 1]['adj_close']) if i >= 1 else float('nan')
            
            # 买入逻辑
            if self.position == 0 and bool(data.iloc[i]['buy_signal']):
                buy_amount = self.cash * (0.95 if self.position_mode == 'full' else 0.50)
                shares = int(buy_amount / open_price / 100) * 100
                if shares > 0:
                    self.buy(date, open_price, shares, reason="RSI超卖")
            
            # 卖出逻辑
            elif self.position > 0:
                sell_signal = False
                sell_reason = ""
                
                if bool(data.iloc[i]['sell_signal']):
                    sell_signal = True
                    rsi_v = float(prev_rsi.iloc[i]) if i < len(prev_rsi) else 0
                    sell_reason = f"RSI超买({rsi_v:.1f})"
                
                if self.avg_cost > 0 and pd.notna(prev_close_val):
                    pct = (prev_close_val - self.avg_cost) / self.avg_cost
                    if pct >= self.take_profit:
                        sell_signal = True
                        sell_reason = f"止盈({pct:.1%})"
                    elif pct <= -self.stop_loss:
                        sell_signal = True
                        sell_reason = f"止损({pct:.1%})"
                
                if sell_signal and self.position > 0:
                    self.sell(date, open_price, reason=sell_reason)
            
            v = self.cash + self.position * close_price if self.position > 0 else self.cash
            self.daily_values.append({'date': date, 'portfolio_value': v})
        
        # 回测结束平仓
        if self.position > 0:
            last_price = float(data.iloc[-1]['adj_close'])
            self.sell(data.iloc[-1]['trade_date'], last_price, reason="回测结束平仓")
        
        ret = self.calc_returns()
        logger.info(f"RSIStrategyPlugin finished: returns={ret:.2f}%, trades={len(self.trades)}")
        return {"returns": ret, "trades": self.trades}
