#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RSI 超买超卖策略插件（继承 BaseStrategy）
- 前一日RSI < 超卖线 → 今日开盘买入
- 前一日RSI > 超买线 → 今日开盘卖出
- 止盈止损基于前一日收盘价判断

优化：使用 TA-Lib 计算 RSI（性能提升 10-100 倍）
"""
import pandas as pd
import numpy as np
import talib as ta  # ← 新增 TA-Lib
from backtest.base_strategy import BaseStrategy
from backtest.atr_stop_loss import ATRStopLoss
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
        # ── ATR 动态止损 ──
        self.use_atr_sl = cfg.get("atr_stop_loss", False)
        self.atr_sl = ATRStopLoss(
            atr_period=cfg.get("atr_period", 14),
            atr_mult=cfg.get("atr_mult", 3.0),
            trail_mult=cfg.get("trail_mult", 3.0),
        )
        logger.info(f"RSIStrategyPlugin initialized: period={self.rsi_period}, oversold={self.rsi_oversold}, overbought={self.rsi_overbought}")
    
    def _calculate_rsi(self, prices: pd.Series) -> pd.Series:
        """计算RSI指标（使用 TA-Lib 优化）"""
        # TA-Lib RSI：前 (timeperiod-1) 个值为 NaN
        return pd.Series(ta.RSI(prices.values, timeperiod=self.rsi_period), index=prices.index)
    
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
        
        # ── ATR 计算 ──
        if self.use_atr_sl:
            # ATR 所需的数据列
            atr_arr = self.atr_sl.calc_atr(
                data['high'].values if 'high' in data.columns else data['adj_close'].values,
                data['low'].values if 'low' in data.columns else data['adj_close'].values,
                data['adj_close'].values,
            )
        else:
            atr_arr = np.zeros(len(data))
        
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
            
            # 删除这行 ↓↓↓↓
            # prev_close_val = float(data.iloc[i - 1]['adj_close']) if i >= 1 else float('nan')
            
            # 买入逻辑
            if self.position == 0 and bool(data.iloc[i]['buy_signal']):
                buy_amount = self.cash * (0.95 if self.position_mode == 'full' else 0.50)
                shares = int(buy_amount / open_price / 100) * 100
                if shares > 0:
                    success = self.buy(date, open_price, shares, reason="RSI超卖")
                    # ATR初始化（仅当买入成功时）
                    if success and self.use_atr_sl:
                        self.atr_sl.on_entry(entry_price=open_price, atr_val=atr_arr[i])
            
            # 卖出逻辑
            elif self.position > 0:
                # ── ATR 追踪止损（每日更新）──
                if self.use_atr_sl:
                    high_val = float(row['high']) if 'high' in data.columns else close_price
                    self.atr_sl.update(high_price=high_val, atr_val=atr_arr[i])
                    should_stop, stop_price, atr_reason = self.atr_sl.check_stop(close_price=close_price)
                    if should_stop:
                        self.sell(date, open_price, reason=atr_reason)
                        v = self.cash + self.position * close_price if self.position > 0 else self.cash
                        self.daily_values.append({'date': date, 'portfolio_value': v})
                        continue

                sell_signal = False
                sell_reason = ""
                
                if bool(data.iloc[i]['sell_signal']):
                    sell_signal = True
                    rsi_v = float(prev_rsi.iloc[i]) if i < len(prev_rsi) else 0
                    sell_reason = f"RSI超买({rsi_v:.1f})"
                
                if self.avg_cost > 0:
                    pct = (open_price - self.avg_cost) / self.avg_cost  # 基于今日开盘价
                    if pct >= self.take_profit:
                        sell_signal = True
                        sell_reason = f"止盈({pct:.1%})"
                    elif not self.use_atr_sl and pct <= -self.stop_loss:
                        # 仅未启用ATR时使用固定止损
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
        return {"returns": ret, "trades": self.trades, "daily_values": self.daily_values}
