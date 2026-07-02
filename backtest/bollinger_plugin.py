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

仓位管理（v2）:
- use_kelly=False: 固定半仓(50%)/全仓(95%)
- use_kelly=True: 半凯利公式动态仓位（默认上限20%）

优化：使用 TA-Lib 计算布林带和 RSI（性能提升 10-100 倍）
"""
import pandas as pd
import numpy as np
import talib as ta  # ← 新增 TA-Lib
from backtest.base_strategy import BaseStrategy
from backtest.atr_stop_loss import ATRStopLoss
from backtest.kelly_sizer import KellySizer
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
        # ── RSI 入场阈值（修复硬编码 40）──
        self.rsi_oversold = cfg.get("rsi_oversold", 40)
        # ── ATR 动态止损 ──
        self.use_atr_sl = cfg.get("atr_stop_loss", False)
        self.atr_sl = ATRStopLoss(
            atr_period=cfg.get("atr_period", 14),
            atr_mult=cfg.get("atr_mult", 3.0),
            trail_mult=cfg.get("trail_mult", 3.0),
        )
        # ── 凯利公式仓位 ──
        self.use_kelly = cfg.get("use_kelly", False)
        if self.use_kelly:
            self.kelly = KellySizer(
                estimated_win_rate=cfg.get("kelly_win_rate", 0.55),
                estimated_win_loss_ratio=cfg.get("kelly_win_loss_ratio", 1.5),
                kelly_fraction=cfg.get("kelly_fraction", 0.5),
                max_position_pct=cfg.get("kelly_max_position", 0.25),
                min_position_pct=cfg.get("kelly_min_position", 0.05),
                safety_discount=cfg.get("kelly_safety_discount", 0.8),
            )
        mode_str = "kelly" if self.use_kelly else self.position_mode
        logger.info(f"BollingerStrategyPlugin initialized: period={self.bb_period}, std={self.bb_std}, "
                     f"position={mode_str}")
    
    def _calculate_bollinger(self, close: pd.Series) -> pd.DataFrame:
        """计算布林带指标（使用 TA-Lib 优化）"""
        # TA-Lib BBANDS 返回 (upper, middle, lower)
        upper, middle, lower = ta.BBANDS(
            close.values,
            timeperiod=self.bb_period,
            nbdevup=self.bb_std,
            nbdevdn=self.bb_std,
            matype=0  # 0 = SMA（简单移动平均）
        )
        return pd.DataFrame({
            'middle': middle, 'upper': upper, 'lower': lower
        })
    
    def _calculate_rsi(self, close: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI指标（使用 TA-Lib 优化）"""
        # TA-Lib RSI：前 (period-1) 个值为 NaN
        return pd.Series(ta.RSI(close.values, timeperiod=period), index=close.index)
    
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
        
        data['buy_signal'] = (prev_close <= prev_lower * 1.02) & (prev_rsi < self.rsi_oversold)
        data['sell_signal_bb'] = prev_close >= prev_upper
        
        # ── ATR 计算 ──
        if self.use_atr_sl:
            high_col = 'high' if 'high' in data.columns else 'adj_close'
            low_col = 'low' if 'low' in data.columns else 'adj_close'
            atr_arr = self.atr_sl.calc_atr(data[high_col].values, data[low_col].values, data['adj_close'].values)
        else:
            atr_arr = np.zeros(len(data))
        
        for i in range(start_idx, len(data)):  # ← 修复：循环起点用 start_idx（避免回测期前数据干扰）
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
                # 计算仓位比例
                if self.use_kelly:
                    position_pct = self.kelly.get_position_pct()
                    if position_pct <= 0:
                        # 凯利公式给出零仓位（期望值为负），跳过
                        v = self.cash
                        self.daily_values.append({'date': date, 'portfolio_value': v})
                        continue
                else:
                    position_pct = 0.95 if self.position_mode == 'full' else 0.50
                buy_amount = self.cash * position_pct
                shares = int(buy_amount / open_price / 100) * 100
                if shares > 0:
                    success = self.buy(date, open_price, shares, reason="布林带下轨+RSI超卖")
                    # ATR初始化（仅当买入成功时）
                    if success and self.use_atr_sl:
                        self.atr_sl.on_entry(entry_price=open_price, atr_val=atr_arr[i])
            
            # 卖出逻辑
            elif self.position > 0:
                # ── ATR 追踪止损 ──
                if self.use_atr_sl:
                    high_val = float(row.get('high', close_price))
                    self.atr_sl.update(high_price=high_val, atr_val=atr_arr[i])
                    should_stop, stop_price, atr_reason = self.atr_sl.check_stop(close_price=close_price)
                    if should_stop:
                        self.sell(date, open_price, reason=atr_reason)
                        v = self.cash + self.position * close_price if self.position > 0 else self.cash
                        self.daily_values.append({'date': date, 'portfolio_value': v})
                        continue

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
                    elif not self.use_atr_sl and profit_pct <= -self.stop_loss:
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
            # 更新最终资产值（扣除卖出费用后的实际现金）
            self.daily_values[-1]['portfolio_value'] = self.cash
        
        ret = self.calc_returns()
        logger.info(f"BollingerStrategyPlugin finished: returns={ret:.2f}%, trades={len(self.trades)}")
        return {"returns": ret, "trades": self.trades, "daily_values": self.daily_values}
