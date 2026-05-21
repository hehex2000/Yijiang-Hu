"""
主动量化策略 - MACD/RSI组合策略
MACD金叉 + RSI未超买(<70) → 买入
MACD死叉 或 RSI超买(>70) 或 止盈止损 → 卖出
止盈25%，止损10%
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from loguru import logger


class MACDRSIStrategy:
    """主动量化：MACD/RSI组合策略"""
    
    def __init__(self, 
                 total_capital: float = 100000.0,
                 macd_fast: int = 12,
                 macd_slow: int = 26,
                 macd_signal: int = 9,
                 rsi_period: int = 14,
                 rsi_overbought: float = 70,
                 rsi_oversold: float = 30,
                 take_profit: float = 0.25,
                 stop_loss: float = 0.10,
                 trading_fee_rate: float = 0.0002,
                 stamp_duty_rate: float = 0.001):
        """
        初始化MACD/RSI组合策略
        
        Args:
            total_capital: 总资金（默认10万）
            macd_fast: MACD快线周期（默认12）
            macd_slow: MACD慢线周期（默认26）
            macd_signal: MACD信号线周期（默认9）
            rsi_period: RSI周期（默认14）
            rsi_overbought: RSI超买线（默认70）
            rsi_oversold: RSI超卖线（默认30）
            take_profit: 止盈线（默认25%）
            stop_loss: 止损线（默认-10%）
            trading_fee_rate: 交易手续费率（默认万分之2 = 0.0002）
            stamp_duty_rate: 印花税率（默认千分之1 = 0.001，仅卖出收取）
        """
        self.total_capital = total_capital
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.trading_fee_rate = trading_fee_rate
        self.stamp_duty_rate = stamp_duty_rate
        
        self.cash = total_capital          # 可用现金
        self.position_shares = 0          # 持仓股数
        self.avg_cost = 0.0               # 平均持仓成本
        self.orders = []                   # 交易记录
        self.daily_values = []             # 每日市值（用于计算绩效）
        
        logger.info(f"MACDRSIStrategy initialized: macd={macd_fast}/{macd_slow}/{macd_signal}, "
                   f"rsi={rsi_period}/{rsi_overbought}/{rsi_oversold}, "
                   f"take_profit={take_profit}, stop_loss={stop_loss}")
    
    def _calculate_macd(self, close_prices: pd.Series) -> tuple:
        """
        计算MACD指标
        
        Returns:
            (dif, dea, macd_hist)
        """
        # 计算EMA
        ema_fast = close_prices.ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = close_prices.ewm(span=self.macd_slow, adjust=False).mean()
        
        # 计算DIF和DEA
        dif = ema_fast - ema_slow
        dea = dif.ewm(span=self.macd_signal, adjust=False).mean()
        
        # 计算MACD柱
        macd_hist = (dif - dea) * 2
        
        return dif, dea, macd_hist
    
    def _calculate_rsi(self, close_prices: pd.Series) -> pd.Series:
        """
        计算RSI指标
        
        Returns:
            RSI序列
        """
        # 计算价格变化
        delta = close_prices.diff()
        
        # 分别计算上涨和下跌
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # 计算平均上涨和下跌
        avg_gain = gain.rolling(window=self.rsi_period).mean()
        avg_loss = loss.rolling(window=self.rsi_period).mean()
        
        # 计算RS和RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def run(self, df: pd.DataFrame) -> List[Dict]:
        """
        运行MACD/RSI组合策略，返回交易记录
        
        Args:
            df: 包含 trade_date, adj_open, adj_close 的 DataFrame
            
        Returns:
            交易记录列表：[{date, action, price, shares, amount, profit, return_pct, reason}]
        """
        logger.info(f"Running MACD/RSI strategy on {len(df)} days of data...")
        
        self.orders = []
        self.daily_values = []
        self.cash = self.total_capital
        self.position_shares = 0
        self.avg_cost = 0.0
        
        # 计算MACD和RSI指标
        dif, dea, macd_hist = self._calculate_macd(df['adj_close'])
        rsi = self._calculate_rsi(df['adj_close'])
        
        # 计算金叉死叉
        golden_cross = (dif > dea) & (dif.shift(1) <= dea.shift(1))
        death_cross = (dif < dea) & (dif.shift(1) >= dea.shift(1))
        
        # 遍历每个交易日
        for idx, row in df.iterrows():
            # 记录每日市值（用当天adj_close计算）
            self._record_daily_value(row)
            
            current_price = row['adj_close']
            current_rsi = rsi.iloc[idx]
            
            # 检查止盈止损（每天检查，用当天收盘价）
            if self.position_shares > 0:
                current_return = (current_price - self.avg_cost) / self.avg_cost
                
                # 止盈
                if current_return >= self.take_profit:
                    self._sell(row, reason="take_profit")
                    continue
                
                # 止损
                elif current_return <= -self.stop_loss:
                    self._sell(row, reason="stop_loss")
                    continue
            
            # 买入信号：MACD金叉 + RSI未超买
            if golden_cross.iloc[idx] and current_rsi < self.rsi_overbought and self.cash > 0:
                self._buy(row, reason="macd_golden_cross_rsi_not_overbought")
            
            # 卖出信号：MACD死叉 或 RSI超买
            elif (death_cross.iloc[idx] or current_rsi > self.rsi_overbought) and self.position_shares > 0:
                self._sell(row, reason="macd_death_cross_or_rsi_overbought")
        
        # 如果回测结束仍有持仓，按最后一天收盘价卖出
        if self.position_shares > 0:
            last_row = df.iloc[-1]
            self._sell(last_row, reason="end_of_backtest")
        
        logger.info(f"✓ MACD/RSI strategy completed: {len(self.orders)} orders")
        return {
            'trades': self.orders,
            'daily_values': self.daily_values
        }
    
    def _record_daily_value(self, row: pd.Series):
        """记录每日市值"""
        current_price = row['adj_close']
        portfolio_value = self.cash + self.position_shares * current_price
        
        self.daily_values.append({
            'date': row['trade_date'],
            'portfolio_value': portfolio_value
        })
    
    def _buy(self, row: pd.Series, reason: str):
        """
        买入操作：买入 50% 可用资金
        
        Args:
            row: 当前行的数据（包含 trade_date, adj_open, adj_close）
            reason: 买入原因
        """
        price = row['adj_close']  # 使用收盘价买入
        buy_amount = self.cash * 0.5  # 买入50%可用资金
        shares = int(buy_amount / price / 100) * 100  # 整百股
        
        if shares > 0:
            actual_cost = shares * price
            trading_fee = max(actual_cost * self.trading_fee_rate, 5)
            total_cost = actual_cost + trading_fee
            
            self.orders.append({
                'date': row['trade_date'],
                'action': 'buy',
                'price': price,
                'shares': shares,
                'amount': actual_cost,
                'trading_fee': trading_fee,
                'total_cost': total_cost,
                'profit': None,
                'return_pct': None,
                'reason': reason
            })
            
            # 更新状态
            self.cash -= total_cost
            self.position_shares += shares
            
            # 更新平均成本（加权平均，含手续费）
            total_cost_for_avg = self.avg_cost * (self.position_shares - shares) + total_cost
            if self.position_shares > 0:
                self.avg_cost = total_cost_for_avg / self.position_shares
            
            logger.debug(f"Buy: {row['trade_date']}, price={price:.2f}, shares={shares}, cash={self.cash:.2f}")
    
    def _sell(self, row: pd.Series, reason: str):
        """
        卖出操作：全仓卖出
        
        Args:
            row: 当前行的数据（包含 trade_date, adj_open, adj_close）
            reason: 卖出原因
        """
        if self.position_shares == 0:
            return
        
        price = row['adj_close']  # 使用收盘价卖出
        shares = self.position_shares
        revenue = shares * price
        
        # 交易手续费（万分之二，最低5元）
        trading_fee = max(revenue * self.trading_fee_rate, 5)
        
        # 印花税（仅卖出收取）
        stamp_duty = revenue * self.stamp_duty_rate
        
        # 实际收入
        actual_revenue = revenue - trading_fee - stamp_duty
        
        # 利润计算
        profit = actual_revenue - self.avg_cost * shares
        return_pct = profit / (self.avg_cost * shares)
        
        self.orders.append({
            'date': row['trade_date'],
            'action': 'sell',
            'price': price,
            'shares': shares,
            'amount': revenue,
            'trading_fee': trading_fee,
            'stamp_duty': stamp_duty,
            'actual_revenue': actual_revenue,
            'profit': profit,
            'return_pct': return_pct,
            'reason': reason
        })
        
        # 更新状态
        self.cash += actual_revenue
        self.position_shares = 0
        self.avg_cost = 0.0
        
        logger.debug(f"Sell: {row['trade_date']}, price={price:.2f}, shares={shares}, profit={profit:.2f}, reason={reason}")
    
    def get_portfolio_value(self, current_price: float) -> float:
        """
        计算当前 portfolio 价值
        
        Args:
            current_price: 当前价格
            
        Returns:
            总市值（现金 + 持仓市值）
        """
        position_value = self.position_shares * current_price
        return self.cash + position_value
