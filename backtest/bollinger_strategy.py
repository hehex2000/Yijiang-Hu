"""
主动量化策略 - 布林带均值回归策略
价格触及下轨 → 买入（超卖）
价格触及上轨 → 卖出（超买）
止盈25%，止损10%
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from loguru import logger


class BollingerStrategy:
    """主动量化：布林带均值回归策略"""
    
    def __init__(self, 
                 total_capital: float = 100000.0,
                 bb_period: int = 20,
                 bb_std: float = 2.0,
                 take_profit: float = 0.25,
                 stop_loss: float = 0.10,
                 position_mode: str = 'half',
                 trading_fee_rate: float = 0.0002,
                 stamp_duty_rate: float = 0.001):
        """
        初始化布林带策略
        
        Args:
            total_capital: 总资金（默认10万）
            bb_period: 布林带周期（默认20）
            bb_std: 标准差倍数（默认2.0）
            take_profit: 止盈线（默认25%）
            stop_loss: 止损线（默认-10%）
            position_mode: 仓位模式，'full'=全仓，'half'=半仓（默认半仓）
            trading_fee_rate: 交易手续费率（默认万分之2 = 0.0002）
            stamp_duty_rate: 印花税率（默认千分之1 = 0.001，仅卖出收取）
        """
        self.total_capital = total_capital
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.position_mode = position_mode
        self.trading_fee_rate = trading_fee_rate
        self.stamp_duty_rate = stamp_duty_rate
        
        self.cash = total_capital      # 可用现金
        self.position_shares = 0      # 持仓股数
        self.avg_cost = 0.0           # 平均持仓成本
        self.orders = []               # 交易记录
        self.daily_values = []        # 每日市值（用于计算绩效）
        
        logger.info(f"BollingerStrategy initialized: period={bb_period}, std={bb_std}, "
                   f"take_profit={take_profit}, stop_loss={stop_loss}")
    
    def _calculate_bollinger(self, close: pd.Series) -> pd.DataFrame:
        """
        计算布林带指标
        
        Args:
            close: 收盘价序列
            
        Returns:
            包含 middle, upper, lower 的 DataFrame
        """
        # 中轨 = SMA
        middle = close.rolling(window=self.bb_period).mean()
        
        # 标准差
        std = close.rolling(window=self.bb_period).std()
        
        # 上轨和下轨
        upper = middle + (std * self.bb_std)
        lower = middle - (std * self.bb_std)
        
        return pd.DataFrame({
            'middle': middle,
            'upper': upper,
            'lower': lower
        })
    
    def run(self, df: pd.DataFrame) -> Dict:
        """
        运行布林带策略，返回交易记录和每日市值
        
        Args:
            df: 包含 trade_date, adj_open, adj_close 的 DataFrame
            
        Returns:
            {
                'trades': 交易记录列表,
                'daily_values': 每日市值列表
            }
        """
        logger.info(f"Running Bollinger strategy on {len(df)} days of data...")
        
        self.orders = []
        self.daily_values = []
        self.cash = self.total_capital
        self.position_shares = 0
        self.avg_cost = 0.0
        
        if len(df) == 0:
            return {'trades': [], 'daily_values': []}
        
        # 确保按日期排序
        df = df.sort_values('trade_date').reset_index(drop=True)
        
        # 计算布林带指标
        bb = self._calculate_bollinger(df['adj_close'])
        df = pd.concat([df, bb], axis=1)
        
        # 初始化持仓状态
        position = 0      # 0=空仓，1=持仓
        entry_price = 0.0 # 买入价格
        
        # 遍历每个交易日
        for i in range(len(df)):
            row = df.iloc[i]
            date = row['trade_date']
            open_price = row['adj_open']
            close_price = row['adj_close']
            
            # 跳过布林带为 NaN 的早期数据
            if pd.isna(row['upper']) or pd.isna(row['lower']):
                # 计算当日市值
                if position == 0:
                    portfolio_value = self.cash
                else:
                    portfolio_value = self.cash + self.position_shares * close_price
                
                self.daily_values.append({
                    'date': date,
                    'portfolio_value': portfolio_value
                })
                continue
            
            upper = row['upper']
            lower = row['lower']
            
            # ── 买入逻辑：价格触及下轨（超卖）───────────────────────────────
            if position == 0 and close_price <= lower:
                # 计算买入股数
                if self.position_mode == 'full':
                    buy_amount = self.cash * 0.95  # 留5%现金
                else:  # half
                    buy_amount = self.cash * 0.50  # 半仓
                
                shares = int(buy_amount / open_price / 100) * 100  # 向下取整到100股
                
                if shares > 0:
                    cost = shares * open_price
                    fee = cost * self.trading_fee_rate
                    total_cost = cost + fee
                    
                    if total_cost <= self.cash:
                        # 买入
                        self.cash -= total_cost
                        self.position_shares += shares
                        
                        # 更新平均成本
                        total_shares = self.position_shares
                        self.avg_cost = (self.avg_cost * (total_shares - shares) + cost) / total_shares
                        
                        entry_price = open_price
                        position = 1
                        
                        self.orders.append({
                            'date': date,
                            'action': 'BUY',
                            'price': open_price,
                            'shares': shares,
                            'amount': cost,
                            'fee': fee,
                            'cash_after': self.cash,
                            'bb_lower': lower,
                            'bb_upper': upper
                        })
                        
                        logger.debug(f"BUY: {date}, price={open_price:.2f}, shares={shares}, BB_Lower={lower:.2f}")
            
            # ── 卖出逻辑：价格触及上轨（超买）或 止盈止损 ─────────────────
            elif position == 1:
                sell_signal = False
                sell_reason = ""
                
                # 1. 价格触及上轨（超买）
                if close_price >= upper:
                    sell_signal = True
                    sell_reason = f"布林带上轨({upper:.2f})"
                
                # 2. 止盈卖出
                profit_pct = (close_price - self.avg_cost) / self.avg_cost
                if profit_pct >= self.take_profit:
                    sell_signal = True
                    sell_reason = f"止盈({profit_pct:.1%})"
                
                # 3. 止损卖出
                if profit_pct <= -self.stop_loss:
                    sell_signal = True
                    sell_reason = f"止损({profit_pct:.1%})"
                
                # 执行卖出
                if sell_signal and self.position_shares > 0:
                    sell_shares = self.position_shares
                    revenue = sell_shares * open_price
                    fee = revenue * self.trading_fee_rate
                    tax = revenue * self.stamp_duty_rate
                    net_revenue = revenue - fee - tax
                    
                    self.cash += net_revenue
                    self.position_shares = 0
                    self.avg_cost = 0.0
                    position = 0
                    
                    self.orders.append({
                        'date': date,
                        'action': 'SELL',
                        'price': open_price,
                        'shares': sell_shares,
                        'amount': revenue,
                        'fee': fee,
                        'tax': tax,
                        'cash_after': self.cash,
                        'reason': sell_reason,
                        'bb_lower': lower,
                        'bb_upper': upper
                    })
                    
                    logger.debug(f"SELL: {date}, price={open_price:.2f}, shares={sell_shares}, reason={sell_reason}")
            
            # ── 计算当日市值 ──────────────────────────────
            if position == 0:
                portfolio_value = self.cash
            else:
                portfolio_value = self.cash + self.position_shares * close_price
            
            self.daily_values.append({
                'date': date,
                'portfolio_value': portfolio_value
            })
        
        # ── 回测结束，如果还有持仓，按最后一日收盘价卖出 ──
        if position == 1 and self.position_shares > 0:
            last_row = df.iloc[-1]
            last_date = last_row['trade_date']
            last_price = last_row['adj_close']
            
            sell_shares = self.position_shares
            revenue = sell_shares * last_price
            fee = revenue * self.trading_fee_rate
            tax = revenue * self.stamp_duty_rate
            net_revenue = revenue - fee - tax
            
            self.cash += net_revenue
            self.position_shares = 0
            
            self.orders.append({
                'date': last_date,
                'action': 'SELL',
                'price': last_price,
                'shares': sell_shares,
                'amount': revenue,
                'fee': fee,
                'tax': tax,
                'cash_after': self.cash,
                'reason': '回测结束平仓',
                'bb_lower': last_row['lower'],
                'bb_upper': last_row['upper']
            })
            
            # 更新最后一天的市值
            self.daily_values[-1]['portfolio_value'] = self.cash
            
            logger.info(f"强制平仓: {last_date}, price={last_price:.2f}")
        
        logger.info(f"✓ Bollinger strategy completed: {len(self.orders)} orders")
        
        return {
            'trades': self.orders,
            'daily_values': self.daily_values
        }


if __name__ == '__main__':
    # 测试代码
    print("布林带策略模块")
    print("=" * 60)
