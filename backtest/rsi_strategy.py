"""
主动量化策略 - RSI 超买超卖策略
RSI < 30（超卖）→ 买入
RSI > 70（超买）→ 卖出
止盈25%，止损10%
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from loguru import logger


class RSIStrategy:
    """主动量化：RSI 超买超卖策略"""
    
    def __init__(self, 
                 total_capital: float = 100000.0,
                 rsi_period: int = 14,
                 rsi_oversold: float = 30,
                 rsi_overbought: float = 70,
                 take_profit: float = 0.25,
                 stop_loss: float = 0.10,
                 position_mode: str = 'half',
                 trading_fee_rate: float = 0.0002,
                 stamp_duty_rate: float = 0.001):
        """
        初始化 RSI 策略
        
        Args:
            total_capital: 总资金（默认10万）
            rsi_period: RSI 计算周期（默认14）
            rsi_oversold: 超卖线（默认30）
            rsi_overbought: 超买线（默认70）
            take_profit: 止盈线（默认25%）
            stop_loss: 止损线（默认-10%）
            position_mode: 仓位模式，'full'=全仓，'half'=半仓（默认半仓）
            trading_fee_rate: 交易手续费率（默认万分之2 = 0.0002）
            stamp_duty_rate: 印花税率（默认千分之1 = 0.001，仅卖出收取）
        """
        self.total_capital = total_capital
        self.rsip_period = rsi_period
        self.rsip_oversold = rsi_oversold
        self.rsip_overbought = rsi_overbought
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.position_mode = position_mode
        self.trading_fee_rate = trading_fee_rate
        self.stamp_duty_rate = stamp_duty_rate
        
        self.cash = total_capital          # 可用现金
        self.position_shares = 0          # 持仓股数
        self.avg_cost = 0.0               # 平均持仓成本
        self.orders = []                   # 交易记录
        self.daily_values = []            # 每日市值（用于计算绩效）
        
        logger.info(f"RSIStrategy initialized: period={rsi_period}, "
                   f"oversold={rsi_oversold}, overbought={rsi_overbought}, "
                   f"take_profit={take_profit}, stop_loss={stop_loss}")
    
    def _calculate_rsi(self, prices: pd.Series) -> pd.Series:
        """
        计算 RSI 指标
        
        Args:
            prices: 价格序列（通常是收盘价）
            
        Returns:
            RSI 序列（0-100）
        """
        # 计算价格变化
        delta = prices.diff()
        
        # 分离上涨和下跌
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        
        # 计算平均上涨和下跌（使用 EMA）
        avg_gain = gain.ewm(com=self.rsip_period - 1, min_periods=self.rsip_period).mean()
        avg_loss = loss.ewm(com=self.rsip_period - 1, min_periods=self.rsip_period).mean()
        
        # 计算 RS 和 RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def run(self, df: pd.DataFrame) -> Dict:
        """
        运行 RSI 策略，返回交易记录和每日市值
        
        Args:
            df: 包含 trade_date, adj_open, adj_close 的 DataFrame
            
        Returns:
            {
                'trades': 交易记录列表,
                'daily_values': 每日市值列表
            }
        """
        logger.info(f"Running RSI strategy on {len(df)} days of data...")
        
        self.orders = []
        self.daily_values = []
        self.cash = self.total_capital
        self.position_shares = 0
        self.avg_cost = 0.0
        
        if len(df) == 0:
            return {'trades': [], 'daily_values': []}
        
        # 确保按日期排序
        df = df.sort_values('trade_date').reset_index(drop=True)
        
        # 计算 RSI 指标
        df['rsi'] = self._calculate_rsi(df['adj_close'])
        
        # 初始化持仓状态
        position = 0          # 0=空仓，1=持仓
        entry_price = 0.0     # 买入价格
        peak_value = 0.0     # 市值峰值（用于动态止损）
        
        # 遍历每个交易日
        for i in range(len(df)):
            row = df.iloc[i]
            date = row['trade_date']
            open_price = row['adj_open']
            close_price = row['adj_close']
            rsi = row['rsi']
            
            # 跳过 RSI 为 NaN 的早期数据
            if pd.isna(rsi):
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
            
            # ── 买入逻辑：RSI < 超卖线 ──────────────────────────────
            if position == 0 and rsi < self.rsip_oversold:
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
                            'rsi': rsi
                        })
                        
                        logger.debug(f"BUY: {date}, price={open_price:.2f}, shares={shares}, RSI={rsi:.1f}")
            
            # ── 卖出逻辑：RSI > 超买线 或 止盈止损 ───────
            elif position == 1:
                sell_signal = False
                sell_reason = ""
                
                # 1. RSI 超买卖出
                if rsi > self.rsip_overbought:
                    sell_signal = True
                    sell_reason = f"RSI超买({rsi:.1f})"
                
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
                        'rsi': rsi
                    })
                    
                    logger.debug(f"SELL: {date}, price={open_price:.2f}, shares={sell_shares}, reason={sell_reason}, RSI={rsi:.1f}")
            
            # ── 计算当日市值 ──────────────────────────────
            if position == 0:
                portfolio_value = self.cash
            else:
                portfolio_value = self.cash + self.position_shares * close_price
            
            # 更新市值峰值
            if portfolio_value > peak_value:
                peak_value = portfolio_value
            
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
                'rsi': df.iloc[-1]['rsi']
            })
            
            # 更新最后一天的市值
            self.daily_values[-1]['portfolio_value'] = self.cash
            
            logger.info(f"强制平仓: {last_date}, price={last_price:.2f}")
        
        logger.info(f"✓ RSI strategy completed: {len(self.orders)} orders")
        
        return {
            'trades': self.orders,
            'daily_values': self.daily_values
        }


if __name__ == '__main__':
    # 测试代码
    print("RSI 策略模块")
    print("=" * 60)
