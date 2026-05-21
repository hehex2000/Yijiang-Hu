"""
主动量化策略 - 双均线策略（MA20 / MA60）
金叉买入50%仓，死叉全仓卖出，止盈30%，止损10%
支持通过参数配置止盈止损
卖出时收取交易手续费（万分之2）和印花税（千分之1）
"""

import pandas as pd
from typing import List, Dict
from loguru import logger


class DualMAStrategy:
    """主动量化：双均线策略"""
    
    def __init__(self, 
                 total_capital: float = 100000.0,
                 take_profit: float = 0.30,
                 stop_loss: float = 0.10,
                 trading_fee_rate: float = 0.0002,
                 stamp_duty_rate: float = 0.001):
        """
        初始化双均线策略
        
        Args:
            total_capital: 总资金（默认10万）
            take_profit: 止盈线（默认30%）
            stop_loss: 止损线（默认-10%）
            trading_fee_rate: 交易手续费率（默认万分之2 = 0.0002）
            stamp_duty_rate: 印花税率（默认千分之1 = 0.001，仅卖出收取）
        """
        self.total_capital = total_capital
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.trading_fee_rate = trading_fee_rate
        self.stamp_duty_rate = stamp_duty_rate
        self.cash = total_capital          # 可用现金
        self.position_shares = 0          # 持仓股数
        self.avg_cost = 0.0               # 平均持仓成本
        self.orders = []                   # 交易记录
        self.daily_values = []            # 每日市值（用于计算绩效）
        
        logger.info(f"DualMAStrategy initialized: capital={total_capital}, take_profit={take_profit}, stop_loss={stop_loss}")
    
    def run(self, df: pd.DataFrame) -> List[Dict]:
        """
        运行双均线策略，返回交易记录
        
        Args:
            df: 包含 trade_date, adj_open, adj_close, ma10, ma60, signal_change 的 DataFrame
            
        Returns:
            交易记录列表：[{date, action, price, shares, amount, profit, return_pct, reason}]
        """
        logger.info(f"Running Dual MA strategy on {len(df)} days of data...")
        
        self.orders = []
        self.daily_values = []
        self.cash = self.total_capital
        self.position_shares = 0
        self.avg_cost = 0.0
        
        # 记录前一天的信号（用于T+1执行）
        prev_signal_change = 0
        
        for idx, row in df.iterrows():
            # 记录每日市值（用当天adj_close计算）
            self._record_daily_value(row)
            
            # T+1执行：如果前一天有信号，今天开盘交易
            if prev_signal_change == 2 and self.cash > 0:
                # 买入：用今天开盘价
                self._buy(row, use_open_price=True)
            elif prev_signal_change == -2 and self.position_shares > 0:
                # 卖出：用今天开盘价
                self._sell(row, reason="death_cross", use_open_price=True)
            
            # 检查止盈止损（每天检查，用当天收盘价）
            if self.position_shares > 0:
                current_return = (row['adj_close'] - self.avg_cost) / self.avg_cost
                
                # 止盈（使用配置参数）
                if current_return >= self.take_profit:
                    self._sell(row, reason="stop_profit", use_open_price=False)
                
                # 止损（使用配置参数，注意：stop_loss是正数，需要加负号）
                elif current_return <= -self.stop_loss:
                    self._sell(row, reason="stop_loss", use_open_price=False)
            
            # 更新前一天的信号（用于明天执行）
            prev_signal_change = row['signal_change']
        
        # 如果回测结束仍有持仓，按最后一天开盘价卖出
        if self.position_shares > 0:
            last_row = df.iloc[-1]
            self._sell(last_row, reason="end_of_backtest", use_open_price=True)
        
        logger.info(f"✓ Dual MA strategy completed: {len(self.orders)} orders")
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
    
    def _buy(self, row: pd.Series, use_open_price: bool = False):
        """
        买入操作：买入 50% 可用资金
        
        Args:
            row: 当前行的数据（包含 trade_date, adj_open, adj_close）
            use_open_price: 是否使用开盘价（True=开盘价, False=收盘价）
        """
        # 选择价格：开盘价 or 收盘价
        if use_open_price:
            price = row['adj_open']
        else:
            price = row['adj_close']
        
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
                'reason': 'golden_cross'
            })
            
            # 更新状态
            self.cash -= total_cost
            self.position_shares += shares
            
            # 更新平均成本（加权平均，含手续费）
            total_cost_for_avg = self.avg_cost * (self.position_shares - shares) + total_cost
            if self.position_shares > 0:
                self.avg_cost = total_cost_for_avg / self.position_shares
            
            logger.debug(f"Buy: {row['trade_date']}, price={price:.2f}, shares={shares}, cash={self.cash:.2f}")
    
    def _sell(self, row: pd.Series, reason: str, use_open_price: bool = False):
        """
        卖出操作：全仓卖出
        
        Args:
            row: 当前行的数据（包含 trade_date, adj_open, adj_close）
            reason: 卖出原因（death_cross, stop_profit, stop_loss, end_of_backtest）
            use_open_price: 是否使用开盘价（True=开盘价, False=收盘价）
        """
        if self.position_shares == 0:
            return
        
        # 选择价格：开盘价 or 收盘价
        if use_open_price:
            price = row['adj_open']
        else:
            price = row['adj_close']
        
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
