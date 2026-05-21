"""
主动量化策略 - 海龟策略（Turtle Strategy）
通道上限为当前日之前n日收盘成交价最大值
通道下限为当前日之前n日收盘成交价最小值
当股价上穿上限时买入（最多加仓max_positions次），下穿下限时卖出（全仓卖出）
止盈take_profit%，止损stop_loss%
交易手续费：万分之2（买入和卖出均收取）
印花税：千分之1（仅卖出收取）
"""

import pandas as pd
from typing import List, Dict
from loguru import logger


class TurtleStrategy:
    """主动量化：海龟策略"""
    
    def __init__(self, 
                 total_capital: float = 100000.0, 
                 channel_period: int = 30,
                 position_mode: str = 'half',
                 max_positions: int = 3,
                 take_profit: float = 0.20,
                 stop_loss: float = 0.10,
                 trading_fee_rate: float = 0.0002,
                 stamp_duty_rate: float = 0.001):
        """
        初始化海龟策略
        
        Args:
            total_capital: 总资金（默认10万）
            channel_period: 通道周期（默认30日）
            position_mode: 仓位模式，'full'=全仓，'half'=半仓（默认半仓）
            max_positions: 最大加仓次数（默认3次）
            take_profit: 止盈线（默认20%）
            stop_loss: 止损线（默认-10%）
            trading_fee_rate: 交易手续费率（默认万分之2 = 0.0002）
            stamp_duty_rate: 印花税率（默认千分之1 = 0.001，仅卖出收取）
        """
        self.total_capital = total_capital
        self.channel_period = channel_period
        self.position_mode = position_mode
        self.max_positions = max_positions
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.trading_fee_rate = trading_fee_rate
        self.stamp_duty_rate = stamp_duty_rate
        
        self.cash = total_capital          # 可用现金
        self.position_shares = 0          # 持仓股数
        self.avg_cost_price = 0.0         # 平均持仓成本价（不含手续费）
        self.avg_total_cost = 0.0         # 平均持仓总成本（含手续费）
        self.position_count = 0            # 当前加仓次数
        self.orders = []                   # 交易记录
        self.daily_values = []            # 每日市值（用于计算绩效）
        
        logger.info(f"TurtleStrategy initialized: capital={total_capital}, channel_period={channel_period}, position_mode={position_mode}, max_positions={max_positions}, take_profit={take_profit}, stop_loss={stop_loss}")
    
    def run(self, df: pd.DataFrame) -> List[Dict]:
        """
        运行海龟策略，返回交易记录
        
        Args:
            df: 包含 trade_date, adj_open, adj_close, upper_channel, lower_channel, turtle_signal_change 的 DataFrame
            
        Returns:
            交易记录列表：[{date, action, price, shares, amount, profit, return_pct, reason}]
        """
        logger.info(f"Running Turtle strategy on {len(df)} days of data...")
        
        self.orders = []
        self.daily_values = []
        self.cash = self.total_capital
        self.position_shares = 0
        self.avg_cost_price = 0.0
        self.avg_total_cost = 0.0
        self.position_count = 0
        
        # 记录前一天的信号（用于T+1执行）
        prev_signal_change = 0
        
        for idx, row in df.iterrows():
            # 记录每日市值（用当天adj_close计算）
            self._record_daily_value(row)
            
            # T+1执行：如果前一天有突破信号，今天开盘买入
            if prev_signal_change == 1 and self.position_count < self.max_positions and self.cash > 0:
                self._buy(row, use_open_price=True)
            
            # T+1执行：如果前一天有跌破信号，今天开盘卖出
            elif prev_signal_change == -1 and self.position_shares > 0:
                self._sell(row, reason="breakdown", use_open_price=True)
            
            # 检查止盈止损（每天检查，用当天收盘价）
            if self.position_shares > 0:
                current_return = (row['adj_close'] - self.avg_cost_price) / self.avg_cost_price
                
                # 止盈
                if current_return >= self.take_profit:
                    self._sell(row, reason="stop_profit", use_open_price=False)
                
                # 止损
                elif current_return <= -self.stop_loss:
                    self._sell(row, reason="stop_loss", use_open_price=False)
            
            # 更新前一天的信号（用于明天执行）
            prev_signal_change = row['turtle_signal_change']
        
        # 如果回测结束仍有持仓，按最后一天开盘价卖出
        if self.position_shares > 0:
            last_row = df.iloc[-1]
            self._sell(last_row, reason="end_of_backtest", use_open_price=True)
        
        logger.info(f"✓ Turtle strategy completed: {len(self.orders)} orders")
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
        买入操作：根据 position_mode 决定买入金额
        
        Args:
            row: 当前行的数据（包含 trade_date, adj_open, adj_close）
            use_open_price: 是否使用开盘价（True=开盘价, False=收盘价）
        """
        # 选择价格：开盘价 or 收盘价
        if use_open_price:
            price = row['adj_open']
        else:
            price = row['adj_close']
        
        # 根据仓位模式决定买入金额
        if self.position_mode == 'full':
            buy_amount = self.cash  # 全仓买入
        else:  # 'half'
            buy_amount = self.cash * 0.5  # 半仓买入
        
        shares = int(buy_amount / price / 100) * 100  # 整百股
        
        if shares > 0:
            actual_cost = shares * price
            
            # 交易手续费（万分之二，最低5元）
            trading_fee = max(actual_cost * self.trading_fee_rate, 5)
            total_cost = actual_cost + trading_fee
            
            # 检查现金是否足够（包含手续费）
            if total_cost > self.cash:
                # 调整买入股数
                shares = int((self.cash / (1 + self.trading_fee_rate)) / price / 100) * 100
                if shares <= 0:
                    return
                actual_cost = shares * price
                trading_fee = actual_cost * self.trading_fee_rate
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
                'reason': 'breakout'
            })
            
            # 更新状态
            self.cash -= total_cost
            self.position_shares += shares
            self.position_count += 1
            
            # 更新平均成本（加权平均，包含手续费）
            old_total = self.avg_total_cost * (self.position_shares - shares)
            new_total = total_cost
            self.avg_total_cost = (old_total + new_total) / self.position_shares
            
            old_price = self.avg_cost_price * (self.position_shares - shares)
            new_price = price * shares
            self.avg_cost_price = (old_price + new_price) / self.position_shares
            
            logger.debug(f"Turtle buy: {row['trade_date']}, price={price:.2f}, shares={shares}, total_cost={total_cost:.2f}, position_count={self.position_count}")
    
    def _sell(self, row: pd.Series, reason: str, use_open_price: bool = False):
        """
        卖出操作
        
        Args:
            row: 当前行的数据（包含 trade_date, adj_open, adj_close）
            reason: 卖出原因（breakdown/stop_profit/stop_loss/end_of_backtest）
            use_open_price: 是否使用开盘价（True=开盘价, False=收盘价）
        """
        # 选择价格：开盘价 or 收盘价
        if use_open_price:
            price = row['adj_open']
        else:
            price = row['adj_close']
        
        shares = self.position_shares
        
        if shares > 0:
            revenue = shares * price
            
            # 交易手续费（万分之二，最低5元）
            trading_fee = max(revenue * self.trading_fee_rate, 5)
            
            # 印花税（仅卖出收取）
            stamp_duty = revenue * self.stamp_duty_rate
            
            # 实际收入
            actual_revenue = revenue - trading_fee - stamp_duty
            
            # 利润计算（基于包含手续费的总成本）
            profit = actual_revenue - self.avg_total_cost * shares
            return_pct = profit / (self.avg_total_cost * shares)
            
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
            self.avg_cost_price = 0.0
            self.avg_total_cost = 0.0
            self.position_count = 0  # 重置加仓次数
            
            logger.debug(f"Turtle sell: {row['trade_date']}, price={price:.2f}, shares={shares}, profit={profit:.2f}, reason={reason}")
    
    def get_daily_values(self) -> List[Dict]:
        """返回每日市值数据（用于计算绩效指标）"""
        return self.daily_values
    
    def get_orders(self) -> List[Dict]:
        """返回交易记录"""
        return self.orders
