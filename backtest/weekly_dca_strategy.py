"""
被动量化策略 - 周度定投策略（Weekly DCA）
每周第一个交易日买入100股
单笔止盈30%，止损-10%
卖出时收取交易手续费（万分之2）和印花税（千分之1）
"""

import pandas as pd
from typing import List, Dict
from loguru import logger


class WeeklyDCAStrategy:
    """被动量化：周度定投策略"""
    
    def __init__(self, total_capital: float = 100000.0, 
                 shares_per_week: int = 100,
                 take_profit: float = 0.30,
                 stop_loss: float = 0.10,
                 trading_fee_rate: float = 0.0002,
                 stamp_duty_rate: float = 0.001):
        """
        初始化周度定投策略
        
        Args:
            total_capital: 总资金（默认10万）
            shares_per_week: 每周买入股数（默认100股）
            take_profit: 止盈线（默认30%）
            stop_loss: 止损线（默认-10%）
            trading_fee_rate: 交易手续费率（默认万分之2 = 0.0002）
            stamp_duty_rate: 印花税率（默认千分之1 = 0.001，仅卖出收取）
        """
        self.total_capital = total_capital
        self.shares_per_week = shares_per_week
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.trading_fee_rate = trading_fee_rate
        self.stamp_duty_rate = stamp_duty_rate
        self.cash = total_capital
        self.lots = []
        self.orders = []
        self.daily_values = []
        
        logger.info(f"WeeklyDCAStrategy initialized with total_capital: {total_capital}, shares_per_week: {shares_per_week}")
    
    def run(self, df: pd.DataFrame) -> List[Dict]:
        """
        运行周度定投策略，返回交易记录
        """
        logger.info(f"Running Weekly DCA strategy on {len(df)} days of data...")
        
        self.orders = []
        self.cash = self.total_capital
        self.lots = []
        self.daily_values = []
        
        # 获取每周第一个交易日
        weekly_days = self._get_weekly_trading_days(df)
        weekly_set = set(weekly_days)
        
        for idx, row in df.iterrows():
            current_date = row['trade_date']
            price = row['adj_close']
            
            self._record_daily_value(price, current_date)
            
            if current_date in weekly_set:
                self._buy_weekly(row)
            
            self._check_stop_profit(row)
            self._check_stop_loss(row)
        
        # 回测结束，按最后一天价格卖出所有剩余持仓
        if len(df) > 0:
            last_row = df.iloc[-1]
            last_date = last_row['trade_date']
            last_price = last_row['adj_close']
            
            for lot in self.lots:
                if not lot['sold']:
                    revenue = lot['shares'] * last_price
                    
                    # 交易手续费（万分之二，最低5元）
                    trading_fee = max(revenue * self.trading_fee_rate, 5)
                    
                    # 印花税（仅卖出收取）
                    stamp_duty = revenue * self.stamp_duty_rate
                    
                    # 实际收入
                    actual_revenue = revenue - trading_fee - stamp_duty
                    
                    profit = actual_revenue - lot['cost_total']
                    profit_pct = profit / lot['cost_total'] if lot['cost_total'] > 0 else 0
                    
                    self.orders.append({
                        'date': last_date,
                        'action': 'sell',
                        'price': last_price,
                        'shares': lot['shares'],
                        'amount': revenue,
                        'trading_fee': trading_fee,
                        'stamp_duty': stamp_duty,
                        'actual_revenue': actual_revenue,
                        'profit': profit,
                        'return_pct': profit_pct,
                        'reason': 'end_of_backtest'
                    })
                    
                    self.cash += actual_revenue
                    lot['sold'] = True
                    
                    logger.debug(f"Weekly DCA end sell: {last_date}, price={last_price:.2f}, profit={profit:.2f}")
        
        logger.info(f"✓ Weekly DCA strategy completed: {len(self.orders)} orders")
        return self.orders
    
    def _buy_weekly(self, row: pd.Series):
        """
        执行周度定投买入
        逻辑：每周买入固定股数（默认100股）
        """
        date = row['trade_date']
        price = row['adj_close']
        shares = self.shares_per_week
        actual_cost = shares * price
        trading_fee = max(actual_cost * self.trading_fee_rate, 5)
        total_cost = actual_cost + trading_fee
        
        if self.cash >= total_cost:
            # 执行买入
            self.orders.append({
                'date': date,
                'action': 'buy',
                'price': price,
                'shares': shares,
                'amount': actual_cost,
                'trading_fee': trading_fee,
                'total_cost': total_cost,
                'profit': None,
                'return_pct': None,
                'reason': 'weekly_dca'
            })
            
            self.lots.append({
                'date': date,
                'shares': shares,
                'cost': price,
                'cost_total': total_cost,
                'sold': False
            })
            
            self.cash -= total_cost
            
            logger.debug(f"Weekly DCA buy: {date}, price={price:.2f}, shares={shares}, total_cost={total_cost:.2f}, cash={self.cash:.2f}")
        else:
            logger.debug(f"Weekly DCA skip: {date}, price={price:.2f}, cash={self.cash:.2f} (insufficient funds)")
    
    def _check_stop_profit(self, row: pd.Series):
        """检查每笔定投的止盈（使用配置参数）"""
        current_date = row['trade_date']
        current_price = row['adj_close']
        
        for lot in self.lots:
            if lot['sold']:
                continue
            
            # 使用含手续费的实际成本价
            actual_cost_per_share = lot['cost_total'] / lot['shares']
            profit_pct = (current_price - actual_cost_per_share) / actual_cost_per_share
            
            if profit_pct >= self.take_profit:
                revenue = lot['shares'] * current_price
                
                # 交易手续费（万分之二，最低5元）
                trading_fee = max(revenue * self.trading_fee_rate, 5)
                
                # 印花税（仅卖出收取）
                stamp_duty = revenue * self.stamp_duty_rate
                
                # 实际收入
                actual_revenue = revenue - trading_fee - stamp_duty
                
                profit = actual_revenue - lot['cost_total']
                
                self.orders.append({
                    'date': current_date,
                    'action': 'sell',
                    'price': current_price,
                    'shares': lot['shares'],
                    'amount': revenue,
                    'trading_fee': trading_fee,
                    'stamp_duty': stamp_duty,
                    'actual_revenue': actual_revenue,
                    'profit': profit,
                    'return_pct': profit / lot['cost_total'] if lot['cost_total'] > 0 else 0,
                    'reason': 'stop_profit'
                })
                
                self.cash += actual_revenue
                lot['sold'] = True
                lot['sell_date'] = current_date
                lot['profit'] = profit
                
                logger.debug(f"Weekly DCA stop profit: {current_date}, price={current_price:.2f}, profit={profit:.2f}")
    
    def _check_stop_loss(self, row: pd.Series):
        """检查每笔定投的止损（使用配置参数）"""
        current_date = row['trade_date']
        current_price = row['adj_close']
        
        for lot in self.lots:
            if lot['sold']:
                continue
            
            # 使用含手续费的实际成本价
            actual_cost_per_share = lot['cost_total'] / lot['shares']
            loss_pct = (current_price - actual_cost_per_share) / actual_cost_per_share
            
            if loss_pct <= -self.stop_loss:
                revenue = lot['shares'] * current_price
                
                # 交易手续费（万分之二，最低5元）
                trading_fee = max(revenue * self.trading_fee_rate, 5)
                
                # 印花税（仅卖出收取）
                stamp_duty = revenue * self.stamp_duty_rate
                
                # 实际收入
                actual_revenue = revenue - trading_fee - stamp_duty
                
                loss = actual_revenue - lot['cost_total']
                
                self.orders.append({
                    'date': current_date,
                    'action': 'sell',
                    'price': current_price,
                    'shares': lot['shares'],
                    'amount': revenue,
                    'trading_fee': trading_fee,
                    'stamp_duty': stamp_duty,
                    'actual_revenue': actual_revenue,
                    'profit': loss,  # 负值，表示亏损
                    'return_pct': loss / lot['cost_total'] if lot['cost_total'] > 0 else 0,
                    'reason': 'stop_loss'
                })
                
                self.cash += actual_revenue
                lot['sold'] = True
                lot['sell_date'] = current_date
                lot['profit'] = loss
                
                logger.debug(f"Weekly DCA stop loss: {current_date}, price={current_price:.2f}, loss={loss:.2f}")
    
    def _record_daily_value(self, current_price: float, current_date: str):
        """记录每日市值"""
        unsold_value = sum(
            lot['shares'] * current_price
            for lot in self.lots if not lot['sold']
        )
        
        total_value = self.cash + unsold_value
        
        self.daily_values.append({
            'date': current_date,
            'portfolio_value': total_value
        })
    
    def _get_weekly_trading_days(self, df: pd.DataFrame) -> List[str]:
        """获取每周第一个交易日"""
        df = df.copy()
        # 直接使用星期一作为每周第一个交易日
        df['weekday'] = pd.to_datetime(df['trade_date']).dt.weekday
        # 筛选星期一（0=周一）的交易日期
        mondays = df[df['weekday'] == 0]['trade_date'].tolist()
        
        logger.debug(f"Found {len(mondays)} weekly trading days (Mondays)")
        return mondays
    
    def get_daily_values(self) -> List[Dict]:
        """返回每日市值数据（用于计算绩效指标）"""
        return self.daily_values
    
    def get_orders(self) -> List[Dict]:
        """返回交易记录"""
        return self.orders
