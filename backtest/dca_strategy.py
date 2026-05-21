"""
被动量化策略 - 月度定投策略（DCA）
每月第一个交易日买入：严格投入8000元（向下取整到100股），不够买100股就跳过
止盈30%，止损-20%
止损触发卖出50%仓位，止盈触发卖出100%仓位
卖出时收取交易手续费（万分之2）和印花税（千分之1）
"""

import pandas as pd
from typing import List, Dict
from loguru import logger


class DCAStrategy:
    """被动量化：月度定投策略"""
    
    def __init__(self, 
                 total_capital: float = 100000.0, 
                 amount_per_month: float = 5000,
                 take_profit: float = 0.30,
                 stop_loss: float = 0.20,
                 enable_tp_sl: bool = True,
                 trading_fee_rate: float = 0.0002,
                 stamp_duty_rate: float = 0.001):
        """
        初始化定投策略
        
        Args:
            total_capital: 总资金（默认10万）
            amount_per_month: 每月定投金额（默认5000元，累积到能买为止）
            take_profit: 止盈线（默认30%，仅 enable_tp_sl=True 时生效）
            stop_loss: 止损线（默认20%，正数，仅 enable_tp_sl=True 时生效）
            enable_tp_sl: 是否启用止盈止损（True=启用，False=不启用）
            trading_fee_rate: 交易手续费率（默认万分之2 = 0.0002）
            stamp_duty_rate: 印花税率（默认千分之1 = 0.001，仅卖出收取）
        """
        self.total_capital = total_capital
        self.amount_per_month = amount_per_month
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.enable_tp_sl = enable_tp_sl
        self.trading_fee_rate = trading_fee_rate
        self.stamp_duty_rate = stamp_duty_rate
        self.cash = 0  # 初始现金为0，每月定投时增加
        self.total_invested = 0  # 总投入金额（用于判断是否达到预算上限）
        self.lots = []
        self.orders = []
        self.daily_values = []
        
        logger.info(f"DCAStrategy initialized: total_capital={total_capital}, monthly={amount_per_month}, take_profit={take_profit}, stop_loss={stop_loss}")
    
    def run(self, df: pd.DataFrame) -> List[Dict]:
        """
        运行月度定投策略，返回交易记录
        """
        logger.info(f"Running DCA strategy on {len(df)} days of data...")
        
        self.orders = []
        self.cash = 0  # 从0开始，每月定投累积
        self.total_invested = 0
        self.lots = []
        self.daily_values = []
        
        monthly_days = self._get_monthly_trading_days(df)
        monthly_set = set(monthly_days)
        
        for idx, row in df.iterrows():
            current_date = row['trade_date']
            price = row['adj_close']
            
            self._record_daily_value(price, current_date)
            
            if current_date in monthly_set:
                self._buy_monthly(row)
            
            # 仅在启用止盈止损时检查
            if self.enable_tp_sl:
                self._check_stop_profit(row)
                self._check_stop_loss(row)
        
        # 回测结束，按最后一天价格卖出所有剩余持仓（一笔全部卖出）
        if len(df) > 0:
            last_row = df.iloc[-1]
            last_date = last_row['trade_date']
            last_price = last_row['adj_close']
            
            # 计算总剩余股数和加权平均成本
            total_shares = 0
            total_cost = 0
            
            for lot in self.lots:
                if not lot['sold']:
                    total_shares += lot['shares']
                    total_cost += lot['cost_total']
            
            if total_shares > 0:
                # 一笔卖出所有剩余股票
                revenue = total_shares * last_price
                
                # 交易手续费（万分之二，最低5元）
                trading_fee = max(revenue * self.trading_fee_rate, 5)
                
                # 印花税（仅卖出收取）
                stamp_duty = revenue * self.stamp_duty_rate
                
                # 实际收入
                actual_revenue = revenue - trading_fee - stamp_duty
                
                # 计算总盈亏
                profit = actual_revenue - total_cost
                profit_pct = profit / total_cost if total_cost > 0 else 0
                
                self.orders.append({
                    'date': last_date,
                    'action': 'sell',
                    'price': last_price,
                    'shares': total_shares,
                    'amount': revenue,
                    'trading_fee': trading_fee,
                    'stamp_duty': stamp_duty,
                    'actual_revenue': actual_revenue,
                    'profit': profit,
                    'return_pct': profit_pct,
                    'reason': 'end_of_backtest'
                })
                
                self.cash += actual_revenue
                
                # 标记所有批次为已卖出
                for lot in self.lots:
                    if not lot['sold']:
                        lot['sold'] = True
                
                logger.debug(f"DCA end sell: {last_date}, price={last_price:.2f}, shares={total_shares}, profit={profit:.2f}")
        
        logger.info(f"✓ DCA strategy completed: {len(self.orders)} orders")
        return {
            'trades': self.orders,
            'daily_values': self.daily_values
        }
    
    def _buy_monthly(self, row: pd.Series):
        """
        执行月度定投买入
        逻辑：每月用 5000 元买尽可能多的 100 股整数倍
              不够买 100 股就累积到下个月
        总投入不超过总预算（total_capital）
        """
        date = row['trade_date']
        price = row['adj_close']
        
        # 每月增加定投金额
        self.cash += self.amount_per_month
        self.total_invested += self.amount_per_month
        
        # 检查是否超过总预算
        if self.total_invested > self.total_capital:
            logger.debug(f"DCA skip: {date}, total invested {self.total_invested:.2f} >= budget")
            return
        
        # 计算用当月 5000 元能买多少股（向下取整到 100 股）
        target_shares = int(self.amount_per_month / price / 100) * 100
        
        # 如果当月 5000 元不够买 100 股，尝试用累积现金买 100 股
        if target_shares < 100:
            if self.cash >= 100 * price:
                target_shares = 100
            else:
                # 累积现金也不够买 100 股，等下个月
                logger.debug(f"DCA skip: {date}, cash={self.cash:.2f}, can't buy 100 shares")
                return
        
        actual_cost = target_shares * price
        trading_fee = max(actual_cost * self.trading_fee_rate, 5)
        total_cost = actual_cost + trading_fee
        
        # 如果累积现金不够支付，等下个月
        if self.cash < total_cost:
            logger.debug(f"DCA skip: {date}, cash={self.cash:.2f} < total_cost={total_cost:.2f}")
            return
        
        # 执行买入
        self.orders.append({
            'date': date,
            'action': 'buy',
            'price': price,
            'shares': target_shares,
            'amount': actual_cost,
            'trading_fee': trading_fee,
            'total_cost': total_cost,
            'profit': None,
            'return_pct': None,
            'reason': 'monthly_dca'
        })
        
        self.lots.append({
            'date': date,
            'shares': target_shares,
            'cost': price,
            'cost_total': total_cost,
            'sold': False
        })
        
        self.cash -= total_cost
        
        logger.debug(f"DCA buy: {date}, price={price:.2f}, shares={target_shares}, total_cost={total_cost:.2f}, cash={self.cash:.2f}")
    
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
                    
                self.cash += actual_revenue  # 止盈卖出的钱回到现金池
                lot['sold'] = True
                lot['sell_date'] = current_date
                lot['profit'] = profit
                    
                logger.debug(f"DCA stop profit: {current_date}, price={current_price:.2f}, profit={profit:.2f}")
    
    def _check_stop_loss(self, row: pd.Series):
        """检查每笔定投的止损（使用配置参数），触发时卖出50%仓位"""
        current_date = row['trade_date']
        current_price = row['adj_close']
        
        for lot in self.lots:
            if lot['sold']:
                continue
                
            # 使用含手续费的实际成本价
            actual_cost_per_share = lot['cost_total'] / lot['shares']
            loss_pct = (current_price - actual_cost_per_share) / actual_cost_per_share
                
            if loss_pct <= -self.stop_loss:
                # 止损触发，只卖出 50% 仓位（向下取整到 100 股）
                sell_shares = int(lot['shares'] * 0.5 / 100) * 100
                
                # 如果 50% 不够 100 股，就卖全部
                if sell_shares < 100:
                    sell_shares = lot['shares']
                
                revenue = sell_shares * current_price
                
                # 交易手续费（万分之二，最低5元）
                trading_fee = max(revenue * self.trading_fee_rate, 5)
                
                # 印花税（仅卖出收取）
                stamp_duty = revenue * self.stamp_duty_rate
                
                # 实际收入
                actual_revenue = revenue - trading_fee - stamp_duty
                
                # 计算盈亏（按卖出比例计算成本）
                sell_cost = lot['cost_total'] * (sell_shares / lot['shares'])
                loss = actual_revenue - sell_cost
                    
                self.orders.append({
                    'date': current_date,
                    'action': 'sell',
                    'price': current_price,
                    'shares': sell_shares,
                    'amount': revenue,
                    'trading_fee': trading_fee,
                    'stamp_duty': stamp_duty,
                    'actual_revenue': actual_revenue,
                    'profit': loss,  # 负值，表示亏损
                    'return_pct': loss / sell_cost if sell_cost > 0 else 0,
                    'reason': 'stop_loss_50pct'
                })
                    
                self.cash += actual_revenue
                
                # 更新批次：减少股数，更新成本
                lot['shares'] -= sell_shares
                lot['cost_total'] -= sell_cost
                
                # 如果卖完了，标记为已卖出
                if lot['shares'] == 0:
                    lot['sold'] = True
                    lot['sell_date'] = current_date
                    lot['profit'] = loss
                    
                logger.debug(f"DCA stop loss 50%: {current_date}, price={current_price:.2f}, sell_shares={sell_shares}, loss={loss:.2f}")
    
    def _record_daily_value(self, current_price: float, current_date: str):
        """记录每日市值和累计投入"""
        unsold_value = sum(
            lot['shares'] * current_price
            for lot in self.lots if not lot['sold']
        )
        
        total_value = self.cash + unsold_value
        
        self.daily_values.append({
            'date': current_date,
            'portfolio_value': total_value,
            'total_invested': self.total_invested
        })
    
    def _get_monthly_trading_days(self, df: pd.DataFrame) -> List[str]:
        """获取每月第一个交易日"""
        df = df.copy()
        df['year_month'] = df['trade_date'].str[:6]
        monthly_first = df.groupby('year_month')['trade_date'].min().tolist()
        
        logger.debug(f"Found {len(monthly_first)} monthly trading days")
        return monthly_first
