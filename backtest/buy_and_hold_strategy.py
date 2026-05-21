"""
被动策略 - 买入持有（Buy & Hold）
回测首日一次性买入，末日全部卖出，无任何止盈止损
仅计算涨跌幅度，卖出时扣除手续费和印花税
"""

import pandas as pd
from typing import List, Dict
from loguru import logger


class BuyAndHoldStrategy:
    """被动买入持有策略"""

    def __init__(self,
                 total_capital: float = 200000.0,
                 trading_fee_rate: float = 0.0002,
                 stamp_duty_rate: float = 0.001):
        """
        Args:
            total_capital: 总资金（默认20万）
            trading_fee_rate: 交易手续费率（默认万分之2）
            stamp_duty_rate: 印花税率（默认千分之1，仅卖出收取）
        """
        self.total_capital = total_capital
        self.trading_fee_rate = trading_fee_rate
        self.stamp_duty_rate = stamp_duty_rate
        self.orders = []
        self.daily_values = []
        logger.info(f"BuyAndHoldStrategy initialized: total_capital={total_capital}")

    def run(self, df: pd.DataFrame) -> List[Dict]:
        """
        运行买入持有策略
        首日全仓买入（向下取整到100股），末日全部卖出
        """
        logger.info(f"Running Buy & Hold on {len(df)} days of data...")

        self.orders = []
        self.daily_values = []

        if len(df) == 0:
            return self.orders

        first_row = df.iloc[0]
        last_row = df.iloc[-1]

        first_date = first_row['trade_date']
        first_price = first_row['adj_close']
        last_date = last_row['trade_date']
        last_price = last_row['adj_close']

        # 首日买入：用全部资金买尽可能多的100股整数倍
        max_shares = int(self.total_capital / first_price / 100) * 100
        if max_shares < 100:
            logger.warning("Not enough capital to buy 100 shares, skipping.")
            return self.orders

        buy_amount = max_shares * first_price
        trading_fee_buy = max(buy_amount * self.trading_fee_rate, 5)
        total_cost = buy_amount + trading_fee_buy

        self.orders.append({
            'date': first_date,
            'action': 'buy',
            'price': first_price,
            'shares': max_shares,
            'amount': buy_amount,
            'trading_fee': trading_fee_buy,
            'total_cost': total_cost,
            'profit': None,
            'return_pct': None,
            'reason': 'buy_and_hold'
        })

        # 记录每日市值
        for idx, row in df.iterrows():
            current_price = row['adj_close']
            portfolio_value = max_shares * current_price
            self.daily_values.append({
                'date': row['trade_date'],
                'portfolio_value': portfolio_value,
                'total_invested': total_cost
            })

        # 末日卖出全部
        revenue = max_shares * last_price
        trading_fee_sell = max(revenue * self.trading_fee_rate, 5)
        stamp_duty = revenue * self.stamp_duty_rate
        actual_revenue = revenue - trading_fee_sell - stamp_duty

        profit = actual_revenue - total_cost
        profit_pct = profit / total_cost if total_cost > 0 else 0

        self.orders.append({
            'date': last_date,
            'action': 'sell',
            'price': last_price,
            'shares': max_shares,
            'amount': revenue,
            'trading_fee': trading_fee_sell,
            'stamp_duty': stamp_duty,
            'actual_revenue': actual_revenue,
            'profit': profit,
            'return_pct': profit_pct,
            'reason': 'buy_and_hold_end'
        })

        logger.info(f"Buy & Hold completed: profit={profit:.2f}, return={profit_pct*100:.2f}%")
        return {
            'trades': self.orders,
            'daily_values': self.daily_values
        }
