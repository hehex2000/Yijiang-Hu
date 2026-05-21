"""
智能切换策略 - 市场状态判断 + 策略自动切换
牛市：买入持有（不卖出，直到牛市结束）
熊市：使用 MACD/RSI 信号交易
"""
import pandas as pd
import numpy as np
from typing import List, Dict
from loguru import logger

from .market_state import MarketStateDetector


class SmartSwitchStrategy:
    def __init__(self,
                 total_capital: float = 200000.0,
                 macd_fast: int = 12,
                 macd_slow: int = 26,
                 macd_signal: int = 9,
                 rsi_period: int = 14,
                 rsi_overbought: float = 70,
                 rsi_oversold: float = 30,
                 trading_fee_rate: float = 0.0002,
                 stamp_duty_rate: float = 0.001,
                 ma_period: int = 200):
        self.total_capital = total_capital
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.trading_fee_rate = trading_fee_rate
        self.stamp_duty_rate = stamp_duty_rate
        self.ma_period = ma_period

        self.market_detector = MarketStateDetector(ma_period=ma_period)

        self.cash = total_capital
        self.position_shares = 0
        self.avg_cost = 0.0
        self.orders = []
        self.daily_values = []

        logger.info(f"SmartSwitchStrategy initialized: MA{ma_period}")

    # ── 指标计算 ──────────────────────────────

    def _calc_macd(self, close: pd.Series):
        ema_f = close.ewm(span=self.macd_fast, adjust=False).mean()
        ema_s = close.ewm(span=self.macd_slow, adjust=False).mean()
        dif = ema_f - ema_s
        dea = dif.ewm(span=self.macd_signal, adjust=False).mean()
        return dif, dea

    def _calc_rsi(self, close: pd.Series):
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        ag = gain.rolling(self.rsi_period).mean()
        al = loss.rolling(self.rsi_period).mean()
        rs = ag / (al + 1e-9)
        return 100 - 100 / (1 + rs)

    # ── 信号判断 ──────────────────────────────

    def _macd_cross_up(self, dif, dea, i: int) -> bool:
        if i < 1:
            return True  # 第一天默认买入
        return (dif.iloc[i - 1] <= dea.iloc[i - 1]) and (dif.iloc[i] > dea.iloc[i])

    def _macd_cross_down(self, dif, dea, i: int) -> bool:
        if i < 1:
            return False
        return (dif.iloc[i - 1] >= dea.iloc[i - 1]) and (dif.iloc[i] < dea.iloc[i])

    def _rsi_ok(self, rsi, i: int) -> bool:
        """RSI 未超买"""
        return pd.notna(rsi.iloc[i]) and rsi.iloc[i] < self.rsi_overbought

    def _rsi_overbought(self, rsi, i: int) -> bool:
        return pd.notna(rsi.iloc[i]) and rsi.iloc[i] > self.rsi_overbought

    # ── 买卖执行 ──────────────────────────────

    def _buy(self, price: float, date_str: str, reason: str):
        if self.position_shares > 0:
            return  # 已持有，不重复买入
        max_shares = int(self.cash / price / 100) * 100
        if max_shares < 100:
            return
        buy_amount = max_shares * price
        fee = max(buy_amount * self.trading_fee_rate, 5)
        self.cash -= (buy_amount + fee)
        self.position_shares = max_shares
        self.avg_cost = price
        self.orders.append({
            'date': date_str, 'action': 'buy',
            'price': price, 'shares': max_shares,
            'amount': buy_amount, 'trading_fee': fee,
            'total_cost': buy_amount + fee,
            'profit': None, 'return_pct': None,
            'reason': reason
        })
        logger.debug(f"Smart BUY {date_str} {price:.2f} x{max_shares} {reason}")

    def _sell(self, price: float, date_str: str, reason: str):
        if self.position_shares == 0:
            return
        revenue = self.position_shares * price
        fee = max(revenue * self.trading_fee_rate, 5)
        duty = revenue * self.stamp_duty_rate
        actual = revenue - fee - duty
        profit = actual - (self.position_shares * self.avg_cost)
        profit_pct = profit / (self.position_shares * self.avg_cost) if self.avg_cost > 0 else 0
        self.cash += actual
        self.orders.append({
            'date': date_str, 'action': 'sell',
            'price': price, 'shares': self.position_shares,
            'amount': revenue, 'trading_fee': fee,
            'stamp_duty': duty, 'actual_revenue': actual,
            'profit': profit, 'return_pct': profit_pct,
            'reason': reason
        })
        logger.debug(f"Smart SELL {date_str} {price:.2f} x{self.position_shares} profit={profit:.2f} {reason}")
        self.position_shares = 0
        self.avg_cost = 0.0

    # ── 主逻辑 ────────────────────────────────

    def run(self, df: pd.DataFrame, stock_code: str = '') -> List[Dict]:
        logger.info(f"SmartSwitchStrategy on {len(df)} days, code={stock_code}")

        self.orders = []
        self.daily_values = []
        self.cash = self.total_capital
        self.position_shares = 0
        self.avg_cost = 0.0

        if len(df) == 0:
            return self.orders

        df = df.copy().reset_index(drop=True)
        dif, dea = self._calc_macd(df['adj_close'])
        rsi = self._calc_rsi(df['adj_close'])

        prev_state = None

        for i in range(len(df)):
            row = df.iloc[i]
            date_str = str(row['trade_date'])
            price = float(row['adj_close'])

            # 市场状态
            state = self.market_detector.get_market_state(date_str)
            is_bull = (state == 'bull')
            is_bear = (state == 'bear')

            # 记录每日市值
            if self.position_shares > 0:
                portfolio_value = self.position_shares * price
            else:
                portfolio_value = self.cash
            self.daily_values.append({
                'date': date_str,
                'portfolio_value': portfolio_value,
                'total_invested': self.total_capital - self.cash
            })

            # ── 空仓：买入逻辑 ─────────────
            if self.position_shares == 0:
                buy = False
                reason = ''
                if is_bull:
                    # 牛市：立即买入，不等待信号
                    buy = True
                    reason = 'bull_enter'
                elif is_bear and self._macd_cross_up(dif, dea, i) and self._rsi_ok(rsi, i):
                    buy = True
                    reason = 'bear_macd_rsi_buy'
                elif state == 'unknown' and self._macd_cross_up(dif, dea, i) and self._rsi_ok(rsi, i):
                    buy = True
                    reason = 'unknown_macd_rsi_buy'

                if buy:
                    self._buy(price, date_str, reason)
                continue

            # ── 持仓：卖出逻辑 ─────────────
            else:  # self.position_shares > 0
                # 牛市：坚决不卖
                if is_bull:
                    continue

                sell = False
                reason = ''
                if is_bear and (self._macd_cross_down(dif, dea, i) or self._rsi_overbought(rsi, i)):
                    sell = True
                    reason = 'bear_macd_rsi_sell'
                elif state == 'unknown' and (self._macd_cross_down(dif, dea, i) or self._rsi_overbought(rsi, i)):
                    sell = True
                    reason = 'unknown_macd_rsi_sell'

                # 最后一天必须卖出
                if i == len(df) - 1:
                    sell = True
                    reason = 'end_of_backtest'

                if sell:
                    self._sell(price, date_str, reason)

        logger.info(f"✓ SmartSwitchStrategy done: {len(self.orders)} orders, cash={self.cash:.2f}")
        return self.orders
