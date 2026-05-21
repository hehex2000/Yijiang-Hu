"""
主动量化策略 - MACD/KDJ组合策略
MACD金叉 或 KDJ金叉 或 J值<20（超卖）→ 买入
MACD死叉 或 KDJ死叉 或 J值>80（超买）或 止盈止损 → 卖出
止盈25%，止损10%
市场常规参数：MACD(12/26/9) + KDJ(9/3/3)
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from loguru import logger


class MACDKDJStrategy:
    """主动量化：MACD/KDJ组合策略"""    
    def __init__(self, 
                 total_capital: float = 100000.0,
                 macd_fast: int = 12,
                 macd_slow: int = 26,
                 macd_signal: int = 9,
                 kdj_n: int = 9,
                 kdj_m1: int = 3,
                 kdj_m2: int = 3,
                 take_profit: float = 0.25,
                 stop_loss: float = 0.10):
        """
        初始化MACD/KDJ策略
        
        Args:
            total_capital: 总资金
            macd_fast: MACD快线周期
            macd_slow: MACD慢线周期
            macd_signal: MACD信号线周期
            kdj_n: KDJ计算周期
            kdj_m1: K值平滑参数
            kdj_m2: D值平滑参数
            take_profit: 止盈比例
            stop_loss: 止损比例
        """
        self.total_capital = total_capital
        self.cash = total_capital
        self.position = 0  # 持仓数量
        self.avg_cost = 0.0  # 平均成本
        
        # MACD参数
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        
        # KDJ参数
        self.kdj_n = kdj_n
        self.kdj_m1 = kdj_m1
        self.kdj_m2 = kdj_m2
        
        # 止盈止损
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        
        # 交易记录
        self.trades: List[Dict] = []
        
        logger.info(f"MACDKDJStrategy initialized: MACD({macd_fast},{macd_slow},{macd_signal}), "
                   f"KDJ({kdj_n},{kdj_m1},{kdj_m2})")
    
    def calculate_macd(self, close: pd.Series) -> tuple:
        """计算MACD指标"""
        ema_fast = close.ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.macd_slow, adjust=False).mean()
        
        dif = ema_fast - ema_slow
        dea = dif.ewm(span=self.macd_signal, adjust=False).mean()
        macd_hist = (dif - dea) * 2
        
        return dif, dea, macd_hist
    
    def calculate_kdj(self, high: pd.Series, low: pd.Series, close: pd.Series) -> tuple:
        """计算KDJ指标"""
        low_list = low.rolling(window=self.kdj_n).min()
        high_list = high.rolling(window=self.kdj_n).max()
        
        rsv = (close - low_list) / (high_list - low_list) * 100
        rsv = rsv.fillna(50)
        
        k_value = rsv.ewm(com=self.kdj_m1-1, adjust=False).mean()
        d_value = k_value.ewm(com=self.kdj_m2-1, adjust=False).mean()
        j_value = 3 * k_value - 2 * d_value
        
        return k_value, d_value, j_value
    
    def run_backtest(self, df: pd.DataFrame) -> Dict:
        """
        运行回测
        
        Args:
            df: 包含 adj_open, adj_high, adj_low, adj_close 的DataFrame
            
        Returns:
            回测结果字典
        """
        logger.info(f"Starting MACD/KDJ backtest with {len(df)} days of data")
        
        # 计算指标
        dif, dea, macd_hist = self.calculate_macd(df['adj_close'])
        k_value, d_value, j_value = self.calculate_kdj(
            df['adj_high'], df['adj_low'], df['adj_close']
        )
        
        # 生成信号
        macd_golden = (dif > dea) & (dif.shift(1) <= dea.shift(1))
        macd_death = (dif < dea) & (dif.shift(1) >= dea.shift(1))
        
        kdj_golden = (k_value > d_value) & (k_value.shift(1) <= d_value.shift(1))
        kdj_death = (k_value < d_value) & (k_value.shift(1) >= d_value.shift(1))
        
        # 买入信号：MACD金叉 OR KDJ金叉 OR J值 < 20（放松条件）
        buy_signal = macd_golden | kdj_golden | (j_value < 20)
        
        # 卖出信号：MACD死叉 OR KDJ死叉 OR J值 > 80
        sell_signal = macd_death | kdj_death | (j_value > 80)
        
        # 回测循环
        position = 0
        cash = self.total_capital
        avg_cost = 0.0
        trades = []
        daily_values = []  # 新增：记录每日市值
        
        for i in range(1, len(df)):
            date = df.iloc[i]['trade_date']
            price = df.iloc[i]['adj_close']
            
            # 记录每日市值
            current_value = cash + position * price
            daily_values.append({
                'trade_date': date,
                'portfolio_value': current_value
            })
            
            # 卖出信号
            if position > 0:
                # 止盈止损
                profit_pct = (price - avg_cost) / avg_cost
                if profit_pct >= self.take_profit:
                    # 止盈卖出
                    revenue = position * price * 0.9998  # 扣除手续费
                    cash += revenue
                    trades.append({
                        'date': date,
                        'action': 'sell',
                        'price': price,
                        'shares': position,
                        'reason': 'take_profit',
                        'revenue': revenue,
                        'profit': profit_pct  # 添加收益率字段
                    })
                    logger.debug(f"Take profit sell: {date}, price={price:.2f}, profit={profit_pct:.2%}")
                    position = 0
                
                elif profit_pct <= -self.stop_loss:
                    # 止损卖出
                    revenue = position * price * 0.9998
                    cash += revenue
                    trades.append({
                        'date': date,
                        'action': 'sell',
                        'price': price,
                        'shares': position,
                        'reason': 'stop_loss',
                        'revenue': revenue,
                        'profit': profit_pct  # 添加收益率字段
                    })
                    logger.debug(f"Stop loss sell: {date}, price={price:.2f}, loss={profit_pct:.2%}")
                    position = 0
                
                # 策略卖出信号
                elif sell_signal.iloc[i]:
                    revenue = position * price * 0.9998
                    cash += revenue
                    trades.append({
                        'date': date,
                        'action': 'sell',
                        'price': price,
                        'shares': position,
                        'reason': 'sell_signal',
                        'revenue': revenue,
                        'profit': (price - avg_cost) / avg_cost  # 添加收益率字段
                    })
                    logger.debug(f"Sell signal: {date}, price={price:.2f}")
                    position = 0
            
            # 买入信号
            if position == 0 and buy_signal.iloc[i]:
                # 计算买入数量（100股整数倍）
                max_shares = int(cash / price / 100) * 100
                if max_shares > 0:
                    cost = max_shares * price * 1.0002  # 含手续费
                    cash -= cost
                    position = max_shares
                    avg_cost = price
                    trades.append({
                        'date': date,
                        'action': 'buy',
                        'price': price,
                        'shares': max_shares,
                        'reason': 'buy_signal',
                        'cost': cost
                    })
                    logger.debug(f"Buy signal: {date}, price={price:.2f}, shares={max_shares}")
        
        # 强制平仓（如果还有持仓）
        if position > 0:
            last_price = df.iloc[-1]['adj_close']
            last_date = df.iloc[-1]['trade_date']
            revenue = position * last_price * 0.9998
            cash += revenue
            trades.append({
                'date': last_date,
                'action': 'sell',
                'price': last_price,
                'shares': position,
                'reason': 'force_close',
                'revenue': revenue,
                'profit': (last_price - avg_cost) / avg_cost  # 添加收益率字段
            })
            logger.debug(f"Force close: {last_date}, price={last_price:.2f}")
            position = 0
        
        # 记录最后一天的市值（强制平仓后）
        if len(daily_values) > 0:
            last_date = df.iloc[-1]['trade_date']
            last_value = cash  # 平仓后全部是现金
            daily_values.append({
                'trade_date': last_date,
                'portfolio_value': last_value
            })
        
        # 保存每日市值
        self.daily_values = daily_values
        
        # 计算回测结果
        final_value = cash
        total_return = (final_value - self.total_capital) / self.total_capital
        
        # 计算胜率
        sell_trades = [t for t in trades if t['action'] == 'sell']
        buy_trades = [t for t in trades if t['action'] == 'buy']
        
        win_count = 0
        for i, sell_trade in enumerate(sell_trades):
            if i < len(buy_trades):
                buy_trade = buy_trades[i]
                if sell_trade['price'] > buy_trade['price']:
                    win_count += 1
        
        win_rate = win_count / len(sell_trades) if sell_trades else 0
        
        result = {
            'total_return': total_return,
            'final_value': final_value,
            'win_rate': win_rate,
            'trade_count': len(trades),
            'trades': trades
        }
        
        logger.info(f"Backtest complete: return={total_return:.2%}, win_rate={win_rate:.2%}, "
                    f"trades={len(trades)}")
        
        return result
    
    def run(self, df: pd.DataFrame) -> List[Dict]:
        """
        运行回测（兼容接口）
        
        Args:
            df: 包含 adj_open, adj_high, adj_low, adj_close 的DataFrame
            
        Returns:
            交易记录列表
        """
        result = self.run_backtest(df)
        self.orders = result['trades']
        # self.daily_values 已在 run_backtest() 中设置
        return {
            'trades': self.orders,
            'daily_values': self.daily_values
        }
