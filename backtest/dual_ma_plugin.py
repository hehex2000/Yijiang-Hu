"""
双均线策略插件（Dual MA Cross）
MA20 上穿 MA60 金叉买入50%仓，死叉全仓卖出
止盈 take_profit%，止损 stop_loss%
继承 BaseStrategy，符合回测平台插件接口

优化：使用 TA-Lib 计算均线（性能提升 10-100 倍）
"""

import pandas as pd
import numpy as np
import talib as ta  # ← 新增 TA-Lib
from backtest.base_strategy import BaseStrategy
from backtest.atr_stop_loss import ATRStopLoss
from loguru import logger


class DualMAStrategyPlugin(BaseStrategy):
    """双均线策略（插件版）"""
    
    def __init__(self, capital: float, cfg: dict):
        super().__init__("DualMAStrategyPlugin", capital, cfg)
        # 从 config.py 读取参数
        self.ma_short = cfg.get("ma_short", 20)
        self.ma_long = cfg.get("ma_long", 60)
        self.position_pct = cfg.get("position_pct", 0.5)  # 单次买入仓位比例
        self.take_profit = cfg.get("take_profit", 0.30)
        self.stop_loss = cfg.get("stop_loss", 0.10)
        # ── ATR 动态止损 ──
        self.use_atr_sl = cfg.get("atr_stop_loss", False)
        self.atr_sl = ATRStopLoss(
            atr_period=cfg.get("atr_period", 14),
            atr_mult=cfg.get("atr_mult", 3.0),
            trail_mult=cfg.get("trail_mult", 3.0),
        )
        
        logger.info(
            f"DualMAStrategyPlugin initialized: "
            f"MA{self.ma_short}/{self.ma_long}, "
            f"position_pct={self.position_pct}, "
            f"tp={self.take_profit}, sl={self.stop_loss}"
        )
    
    def run(self, df: pd.DataFrame, start_idx: int = 0) -> dict:
        """
        运行双均线策略
        
        Args:
            df: 股票数据 DataFrame（需包含 ma{ma_short}, ma{ma_long} 列）
            start_idx: 回测起始位置
        
        Returns:
            {"returns": float, "trades": list, "daily_values": list}
        """
        logger.info(f"Running DualMAStrategyPlugin on {len(df)} days of data...")
        
        # 重置状态
        self.trades = []
        self.daily_values = []
        self.position = 0
        self.avg_cost = 0.0
        self.cash = self.capital
        
        if len(df) == 0:
            return {"returns": 0.0, "trades": [], "daily_values": []}
        
        df = df.sort_values('trade_date').reset_index(drop=True)
        
        # 跳过 start_idx 之前的数据
        if start_idx > 0:
            df = df.iloc[start_idx:].reset_index(drop=True)
        
        # ═══ 检查必要的均线列是否存在 ═══
        ma_short_col = f'ma{self.ma_short}'
        ma_long_col = f'ma{self.ma_long}'
        
        # ═══ 尝试找到收盘价列 ═══
        close_col = None
        for col in ['adj_close', 'close', '收盘价', 'close_price']:
            if col in df.columns:
                close_col = col
                break
        
        if close_col is None:
            logger.error("Cannot find close price column in DataFrame")
            return {"returns": 0.0, "trades": [], "daily_values": []}
        
        # ═══ 如果均线列不存在，自动计算（使用 TA-Lib 优化） ═══
        if ma_short_col not in df.columns:
            # TA-Lib SMA：前 (timeperiod-1) 个值为 NaN
            df[ma_short_col] = ta.SMA(df[close_col].values, timeperiod=self.ma_short)
            logger.info(f"Auto-calculated {ma_short_col} using TA-Lib SMA")
        
        if ma_long_col not in df.columns:
            df[ma_long_col] = ta.SMA(df[close_col].values, timeperiod=self.ma_long)
            logger.info(f"Auto-calculated {ma_long_col} using TA-Lib SMA")
        
        # ═════ 预计算金叉/死叉信号 ═════
        df['prev_' + ma_short_col] = df[ma_short_col].shift(1)
        df['prev_' + ma_long_col] = df[ma_long_col].shift(1)
        
        # 金叉：昨天 short_ma <= long_ma，今天 short_ma > long_ma
        golden_cross = (df['prev_' + ma_short_col] <= df['prev_' + ma_long_col]) & (df[ma_short_col] > df[ma_long_col])
        
        # 死叉：昨天 short_ma >= long_ma，今天 short_ma < long_ma
        death_cross = (df['prev_' + ma_short_col] >= df['prev_' + ma_long_col]) & (df[ma_short_col] < df[ma_long_col])
        
        df['signal_change'] = 0
        df.loc[golden_cross, 'signal_change'] = 1
        df.loc[death_cross, 'signal_change'] = -1
        
        logger.info(f"Signal computation complete: {golden_cross.sum()} golden crosses, {death_cross.sum()} death crosses")
        
        # ── ATR 计算 ──
        if self.use_atr_sl:
            high_col = 'high' if 'high' in df.columns else close_col
            low_col = 'low' if 'low' in df.columns else close_col
            atr_arr = self.atr_sl.calc_atr(df[high_col].values, df[low_col].values, df[close_col].values)
        else:
            atr_arr = np.zeros(len(df))
        
        # 记录前一天的信号（用于T+1执行）
        prev_signal = 0
        
        for idx, row in df.iterrows():
            current_date = row['trade_date']
            price = self._get_price(row)
            
            if price <= 0:
                continue
            
            # 记录每日市值
            self._record_daily_value(price, current_date)
            
            # 获取当前信号
            current_signal = self._get_signal(row, ma_short_col, ma_long_col)
            
            # T+1执行：如果前一天有金叉/死叉信号，今天开盘交易
            if prev_signal == 1 and self.cash > 0:  # 金叉买入
                position_before = self.position
                self._buy_golden_cross(row)
                # 金叉买入后设置ATR初始止损（仅当实际买入成功时）
                if self.use_atr_sl and self.position > position_before:
                    self.atr_sl.on_entry(entry_price=self.avg_cost, atr_val=atr_arr[idx])
            elif prev_signal == -1 and self.position > 0:  # 死叉卖出
                # 死叉卖出前重置ATR状态
                if self.use_atr_sl:
                    self.atr_sl.reset()
                self._sell_death_cross(row)
            
            # 检查止盈止损（每天检查，用收盘价）
            if self.position > 0:
                # ── ATR 追踪止损（优先于固定止损）──
                if self.use_atr_sl:
                    high_val = float(row.get('high', price))
                    self.atr_sl.update(high_price=high_val, atr_val=atr_arr[idx])
                    should_stop, stop_price, atr_reason = self.atr_sl.check_stop(close_price=price)
                    if should_stop:
                        self.sell(current_date, price, self.position, reason=atr_reason)
                        continue
                self._check_take_profit_stop_loss(row)
            
            # 更新前一天的信号
            prev_signal = current_signal
        
        # 回测结束，卖出所有剩余持仓
        if len(df) > 0:
            last_row = df.iloc[-1]
            last_date = last_row['trade_date']
            last_price = self._get_price(last_row)
            self._sell_all_at_end(last_date, last_price)
        
        # 计算收益率
        returns = self.calc_returns()
        
        logger.info(f"✓ DualMAStrategyPlugin completed: {len(self.trades)} trades, returns={returns:.2f}%")
        
        return {
            "returns": returns,
            "trades": self.trades,
            "daily_values": self.daily_values,
        }
    
    def _get_price(self, row: pd.Series) -> float:
        """获取复权收盘价（兼容不同列名）"""
        for col in ['adj_close', 'close', '收盘价']:
            if col in row and pd.notna(row[col]):
                return float(row[col])
        return 0.0
    
    def _get_signal(self, row: pd.Series, ma_short_col: str, ma_long_col: str) -> int:
        """
        获取均线信号
        返回：
            1  = 金叉（短均线上穿长均线）
            -1 = 死叉（短均线下穿长均线）
            0  = 无信号
        """
        if ma_short_col not in row or ma_long_col not in row:
            return 0
        
        if pd.isna(row[ma_short_col]) or pd.isna(row[ma_long_col]):
            return 0
        
        # 需要前一天的数据来判断金叉/死叉
        idx = row.name
        if idx == 0:
            return 0
        
            # 前一天数据通过 signal_change 列获取，无需单独变量
        # 更简单的方法：直接比较当前和前一天的均线差值
        # 金叉：昨天 short_ma < long_ma，今天 short_ma > long_ma
        # 死叉：昨天 short_ma > long_ma，今天 short_ma < long_ma
        
        # 由于我们无法在这里访问前一天的数据，改用 signal_change 列
        if 'signal_change' in row:
            return int(row['signal_change'])
        
        return 0
    
    def _buy_golden_cross(self, row: pd.Series):
        """金叉买入（T+1，用开盘价）"""
        date = row['trade_date']
        price = self._get_open_price(row)
        
        if price <= 0:
            return
        
        # 买入 position_pct% 可用资金
        reason = "金叉买入"
        # 计算可买股数（A股最小交易单位100股）
        cash_to_use = self.cash * self.position_pct
        shares = int(cash_to_use / price / 100) * 100
        
        if shares > 0:
            success = self._kelly_buy(date, price, shares, reason)
            
            if success:
                logger.debug(f"Dual MA buy: {date}, price={price:.2f}, shares={shares}")
        else:
            logger.debug(f"Dual MA buy: insufficient cash to buy at {date}, price={price:.2f}")
    
    def _sell_death_cross(self, row: pd.Series):
        """死叉卖出（T+1，用开盘价）"""
        date = row['trade_date']
        price = self._get_open_price(row)
        
        if price <= 0 or self.position <= 0:
            return
        
        # 全仓卖出
        reason = "死叉卖出"
        success = self.sell(date, price, self.position, reason)
        
        if success:
            logger.debug(f"Dual MA sell: {date}, price={price:.2f}, shares={self.position}")
    
    def _check_take_profit_stop_loss(self, row: pd.Series):
        """检查止盈止损（用收盘价）"""
        if self.avg_cost <= 0 or self.position <= 0:
            return
        
        current_date = row['trade_date']
        current_price = self._get_price(row)
        
        if current_price <= 0:
            return
        
        profit_pct = (current_price - self.avg_cost) / self.avg_cost
        
        # 止盈
        if profit_pct >= self.take_profit:
            reason = "止盈"
            success = self.sell(current_date, current_price, self.position, reason)
            if success:
                logger.debug(f"Dual MA take profit: {current_date}, profit={profit_pct:.2%}")
        
        # 止损（仅当非ATR模式时使用固定止损）
        elif not self.use_atr_sl and profit_pct <= -self.stop_loss:
            reason = "止损"
            success = self.sell(current_date, current_price, self.position, reason)
            if success:
                logger.debug(f"Dual MA stop loss: {current_date}, loss={profit_pct:.2%}")
    
    def _sell_all_at_end(self, last_date: str, last_price: float):
        """回测结束时卖出所有剩余持仓"""
        if self.position > 0:
            reason = "回测结束"
            success = self.sell(last_date, last_price, self.position, reason)
            
            if success:
                logger.debug(f"Dual MA end sell: {last_date}, price={last_price:.2f}, shares={self.position}")
    
    def _record_daily_value(self, current_price: float, current_date: str):
        """记录每日市值"""
        portfolio_value = self.get_portfolio_value(current_price)
        self.daily_values.append({
            'date': current_date,
            'portfolio_value': portfolio_value,
        })
    
    def _get_open_price(self, row: pd.Series) -> float:
        """获取开盘价（兼容不同列名）"""
        for col in ['adj_open', 'open', '开盘价']:
            if col in row and pd.notna(row[col]):
                return float(row[col])
        # 如果没有开盘价，用收盘价代替
        return self._get_price(row)
