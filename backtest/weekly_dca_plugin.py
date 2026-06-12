"""
周定投策略插件（Weekly DCA）
每周第一个交易日买入指定股数
止盈 take_profit%，止损 stop_loss%
继承 BaseStrategy，符合回测平台插件接口
"""

import pandas as pd
from backtest.base_strategy import BaseStrategy
from loguru import logger


class WeeklyDCAStrategyPlugin(BaseStrategy):
    """周定投策略（插件版）"""
    
    def __init__(self, capital: float, cfg: dict):
        super().__init__("WeeklyDCAStrategyPlugin", capital, cfg)
        # 从 config.py 读取参数
        self.shares_per_week = cfg.get("shares_per_week", 100)
        self.take_profit = cfg.get("take_profit", 0.30)
        self.stop_loss = cfg.get("stop_loss", 0.10)
        # 内部状态
        self.lots = []  # 持仓批次: {date, shares, cost_per_share, cost_total, sold}
        self.weekly_days = []
        
        logger.info(
            f"WeeklyDCAStrategyPlugin initialized: "
            f"shares/week={self.shares_per_week}, "
            f"tp={self.take_profit}, sl={self.stop_loss}"
        )
    
    def run(self, df: pd.DataFrame, start_idx: int = 0) -> dict:
        """
        运行周定投策略
        
        Args:
            df: 股票数据 DataFrame
            start_idx: 回测起始位置
        
        Returns:
            {"returns": float, "trades": list, "daily_values": list}
        """
        logger.info(f"Running WeeklyDCAStrategyPlugin on {len(df)} days of data...")
        
        # 重置状态
        self.trades = []
        self.daily_values = []
        self.position = 0
        self.avg_cost = 0.0
        self.cash = self.capital
        self.lots = []
        
        if len(df) == 0:
            return {"returns": 0.0, "trades": [], "daily_values": []}
        
        df = df.sort_values('trade_date').reset_index(drop=True)
        
        # 获取每周第一个交易日
        self.weekly_days = self._get_weekly_trading_days(df)
        weekly_set = set(self.weekly_days)
        
        # 跳过 start_idx 之前的数据
        if start_idx > 0:
            df = df.iloc[start_idx:].reset_index(drop=True)
        
        for idx, row in df.iterrows():
            current_date = row['trade_date']
            price = self._get_price(row)
            
            if price <= 0:
                continue
            
            # 记录每日市值
            self._record_daily_value(price, current_date)
            
            # 每周定投买入
            if current_date in weekly_set:
                self._buy_weekly(row)
            
            # 检查止盈止损
            self._check_take_profit(row)
            self._check_stop_loss(row)
        
        # 回测结束，卖出所有剩余持仓
        if len(df) > 0:
            last_row = df.iloc[-1]
            last_date = last_row['trade_date']
            last_price = self._get_price(last_row)
            self._sell_all_at_end(last_date, last_price)
        
        # 计算收益率
        returns = self.calc_returns()
        
        logger.info(f"✓ WeeklyDCAStrategyPlugin completed: {len(self.trades)} trades, returns={returns:.2f}%")
        
        return {
            "returns": returns,
            "trades": self.trades,
            "daily_values": self.daily_values,
        }
    
    def _get_price(self, row: pd.Series) -> float:
        """获取复权收盘价（兼容不同列名）"""
        for col in ['adj_close', 'adj_close', 'close', '收盘价']:
            if col in row and pd.notna(row[col]):
                return float(row[col])
        return 0.0
    
    def _buy_weekly(self, row: pd.Series):
        """
        执行周度定投买入
        逻辑：每周买入固定股数（默认100股）
        """
        date = row['trade_date']
        price = self._get_price(row)
        
        if price <= 0:
            return
        
        shares = self.shares_per_week
        reason = "周定投"
        
        # 使用基类的 buy() 方法
        success = self.buy(date, price, shares, reason)
        
        if success:
            # 记录批次（用于止盈止损）
            self.lots.append({
                'date': date,
                'shares': shares,
                'cost_per_share': price,
                'cost_total': shares * price,
                'sold': False,
            })
            logger.debug(f"Weekly DCA buy: {date}, price={price:.2f}, shares={shares}")
        else:
            logger.debug(f"Weekly DCA skip: {date}, insufficient funds")
    
    def _check_take_profit(self, row: pd.Series):
        """检查每笔持仓的止盈"""
        current_date = row['trade_date']
        current_price = self._get_price(row)
        
        if current_price <= 0:
            return
        
        for lot in self.lots:
            if lot['sold']:
                continue
            
            profit_pct = (current_price - lot['cost_per_share']) / lot['cost_per_share']
            
            if profit_pct >= self.take_profit:
                # 止盈：卖出 100% 仓位
                success = self.sell(current_date, current_price, lot['shares'], reason="止盈")
                
                if success:
                    lot['sold'] = True
                    logger.debug(f"Weekly DCA take profit: {current_date}, profit={profit_pct:.2%}")
    
    def _check_stop_loss(self, row: pd.Series):
        """检查每笔持仓的止损（触发时卖出50%仓位）"""
        current_date = row['trade_date']
        current_price = self._get_price(row)
        
        if current_price <= 0:
            return
        
        for lot in self.lots:
            if lot['sold']:
                continue
            
            loss_pct = (current_price - lot['cost_per_share']) / lot['cost_per_share']
            
            if loss_pct <= -self.stop_loss:
                # 止损：卖出 50% 仓位
                sell_shares = int(lot['shares'] * 0.5 / 100) * 100
                if sell_shares < 100:
                    sell_shares = lot['shares']  # 不够100股就全卖
                
                success = self.sell(current_date, current_price, sell_shares, reason="止损50%")
                
                if success:
                    lot['shares'] -= sell_shares
                    if lot['shares'] == 0:
                        lot['sold'] = True
                    logger.debug(f"Weekly DCA stop loss 50%: {current_date}, loss={loss_pct:.2%}")
    
    def _sell_all_at_end(self, last_date: str, last_price: float):
        """回测结束时卖出所有剩余持仓"""
        total_shares = sum(lot['shares'] for lot in self.lots if not lot['sold'])
        
        if total_shares > 0:
            success = self.sell(last_date, last_price, total_shares, reason="回测结束")
            
            if success:
                for lot in self.lots:
                    if not lot['sold']:
                        lot['sold'] = True
                
                logger.debug(f"Weekly DCA end sell: {last_date}, price={last_price:.2f}, shares={total_shares}")
    
    def _record_daily_value(self, current_price: float, current_date: str):
        """记录每日市值"""
        portfolio_value = self.get_portfolio_value(current_price)
        self.daily_values.append({
            'date': current_date,
            'portfolio_value': portfolio_value,
        })
    
    def _get_weekly_trading_days(self, df: pd.DataFrame) -> list:
        """获取每周第一个交易日（每周一）"""
        df = df.copy()
        df['weekday'] = pd.to_datetime(df['trade_date']).dt.weekday
        # 筛选星期一（0=周一）的交易日期
        mondays = df[df['weekday'] == 0]['trade_date'].tolist()
        
        logger.debug(f"Found {len(mondays)} weekly trading days (Mondays)")
        return mondays
