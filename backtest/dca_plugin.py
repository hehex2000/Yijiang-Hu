"""
月定投策略插件（Monthly DCA）
每月第一个交易日买入：严格投入指定金额（向下取整到100股），不够买100股就跳过
止盈 take_profit%，止损 stop_loss%
止损触发卖出50%仓位，止盈触发卖出100%仓位
继承 BaseStrategy，符合回测平台插件接口
"""

import pandas as pd
from backtest.base_strategy import BaseStrategy
from loguru import logger


class DCAStrategyPlugin(BaseStrategy):
    """月定投策略（插件版）"""
    
    def __init__(self, capital: float, cfg: dict):
        super().__init__("DCAStrategyPlugin", capital, cfg)
        # 从 config.py 读取参数
        self.amount_per_month = cfg.get("amount_per_month", 5000)
        self.take_profit = cfg.get("take_profit", 0.30)
        self.stop_loss = cfg.get("stop_loss", 0.20)
        self.enable_tp_sl = cfg.get("enable_tp_sl", True)
        # 内部状态
        self.lots = []  # 持仓批次: {date, shares, cost_per_share, cost_total, sold}
        self.monthly_days = []
        
        logger.info(
            f"DCAStrategyPlugin initialized: "
            f"monthly={self.amount_per_month}, "
            f"tp={self.take_profit}, sl={self.stop_loss}"
        )
    
    def run(self, df: pd.DataFrame, start_idx: int = 0) -> dict:
        """
        运行月定投策略
        
        Args:
            df: 股票数据 DataFrame
            start_idx: 回测起始位置
        
        Returns:
            {"returns": float, "trades": list, "daily_values": list}
        """
        logger.info(f"Running DCAStrategyPlugin on {len(df)} days of data...")
        
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
        
        # 获取每月第一个交易日
        self.monthly_days = self._get_monthly_trading_days(df)
        monthly_set = set(self.monthly_days)
        
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
            
            # 每月定投买入
            if current_date in monthly_set:
                self._buy_monthly(row)
            
            # 检查止盈止损
            if self.enable_tp_sl:
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
        
        logger.info(f"✓ DCAStrategyPlugin completed: {len(self.trades)} trades, returns={returns:.2f}%")
        
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
    
    def _buy_monthly(self, row: pd.Series):
        """
        执行月度定投买入
        逻辑：每月用 amount_per_month 元买尽可能多的 100 股整数倍
              不够买 100 股就累积到下个月
        """
        date = row['trade_date']
        price = self._get_price(row)
        
        if price <= 0:
            return
        
        # 计算能买多少股（向下取整到 100 股）
        target_shares = int(self.amount_per_month / price / 100) * 100
        
        if target_shares < 100:
            logger.debug(f"DCA skip: {date}, can't buy 100 shares with {self.amount_per_month}")
            return
        
        # 使用基类的 buy() 方法
        reason = "月定投"
        success = self.buy(date, price, target_shares, reason)
        
        if success:
            # 记录批次（用于止盈止损）
            self.lots.append({
                'date': date,
                'shares': target_shares,
                'cost_per_share': price,
                'cost_total': target_shares * price,
                'sold': False,
            })
            logger.debug(f"DCA buy: {date}, price={price:.2f}, shares={target_shares}")
    
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
                    logger.debug(f"DCA take profit: {current_date}, profit={profit_pct:.2%}")
    
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
                    logger.debug(f"DCA stop loss 50%: {current_date}, loss={loss_pct:.2%}")
    
    def _sell_all_at_end(self, last_date: str, last_price: float):
        """回测结束时卖出所有剩余持仓"""
        total_shares = sum(lot['shares'] for lot in self.lots if not lot['sold'])
        
        if total_shares > 0:
            success = self.sell(last_date, last_price, total_shares, reason="回测结束")
            
            if success:
                for lot in self.lots:
                    if not lot['sold']:
                        lot['sold'] = True
                
                logger.debug(f"DCA end sell: {last_date}, price={last_price:.2f}, shares={total_shares}")
    
    def _record_daily_value(self, current_price: float, current_date: str):
        """记录每日市值"""
        portfolio_value = self.get_portfolio_value(current_price)
        self.daily_values.append({
            'date': current_date,
            'portfolio_value': portfolio_value,
        })
    
    def _get_monthly_trading_days(self, df: pd.DataFrame) -> list:
        """获取每月第一个交易日"""
        df = df.copy()
        df['year_month'] = df['trade_date'].astype(str).str[:6]
        monthly_first = df.groupby('year_month')['trade_date'].min().tolist()
        logger.debug(f"Found {len(monthly_first)} monthly trading days")
        return monthly_first
