"""
周定投策略插件（Weekly DCA）
每周第一个交易日买入指定股数
止盈 take_profit%，止损 stop_loss%
继承 BaseStrategy，符合回测平台插件接口

重要修复（2026-06-12）：
  v1: 修复多处逻辑bug
  - _get_weekly_trading_days 改为按 ISO 周取每周首个交易日（而非只找周一）
  - _record_daily_value 改为在买卖操作之后调用，确保 daily_values 反映真实收盘市值
  - 最终清仓后更新 daily_values，calc_returns 使用最终值
  - _sell_all_at_end 不再重复更新 daily_values（已在循环最后一天记录）
  - _get_price 修复重复列名
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
        self.enable_tp_sl = cfg.get("enable_tp_sl", True)
        # 内部状态
        self.lots = []  # 持仓批次: {date, shares, cost_per_share, cost_total, sold}
        self.weekly_dates = []
        
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
        
        # ── 计算每交易日的周内序号（周一定投实际使用每周首个交易日）
        df['_date_dt'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
        # ISO 周：2020-01 → 2020年第1周
        df['_iso_year'] = df['_date_dt'].dt.isocalendar().year.astype(str)
        df['_iso_week'] = df['_date_dt'].dt.isocalendar().week.astype(str).str.zfill(2)
        df['_year_week'] = df['_iso_year'] + '-' + df['_iso_week']
        
        # 每周第一个交易日（按 _year_week 分组取最小 trade_date）
        first_of_week = df.groupby('_year_week')['trade_date'].min()
        weekly_set = set(first_of_week.values.tolist())
        
        logger.debug(f"Found {len(weekly_set)} weekly trading days (first of week)")
        
        # ── 跳过 start_idx 之前的数据
        if start_idx > 0:
            df = df.iloc[start_idx:].reset_index(drop=True)
        
        for idx, row in df.iterrows():
            current_date = row['trade_date']
            price = self._get_price(row)
            
            if price <= 0:
                continue
            
            # ====================================================
            # 【顺序修正】先执行买卖操作，再记录当日收盘市值
            # ====================================================
            
            # 1️⃣ 每周定投买入
            if current_date in weekly_set:
                self._buy_weekly(row)
            
            # 2️⃣ 检查止盈止损
            if self.enable_tp_sl:
                self._check_take_profit(row)
                self._check_stop_loss(row)
            
            # 3️⃣ 记录每日市值（此时已反映当日所有操作）
            self._record_daily_value(price, current_date)
        
        # ── 回测结束，卖出所有剩余持仓 ──
        if len(df) > 0:
            last_row = df.iloc[-1]
            last_date = last_row['trade_date']
            last_price = self._get_price(last_row)
            self._sell_all_at_end(last_date, last_price)
        else:
            # 没有数据：回测结束也记录最终值
            self._record_daily_value(0, "end")
        
        # ── 计算收益率 ──
        returns = self.calc_returns()
        
        # 🔍 DEBUG: 打印关键数据
        if len(self.trades) > 0:
            total_buy_cost = sum(t['cost'] for t in self.trades if t['action'] == 'BUY')
            total_sell_rev = sum(t['revenue'] for t in self.trades if t['action'] == 'SELL')
            print(f"  [DEBUG DCA] capital={self.capital:.2f} cash_final={self.cash:.2f} "
                  f"buy_cost={total_buy_cost:.2f} sell_rev={total_sell_rev:.2f} "
                  f"returns={returns:.2f}% trades={len(self.trades)}")
        
        logger.info(f"✓ WeeklyDCAStrategyPlugin completed: {len(self.trades)} trades, returns={returns:.2f}%")
        
        return {
            "returns": returns,
            "trades": self.trades,
            "daily_values": self.daily_values,
        }
    
    def _get_price(self, row: pd.Series) -> float:
        """获取复权收盘价"""
        for col in ['adj_close', 'close']:
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
        """回测结束时卖出所有剩余持仓，并记录最终市值"""
        total_shares = sum(lot['shares'] for lot in self.lots if not lot['sold'])
        
        if total_shares > 0:
            success = self.sell(last_date, last_price, total_shares, reason="回测结束")
            
            if success:
                for lot in self.lots:
                    if not lot['sold']:
                        lot['sold'] = True
                
                # 【修复】最终清仓后追加一条 daily_value，确保 calc_returns 用真实值
                self._record_daily_value(last_price, last_date + "_结束")
                
                logger.debug(f"Weekly DCA end sell: {last_date}, price={last_price:.2f}, shares={total_shares}")
    
    def _record_daily_value(self, current_price: float, current_date: str):
        """记录每日市值"""
        portfolio_value = self.get_portfolio_value(current_price)
        self.daily_values.append({
            'date': current_date,
            'portfolio_value': portfolio_value,
        })
