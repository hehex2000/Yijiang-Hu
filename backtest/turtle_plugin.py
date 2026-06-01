"""
海龟策略插件（Turtle Strategy）
通道上限为当前日之前n日收盘成交价最大值
通道下限为当前日之前n日收盘成交价最小值
当股价上穿上限时买入（最多加仓max_positions次），下穿下限时卖出（全仓卖出）
止盈take_profit%，止损stop_loss%
交易手续费：万分之2（买入和卖出均收取）
印花税：千分之1（仅卖出收取）
"""
import pandas as pd
from backtest.base_strategy import BaseStrategy
from loguru import logger


class TurtleStrategyPlugin(BaseStrategy):
    """海龟策略（插件版）"""
    
    def __init__(self, capital: float, cfg: dict):
        super().__init__("TurtleStrategyPlugin", capital, cfg)
        self.channel_period = cfg.get("channel_period", 30)
        self.position_mode = cfg.get("position_mode", "half")
        self.max_positions = cfg.get("max_positions", 3)
        self.take_profit = cfg.get("take_profit", 0.20)
        self.stop_loss = cfg.get("stop_loss", 0.10)
        self.position_count = 0
        logger.info(f"TurtleStrategyPlugin initialized: period={self.channel_period}, max_positions={self.max_positions}")
    
    def run(self, df: pd.DataFrame, start_idx: int = 0) -> dict:
        """
        运行海龟策略
        返回: {"returns": float, "trades": list}
        """
        logger.info(f"Running TurtleStrategyPlugin on {len(df)} days of data...")
        self.trades = []
        self.daily_values = []
        self.position = 0
        self.cash = self.capital
        self.position_count = 0
        
        if len(df) == 0:
            return {"returns": 0.0, "trades": []}
        
        df = df.sort_values('trade_date').reset_index(drop=True)
        
        # 列名映射（兼容有无复权列）
        data = df.copy()
        if "adj_close" not in data.columns:
            if "close" in data.columns:
                data["adj_close"] = data["close"]
            else:
                logger.error("缺少必要列: close / adj_close")
                return {"returns": 0.0, "trades": []}
        if "adj_open" not in data.columns:
            if "open" in data.columns:
                data["adj_open"] = data["open"]
            else:
                logger.error("缺少必要列: open / adj_open")
                return {"returns": 0.0, "trades": []}
        
        # 计算海龟通道（不包含当前行，用shift(1)）
        data['upper_channel'] = data['adj_close'].shift(1).rolling(window=self.channel_period).max()
        data['lower_channel'] = data['adj_close'].shift(1).rolling(window=self.channel_period).min()
        
        # 突破信号（上穿/下穿）
        data['prev_close'] = data['adj_close'].shift(1)
        data['prev_upper'] = data['upper_channel'].shift(1)
        data['prev_lower'] = data['lower_channel'].shift(1)
        
        data['breakout'] = (data['prev_close'] <= data['prev_upper']) & (data['adj_close'] > data['upper_channel'])
        data['breakdown'] = (data['prev_close'] >= data['prev_lower']) & (data['adj_close'] < data['lower_channel'])
        
        # 记录前一天的信号（用于T+1执行）
        prev_breakout = False
        prev_breakdown = False
        
        for i in range(len(data)):
            row = data.iloc[i]
            date = row['trade_date']
            open_price = row['adj_open']
            close_price = row['adj_close']
            
            # 跳过早期无指标数据
            if pd.isna(row['upper_channel']) or pd.isna(row['lower_channel']):
                v = self.cash + self.position * close_price if self.position > 0 else self.cash
                self.daily_values.append({'date': date, 'portfolio_value': v})
                continue
            
            # T+1执行：如果前一天有突破信号，今天开盘买入
            if prev_breakout and self.position_count < self.max_positions and self.cash > 0:
                buy_amount = self.cash * (0.95 if self.position_mode == 'full' else 0.50)
                shares = int(buy_amount / open_price / 100) * 100
                if shares > 0:
                    self.buy(date, open_price, shares, reason="海龟突破")
                    self.position_count += 1
            
            # T+1执行：如果前一天有跌破信号，今天开盘卖出
            elif prev_breakdown and self.position > 0:
                self.sell(date, open_price, reason="海龟跌破")
                self.position_count = 0  # 重置加仓次数
            
            # 检查止盈止损（每天检查，用当天收盘价）
            if self.position > 0:
                current_return = (close_price - self.avg_cost) / self.avg_cost if self.avg_cost > 0 else 0
                
                # 止盈
                if current_return >= self.take_profit:
                    self.sell(date, open_price, reason=f"止盈({current_return:.1%})")
                    self.position_count = 0
                
                # 止损
                elif current_return <= -self.stop_loss:
                    self.sell(date, open_price, reason=f"止损({current_return:.1%})")
                    self.position_count = 0
            
            # 更新前一天的信号（用于明天执行）
            prev_breakout = bool(data.iloc[i]['breakout'])
            prev_breakdown = bool(data.iloc[i]['breakdown'])
            
            v = self.cash + self.position * close_price if self.position > 0 else self.cash
            self.daily_values.append({'date': date, 'portfolio_value': v})
        
        # 如果回测结束仍有持仓，按最后一天开盘价卖出
        if self.position > 0:
            last_price = float(data.iloc[-1]['adj_close'])
            self.sell(data.iloc[-1]['trade_date'], last_price, reason="回测结束平仓")
        
        ret = self.calc_returns()
        logger.info(f"TurtleStrategyPlugin finished: returns={ret:.2f}%, trades={len(self.trades)}")
        return {"returns": ret, "trades": self.trades}
