"""
策略基类 — 所有策略必须继承此类并实现 run() 方法
"""
import pandas as pd
import sqlite3
from pathlib import Path

# 数据库路径（共享）
DB_PATH = str(Path(__file__).parent.parent / "data" / "stock_db.db")


class BaseStrategy:
    """
    策略基类 — 定义标准接口
    
    新策略只需：
    1. 继承此类
    2. 实现 run(df, start_idx=0) 方法
    3. 返回 {"returns": float, "trades": list}
    """

    def __init__(self, name, capital, cfg):
        """
        Args:
            name: 策略名称（用于显示）
            capital: 初始资金
            cfg: 策略配置（从 config.py 读取）
        """
        self.name = name
        self.capital = capital
        self.cfg = cfg
        self.trades = []
        self.daily_values = []
        self.position = 0      # 持仓数量
        self.avg_cost = 0.0    # 平均持仓成本
        self.cash = capital     # 可用资金

    def run(self, df, start_idx=0):
        """
        运行策略（子类必须实现）
        
        Args:
            df: 股票数据 DataFrame，必须包含列：
                    - trade_date, open, high, low, close, volume
                    - adj_open, adj_close（复权价）
            start_idx: 回测起始位置（跳过此前数据）
        
        Returns:
            {
                "returns": float,      # 收益率（%）
                "trades": list,       # 交易记录
                "daily_values": list,  # 每日资产值（用于绘图）
            }
        """
        raise NotImplementedError("子类必须实现 run() 方法")

    def buy(self, date, price, shares, reason=""):
        """买入操作（子类可调用）"""
        cost = shares * price
        fee = cost * 0.0002      # 买入手续费
        total_cost = cost + fee
        if total_cost > self.cash:
            return False  # 资金不足
        
        self.cash -= total_cost
        
        # 更新平均持仓成本（加权平均）
        if self.position > 0:
            total_cost_base = self.avg_cost * self.position + cost
            self.position += shares
            self.avg_cost = total_cost_base / self.position
        else:
            # 第一次买入
            self.avg_cost = price
            self.position = shares
        
        trade = {
            "date": date,
            "action": "BUY",
            "price": price,
            "shares": shares,
            "cost": total_cost,
            "reason": reason,
        }
        self.trades.append(trade)
        return True

    def sell(self, date, price, shares=None, reason=""):
        """卖出操作（子类可调用）"""
        if shares is None:
            shares = self.position

        if shares > self.position:
            shares = self.position

        if shares == 0:
            return False

        revenue = shares * price
        fee = revenue * 0.0002   # 卖出手续费
        tax = revenue * 0.001    # 印花税（A股卖出收）
        net_revenue = revenue - fee - tax

        self.cash += net_revenue

        # 更新持仓
        self.position -= shares
        if self.position == 0:
            self.avg_cost = 0.0  # 清仓后重置平均成本

        trade = {
            "date": date,
            "action": "SELL",
            "price": price,
            "shares": shares,
            "revenue": net_revenue,
            "reason": reason,
        }
        self.trades.append(trade)
        return True

    def get_portfolio_value(self, current_price):
        """计算当前总资产价值"""
        return self.cash + self.position * current_price

    def calc_returns(self):
        """计算收益率（只返回 float，交易次数通过 len(self.trades) 获取）"""
        if not self.daily_values:
            return 0.0
        final_value = self.daily_values[-1]["portfolio_value"]
        ret = (final_value / self.capital - 1) * 100
        return ret  # 只返回 float，不返回 tuple

    @staticmethod
    def load_stock_data(ts_code, start_date, end_date):
        """
        加载股票数据（公共方法，所有策略可用）
        
        Returns:
            DataFrame with columns: trade_date, open, high, low, close, volume, adj_open, adj_close
        """
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql(
            """
            SELECT trade_date, open, high, low, close, volume,
                   adj_open, adj_close
            FROM daily
            WHERE ts_code = ? AND trade_date BETWEEN ? AND ?
            ORDER BY trade_date
            """,
            conn,
            params=(ts_code, start_date, end_date),
        )
        conn.close()
        return df
