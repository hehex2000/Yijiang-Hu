"""
策略基类 — 所有策略必须继承此类并实现 run() 方法
"""
import pandas as pd
import sqlite3
from pathlib import Path
from backtest.kelly_sizer import KellySizer

# 数据库路径（共享）—— 统一指向平台唯一主库（2026-07-08 修正：原 data/stock_db.db 不存在）
DB_PATH = r"D:\tu-shareData\astock_daily.db"


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

        # ── 凯利公式控仓（总持仓封顶模式）──
        # use_kelly=True 时，约束「持仓市值 ≤ kelly_cap × 总资金」。
        # kelly_cap 来自 KellySizer（全凯利 f* × 半凯利折扣 × 安全折扣，夹在[min,max]）。
        # 这是平台已有 kelly_sizer 框架的统一推广：所有择时策略（除买入持有）共用，
        # 买入持有不读此段 → 天然排除。
        self.use_kelly = bool(cfg.get("use_kelly", False))
        if self.use_kelly:
            self.kelly = KellySizer(
                estimated_win_rate=cfg.get("kelly_win_rate", 0.55),
                estimated_win_loss_ratio=cfg.get("kelly_win_loss_ratio", 1.5),
                kelly_fraction=cfg.get("kelly_fraction", 0.5),
                max_position_pct=cfg.get("kelly_max_position", 0.25),
                min_position_pct=cfg.get("kelly_min_position", 0.05),
                safety_discount=cfg.get("kelly_safety_discount", 0.8),
            )
            self.kelly_cap = self.kelly.get_position_pct()  # 封顶比例（0.05~0.25）
        else:
            self.kelly = None
            self.kelly_cap = None  # None ⇒ 不封顶（等价于封顶比例为1.0）

        # 凯利封顶基准：多股分仓时按「组合总资金」封顶（避免每票切片太小把高价股禁仓），
        # 单股回测时 total_capital 未设 → 退化为 self.capital，行为不变。
        self.total_capital = cfg.get("total_capital", self.capital)

        # ── 真实分科目成本开关（opt-in，默认关 → 维持历史简单成本口径）──
        # 开时 buy/sell 改用 _real_fee_buy/_real_fee_sell：
        #   佣金万2.5 + 滑点0.1% + 过户费0.001%（双边）+ 日期感知印花税(2023-08-28前0.1%后0.05%)。
        # 关时维持原 base 简单口径（佣金万2 + 卖出印花税0.1%固定，无滑点/过户费），
        # 与历史回测数字完全一致，其他插件不受影响。
        self.real_cost = bool(cfg.get("real_cost", False))

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

    @staticmethod
    def _stamp_duty_rate(trade_date) -> float:
        """日期感知印花税：2023-08-28 起减半（0.1% → 0.05%）。

        trade_date 缺省/不可解析时回退 0.1%（保守）。
        """
        try:
            td = int(trade_date)
        except (TypeError, ValueError):
            return 0.001
        return 0.0005 if td >= 20230828 else 0.001

    def _real_fee_buy(self, amount: float, trade_date) -> float:
        """真实买入成本：佣金万2.5 + 滑点0.1% + 过户费0.001%（双边统一）。"""
        commission = max(amount * 0.00025, 5.0)
        slippage = amount * 0.001
        transfer = amount * 0.00001
        return commission + slippage + transfer

    def _real_fee_sell(self, amount: float, trade_date):
        """真实卖出成本：佣金万2.5 + 滑点0.1% + 过户费0.001% + 日期感知印花税。"""
        commission = max(amount * 0.00025, 5.0)
        slippage = amount * 0.001
        transfer = amount * 0.00001
        stamp = amount * self._stamp_duty_rate(trade_date)
        return commission + slippage + transfer, stamp

    @staticmethod
    def cap_by_kelly(capital, position, cash, kelly_cap, price, intended_shares):
        """
        按凯利「总持仓封顶」缩放意图买入股数。

        约束：买入后持仓市值 ≤ kelly_cap × capital。
        - kelly_cap 为 None（未启用凯利）或 >= 1.0 时不封顶，直接返回意图股数。
        - 当前已持仓市值用 (position × price) 估算；剩余空间 = 上限 − 已持仓。
        - 最终取 min(意图金额, 剩余空间, 可用现金)，再向下取整到百股。
        """
        if kelly_cap is None or kelly_cap >= 1.0:
            return intended_shares
        if price is None or price <= 0 or intended_shares <= 0:
            return 0
        cap_value = kelly_cap * capital
        cur_value = position * price
        room_value = cap_value - cur_value
        if room_value <= 0:
            return 0
        intended_value = intended_shares * price
        allowed_value = min(intended_value, room_value, cash)
        if allowed_value <= 0:
            return 0
        return max(0, int(allowed_value / price / 100) * 100)

    def kelly_room_shares(self, price, intended_shares):
        """
        返回经凯利总持仓封顶后的实际可买股数。
        供「自管现金」类策略（如网格）使用：它们不调 self.buy()，
        需自行用返回值缩放后维护 cash/position。
        """
        return BaseStrategy.cap_by_kelly(
            self.total_capital, self.position, self.cash,
            self.kelly_cap if self.use_kelly else None,
            price, intended_shares,
        )

    def _kelly_buy(self, date, price, intended_shares, reason=""):
        """
        带凯利总持仓封顶的买入。封顶后委托基类 buy() 执行。
        所有「调 self.buy() 开仓」的插件统一改用本方法即可自动受凯利控仓约束。
        """
        sh = self.kelly_room_shares(price, intended_shares)
        if sh > 0:
            return self.buy(date, price, sh, reason)
        return False

    def buy(self, date, price, shares, reason=""):
        """买入操作（子类可调用）"""
        cost = shares * price
        fee = self._real_fee_buy(cost, date) if self.real_cost else max(cost * 0.0002, 5.0)
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
        if self.real_cost:
            fee, tax = self._real_fee_sell(revenue, date)
        else:
            fee = max(revenue * 0.0002, 5.0)   # 万分之二，最低5元
            tax = revenue * 0.001    # 印花税（A股卖出收，固定0.1%，历史口径）
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
        # 主库 daily 表无 adj_open/adj_close 列，用 adj_factor 现场前复权补出
        df = pd.read_sql(
            """
            SELECT d.trade_date, d.open, d.high, d.low, d.close, d.vol,
                   d.close * f.adj_factor / m.maxf AS adj_close,
                   d.open  * f.adj_factor / m.maxf AS adj_open
            FROM daily d
            LEFT JOIN adj_factor f
                   ON d.ts_code = f.ts_code AND d.trade_date = f.trade_date
            LEFT JOIN (SELECT ts_code, MAX(adj_factor) AS maxf
                       FROM adj_factor GROUP BY ts_code) m
                   ON d.ts_code = m.ts_code
            WHERE d.ts_code = ? AND d.trade_date BETWEEN ? AND ?
            ORDER BY d.trade_date
            """,
            conn,
            params=(ts_code, start_date, end_date),
        )
        conn.close()
        return df
