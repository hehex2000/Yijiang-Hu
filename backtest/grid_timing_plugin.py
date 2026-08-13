"""
网格择时策略 - 插件化版本（继承BaseStrategy）

适配个股场景的网格策略：
- 固定中枢：基于回测起始价格，定期（60天）更新（避免下跌市中网格下移）
- 等比网格：3%间距（个股波动大）
- 强趋势保护：价格>MA50才开仓（避免熊市接飞刀）
- 单档买卖：每档独立跟踪，买了哪档就卖哪档
- 总止损：从峰值回撤8%强制清仓
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.base_strategy import BaseStrategy


class GridTimingPlugin(BaseStrategy):
    """
    网格择时策略（固定中枢+趋势保护版）
    """

    def __init__(self, capital: float, cfg: dict):
        super().__init__(
            name=cfg.get("name", "网格择时策略"),
            capital=capital,
            cfg=cfg,
        )

        # ── 网格参数 ──
        self.grid_pct = float(cfg.get("grid_pct", 0.03))          # 每档3%
        self.grid_levels = int(cfg.get("grid_levels", 4))         # 上下各4档
        self.center_update_days = int(cfg.get("center_update_days", 60))  # 中枢更新周期

        # ── 仓位管理 ──
        self.invest_ratio = float(cfg.get("invest_ratio", 0.6))   # 60%资金用于网格

        # ── 风控参数 ──
        self.total_stop_loss = float(cfg.get("total_stop_loss", 0.08))  # 总止损8%

        # ── 状态 ──
        self.position = 0
        self.avg_cost = 0.0
        self.daily_values = []
        self.trades = []

    def run(self, df: pd.DataFrame, start_idx: int = 0) -> dict:
        """运行策略"""
        self.daily_values = []
        self.trades = []
        self.cash = self.capital
        self.position = 0
        self.avg_cost = 0.0

        if df is None or len(df) == 0:
            return {"returns": 0.0, "trades": [], "daily_values": []}

        data = df.copy()

        # ── 列名映射 ──
        if "adj_close" not in data.columns:
            data["adj_close"] = data.get("close", pd.Series(dtype=float))
        if "adj_open" not in data.columns:
            data["adj_open"] = data.get("open", data["adj_close"])
        if "adj_high" not in data.columns:
            data["adj_high"] = data.get("high", data["adj_close"])
        if "adj_low" not in data.columns:
            data["adj_low"] = data.get("low", data["adj_close"])

        if "trade_date" not in data.columns:
            raise KeyError("缺少必要列: trade_date")
        data = data.sort_values("trade_date").reset_index(drop=True)

        close = data["adj_close"].values
        high = data["adj_high"].values
        low = data["adj_low"].values
        open_p = data["adj_open"].values
        n = len(close)

        # ── MA50趋势过滤 ──
        ma50 = pd.Series(close).rolling(window=50, min_periods=1).mean().values

        # ── 初始化 ──
        loop_start = max(start_idx, 50)
        center_price = close[loop_start]  # 固定中枢（起始价格）
        last_center_update = loop_start

        # 每档的状态：{'shares': 0, 'buy_price': 0.0, 'active': False}
        lots = [{'shares': 0, 'buy_price': 0.0, 'active': False} for _ in range(self.grid_levels)]

        peak_value = self.capital
        total_stop_triggered = False

        # ── 主循环 ──
        for i in range(n):
            if i < loop_start:
                cv = self.cash + self.position * close[i] if self.position > 0 else self.cash
                self.daily_values.append({"date": data["trade_date"].iloc[i], "portfolio_value": cv})
                continue

            date = data["trade_date"].iloc[i]
            co = open_p[i]
            cc = close[i]
            ch = high[i]
            cl = low[i]

            # ── 总资产 & 峰值 ──
            total_value = self.cash + self.position * cc
            if total_value > peak_value:
                peak_value = total_value

            # ── 总止损（从峰值回撤8%）──
            if not total_stop_triggered:
                dd = (peak_value - total_value) / peak_value
                if dd >= self.total_stop_loss:
                    if self.position > 0:
                        self._sell_all(date, co, f"总止损{dd:.1%}")
                    total_stop_triggered = True

            if total_stop_triggered:
                self.daily_values.append({"date": date, "portfolio_value": self.cash})
                continue

            # ── 中枢定期更新（每60天，且只在趋势向上时）──
            if i - last_center_update >= self.center_update_days:
                if cc > ma50[i]:  # 只在上升趋势中更新中枢
                    center_price = cc
                last_center_update = i

            # ── 计算网格线 ──
            buy_levels = [center_price * (1 - self.grid_pct * (j + 1)) for j in range(self.grid_levels)]
            sell_levels = [center_price * (1 + self.grid_pct * (j + 1)) for j in range(self.grid_levels)]

            # ── 趋势保护：价格>MA50才开新仓 ──
            trend_bullish = cc > ma50[i]

            # ── 买入：价格触及下方网格 + 趋势允许 ──
            if trend_bullish and self.position < self.capital * self.invest_ratio / cc:
                for j in range(self.grid_levels):
                    if not lots[j]['active'] and cl <= buy_levels[j]:
                        # 计算这档的买入金额（均分）
                        lot_amount = self.capital * self.invest_ratio / self.grid_levels
                        shares = int(lot_amount / co / 100) * 100
                        # 凯利总持仓封顶（自管现金策略用 kelly_room_shares 缩放；
                        # 多档累计到上限后自动停买，保留网格档位结构）
                        shares = self.kelly_room_shares(co, shares)
                        if shares < 100:
                            continue
                        cost = shares * co
                        fee = max(cost * 0.00025, 5.0) + cost * 0.001  # 佣金+滑点（买入无印花税）
                        if cost + fee > self.cash:
                            continue
                        self.cash -= (cost + fee)
                        total_cost = self.avg_cost * self.position + cost
                        self.position += shares
                        self.avg_cost = total_cost / self.position
                        lots[j] = {'shares': shares, 'buy_price': co, 'active': True}
                        self.trades.append({
                            "date": date, "action": "BUY", "price": co,
                            "shares": shares, "cost": cost + fee,
                            "reason": f"网格L{j+1}买入"
                        })
                        break  # 每天最多买一档

            # ── 卖出：价格触及上方网格 ──
            if self.position > 0:
                for j in range(self.grid_levels):
                    if ch >= sell_levels[j]:
                        # 卖出最早买入的那一档（FIFO）
                        for k in range(self.grid_levels):
                            if lots[k]['active']:
                                shares = lots[k]['shares']
                                if shares > self.position:
                                    shares = self.position
                                revenue = shares * co
                                fee = max(revenue * 0.00025, 5.0) + revenue * 0.001 + revenue * 0.001  # 佣金+印花税+滑点
                                self.cash += (revenue - fee)
                                self.position -= shares
                                if self.position <= 0:
                                    self.position = 0
                                    self.avg_cost = 0.0
                                lots[k] = {'shares': 0, 'buy_price': 0.0, 'active': False}
                                self.trades.append({
                                    "date": date, "action": "SELL", "price": co,
                                    "shares": shares, "cost": revenue - fee,
                                    "reason": f"网格L{j+1}卖出"
                                })
                                break
                        break  # 每天最多卖一档

            cv = self.cash + self.position * cc
            self.daily_values.append({"date": date, "portfolio_value": cv})

        # ── 强制平仓 ──
        if self.position > 0:
            last_row = data.iloc[-1]
            revenue = self.position * last_row["adj_open"]
            fee = max(revenue * 0.00025, 5.0) + revenue * 0.001 + revenue * 0.001
            self.cash += (revenue - fee)
            self.trades.append({
                "date": last_row["trade_date"], "action": "SELL",
                "price": last_row["adj_open"], "shares": self.position,
                "cost": revenue - fee, "reason": "回测结束平仓"
            })
            self.position = 0

        returns = self.calc_returns()
        return {"returns": returns, "trades": self.trades, "daily_values": self.daily_values}

    def _sell_all(self, date, price, reason=""):
        """全部卖出"""
        if self.position == 0:
            return
        revenue = self.position * price
        fee = max(revenue * 0.00025, 5.0) + revenue * 0.001 + revenue * 0.001
        self.cash += (revenue - fee)
        self.trades.append({
            "date": date, "action": "SELL", "price": price,
            "shares": self.position, "cost": revenue - fee, "reason": reason
        })
        self.position = 0
        self.avg_cost = 0.0
