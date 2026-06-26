"""
RSI趋势策略 - 插件化版本（继承BaseStrategy）

策略逻辑：
- RSI上穿50 → 买入（趋势由空转多）
- RSI下穿50 → 卖出（趋势由多转空）
- 止盈 +50%，止损 -15%
"""
from __future__ import annotations

import pandas as pd
import talib as ta
from loguru import logger

from backtest.base_strategy import BaseStrategy
from backtest.atr_stop_loss import ATRStopLoss


class RSITrendPlugin(BaseStrategy):
    """
    RSI趋势跟踪策略（插件化版本）

    使用方法：
    1. 在 config.py 添加配置
    2. 系统自动发现并加载此策略
    3. 无需修改 run_backtest.py！
    """

    def __init__(self, capital: float, cfg: dict):
        """
        Args:
            capital: 初始资金
            cfg: 策略配置（从 config.py 的 STRATEGIES["rsi_trend"] 读取）
        """
        super().__init__(
            name=cfg.get("name", "RSI趋势策略"),
            capital=capital,
            cfg=cfg,
        )

        # ── 从配置解析参数（带默认值）─────────────────
        self.rsi_period = int(cfg.get("rsi_period", 14))
        self.rsi_center = float(cfg.get("rsi_center", 50))
        self.take_profit = float(cfg.get("take_profit", 0.50))
        self.stop_loss = float(cfg.get("stop_loss", 0.15))
        self.position_mode = cfg.get("position_mode", "half")  # half | full

        # ── ATR 动态止损 ──
        self.use_atr_sl = cfg.get("atr_stop_loss", False)
        self.atr_sl = ATRStopLoss(
            atr_period=cfg.get("atr_period", 14),
            atr_mult=cfg.get("atr_mult", 3.0),
            trail_mult=cfg.get("trail_mult", 3.0),
        )

        # ── 内部状态 ─────────────────────────────
        self.position = 0
        self.avg_cost = 0.0
        self.daily_values = []
        self.trades = []

        logger.info(
            "RSITrendPlugin initialized: period={}, center={}, TP={}, SL={}",
            self.rsi_period,
            self.rsi_center,
            self.take_profit,
            self.stop_loss,
        )

    def _calculate_rsi(self, prices: pd.Series) -> pd.Series:
        """计算RSI指标（使用TA-Lib）"""
        rsi = ta.RSI(prices.values, timeperiod=self.rsi_period)
        return pd.Series(rsi, index=prices.index)

    def run(self, df: pd.DataFrame, start_idx: int = 0) -> dict:
        """
        运行策略（必须实现的方法）

        Args:
            df: 股票数据，必须包含列：
                - trade_date, adj_open, adj_close
            start_idx: 回测起始位置（跳过此前数据）

        Returns:
            {
                "returns": float,      # 收益率（%）
                "trades": list,       # 交易记录
                "daily_values": list,  # 每日资产值
            }
        """
        logger.info("Running RSITrendPlugin on {} days of data...", len(df))

        # ── 初始化 ──────────────────────────────────
        self.daily_values = []
        self.trades = []
        self.cash = self.capital
        self.position = 0
        self.avg_cost = 0.0

        if df is None or len(df) == 0:
            return {"returns": 0.0, "trades": [], "daily_values": []}

        data = df.copy()

        # ── 列名映射（兼容有无复权列）─────────
        if "adj_close" not in data.columns:
            if "close" in data.columns:
                data["adj_close"] = data["close"]
            else:
                raise KeyError("缺少必要列: close / adj_close")

        if "adj_open" not in data.columns:
            if "open" in data.columns:
                data["adj_open"] = data["open"]
            else:
                raise KeyError("缺少必要列: open / adj_open")

        # 确保有 trade_date 列
        if "trade_date" not in data.columns:
            raise KeyError("缺少必要列: trade_date")
        data = data.sort_values("trade_date").reset_index(drop=True)

        # ── 计算RSI ───────────────────────────────
        data["rsi"] = self._calculate_rsi(data["adj_close"])

        # ── 生成信号 ───────────────────────────────
        rsi_y = data["rsi"].shift(1)
        rsi_yy = data["rsi"].shift(2)
        data["buy_signal"] = (rsi_y > self.rsi_center) & (rsi_yy <= self.rsi_center)
        data["sell_signal"] = (rsi_y < self.rsi_center) & (rsi_yy >= self.rsi_center)

        # ── ATR 计算 ──
        if self.use_atr_sl:
            high_col = 'high' if 'high' in data.columns else 'adj_close'
            low_col = 'low' if 'low' in data.columns else 'adj_close'
            atr_arr = self.atr_sl.calc_atr(data[high_col].values, data[low_col].values, data['adj_close'].values)
        else:
            atr_arr = np.zeros(len(data))

        # ── 主循环 ──────────────────────────────────
        n = len(data)
        loop_start = max(start_idx, self.rsi_period + 2)

        for i in range(n):
            if i < loop_start:
                # 跳过起始数据（不交易）
                if self.position > 0:
                    current_value = self.cash + self.position * data["adj_close"].iloc[i]
                else:
                    current_value = self.cash
                self.daily_values.append({
                    "date": data["trade_date"].iloc[i],
                    "portfolio_value": current_value,
                })
                continue

            row = data.iloc[i]
            date = row["trade_date"]
            open_price = row["adj_open"]
            close_price = row["adj_close"]
            prev_close = data["adj_close"].iloc[i - 1] if i > 0 else close_price

            # ── 止损止盈 + ATR追踪止损 ──
            if self.position > 0:
                # ── ATR 追踪止损 ──
                if self.use_atr_sl:
                    high_val = float(row.get('high', close_price))
                    self.atr_sl.update(high_price=high_val, atr_val=atr_arr[i])
                    should_stop, stop_price, atr_reason = self.atr_sl.check_stop(close_price=close_price)
                    if should_stop:
                        self._sell(date, open_price, reason=atr_reason)
                        continue

                pnl_pct = (prev_close / self.avg_cost) - 1

                # 止盈
                if pnl_pct >= self.take_profit:
                    self._sell(date, open_price, reason=f"止盈 {pnl_pct:.1%}")
                    continue

                # 止损（仅非ATR模式）
                if not self.use_atr_sl and pnl_pct <= -self.stop_loss:
                    self._sell(date, open_price, reason=f"止损 {pnl_pct:.1%}")
                    continue

            # ── 买入信号 ─────────────────────────────
            if self.position == 0 and row["buy_signal"]:
                self._buy(date, open_price)
                # ATR初始化（仅当买入成功时）
                if self.use_atr_sl and self.position > 0:
                    self.atr_sl.on_entry(entry_price=self.avg_cost, atr_val=atr_arr[i])
                continue

            # ── 卖出信号 ─────────────────────────────
            if self.position > 0 and row["sell_signal"]:
                self._sell(date, open_price, reason="RSI下穿中心")
                continue

            # ── 记录每日资产值 ─────────────────────
            if self.position > 0:
                current_value = self.cash + self.position * close_price
            else:
                current_value = self.cash

            self.daily_values.append({
                "date": date,
                "portfolio_value": current_value,
            })

        # ── 强制平仓（回测结束时还有持仓）─────────
        if self.position > 0:
            last_row = data.iloc[-1]
            self._sell(
                last_row["trade_date"],
                last_row["adj_open"],
                reason="回测结束平仓"
            )

        # ── 计算收益率 ─────────────────────────────
        returns = self.calc_returns()

        logger.info(f"RSITrendPlugin finished: returns={returns:.2f}%, trades={len(self.trades)}")

        return {
            "returns": returns,
            "trades": self.trades,
            "daily_values": self.daily_values,
        }

    def _buy(self, date, price):
        """买入操作（复用基类 buy()，含手续费计算）"""
        if self.position > 0:
            return  # 已持仓，不再买入

        # 仓位管理
        if self.position_mode == "half":
            invest = self.cash * 0.5
        else:
            invest = self.cash * 0.95  # 留5%现金

        shares = int(invest / price / 100) * 100  # 整百股
        if shares <= 0:
            return

        self.buy(date, price, shares, "RSI上穿中心")

    def _sell(self, date, price, reason=""):
        """卖出操作（复用基类 sell()，含手续费+印花税）"""
        if self.position == 0:
            return

        self.sell(date, price, reason=reason)
