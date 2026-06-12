"""
能量指标策略插件（AR/BR/CR/VR 共振）
继承 BaseStrategy，符合回测平台插件接口

策略逻辑：
- AR/BR/CR/VR 同时 < 100 且 BR < AR（底背离）→ 买入
- 任一指标 > 150（非理性繁荣）→ 卖出
- 止盈 +25%，止损 -10%
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from loguru import logger

from backtest.base_strategy import BaseStrategy
from backtest.energy_indicators import (
    calculate_ar, calculate_br, calculate_cr, calculate_vr,
    calculate_all_energy_indicators, generate_energy_signals
)


class EnergyPlugin(BaseStrategy):
    """
    能量指标共振策略（插件化版本）
    
    使用方法：
    1. 在 config.py 添加 STRATEGIES["energy"] 配置
    2. 系统自动发现并加载此策略
    3. 无需修改 run_backtest.py！
    """

    def __init__(self, capital: float, cfg: dict):
        """
        Args:
            capital: 初始资金
            cfg: 策略配置（从 config.py 的 STRATEGIES["energy"] 读取）
        """
        super().__init__(
            name=cfg.get("name", "能量指标策略"),
            capital=capital,
            cfg=cfg,
        )

        # ── 从配置解析参数（带默认值）─────────────────
        self.indicator_period = int(cfg.get("indicator_period", 26))
        self.buy_threshold = float(cfg.get("buy_threshold", 100))
        self.sell_threshold = float(cfg.get("sell_threshold", 150))
        self.position_pct = float(cfg.get("position_pct", 0.5))  # 单次买入仓位比例
        self.take_profit = float(cfg.get("take_profit", 0.25))
        self.stop_loss = float(cfg.get("stop_loss", 0.10))
        
        logger.info(
            "EnergyPlugin initialized: period={}, buy_thr={}, sell_thr={}, "
            "pos_pct={}, TP={}, SL={}",
            self.indicator_period,
            self.buy_threshold,
            self.sell_threshold,
            self.position_pct,
            self.take_profit,
            self.stop_loss,
        )

    def run(self, df: pd.DataFrame, start_idx: int = 0) -> dict:
        """
        运行能量指标策略
        
        Args:
            df: 股票数据 DataFrame（需包含 adj_open, adj_high, adj_low, adj_close, volume 列）
            start_idx: 回测起始位置（跳过此前数据）
        
        Returns:
            {
                "returns": float,      # 收益率（%）
                "trades": list,       # 交易记录
                "daily_values": list,  # 每日资产值
            }
        """
        logger.info(f"Running EnergyPlugin on {len(df)} days of data...")

        # ── 初始化 ──────────────────────────────────
        self.trades = []
        self.daily_values = []
        self.position = 0
        self.avg_cost = 0.0
        self.cash = self.capital

        if len(df) == 0:
            return {"returns": 0.0, "trades": [], "daily_values": []}

        data = df.copy()
        
        # ── 列名映射（兼容有无复权列，但保留已有的复权列）─────────
        # 只映射那些尚未存在的列名，避免覆盖已有复权列
        existing_cols = set(data.columns)
        col_map = {"开盘": "adj_open", "最高": "adj_high", "最低": "adj_low", "收盘": "adj_close",
                   "open": "adj_open", "high": "adj_high", "low": "adj_low", "close": "adj_close"}
        col_map = {k: v for k, v in col_map.items() 
                   if k in existing_cols and v not in existing_cols}
        data = data.rename(columns=col_map)
        
        # 确保有必要的列
        if "adj_close" not in data.columns:
            logger.error("EnergyPlugin: 缺少 adj_close 列")
            return {"returns": 0.0, "trades": [], "daily_values": []}
        
        if "adj_open" not in data.columns:
            data["adj_open"] = data["adj_close"]
        if "adj_high" not in data.columns:
            data["adj_high"] = data["adj_close"]
        if "adj_low" not in data.columns:
            data["adj_low"] = data["adj_close"]
        if "volume" not in data.columns and "vol" in data.columns:
            data["volume"] = data["vol"]
        if "volume" not in data.columns:
            logger.error("EnergyPlugin: 缺少 volume 列（VR指标需要）")
            return {"returns": 0.0, "trades": [], "daily_values": []}

        data = data.sort_values("trade_date").reset_index(drop=True)

        # ── 计算能量指标 ───────────────────────────────
        n = self.indicator_period
        min_rows = n + 1
        
        if len(data) < min_rows:
            logger.warning(f"EnergyPlugin: 数据不足 {len(data)} < {min_rows}")
            return {"returns": 0.0, "trades": [], "daily_values": []}

        data = calculate_all_energy_indicators(data, n=self.indicator_period)
        data = generate_energy_signals(data, n=self.indicator_period)

        # ── 主循环 ──────────────────────────────────
        total_rows = len(data)
        loop_start = max(start_idx, min_rows)

        prev_buy_signal = False
        prev_sell_signal = False

        for i in range(total_rows):
            row = data.iloc[i]
            date = row["trade_date"]
            close_price = row["adj_close"]
            open_price = row["adj_open"]

            # ── 跳过起始数据（不交易）─────────
            if i < loop_start:
                # 记录每日资产值
                if self.position > 0:
                    current_value = self.cash + self.position * close_price
                else:
                    current_value = self.cash
                self.daily_values.append({
                    "date": date,
                    "portfolio_value": current_value,
                })
                continue

            # ── 止损止盈检查（以前一日收盘价判断，T+1 执行）─────
            if self.position > 0:
                prev_close = data["adj_close"].iloc[i - 1] if i > 0 else close_price
                pnl_pct = (prev_close / self.avg_cost) - 1

                # 止盈（T+1 执行）
                if pnl_pct >= self.take_profit:
                    self._sell(date, open_price, reason=f"止盈 {pnl_pct:.1%}")
                    current_value = self.cash
                    self.daily_values.append({"date": date, "portfolio_value": current_value})
                    continue

                # 止损（T+1 执行）
                if pnl_pct <= -self.stop_loss:
                    self._sell(date, open_price, reason=f"止损 {pnl_pct:.1%}")
                    current_value = self.cash
                    self.daily_values.append({"date": date, "portfolio_value": current_value})
                    continue

            # ── 买入信号（T+1 执行）─────────
            if self.position == 0 and row["buy_signal"] and not prev_buy_signal:
                self._buy(date, open_price)
                prev_buy_signal = True
                current_value = self.cash + self.position * close_price
                self.daily_values.append({"date": date, "portfolio_value": current_value})
                continue

            # ── 卖出信号（T+1 执行）─────────
            if self.position > 0 and row["sell_signal"] and not prev_sell_signal:
                self._sell(date, open_price, reason="能量指标过热")
                prev_sell_signal = True
                current_value = self.cash
                self.daily_values.append({"date": date, "portfolio_value": current_value})
                continue

            # ── 重置信号标志 ─────────────────────
            if not row["buy_signal"]:
                prev_buy_signal = False
            if not row["sell_signal"]:
                prev_sell_signal = False

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

        logger.info(f"EnergyPlugin finished: returns={returns:.2f}%, trades={len(self.trades)}")

        return {
            "returns": returns,
            "trades": self.trades,
            "daily_values": self.daily_values,
        }

    def _buy(self, date, price):
        """买入操作（复用基类 buy()，含手续费计算）"""
        if self.position > 0:
            return  # 已持仓，不再买入（单次买入模式）

        # 仓位管理
        invest = self.cash * self.position_pct
        shares = int(invest / price / 100) * 100  # 整百股
        if shares <= 0:
            return

        self.buy(date, price, shares, "能量指标共振买入")

    def _sell(self, date, price, reason=""):
        """卖出操作（复用基类 sell()，含手续费+印花税）"""
        if self.position == 0:
            return

        self.sell(date, price, reason=reason)
