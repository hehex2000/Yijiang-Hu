"""
动态网格策略插件（VR 驱动）
继承 BaseStrategy，符合回测平台插件接口

策略逻辑：
- 初始建立 N 层买入挂单（价格向下台阶分布）
- 价格回落到挂单价位 → 买入
- 持仓后，价格达到（买入价 × (1 + 当前网格间距)）→ 卖出
- 网格间距由 VR 指标动态决定：
  - VR < 80：3%（低活跃区，高频捕捉小波动）
  - 80 ≤ VR ≤ 120：5%（正常市场）
  - VR > 120：8%（高活跃区，防止过早卖飞）
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from loguru import logger
from typing import List, Dict, Optional

from backtest.base_strategy import BaseStrategy
from backtest.energy_indicators import (
    calculate_vr,
    calculate_all_energy_indicators,
)


class EnergyGridPlugin(BaseStrategy):
    """
    动态网格策略（插件化版本，VR 驱动）

    使用方法：
    1. 在 config.py 添加 STRATEGIES["energy_grid"] 配置
    2. 系统自动发现并加载此策略
    3. 无需修改 run_backtest.py！
    """

    def __init__(self, capital: float, cfg: dict):
        """
        Args:
            capital: 初始资金
            cfg: 策略配置（从 config.py 的 STRATEGIES["energy_grid"] 读取）
        """
        super().__init__(
            name=cfg.get("name", "动态网格策略(VR驱动)"),
            capital=capital,
            cfg=cfg,
        )

        # ── 从配置解析参数（带默认值）─────────────────
        self.grid_levels = int(cfg.get("grid_levels", 10))
        self.base_grid_pct = float(cfg.get("base_grid_pct", 0.05))
        self.low_vr_grid_pct = float(cfg.get("low_vr_grid_pct", 0.03))
        self.high_vr_grid_pct = float(cfg.get("high_vr_grid_pct", 0.08))
        self.vr_period = int(cfg.get("vr_period", 26))
        self.position_per_grid = float(cfg.get("position_per_grid", 0.1))

        # ── 内部状态 ─────────────────────────────
        self.lots: List[Dict] = []  # 网格层状态
        self.current_grid_pct = self.base_grid_pct  # 当前网格间距

        # 校验资金分配比例
        total_allocation = self.grid_levels * self.position_per_grid
        if total_allocation > 1.0:
            logger.warning(
                f"EnergyGridPlugin: 网格总资金需求 ({total_allocation:.0%}) 超过总资金 (100%)！"
                f"调整 position_per_grid 或减少 grid_levels。"
            )

        logger.info(
            "EnergyGridPlugin initialized: levels={}, base_pct={}, "
            "low_vr_pct={}, high_vr_pct={}, vr_period={}",
            self.grid_levels,
            self.base_grid_pct,
            self.low_vr_grid_pct,
            self.high_vr_grid_pct,
            self.vr_period,
        )

    def _setup_grid(self, current_price: float):
        """
        基于当前价格初始化网格

        生成 grid_levels 层买入挂单（价格向下），等待价格回落到这些位置。
        卖出目标由各层实际买入价 + 当前 VR 间距动态计算。

        Args:
            current_price: 当前参考价格
        """
        self.lots = []

        # 所有网格层均为买入挂单，卖出目标由各层实际买入价 + 当前 VR 间距动态计算
        for i in range(self.grid_levels):
            buy_price = current_price * (1 - self.current_grid_pct * (i + 1))
            self.lots.append({
                'buy_price': buy_price,
                'shares': 0,
                'total_cost': 0.0,
                'grid_level': -(i + 1),
                'status': 'pending'  # pending / active / closed
            })

        logger.debug(f"_setup_grid: {len(self.lots)} levels, pct={self.current_grid_pct:.2%}")

    def _get_grid_target(self, buy_price: float, grid_pct: float) -> float:
        """计算网格层的卖出目标价"""
        return buy_price * (1 + grid_pct)

    def _update_vr_grid(self, vr_value: Optional[float]):
        """根据 VR 值更新当前网格间距"""
        if vr_value is None:
            return

        old_pct = self.current_grid_pct

        if vr_value < 80:
            self.current_grid_pct = self.low_vr_grid_pct
        elif vr_value > 120:
            self.current_grid_pct = self.high_vr_grid_pct
        else:
            self.current_grid_pct = self.base_grid_pct

        if abs(old_pct - self.current_grid_pct) > 0.001:
            logger.debug(f"VR={vr_value:.1f} → 网格间距 {old_pct:.2%} → {self.current_grid_pct:.2%}")

    def run(self, df: pd.DataFrame, start_idx: int = 0) -> dict:
        """
        运行动态网格策略

        Args:
            df: 股票数据 DataFrame（需包含 adj_close, volume 列）
            start_idx: 回测起始位置（跳过此前数据）

        Returns:
            {
                "returns": float,      # 收益率（%）
                "trades": list,       # 交易记录
                "daily_values": list,  # 每日资产值
            }
        """
        logger.info(f"Running EnergyGridPlugin on {len(df)} days of data...")

        # ── 初始化 ──────────────────────────────────
        self.trades = []
        self.daily_values = []
        self.position = 0  # 总持仓（所有网格层之和）
        self.avg_cost = 0.0
        self.cash = self.capital
        self.lots = []
        self.current_grid_pct = self.base_grid_pct

        if len(df) == 0:
            return {"returns": 0.0, "trades": [], "daily_values": []}

        data = df.copy()

        # ── 列名映射（兼容有无复权列，但保留已有的复权列）─────────
        existing_cols = set(data.columns)
        col_map = {
            "收盘": "adj_close", "close": "adj_close",
            "开盘": "adj_open", "open": "adj_open",
            "最高": "adj_high", "high": "adj_high",
            "最低": "adj_low", "low": "adj_low",
        }
        col_map = {k: v for k, v in col_map.items() 
                   if k in existing_cols and v not in existing_cols}
        data = data.rename(columns=col_map)

        if "adj_close" not in data.columns:
            logger.error("EnergyGridPlugin: 缺少 adj_close 列")
            return {"returns": 0.0, "trades": [], "daily_values": []}

        if "volume" not in data.columns and "vol" in data.columns:
            data["volume"] = data["vol"]

        data = data.sort_values("trade_date").reset_index(drop=True)

        # ── 计算 VR 指标 ─────────────────────────────
        min_rows = self.vr_period + 1
        if len(data) < min_rows:
            logger.warning(f"EnergyGridPlugin: 数据不足 {len(data)} < {min_rows}")
            return {"returns": 0.0, "trades": [], "daily_values": []}

        data = calculate_all_energy_indicators(data, n=self.vr_period)

        # ── 初始化网格 ─────────────────────────────
        first_price = data["adj_close"].iloc[max(start_idx, min_rows)]
        self._setup_grid(first_price)

        # ── 主循环 ──────────────────────────────────
        total_rows = len(data)
        loop_start = max(start_idx, min_rows)

        for i in range(total_rows):
            row = data.iloc[i]
            date = row["trade_date"]
            close_price = row["adj_close"]

            # ── 跳过起始数据（不交易）─────────
            if i < loop_start:
                current_value = self.cash + self.position * close_price
                self.daily_values.append({
                    "date": date,
                    "portfolio_value": current_value,
                })
                continue

            # ── 更新 VR 网格间距 ─────────────────
            vr_val = row.get("vr", None)
            self._update_vr_grid(vr_val)

            # ── 重建网格（如果 VR 间距发生变化）─────────
            # 简化：不重建，仅用当前间距计算卖出目标
            # （重建会丢失已有持仓，实际中应逐步调整）

            # ── 检查挂单：买入网格层（仅用收盘价判断）─────────
            just_bought = set()  # 本日买入的 lot index，防止同日买卖
            for j, lot in enumerate(self.lots):
                if lot['status'] == 'pending':
                    if close_price <= lot['buy_price']:
                        # 执行买入
                        shares = int(self.cash * self.position_per_grid / close_price / 100) * 100
                        if shares <= 0:
                            continue

                        cost = shares * close_price
                        fee = max(cost * 0.0002, 5.0)
                        total_cost = cost + fee

                        if total_cost > self.cash:
                            continue

                        self.cash -= total_cost
                        self.position += shares
                        lot['shares'] = lot.get('shares', 0) + shares
                        lot['total_cost'] = lot.get('total_cost', 0.0) + cost
                        lot['status'] = 'active'
                        lot['buy_price'] = close_price  # 真实成交价

                        trade = {
                            "date": date,
                            "action": "BUY",
                            "price": close_price,
                            "shares": shares,
                            "cost": total_cost,
                            "reason": f"网格买入 L{lot['grid_level']}",
                        }
                        self.trades.append(trade)
                        just_bought.add(j)

            # ── 检查止盈：活跃网格层（跳过本日刚买入的）─────────
            for j, lot in enumerate(self.lots):
                if lot['status'] == 'active' and j not in just_bought:
                    target_price = self._get_grid_target(lot['buy_price'], self.current_grid_pct)
                    if close_price >= target_price:
                        # 执行卖出（全部持仓）
                        shares = lot['shares']
                        revenue = shares * close_price
                        fee = max(revenue * 0.0002, 5.0)
                        tax = revenue * 0.001
                        net_revenue = revenue - fee - tax

                        self.cash += net_revenue
                        self.position -= shares

                        profit = net_revenue - lot['total_cost']

                        trade = {
                            "date": date,
                            "action": "SELL",
                            "price": close_price,
                            "shares": shares,
                            "revenue": net_revenue,
                            "profit": profit,
                            "reason": f"网格止盈 L{lot['grid_level']}",
                        }
                        self.trades.append(trade)

                        lot['shares'] = 0
                        lot['total_cost'] = 0.0
                        lot['status'] = 'closed'

            # ── 记录每日资产值 ─────────────────────
            current_value = self.cash + self.position * close_price
            self.daily_values.append({
                "date": date,
                "portfolio_value": current_value,
            })

        # ── 强制平仓（回测结束时还有持仓）─────────
        if self.position > 0:
            last_row = data.iloc[-1]
            # 按网格层平仓
            for lot in self.lots:
                if lot['status'] == 'active':
                    shares = lot['shares']
                    price = last_row["adj_open"]
                    revenue = shares * price
                    fee = max(revenue * 0.0002, 5.0)
                    tax = revenue * 0.001
                    net_revenue = revenue - fee - tax

                    self.cash += net_revenue
                    self.position -= shares

                    trade = {
                        "date": last_row["trade_date"],
                        "action": "SELL",
                        "price": price,
                        "shares": shares,
                        "revenue": net_revenue,
                        "reason": "回测结束平仓",
                    }
                    self.trades.append(trade)

        # ── 计算收益率 ─────────────────────────────
        returns = self.calc_returns()

        logger.info(f"EnergyGridPlugin finished: returns={returns:.2f}%, trades={len(self.trades)}")

        return {
            "returns": returns,
            "trades": self.trades,
            "daily_values": self.daily_values,
        }
