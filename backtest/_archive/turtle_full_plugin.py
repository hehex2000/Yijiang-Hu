"""
海龟策略插件（完整版 - Turtle Strategy Full）
==============================================
实现原版海龟交易法则：
1. 双周期系统（20日短期 + 55日长期）
2. ATR动态仓位计算
3. 金字塔加仓（最多4次，0.5×ATR步长）
4. ATR动态止损（4×ATR）
5. ATR追踪止损（4×ATR）
6. 趋势过滤（价格 > MA200）
7. 成交量过滤（突破日放量）

使用方法：
1. 将此文件放到 backtest/turtle_full_plugin.py
2. 在 config.py 添加 STRATEGIES["turtle_full"] = {...}
3. 运行 python run_backtest.py（自动发现插件）
"""

import pandas as pd
import numpy as np
import talib as ta  # ← 新增 TA-Lib
from backtest.base_strategy import BaseStrategy
from loguru import logger


class TurtleFullStrategyPlugin(BaseStrategy):
    """海龟策略（完整版 - 双周期+ATR动态风控）"""
    
    def __init__(self, capital: float, cfg: dict):
        super().__init__("TurtleFullStrategyPlugin", capital, cfg)
        
        # ── 双周期趋势识别（海龟原版）──
        self.short_period = cfg.get("short_period", 20)      # 短期系统：20日高点突破
        self.long_period = cfg.get("long_period", 55)       # 长期系统：55日高点突破
        self.short_exit_period = cfg.get("short_exit_period", 10)   # 短期离场：10日低点跌破
        self.long_exit_period = cfg.get("long_exit_period", 20)    # 长期离场：20日低点跌破
        
        # ── ATR 波动量化 ──
        self.atr_period = cfg.get("atr_period", 14)        # ATR 计算周期
        self.risk_pct = cfg.get("risk_pct", 0.02)         # 单笔风险：总资金的 2%
        self.max_risk_per_day = cfg.get("max_risk_per_day", 0.02)  # 单日最大亏损：总资金的 2%
        self.max_pos_pct = cfg.get("max_pos_pct", 1.0)     # 单品种最大仓位：总资金的 100%
        
        # ── 金字塔加仓（海龟原版）──
        self.max_adds = cfg.get("max_adds", 4)             # 最多加仓次数（海龟原版）
        self.add_step_atr = cfg.get("add_step_atr", 0.5)   # 加仓步长：0.5×ATR
        self.pos_unit_decay = cfg.get("pos_unit_decay", True)  # 加仓单位递减（True=海龟原版）
        
        # ── 动态止损（基于 ATR）──
        self.stop_atr_mult = cfg.get("stop_atr_mult", 4.0)    # 止损距离：4×ATR（与追踪止损一致）
        self.trail_atr_mult = cfg.get("trail_atr_mult", 4.0)  # 追踪止损：4×ATR（放宽，拿住趋势）
        
        # ── A股本土化过滤 ──
        self.trend_filter = cfg.get("trend_filter", False)      # 趋势过滤：价格 > MA200 才做多
        self.trend_ma_period = cfg.get("trend_ma_period", 200)
        self.volume_filter = cfg.get("volume_filter", False)     # 成交量过滤：突破日放量
        self.volume_ma_period = cfg.get("volume_ma_period", 20)
        self.min_listed_days = cfg.get("min_listed_days", 250)  # 上市不足 250 个交易日过滤
        
        # ── 系统选择 ──
        self.use_short_system = cfg.get("use_short_system", True)
        self.use_long_system = cfg.get("use_long_system", True)
        self.system_weight = cfg.get("system_weight", [0.5, 0.5])  # 短期/长期系统资金分配权重
        
        # ── 手续费模型（更真实）──
        self.buy_cost = cfg.get("buy_cost", 1.0012)          # 买入成本系数（含佣金+印花税+滑点）
        self.sell_cost = cfg.get("sell_cost", 0.9985)        # 卖出成本系数
        
        # ── 状态变量 ──
        self.position = 0           # 持仓数量（股）
        self.cash = capital         # 可用资金
        self.avg_cost = 0.0       # 加权平均成本
        self.units = []            # 加仓单位记录 [(price, shares), ...]
        self.n_adds = 0           # 已加仓次数
        self.entry_atr = 0.0      # 入场时的ATR值
        self.stop_price = 0.0      # 当前止损价
        self.trail_price = 0.0     # 追踪最高价（用于追踪止损）
        
        logger.info(
            f"TurtleFullStrategyPlugin initialized: "
            f"short={self.short_period}, long={self.long_period}, "
            f"ATR_period={self.atr_period}, risk_pct={self.risk_pct:.2%}, "
            f"max_adds={self.max_adds}, stop_mult={self.stop_atr_mult}, "
            f"trail_mult={self.trail_atr_mult}, "
            f"trend_filter={self.trend_filter}, volume_filter={self.volume_filter}"
        )
    
    def calc_atr(self, high, low, close, period=14):
        """计算 ATR（使用 TA-Lib 优化）"""
        # TA-Lib ATR 返回 numpy ndarray，前 (period-1) 个值为 NaN
        atr = ta.ATR(high, low, close, timeperiod=period)
        return atr
    
    def run(self, df: pd.DataFrame, start_idx: int = 0) -> dict:
        """
        运行海龟策略（完整版）
        返回: {"returns": float, "trades": list}
        """
        logger.info(f"Running TurtleFullStrategyPlugin on {len(df)} days of data...")
        self.trades = []
        self.daily_values = []
        self.position = 0
        self.cash = self.capital
        self.avg_cost = 0.0
        self.units = []
        self.n_adds = 0
        self.entry_atr = 0.0
        self.stop_price = 0.0
        self.trail_price = 0.0
        
        if df is None or len(df) == 0:
            return {"returns": 0.0, "trades": [], "daily_values": []}
        
        # ── 数据检查 ──────────────────────────────────
        # 列名标准化（兼容中英文）
        col_map = {"开盘": "open", "最高": "high", "最低": "low", "收盘": "close", "成交量": "volume"}
        df = df.rename(columns=col_map)
        
        req_cols = ["open", "high", "low", "close"]
        for c in req_cols:
            if c not in df.columns:
                logger.error(f"Missing required column: {c}")
                return {"returns": 0.0, "trades": [], "daily_values": []}
        
        # 确保数值列为 float
        for c in req_cols + (["volume"] if "volume" in df.columns else []):
            df[c] = df[c].astype(float)
        
        # 按日期排序
        df = df.sort_values("trade_date").reset_index(drop=True)
        
        # ── 计算指标 ──────────────────────────────────
        n = len(df)
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        open_price = df["open"].values
        
        # 短期通道（20日高点/10日低点）
        short_high = df["close"].rolling(self.short_period).max().shift(1).values
        short_low = df["close"].rolling(self.short_exit_period).min().shift(1).values
        
        # 长期通道（55日高点/20日低点）
        long_high = df["close"].rolling(self.long_period).max().shift(1).values
        long_low = df["close"].rolling(self.long_exit_period).min().shift(1).values
        
        # ATR
        atr = self.calc_atr(high, low, close, self.atr_period)
        
        # 趋势过滤：MA200
        if self.trend_filter:
            ma200 = df["close"].rolling(self.trend_ma_period).mean().values
        else:
            ma200 = np.full(n, np.nan)
        
        # 成交量过滤
        if self.volume_filter and "volume" in df.columns:
            vol = df["volume"].values
            vol_ma = df["volume"].rolling(self.volume_ma_period).mean().values
        else:
            vol = np.full(n, np.nan)
            vol_ma = np.full(n, np.nan)
        
        # ── 回测循环 ──────────────────────────────────
        # 注意：信号用今日收盘计算，但实际买入用次日开盘价（避免未来函数）
        for i in range(n):
            if i < start_idx:
                # 记录每日价值（未进入回测区间）
                v = self.cash + self.position * close[i] if self.position > 0 else self.cash
                self.daily_values.append({"date": df.iloc[i]["trade_date"], "portfolio_value": v})
                continue
            
            p_close = close[i]
            atr_val = atr[i] if i < len(atr) and not np.isnan(atr[i]) else 0.0
            
            # ── 更新追踪止损价 ──────────────────────────────
            if self.position > 0:
                if p_close > self.trail_price or self.trail_price == 0.0:
                    self.trail_price = p_close
                
                # 追踪止损（收盘价跌破 trail_price - trail_atr_mult * ATR）
                if atr_val > 0:
                    trail_stop = self.trail_price - self.trail_atr_mult * atr_val
                    if p_close < trail_stop:
                        # 用次日开盘价卖出（更真实）
                        if i + 1 < n:
                            self.sell(df.iloc[i+1]["trade_date"], open_price[i+1], reason=f"追踪止损({p_close:.2f} < {trail_stop:.2f})")
                        else:
                            self.sell(df.iloc[i]["trade_date"], p_close, reason=f"追踪止损({p_close:.2f} < {trail_stop:.2f})")
                        continue
            
            # ── 检查止损 ──────────────────────────────
            if self.position > 0 and atr_val > 0:
                if p_close < self.stop_price:
                    # 用次日开盘价卖出（更真实）
                    if i + 1 < n:
                        self.sell(df.iloc[i+1]["trade_date"], open_price[i+1], reason=f"ATR止损({p_close:.2f} < {self.stop_price:.2f})")
                    else:
                        self.sell(df.iloc[i]["trade_date"], p_close, reason=f"ATR止损({p_close:.2f} < {self.stop_price:.2f})")
                    continue
            
            # ── 短期系统：20日高点突破买入 ───────────────────
            if self.use_short_system and self.position == 0 and i >= self.short_period:
                # 突破条件：昨日收盘 ≤ 20日高点，今日收盘 > 20日高点
                prev_close = close[i-1] if i > 0 else p_close
                if prev_close <= short_high[i-1] and p_close > short_high[i-1]:
                    # 趋势过滤
                    if self.trend_filter and not np.isnan(ma200[i]) and p_close < ma200[i]:
                        continue  # 价格低于MA200，不做多
                    
                    # 成交量过滤
                    if self.volume_filter and not np.isnan(vol[i]) and not np.isnan(vol_ma[i]):
                        if vol[i] < vol_ma[i] * 1.2:
                            continue  # 突破日未放量，跳过
                    
                    # 计算仓位（ATR动态仓位）
                    if atr_val > 0:
                        risk_amount = self.capital * self.risk_pct
                        unit_shares = int(risk_amount / (atr_val * 2))  # 简化：每单位风险 = 2×ATR
                        unit_shares = (unit_shares // 100) * 100  # 整百股
                        
                        if unit_shares > 0 and self.cash > 0:
                            # 用次日开盘价买入（避免未来函数）
                            if i + 1 < n:
                                buy_amount = min(unit_shares * open_price[i+1], self.cash * 0.5)  # 首次最多用50%资金
                                shares = int(buy_amount / open_price[i+1] / 100) * 100
                                if shares > 0:
                                    self.buy(df.iloc[i+1]["trade_date"], open_price[i+1], shares, reason=f"短期突破({self.short_period}日)")
                                    self.entry_atr = atr_val
                                    self.stop_price = p_close - self.stop_atr_mult * atr_val
                                    self.trail_price = p_close
                            else:
                                # 最后一天，无法买入
                                continue
            
            # ── 长期系统：55日高点突破买入 ───────────────────
            if self.use_long_system and self.position == 0 and i >= self.long_period:
                prev_close = close[i-1] if i > 0 else p_close
                if prev_close <= long_high[i-1] and p_close > long_high[i-1]:
                    # 趋势过滤
                    if self.trend_filter and not np.isnan(ma200[i]) and p_close < ma200[i]:
                        continue
                    
                    # 成交量过滤
                    if self.volume_filter and not np.isnan(vol[i]) and not np.isnan(vol_ma[i]):
                        if vol[i] < vol_ma[i] * 1.2:
                            continue
                    
                    # 计算仓位
                    if atr_val > 0:
                        risk_amount = self.capital * self.risk_pct
                        unit_shares = int(risk_amount / (atr_val * 2))
                        unit_shares = (unit_shares // 100) * 100
                        
                        if unit_shares > 0 and self.cash > 0:
                            # 用次日开盘价买入（避免未来函数）
                            if i + 1 < n:
                                buy_amount = min(unit_shares * open_price[i+1], self.cash * 0.5)
                                shares = int(buy_amount / open_price[i+1] / 100) * 100
                                if shares > 0:
                                    self.buy(df.iloc[i+1]["trade_date"], open_price[i+1], shares, reason=f"长期突破({self.long_period}日)")
                                    self.entry_atr = atr_val
                                    self.stop_price = p_close - self.stop_atr_mult * atr_val
                                    self.trail_price = p_close
                            else:
                                # 最后一天，无法买入
                                continue
            
            # ── 金字塔加仓 ────────────────────────────────────
            if self.position > 0 and self.n_adds < self.max_adds and atr_val > 0:
                # 价格比上次买入价上涨了 0.5×ATR
                if p_close >= self.avg_cost + self.add_step_atr * atr_val:
                    # 加仓单位递减（海龟原版）
                    if self.pos_unit_decay:
                        decay_factor = 1.0 / (self.n_adds + 1)
                    else:
                        decay_factor = 1.0
                    
                    risk_amount = self.capital * self.risk_pct * decay_factor
                    add_shares = int(risk_amount / (atr_val * 2))
                    add_shares = (add_shares // 100) * 100
                    
                    if add_shares > 0 and self.cash > add_shares * open_price[i]:
                        # 用次日开盘价加仓（避免未来函数）
                        if i + 1 < n:
                            self.buy(df.iloc[i+1]["trade_date"], open_price[i+1], add_shares, reason=f"金字塔加仓({self.n_adds+1}/{self.max_adds})")
                            self.n_adds += 1
                            # 重置止损价（加仓后重新计算）
                            self.stop_price = p_close - self.stop_atr_mult * atr_val
                        else:
                            # 最后一天，无法加仓
                            continue
            
            # ── 离场信号 ──────────────────────────────────────
            if self.position > 0:
                # 短期离场：10日低点跌破
                if self.use_short_system and i >= self.short_exit_period:
                    prev_close = close[i-1] if i > 0 else p_close
                    if prev_close >= short_low[i-1] and p_close < short_low[i]:
                        # 用次日开盘价卖出（更真实）
                        if i + 1 < n:
                            self.sell(df.iloc[i+1]["trade_date"], open_price[i+1], reason=f"短期离场({self.short_exit_period}日低点)")
                        else:
                            self.sell(df.iloc[i]["trade_date"], p_close, reason=f"短期离场({self.short_exit_period}日低点)")
                        continue
                
                # 长期离场：20日低点跌破
                if self.use_long_system and i >= self.long_exit_period:
                    prev_close = close[i-1] if i > 0 else p_close
                    if prev_close >= long_low[i-1] and p_close < long_low[i]:
                        # 用次日开盘价卖出（更真实）
                        if i + 1 < n:
                            self.sell(df.iloc[i+1]["trade_date"], open_price[i+1], reason=f"长期离场({self.long_exit_period}日低点)")
                        else:
                            self.sell(df.iloc[i]["trade_date"], p_close, reason=f"长期离场({self.long_exit_period}日低点)")
                        continue
            
            # 记录每日价值
            v = self.cash + self.position * p_close if self.position > 0 else self.cash
            self.daily_values.append({"date": df.iloc[i]["trade_date"], "portfolio_value": v})
        
        # ── 回测结束，平仓 ──────────────────────────────────
        if self.position > 0:
            last_price = close[-1]
            self.sell(df.iloc[-1]["trade_date"], last_price, reason="回测结束平仓")
        
        ret = self.calc_returns()
        logger.info(f"TurtleFullStrategyPlugin finished: returns={ret:.2f}%, trades={len(self.trades)}")
        return {"returns": ret, "trades": self.trades, "daily_values": self.daily_values}
