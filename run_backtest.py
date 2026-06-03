# -*- coding: utf-8 -*-
"""
多因子选股 + 回测系统 — 统一入口
==================================
使用方法:
  1. 修改 config.py 配置参数
  2. 运行: python run_backtest.py
"""
import sys, os, sqlite3, glob, argparse
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger

# 简化日志格式（只显示消息，不显示时间/文件/行号）
# 级别=ERROR：屏蔽所有 INFO/WARNING，只显示严重错误
logger.remove()
logger.add(sys.stderr, format="{message}", level="ERROR")


# ════════════════════════════════════════════════════════
# 加载配置
# ════════════════════════════════════════════════════════
from config import (
    DATA, SELECTION, FACTOR_CALCULATOR, FACTOR_PROCESSOR,
    BACKTEST, STRATEGIES, OUTPUT, INDUSTRY_MOMENTUM,
)

DB_PATH = DATA["local_db_path"]


# ══════════════════════════════════════════════════════
# 策略插件自动发现（无需修改此文件即可添加新策略）
# ══════════════════════════════════════════════════════

def load_strategy_plugins():
    """
    自动扫描 backtest/*.py，发现所有 BaseStrategy 子类
    
    使用方法：
    1. 在 backtest/ 创建新文件（如 my_strategy.py）
    2. 定义 class MyStrategy(BaseStrategy):
    3. 实现 run(self, df, start_idx=0) 方法
    4. 在 config.py 添加 STRATEGIES["my_strategy"] = {...}
    5. 运行 run_backtest.py（自动发现，无需修改此文件！）
    
    Returns:
        dict: {config_key: strategy_class}
    """
    import importlib.util
    from pathlib import Path
    
    plugins = {}
    backtest_dir = Path(__file__).parent / "backtest"
    
    if not backtest_dir.exists():
        print("  [WARN] backtest/ 目录不存在")
        return plugins
    
    for py_file in backtest_dir.glob("*.py"):
        if py_file.name.startswith("_") or py_file.name == "base_strategy.py":
            continue  # 跳过私有文件和基类
            
        module_name = py_file.stem  # e.g., "rsi_trend_plugin"
        
        try:
            # 动态导入模块
            spec = importlib.util.spec_from_file_location(module_name, py_file)
            if spec is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # 找到所有 BaseStrategy 子类
            from backtest.base_strategy import BaseStrategy
            
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and 
                    issubclass(attr, BaseStrategy) and 
                    attr != BaseStrategy):
                    # 配置键名 = 文件名（去掉 _plugin 后缀）
                    config_key = module_name.replace("_plugin", "")
                    plugins[config_key] = attr
                    break  # 每个文件一个策略类
                    
        except Exception as e:
            print(f"  [WARN] 无法加载策略插件 {py_file.name}: {e}")
    
    return plugins


def ts_code(c):
    """简单代码 → Tushare格式"""
    return c + (".SH" if c.startswith("6") else ".SZ")


def load_stock_prices(code, start, end, conn, lookback_days=250):
    """
    从本地数据库加载日线行情数据（使用外部传入的连接）
    参数：
        lookback_days: 回溯天数（用于计算MA200等指标）
    """
    # 计算起始日期（减去 lookback_days）
    start_dt = pd.Timestamp(start)
    lookback_start = (start_dt - pd.Timedelta(days=lookback_days)).strftime("%Y%m%d")
    
    df = pd.read_sql_query(
        "SELECT trade_date, open, high, low, close, vol FROM daily WHERE ts_code=? AND trade_date BETWEEN ? AND ? ORDER BY trade_date",
        conn, params=(ts_code(code), lookback_start, end))
    if df.empty:
        return None
    # 确保数值列为 float
    for col in ['open', 'high', 'low', 'close', 'vol']:
        if col in df.columns:
            df[col] = df[col].astype(float)
    # 统一列名：vol → volume（兼容策略代码）
    if 'vol' in df.columns and 'volume' not in df.columns:
        df = df.rename(columns={'vol': 'volume'})
    return df


def load_benchmark(code, start, end, conn):
    """加载基准指数"""
    df = pd.read_sql_query(
        "SELECT trade_date, close FROM index_daily WHERE ts_code=? AND trade_date BETWEEN ? AND ? ORDER BY trade_date",
        conn, params=(code, start, end))
    if df.empty or len(df) < 2:
        return None
    df["close"] = df["close"].astype(float)
    return df


# ════════════════════════════════════════════════════════
# 策略引擎
# ════════════════════════════════════════════════════════

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def calculate_max_drawdown(portfolio_values):
    """
    计算最大回撤
    Max Drawdown = max((历史最高净值 - 当前净值) / 历史最高净值 * 100)
    """
    if not portfolio_values or len(portfolio_values) < 2:
        return 0.0
    
    peak = portfolio_values[0]
    max_dd = 0.0
    
    for val in portfolio_values:
        if val > peak:
            peak = val
        dd = (peak - val) / peak * 100
        if dd > max_dd:
            max_dd = dd
    
    return max_dd


def backtest_buy_hold(df, capital, start_idx=0):
    """买入持有"""
    close = df["close"].values
    n = len(close)
    
    # 买入持有：全部资金在 start_idx 买入，一直持有
    p0 = close[start_idx]
    shares = int(capital / p0 / 100) * 100
    if shares == 0:
        return 0.0, 0, 0.0
    
    cash = capital - shares * p0 * 1.0002  # 买入成本
    
    portfolio_values = []
    for i in range(n):
        if i < start_idx:
            portfolio_values.append(capital)  # 未买入前，保持现金
        else:
            portfolio_values.append(cash + shares * close[i])
    
    final = portfolio_values[-1]
    ret = (final / capital - 1) * 100
    max_dd = calculate_max_drawdown(portfolio_values)

    # 调试打印 → 写到文件（避免 stderr 被忽略）
    with open("debug_buy_hold.log", "a", encoding="utf-8") as f:
        f.write(f"pv[0]={portfolio_values[0]:.2f}, pv[-1]={portfolio_values[-1]:.2f}, "
                 f"min={min(portfolio_values):.2f}, max={max(portfolio_values):.2f}, "
                 f"max_dd={max_dd:.2f}%\n")
    
    return ret, 0, max_dd


def backtest_rsi(df, capital, cfg, start_idx=0):
    """RSI超卖买入 / 超买卖出"""
    period = cfg.get("rsi_period", 14)
    ovs = cfg.get("oversold", 40)
    ovb = cfg.get("overbought", 60)
    tp = cfg.get("take_profit", 0.50)
    sl = cfg.get("stop_loss", 0.15)
    close = df["close"].values
    n = len(close)

    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period-1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period-1, min_periods=period).mean()
    rsi = 100 - (100 / (1 + avg_gain / avg_loss.replace(0, 1e-10)))
    rsi = rsi.values

    cash, pos, cost = capital, 0, 0.0
    trades = 0
    portfolio_values = []
    
    for i in range(n):
        if i < start_idx:
            portfolio_values.append(capital)
            continue
        p = close[i]
        prev_rsi = rsi[i-1] if i > 0 else 50
        
        if pos == 0 and prev_rsi < ovs:
            amt = cash * 0.5
            pos = int(amt / p / 100) * 100
            if pos > 0:
                cash -= pos * p * 1.0002
                cost, trades = p, trades + 1
        elif pos > 0:
            if prev_rsi > ovb or (p > cost * (1+tp)) or (p < cost * (1-sl)):
                cash += pos * p * 0.9988
                pos, cost = 0, 0.0
                trades += 1
        
        portfolio_values.append(cash + pos * p)
    
    # 循环结束后计算最终收益率和最大回撤
    final = portfolio_values[-1] if portfolio_values else capital
    ret = (final / capital - 1) * 100
    max_dd = calculate_max_drawdown(portfolio_values)
    
    return ret, trades, max_dd


def backtest_macd_kdj(df, capital, cfg, start_idx=0):
    """MACD金叉+KDJ超卖买入"""
    fast, slow, sig = cfg.get("fast", 12), cfg.get("slow", 26), cfg.get("signal", 9)
    kp = cfg.get("kdj_period", 9)
    tp, sl = cfg.get("take_profit", 0.50), cfg.get("stop_loss", 0.15)
    close = df["close"].values
    s = df["close"]
    n = len(close)
    if n < 35:
        return None, 0, 0.0

    dif = ema(s, fast) - ema(s, slow)
    dea = ema(dif, sig)
    hist = (dif - dea).values

    low_n, high_n = s.rolling(kp).min(), s.rolling(kp).max()
    rsv = (s - low_n) / (high_n - low_n + 1e-10) * 100
    k = ema(rsv, 3)
    d = ema(k, 3)
    j_vals = (3*k - 2*d).values
    k_vals, d_vals = k.values, d.values

    cash, pos, cost = capital, 0, 0.0
    trades = 0
    portfolio_values = []
    
    for i in range(n):
        if i < start_idx:
            portfolio_values.append(capital)
            continue
        p = close[i]
        if pos == 0 and i >= 34:
            macd_gold = (hist[i-1] <= 0 < hist[i]) or (dif.iloc[i] > dea.iloc[i] and dif.iloc[i-1] <= dea.iloc[i-1])
            kdj_low = j_vals[i] < 30 and k_vals[i] > d_vals[i]
            if macd_gold or kdj_low:
                amt = cash * 0.5
                pos = int(amt / p / 100) * 100
                if pos > 0:
                    cash -= pos * p * 1.0002
                    cost, trades = p, trades + 1
        elif pos > 0 and i >= 34:
            macd_dead = (hist[i-1] >= 0 > hist[i]) or (dif.iloc[i] < dea.iloc[i] and dif.iloc[i-1] >= dea.iloc[i-1])
            kdj_high = j_vals[i] > 80
            if macd_dead or kdj_high or p > cost * (1+tp) or p < cost * (1-sl):
                cash += pos * p * 0.9988
                pos, cost = 0, 0.0
                trades += 1
        
        portfolio_values.append(cash + pos * p)
    
    final = portfolio_values[-1] if portfolio_values else capital
    ret = (final / capital - 1) * 100
    max_dd = calculate_max_drawdown(portfolio_values)
    
    return ret, trades, max_dd


def backtest_bollinger(df, capital, cfg, start_idx=0):
    """布林带：跌破下轨买，突破上轨卖"""
    period = cfg.get("period", 20)
    std_n = cfg.get("std", 2)
    tp, sl = cfg.get("take_profit", 0.50), cfg.get("stop_loss", 0.15)
    close = df["close"].values
    s = df["close"]
    n = len(close)

    mid = s.rolling(period).mean()
    std = s.rolling(period).std()
    upper = (mid + std_n * std).values
    lower = (mid - std_n * std).values

    cash, pos, cost = capital, 0, 0.0
    trades = 0
    portfolio_values = []
    
    for i in range(n):
        if i < start_idx:
            portfolio_values.append(capital)
            continue
        p = close[i]
        if i >= period and pos == 0 and p <= lower[i] > 0:
            amt = cash * 0.5
            pos = int(amt / p / 100) * 100
            if pos > 0:
                cash -= pos * p * 1.0002
                cost, trades = p, trades + 1
        elif pos > 0 and i >= period:
            if p >= upper[i] or p > cost * (1+tp) or p < cost * (1-sl):
                cash += pos * p * 0.9988
                pos, cost = 0, 0.0
                trades += 1
        portfolio_values.append(cash + pos * p)
    
    final = portfolio_values[-1] if portfolio_values else capital
    ret = (final / capital - 1) * 100
    max_dd = calculate_max_drawdown(portfolio_values)
    
    return ret, trades, max_dd


def backtest_turtle(df, capital, cfg, start_idx=0):
    """
    海龟策略（双周期 + ATR 动态风控）
    基于原版海龟法则 + A股本土化改良

    系统1（短期）：20日高点突破买入，10日低点跌破卖出
    系统2（长期）：55日高点突破买入，20日低点跌破卖出

    ATR 波动量化 → 动态仓位 + 动态止损
    金字塔加仓：价格向有利方向变动 0.5×ATR 时加仓，最多4次
    1% 风险原则：每笔交易风险 ≤ 总资金 × 1%
    """
    import numpy as np

    # ── 解析配置 ──────────────────────────────────────────
    short_period = cfg.get("short_period", 20)
    long_period = cfg.get("long_period", 55)
    short_exit = cfg.get("short_exit_period", 10)
    long_exit = cfg.get("long_exit_period", 20)
    atr_period = cfg.get("atr_period", 14)
    risk_pct = cfg.get("risk_pct", 0.01)          # 1% 风险
    max_risk_per_day = cfg.get("max_risk_per_day", 0.02)
    max_pos_pct = cfg.get("max_pos_pct", 0.10)
    max_adds = cfg.get("max_adds", 4)
    add_step_atr = cfg.get("add_step_atr", 0.5)
    pos_unit_decay = cfg.get("pos_unit_decay", True)
    stop_atr_mult = cfg.get("stop_atr_mult", 2.0)
    trail_atr_mult = cfg.get("trail_atr_mult", 2.0)
    volume_filter = cfg.get("volume_filter", True)
    volume_ma_period = cfg.get("volume_ma_period", 20)
    min_listed_days = cfg.get("min_listed_days", 250)
    # ── 趋势过滤（A股本土化）──
    trend_filter = cfg.get("trend_filter", True)      # 趋势过滤：价格 > MA200 才做多
    trend_ma_period = cfg.get("trend_ma_period", 200)
    # ── 手续费模型（更真实）──
    buy_cost = cfg.get("buy_cost", 1.0012)          # 买入成本系数（含佣金+印花税+滑点）
    sell_cost = cfg.get("sell_cost", 0.9985)        # 卖出成本系数
    use_short = cfg.get("use_short_system", True)
    use_long = cfg.get("use_long_system", True)
    system_weight = cfg.get("system_weight", [0.5, 0.5])

    # ── 数据检查 ──────────────────────────────────────────
    # ── 列名标准化（兼容中英文）─────────────────────────
    col_map = {"开盘": "open", "最高": "high", "最低": "low", "收盘": "close"}
    df = df.rename(columns=col_map)
    req_cols = ["open", "high", "low", "close"]
    for c in req_cols:
        if c not in df.columns:
            return None, 0
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    n = len(close)
    min_n = max(long_period, atr_period, short_exit, long_exit) + 1
    if n < min_n:
        return None, 0

    # ── 成交量过滤（A股本土化）──────────────────────────
    volume_ok = np.ones(n, dtype=bool)
    if volume_filter and "volume" in df.columns:
        vol = df["volume"].values
        vol_ma = df["volume"].rolling(volume_ma_period).mean().values
        # 突破日成交量需大于20日均量
        for i in range(volume_ma_period, n):
            if vol[i] < vol_ma[i] * 1.2:   # 放量突破才有效
                volume_ok[i] = False


    # ── 计算 ATR（真实波幅均值）────────────────────────
    tr = np.zeros(n)
    for i in range(1, n):
        h = high[i]
        l = low[i]
        c_prev = close[i-1]
        tr[i] = max(h - l, abs(h - c_prev), abs(l - c_prev))
    atr = np.zeros(n)
    # 初始 ATR = TR 的简单平均
    for i in range(atr_period, n):
        atr[i] = np.mean(tr[i-atr_period+1:i+1])
    # 后续用指数平滑（海龟原版）
    for i in range(atr_period+1, n):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period

    # ── 计算趋势过滤均线 ───────────────────────
    trend_ma = np.zeros(n)
    if trend_filter:
        ma200 = df["close"].rolling(trend_ma_period).mean().values
        trend_ma = ma200

    # ── 计算突破/离场通道 ───────────────────────────────
    # 系统1（短期）
    s1_entry = np.zeros(n)   # 20日高点
    s1_exit = np.zeros(n)     # 10日低点
    for i in range(short_period, n):
        s1_entry[i] = np.max(high[i-short_period:i])
    for i in range(short_exit, n):
        s1_exit[i] = np.min(low[i-short_exit:i])

    # 系统2（长期）
    s2_entry = np.zeros(n)   # 55日高点
    s2_exit = np.zeros(n)     # 20日低点
    for i in range(long_period, n):
        s2_entry[i] = np.max(high[i-long_period:i])
    for i in range(long_exit, n):
        s2_exit[i] = np.min(low[i-long_exit:i])

    # ── 回测循环 ─────────────────────────────────────────
    cash = float(capital)
    daily_pnl = []   # 记录每日盈亏（用于单日亏损风控）
    max_capital = cash   # 历史最高资金（用于计算回撤）

    # 系统1 状态
    s1_pos = 0          # 持仓股数
    s1_cost = 0.0       # 持仓成本（均价）
    s1_entry_price = 0.0 # 初始建仓价格
    s1_adds = 0         # 已加仓次数
    s1_last_add_price = 0.0
    s1_highest = 0.0    # 持仓期间最高价（追踪止损用）
    s1_stop_price = 0.0  # 当前止损价

    # 系统2 状态
    s2_pos = 0
    s2_cost = 0.0
    s2_entry_price = 0.0
    s2_adds = 0
    s2_last_add_price = 0.0
    s2_highest = 0.0
    s2_stop_price = 0.0

    trades = 0
    portfolio_values = []  # 记录每日总资产（用于计算收益）

    # 计算策略实际起始位置（跳过回溯期）
    loop_start = max(start_idx, long_period, atr_period, long_exit)
    if loop_start >= n:
        return 0.0, 0

    for i in range(n):
        # 跳过回溯期（只记录资产，不执行交易逻辑）
        if i < loop_start:
            portfolio_values.append(cash + s1_pos * close[i] + s2_pos * close[i])
            continue

        p = close[i]
        if p <= 0:
            portfolio_values.append(cash + s1_pos * p + s2_pos * p)
            continue

        atr_i = atr[i] if i >= atr_period else atr[atr_period] if atr_period < n else 0
        if atr_i <= 0:
            portfolio_values.append(cash + s1_pos * p + s2_pos * p)
            continue

        # ── 风控：单日亏损上限检查 ──────────────────────
        today_pnl = 0
        if len(portfolio_values) > 0:
            prev_val = portfolio_values[-1]
            curr_val = cash + s1_pos * p + s2_pos * p
            today_pnl = curr_val - prev_val
            if today_pnl < -max_risk_per_day * capital:
                # 触发单日亏损上限，强制平仓
                if s1_pos > 0:
                    cash += s1_pos * p * sell_cost
                    s1_pos = 0; s1_adds = 0
                    trades += 1
                if s2_pos > 0:
                    cash += s2_pos * p * sell_cost
                    s2_pos = 0; s2_adds = 0
                    trades += 1

        # ── 系统1（短期）────────────────────────────────
        if use_short:
            # 入场：突破20日高点
            if s1_pos == 0 and i >= short_period:
                if p > s1_entry[i] and volume_ok[i] and (not trend_filter or p > trend_ma[i]):
                    # 计算建仓单位（1% 风险原则）
                    # 股数 = (总资金 × 1%) / ATR
                    risk_capital = capital * risk_pct  # 用初始总资金，不是当前现金
                    unit_shares = int(risk_capital / atr_i) if atr_i > 0 else 0
                    unit_shares = max((unit_shares // 100) * 100, 100)  # 取整到100股，最少100股
                    if unit_shares > 0:
                        cost = unit_shares * p * buy_cost
                        if cost <= cash:
                            cash -= cost
                            s1_pos = unit_shares
                            s1_cost = p
                            s1_entry_price = p
                            s1_last_add_price = p
                            s1_adds = 1
                            s1_highest = p
                            s1_stop_price = p - stop_atr_mult * atr_i
                            trades += 1

            # 加仓：价格向有利方向变动 0.5×ATR
            elif s1_pos > 0 and s1_adds < max_adds:
                add_threshold = s1_last_add_price + add_step_atr * atr_i
                if p >= add_threshold:
                    # 计算加仓单位（海龟原版：递减）
                    if pos_unit_decay:
                        # 加仓单位 = 初始建仓股数 / (加仓次数 + 1)
                        base_units = int(capital * risk_pct / atr_i) if atr_i > 0 else 0
                        add_shares = max(int(base_units / (s1_adds + 1)) // 100 * 100, 100)
                    else:
                        add_shares = max(int(capital * risk_pct / atr_i) // 100 * 100, 100)
                    add_shares = max(add_shares, 100)
                    cost = add_shares * p * buy_cost
                    if add_shares > 0 and cost <= cash:
                        cash -= cost
                        s1_cost = (s1_cost * s1_pos + p * add_shares) / (s1_pos + add_shares)
                        s1_pos += add_shares
                        s1_last_add_price = p
                        s1_adds += 1
                        s1_highest = max(s1_highest, p)
                        trades += 1

            # 止损：跌破止损价 或 跌破10日低点
            if s1_pos > 0:
                s1_highest = max(s1_highest, p)
                # 追踪止损价（随着最高价上移）
                trail_stop = s1_highest - trail_atr_mult * atr_i
                s1_stop_price = max(s1_stop_price, trail_stop)
                if p <= s1_stop_price or (i >= short_exit and p < s1_exit[i]):
                    cash += s1_pos * p * sell_cost
                    s1_pos = 0; s1_adds = 0; s1_stop_price = 0
                    trades += 1

        # ── 系统2（长期）────────────────────────────────
        if use_long:
            if s2_pos == 0 and i >= long_period:
                if p > s2_entry[i] and volume_ok[i] and (not trend_filter or p > trend_ma[i]):
                    # 计算建仓单位（1% 风险原则）
                    risk_capital = capital * risk_pct  # 用初始总资金
                    unit_shares = int(risk_capital / atr_i) if atr_i > 0 else 0
                    unit_shares = max((unit_shares // 100) * 100, 100)  # 取整到100股，最少100股
                    if unit_shares > 0:
                        cost = unit_shares * p * buy_cost
                        if cost <= cash:
                            cash -= cost
                            s2_pos = unit_shares
                            s2_cost = p
                            s2_entry_price = p
                            s2_last_add_price = p
                            s2_adds = 1
                            s2_highest = p
                            s2_stop_price = p - stop_atr_mult * atr_i
                            trades += 1

            elif s2_pos > 0 and s2_adds < max_adds:
                add_threshold = s2_last_add_price + add_step_atr * atr_i
                if p >= add_threshold:
                    if pos_unit_decay:
                        # 加仓单位 = 初始建仓股数 / (加仓次数 + 1)
                        base_units = int(capital * risk_pct / atr_i) if atr_i > 0 else 0
                        add_shares = max(int(base_units / (s2_adds + 1)) // 100 * 100, 100)
                    else:
                        add_shares = s2_pos
                    add_shares = max(add_shares, 100)
                    cost = add_shares * p * buy_cost
                    if add_shares > 0 and cost <= cash:
                        cash -= cost
                        s2_cost = (s2_cost * s2_pos + p * add_shares) / (s2_pos + add_shares)
                        s2_pos += add_shares
                        s2_last_add_price = p
                        s2_adds += 1
                        s2_highest = max(s2_highest, p)
                        trades += 1

            if s2_pos > 0:
                s2_highest = max(s2_highest, p)
                trail_stop = s2_highest - trail_atr_mult * atr_i
                s2_stop_price = max(s2_stop_price, trail_stop)
                if p <= s2_stop_price or (i >= long_exit and p < s2_exit[i]):
                    cash += s2_pos * p * sell_cost
                    s2_pos = 0; s2_adds = 0; s2_stop_price = 0
                    trades += 1

        # 记录当日总资产
        total_val = cash + s1_pos * p + s2_pos * p
        portfolio_values.append(total_val)
        if total_val > max_capital:
            max_capital = total_val

    # ── 期末清仓 ─────────────────────────────────────────
    final_price = close[-1]
    if s1_pos > 0:
        cash += s1_pos * final_price * sell_cost
    if s2_pos > 0:
        cash += s2_pos * final_price * sell_cost

    final = cash
    ret = (final / capital - 1) * 100
    max_dd = calculate_max_drawdown(portfolio_values)
    return ret, trades, max_dd


# ════════════════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════
# RSI趋势策略（使用RSITrendStrategy类）
# ════════════════════════════════════════════════════

def backtest_rsi_trend(df, capital, cfg, start_idx=0):
    """
    RSI趋势跟踪策略（调用RSITrendPlugin类）
    RSI上穿50 → 买入，RSI下穿50 → 卖出
    """
    try:
        from backtest.rsi_trend_plugin import RSITrendPlugin
    except ImportError:
        print("  [ERR] 无法导入 RSITrendPlugin")
        return None, 0, 0.0

    # ── 数据检查 ──────────────────────────
    if df is None or len(df) == 0:
        return None, 0, 0.0

    # 确保有必要的列
    if "adj_open" not in df.columns:
        df["adj_open"] = df["open"]
    if "adj_close" not in df.columns:
        df["adj_close"] = df["close"]

    # ── 初始化策略 ──────────────────────────
    strategy = RSITrendPlugin(
        capital=capital,
        cfg=cfg,
    )

    # ── 运行策略 ──────────────────────────
    result = strategy.run(df, start_idx=start_idx)

    # ── 计算收益率 ──────────────────────────
    trades = result.get("trades", [])
    daily_values = result.get("daily_values", [])

    if not daily_values:
        return 0.0, len(trades), 0.0

    # 计算最终资产
    final_value = daily_values[-1]["portfolio_value"] if daily_values else capital
    ret = (final_value / capital - 1) * 100
    
    # 计算最大回撤
    portfolio_values = [v["portfolio_value"] for v in daily_values]
    max_dd = calculate_max_drawdown(portfolio_values)

    return ret, len(trades), max_dd

def run_selection():
    """执行多因子选股，返回 TOP N 股票列表"""
    from src.data_fetcher import DataFetcher
    from src.factor_calculator import FactorCalculator
    from src.factor_processor import FactorProcessor
    from src.stock_selector import StockSelector

    print(f"\n{'='*60}")
    print(f"  选股阶段 — {SELECTION['date']} | 池:{SELECTION['stock_pool']} | TOP {SELECTION['top_n']}")
    print(f"{'='*60}")

    df_config = {
        "primary_source": DATA["primary_source"],
        "tushare_token": DATA.get("tushare_token", ""),
        "local_db_path": DATA["local_db_path"],
        "use_akshare_backup": DATA["use_akshare_backup"],
        "use_tushare_backup": DATA["use_tushare_backup"],
    }
    fetcher = DataFetcher(**df_config)

    # 启用行业动量因子（如果配置中已启用）
    if INDUSTRY_MOMENTUM.get("enabled", False):
        FACTOR_CALCULATOR["enable_industry_momentum"] = True
        print(f"  [OK] 行业动量因子已启用")

    calculator = FactorCalculator(**FACTOR_CALCULATOR)
    processor = FactorProcessor(config=FACTOR_PROCESSOR)
    selector = StockSelector(config={"top_n": SELECTION["top_n"]})

    # 根据 stock_pool 配置动态选择股票池
    pool_map = {
        "hs300": fetcher.get_hs300_components,
        "zz500": lambda: fetcher.get_zz500_components() if hasattr(fetcher, 'get_zz500_components') else fetcher.get_hs300_components(),
        "zz800": fetcher.get_zz800_components,
        "all":   lambda: fetcher.get_all_stocks(),
    }
    get_pool = pool_map.get(SELECTION["stock_pool"], fetcher.get_hs300_components)
    
    # 详细调试信息
    print(f"  [DEBUG] stock_pool配置: {SELECTION['stock_pool']}")
    print(f"  [DEBUG] 调用函数: {get_pool.__name__ if hasattr(get_pool, '__name__') else get_pool}")
    
    pool = get_pool()
    
    # 验证股票池数量
    expected_max = {"hs300": 300, "zz500": 500, "zz800": 800}.get(SELECTION["stock_pool"], 10000)
    print(f"  [DEBUG] 实际获取数量: {len(pool)} 只")
    if len(pool) > expected_max * 1.5:  # 允许50%误差（如有重复）
        print(f"  [WARN] 警告: 股票池数量异常！配置={SELECTION['stock_pool']}, 预期≤{expected_max}, 实际={len(pool)}")
        print(f"  [WARN] 可能误用了全市场数据，请检查 get_zz800_components 实现")
    print(f"  股票池 [{SELECTION['stock_pool']}]: {len(pool)} 只")

    factors = calculator.calculate_all_factors(
        pool["code"].tolist(), fetcher,
        start_date=None,          # 自动计算（end_date 前 2 年）
        end_date=SELECTION["date"],
        max_workers=5
    )
    print(f"  因子计算: {len(factors)} 只 × {len([c for c in factors.columns if c.startswith(('VF','GF','QF','MF','TF','LVF','MWF'))])} 因子")

    processed = processor.process(factors)
    selected = selector.select(processed, top_n=SELECTION["top_n"])

    # 补全股票名称（容错）
    names = {}
    try:
        conn = sqlite3.connect(DB_PATH)
        for code in selected["code"].tolist():
            r = conn.execute("SELECT name FROM stock_basic WHERE ts_code=?", (ts_code(code),)).fetchone()
            names[code] = r[0] if r else ""
        conn.close()
    except Exception:
        pass
    selected["name"] = selected["code"].map(names).fillna("")

    selector.print_top_stocks(selected, n=min(20, len(selected)))

    # 保存选股结果 CSV（命名规则: multi-YYYYMM-selection.csv）
    if OUTPUT.get("save_csv"):
        os.makedirs(OUTPUT["dir"], exist_ok=True)
        from datetime import datetime
        ym = SELECTION["date"][:6]  # "20220103" -> "202201"
        csv_path = os.path.join(OUTPUT["dir"], f"multi-{ym}-selection.csv")
        selected[["code", "name"]].to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"  选股结果已保存 → {csv_path}")

    return selected


def run_backtest(stocks):
    """对股票列表执行所有已启用的策略"""
    capital = BACKTEST["initial_capital"]
    start, end = BACKTEST["start_date"], BACKTEST["end_date"]
    benchmark = BACKTEST["benchmark"]

    # ── 补全股票名称（CSV 中可能没有 name 列，强制从数据库补全）───
    try:
        _conn = sqlite3.connect(DB_PATH)
        _names = {}
        for _code in stocks["code"].tolist():
            _r = _conn.execute(
                "SELECT name FROM stock_basic WHERE ts_code=?",
                (ts_code(str(_code)),)
            ).fetchone()
            _names[_code] = _r[0] if _r else str(_code)
        _conn.close()
        stocks["name"] = stocks["code"].map(_names).fillna("")
    except Exception as _e:
        print(f"  [WARN] 补全股票名称失败: {_e}")
        stocks["name"] = stocks.get("name", "")

    # 使用持久连接（避免 sandbox 反复拦截）
    conn = sqlite3.connect(DB_PATH)

    # 加载基准
    idx = load_benchmark(benchmark, start, end, conn)
    idx_ret = (idx["close"].iloc[-1] / idx["close"].iloc[0] - 1) * 100 if idx is not None else 0

    # 预加载所有股票数据（含250天回溯，用于计算MA200）
    print(f"\n  加载股票数据 ({start} → {end})...")
    stock_data = {}
    bh_results = {}
    # 找到回测起始日在 df 中的索引（用于 bh 计算和策略起始位置）
    for _, row in stocks.iterrows():
        code, name = row["code"], row.get("name", "")
        df = load_stock_prices(code, start, end, conn, lookback_days=250)
        if df is not None and len(df) >= 30:
            # 找到第一个 >= start 的交易日索引
            start_idx = df[df["trade_date"] >= start].index.min()
            if pd.isna(start_idx):
                continue
            start_idx = int(start_idx)
            stock_data[code] = (name, df, start_idx)
            # BH 收益 = 回测起始日买入 → 结束日卖出
            bh_ret = (df["close"].iloc[-1] / df["close"].iloc[start_idx] - 1) * 100
            bh_results[code] = bh_ret
    conn.close()
    print(f"  有效股票: {len(stock_data)}/{len(stocks)}")

    # ═══ 加载策略插件（自动发现 backtest/*.py）═══
    print(f"\n  正在加载策略插件...")
    plugins = load_strategy_plugins()
    print(f"  已发现 {len(plugins)} 个策略插件: {', '.join(plugins.keys())}")

    # ═─ 硬编码策略函数（向后兼容，逐步迁移到插件）───
    strategy_funcs = {
        "buy_hold": (backtest_buy_hold, 0),
        "rsi": (backtest_rsi, 0),
        "macd_kdj": (backtest_macd_kdj, 0),
        "bollinger": (backtest_bollinger, 0),
        "turtle": (backtest_turtle, 0),
        "rsi_trend": (backtest_rsi_trend, 0),
    }

    enabled = [(k, v) for k, v in STRATEGIES.items() if v.get("enabled")]
    print(f"\n{'='*100}")
    print(f"  回测阶段 — {start} → {end} | 基准{benchmark}: {idx_ret:+.2f}% | 资金{capital/10000:.0f}万/只")
    print(f"  启用策略: {', '.join([s['name'] for _, s in enabled])}")
    print(f"{'='*100}")

    all_summaries = {}

    for skey, scfg in enabled:
        sname = scfg["name"]
        print(f"\n{'─'*100}")
        print(f"  【{sname}】")
        print(f"  {'代码':<8} {'名称':<8} {'收益率':>8} {'超额':>8} {'vs买入持有':>10} {'交易':>6} {'跑赢':>6} {'最大回撤':>10}")
        print(f"  {'─'*60}")

        results = []
        for code, (name, df, start_idx) in stock_data.items():
            try:
                # ═─ 优先使用插件（类方式）══─
                if skey in plugins:
                    strategy_class = plugins[skey]
                    strategy = strategy_class(capital, scfg)
                    result = strategy.run(df, start_idx)
                    ret = result.get("returns", 0.0)
                    trades = len(result.get("trades", []))
                    # 计算最大回撤
                    daily_values = result.get("daily_values", [])
                    if daily_values:
                        portfolio_values = [v["portfolio_value"] for v in daily_values]
                        max_dd = calculate_max_drawdown(portfolio_values)
                    else:
                        max_dd = 0.0
                    
                # ═─ 回退到硬编码函数（兼容旧策略）══─
                elif skey in strategy_funcs:
                    func = strategy_funcs[skey][0]
                    if skey == "buy_hold":
                        ret, trades, max_dd = func(df, capital, start_idx)
                    else:
                        ret, trades, max_dd = func(df, capital, scfg, start_idx)
                else:
                    print(f"  [ERR] 未找到策略: {skey}")
                    continue
                
                if ret is None:
                    continue
                exc = ret - idx_ret
                bh = bh_results.get(code, 0)
                vs_bh = ret - bh if skey != "buy_hold" else 0.0
                beat = "[OK]" if ret > idx_ret else "[ERR]"
                print(f"  {code:<8} {name:<8} {ret:>+7.2f}% {exc:>+7.2f}% {vs_bh:>+9.2f}% {trades:>6} {beat:>6}  {max_dd:>6.2f}%")
                results.append({"ret": ret, "exc": exc, "vs_bh": vs_bh, "trades": trades, "beat": ret > idx_ret, "max_dd": max_dd})
            except Exception as e:
                print(f"  {code:<8} {name:<8} {'ERR':>8} ({e})")

        if results:
            rets = [r["ret"] for r in results]
            # 买入持有策略不计算"优于BH"（自己不需要比自己）
            n_better_bh = 0 if skey == "buy_hold" else sum(1 for r in results if r["vs_bh"] > 0)
            s = all_summaries[sname] = {
                "mean": np.mean(rets), "median": np.median(rets),
                "best": np.max(rets), "worst": np.min(rets),
                "n_pos": sum(1 for r in rets if r > 0),
                "n_beat": sum(1 for r in results if r["beat"]),
                "n_better_bh": n_better_bh,
                "n": len(results),
                "trades_mean": np.mean([r["trades"] for r in results]),
                "max_dd_mean": np.mean([r["max_dd"] for r in results]),
            }
            # 买入持有不显示"优于BH"
            if skey == "buy_hold":
                better_bh_str = "优于BH N/A"
            else:
                better_bh_str = f"优于BH {s['n_better_bh']}/{len(results)}"
            print(f"  {'─'*60}")
            print(f"  汇总: 均值{s['mean']:+.2f}% 中位数{s['median']:+.2f}% "
                  f"正收益{s['n_pos']}/{len(results)} 跑赢{s['n_beat']}/{len(results)} "
                  f"{better_bh_str} 均交易{s['trades_mean']:.1f}次 "
                  f"均最大回撤{s['max_dd_mean']:.2f}%")

    # ══ 总表 ══
    print(f"\n\n{'='*100}")
    print(f"  【策略对比总表】")
    print(f"{'='*100}")
    print(f"  {'策略':<16} {'均值':>8} {'中位数':>8} {'正收益':>8} {'跑赢指数':>8} {'优于BH':>8} {'均交易':>6} {'均最大回撤':>12} {'最佳':>10} {'最差':>10}")
    print(f"  {'─'*90}")
    for skey, scfg in enabled:
        s = all_summaries.get(scfg["name"], {})
        if s:
            better_bh_display = f"{s['n_better_bh']}/{s['n']}" if skey != "buy_hold" else "N/A"
            print(f"  {scfg['name']:<16} {s['mean']:>+7.2f}% {s['median']:>+7.2f}% "
                  f"{s['n_pos']}/{s['n']:>4}  {s['n_beat']}/{s['n']:>4}  {better_bh_display:>6}  "
                  f"{s['trades_mean']:>5.1f}  {s['max_dd_mean']:>11.2f}%  {s['best']:>+9.2f}% {s['worst']:>+9.2f}%")
    bh_mean = np.mean(list(bh_results.values()))
    print(f"  {'─'*90}")
    print(f"  {'买入持有(等权)':<16} {bh_mean:>+7.2f}%")
    print(f"  {benchmark + '基准':<16} {idx_ret:>+7.2f}%")
    print(f"{'='*100}\n")

    # ── 保存结果 ──
    if OUTPUT.get("save_csv"):
        os.makedirs(OUTPUT["dir"], exist_ok=True)
        rows = []
        for skey, scfg in enabled:
            s = all_summaries.get(scfg["name"])
            if s:
                rows.append({
                    "策略": scfg["name"], "均值收益%": round(s["mean"], 2),
                    "中位数%": round(s["median"], 2), "正收益比": f"{s['n_pos']}/{s['n']}",
                    "跑赢指数比": f"{s['n_beat']}/{s['n']}", "优于买入持有比": f"{s['n_better_bh']}/{s['n']}",
                    "平均交易次数": round(s["trades_mean"], 1),
                    "均最大回撤%": round(s["max_dd_mean"], 2),
                    "最佳%": round(s["best"], 2), "最差%": round(s["worst"], 2),
                })
        rows.append({"策略": "买入持有(等权)", "均值收益%": round(bh_mean, 2)})
        rows.append({"策略": f"{benchmark}基准", "均值收益%": round(idx_ret, 2)})
        pd.DataFrame(rows).to_csv(
            os.path.join(OUTPUT["dir"], f"backtest_summary_{BACKTEST['start_date'][:4]}.csv"),
            index=False, encoding="utf-8-sig")
        print(f"  报告已保存 → {OUTPUT['dir']}/backtest_summary_{BACKTEST['start_date'][:4]}.csv")


# ════════════════════════════════════════════════════════
# 入口
# ════════════════════════════════════════════════════════
if __name__ == "__main__":
    # ── 命令行参数解析 ──────────────────────────────────────────────────
    # 用法:
    #   python run_backtest.py                # 使用 config.py 默认配置
    #   python run_backtest.py --source multi # 使用最新 multi-*.csv
    #   python run_backtest.py --source ml    # 使用最新 ml-*.csv
    # ─────────────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="多因子选股 + 回测系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例:
  python run_backtest.py                # 使用 config.py 默认配置
  python run_backtest.py --source multi # 使用最新 multi-*.csv 进行回测
  python run_backtest.py --source ml    # 使用最新 ml-*.csv 进行回测
  python run_backtest.py --list         # 列出所有可用的 CSV 文件
"""
    )
    parser.add_argument(
        "--source", "-s",
        type=str,
        choices=["multi", "ml", "csv", "manual"],
        default=None,
        help="选股策略来源: multi(多因子) / ml(机器学习) / csv(指定文件) / manual(手动列表)"
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        default=None,
        help="手动指定 CSV 文件路径（--source csv 时生效）"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="列出 data/results/ 下所有可用的 CSV 文件"
    )
    args = parser.parse_args()

    # 列出可用 CSV 文件
    if args.list:
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT["dir"])
        csv_files = sorted(glob.glob(os.path.join(results_dir, "*.csv")))
        print("\n  可用 CSV 文件 (data/results/):")
        print("  " + "-" * 50)
        for f in csv_files:
            mtime = os.path.getmtime(f)
            mtime_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
            size_kb = os.path.getsize(f) / 1024
            fname = os.path.basename(f)
            # 标记类型
            tag = ""
            if fname.startswith("ml-"):
                tag = "  [ML机器学习]"
            elif fname.startswith("multi-"):
                tag = "  [多因子选股]"
            elif fname.startswith("backtest_comparison"):
                tag = "  [回测结果]"
            print(f"  {fname:<35} {mtime_str}  {size_kb:.1f}KB{tag}")
        print("  " + "-" * 50 + "\n")
        sys.exit(0)

    print("\n" + "=" * 60)
    print("  多因子选股 + 回测系统")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 根据命令行参数决定股票池来源
    if args.source == "multi":
        # 自动匹配最新的 multi-*.csv
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT["dir"])
        multi_files = sorted(glob.glob(os.path.join(results_dir, "multi-*.csv")), key=os.path.getmtime, reverse=True)
        if not multi_files:
            print(f"\n  [ERROR] 未找到 multi-*.csv 文件在 {results_dir}")
            print(f"  请先运行多因子选股 (src/stock_selector.py)\n")
            sys.exit(1)
        csv_path = multi_files[0]
        stocks = pd.read_csv(csv_path, dtype={"code": str})
        stocks["name"] = stocks.get("name", "")
        print(f"\n  使用多因子选股结果: {os.path.basename(csv_path)} ({len(stocks)} 只)")
        run_backtest(stocks)
        sys.exit(0)

    elif args.source == "ml":
        # 自动匹配最新的 ml-*.csv
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT["dir"])
        ml_files = sorted(glob.glob(os.path.join(results_dir, "ml-*.csv")), key=os.path.getmtime, reverse=True)
        if not ml_files:
            print(f"\n  [ERROR] 未找到 ml-*.csv 文件在 {results_dir}")
            print(f"  请先运行机器学习选股 (ml_stock_selector_v4.py)\n")
            sys.exit(1)
        csv_path = ml_files[0]
        stocks = pd.read_csv(csv_path, dtype={"code": str})
        stocks["name"] = stocks.get("name", "")
        print(f"\n  使用机器学习选股结果: {os.path.basename(csv_path)} ({len(stocks)} 只)")
        run_backtest(stocks)
        sys.exit(0)

    elif args.source == "csv":
        # 使用手动指定的 CSV 文件
        csv_path = args.file if args.file else BACKTEST.get("stocks_file", "")
        if not csv_path or not os.path.exists(csv_path):
            print(f"\n  [ERROR] CSV 文件不存在: {csv_path}")
            print(f"  请使用 --file 指定有效路径\n")
            sys.exit(1)
        stocks = pd.read_csv(csv_path, dtype={"code": str})
        stocks["name"] = stocks.get("name", "")
        print(f"\n  从 CSV 加载股票池: {os.path.basename(csv_path)} ({len(stocks)} 只)")
        run_backtest(stocks)
        sys.exit(0)

    elif args.source == "manual":
        # 手动股票列表
        manual = BACKTEST.get("stocks_manual", [])
        stocks = pd.DataFrame(manual, columns=["code", "name"])
        print(f"\n  使用手动股票池: {len(stocks)} 只")
        run_backtest(stocks)
        sys.exit(0)

    else:
        # 使用 config.py 中的默认配置
        source = BACKTEST.get("stocks_source", "selection")

        if source == "selection":
            # 运行选股 → 回测
            stocks = run_selection()
        elif source == "csv":
            # 从 CSV 读取上次选股结果
            csv_path = BACKTEST.get("stocks_file", "")
            if not csv_path or not os.path.exists(csv_path):
                # 尝试自动匹配
                csv_path = os.path.join(OUTPUT["dir"], f"selection_{SELECTION['date']}.csv")
            if os.path.exists(csv_path):
                stocks = pd.read_csv(csv_path, dtype={"code": str})
                stocks["name"] = stocks.get("name", "")
                print(f"\n  从 CSV 加载股票池: {csv_path} ({len(stocks)} 只)")
            else:
                print(f"\n  CSV 文件不存在: {csv_path}，回退到手动模式")
                manual = BACKTEST.get("stocks_manual", [])
                stocks = pd.DataFrame(manual, columns=["code", "name"])
        elif source == "manual":
            # 手动股票列表
            manual = BACKTEST.get("stocks_manual", [])
            stocks = pd.DataFrame(manual, columns=["code", "name"])
            print(f"\n  使用手动股票池: {len(stocks)} 只")
        else:
            manual = BACKTEST.get("stocks_manual", [])
            stocks = pd.DataFrame(manual, columns=["code", "name"])

        run_backtest(stocks)

    print(f"\n{'='*60}")
    print("  完成！修改 config.py 参数后可再次运行。")
    print(f"{'='*60}\n")
