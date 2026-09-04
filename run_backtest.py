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


import numpy as np
import pandas as pd
from datetime import datetime
from loguru import logger

# 完全静音 loguru 终端输出（回测平台的主输出使用 print()，不需要 loguru）
logger.remove()
logger.add(lambda _: None, level="CRITICAL")  # 空 sink，只吞不吐


# ════════════════════════════════════════════════════════
# 加载配置
# ════════════════════════════════════════════════════════
from config import (
    DATA, SELECTION, FACTOR_CALCULATOR, FACTOR_PROCESSOR,
    BACKTEST, STRATEGIES, OUTPUT, INDUSTRY_MOMENTUM,
    VALUE_STRATEGY, DIVIDEND_LOW_VOL, DOGS_OF_MARKET,
)

# 印花税率：复用共享引擎的「分段口径」（2023-08-28 起千1→千0.5），保证全平台一致
from run_monthly_rebalance import stamp_duty_rate
import bench_index as bi

# 凯利公式控仓（总持仓封顶）：让内置回测函数与插件策略共用同一套 kelly_sizer 框架
from backtest.base_strategy import BaseStrategy


def _make_kelly_cap(cfg):
    """读取 cfg 的 kelly 段，返回封顶比例（0.05~0.25）；未启用 use_kelly 返回 None。"""
    if not cfg.get("use_kelly", False):
        return None
    from backtest.kelly_sizer import KellySizer
    _k = KellySizer(
        estimated_win_rate=cfg.get("kelly_win_rate", 0.55),
        estimated_win_loss_ratio=cfg.get("kelly_win_loss_ratio", 1.5),
        kelly_fraction=cfg.get("kelly_fraction", 0.5),
        max_position_pct=cfg.get("kelly_max_position", 0.25),
        min_position_pct=cfg.get("kelly_min_position", 0.05),
        safety_discount=cfg.get("kelly_safety_discount", 0.8),
    )
    return _k.get_position_pct()


DB_PATH = DATA["local_db_path"]


# ══════════════════════════════════════════════════════
# 策略插件自动发现（无需修改此文件即可添加新策略）
# ══════════════════════════════════════════════════════

# ---- 胜率计算函数 ----
def calc_win_rate_from_trades(trade_records):
    """
    从交易记录计算胜率
    trade_records: list of dicts with keys: action, price, shares
    Returns: (win_rate_pct, win_count, total_closed)
    """
    if not trade_records:
        return 0.0, 0, 0
    
    # FIFO 匹配买卖对
    pending_buys = []  # [{"price": p, "shares": s}]
    win = 0
    total = 0
    
    for t in trade_records:
        action = t.get("action", "")
        shares = t.get("shares", 0)
        price = t.get("price", 0.0)
        
        if action.startswith("BUY"):
            pending_buys.append({"price": price, "shares": shares})
        elif action.startswith("SELL"):
            remaining = shares
            while remaining > 0 and pending_buys:
                first = pending_buys[0]
                match_shares = min(first["shares"], remaining)
                pnl = (price - first["price"]) * match_shares
                total += 1
                if pnl > 0:
                    win += 1
                
                first["shares"] -= match_shares
                remaining -= match_shares
                
                if first["shares"] <= 0:
                    pending_buys.pop(0)
    
    wr = (win / total * 100) if total > 0 else 0.0
    return wr, win, total


# ============================================================
#  统一交易成本模型（佣金 + 印花税 + 滑点）
#  口径与 run_monthly_rebalance.calc_fee 对齐，供所有回测函数复用
# ============================================================
COMMISSION_RATE_RB = 0.00025   # 佣金率
COMMISSION_MIN_RB  = 5.0        # 最低佣金（元）
STAMP_DUTY_RATE_RB = 0.001      # 印花税率（仅卖出收取）
SLIPPAGE_RATE_RB   = 0.001      # 滑点率（买卖均含，模拟冲击成本）

# ── 流动性过滤（保守阈值，主要跑大盘股时几乎不剔除成分股）────
# daily.amount 单位为"千元"，故阈值以元传入时需 ×1000 换算
LIQUIDITY_MIN_AVG_AMOUNT = 50_000_000   # 选股日往前 LOOKBACK 日，日均成交额下限（元）：5000万
LIQUIDITY_LOOKBACK       = 20           # 滚动窗口（交易日）

def calc_fee_rb(buy_or_sell, price, shares, trade_date=None, stamp_duty=None):
    """计算单笔交易的总成本（元）。ETF/配对等免印花税可传 stamp_duty=0。
    stamp_duty 省略时按成交日分段印花税率（2023-08-28 起千1→千0.5）。"""
    amount = price * shares
    commission = max(amount * COMMISSION_RATE_RB, COMMISSION_MIN_RB)
    slippage = amount * SLIPPAGE_RATE_RB
    if buy_or_sell == "buy":
        return commission + slippage
    if stamp_duty is None:
        stamp_duty = stamp_duty_rate(trade_date)
    return commission + stamp_duty * amount + slippage


def prefilter_by_liquidity(conn, pool_df, as_of_date_fmt,
                           min_avg_amount=LIQUIDITY_MIN_AVG_AMOUNT,
                           lookback=LIQUIDITY_LOOKBACK):
    """
    流动性预过滤：剔除选股日往前 lookback 个交易日，日均成交额低于阈值的股票。
    阈值默认 5000万（保守）。沪深300/中证500 成分股几乎不会被剔除，
    仅挡掉真正的小盘/僵尸股，避免回测里"想买买不进、想卖卖不出"的流动性幻觉。
    daily.amount 单位为千元 → avg_amt * 1000 = 元。

    Args:
        conn: sqlite3 连接（指向唯一数据库 astock_daily.db）
        pool_df: 含 'code' 列（6位）的 DataFrame
        as_of_date_fmt: 选股日 YYYYMMDD
    Returns:
        过滤后的 DataFrame（保留达标股票）
    """
    if pool_df is None or len(pool_df) == 0:
        return pool_df
    try:
        win = pd.read_sql_query(
            "SELECT trade_date FROM daily WHERE trade_date <= ? "
            "ORDER BY trade_date DESC LIMIT ?",
            conn, params=(as_of_date_fmt, lookback))
        if len(win) == 0:
            return pool_df
        win_start = win['trade_date'].min()
        codes = [ts_code(c) for c in pool_df['code']]
        ph = ",".join("?" * len(codes))
        amt = pd.read_sql_query(
            f"SELECT ts_code, AVG(amount) AS avg_amt FROM daily "
            f"WHERE ts_code IN ({ph}) AND trade_date >= ? AND trade_date <= ? "
            f"GROUP BY ts_code",
            conn, params=codes + [win_start, as_of_date_fmt])
        avg_map = dict(zip(amt['ts_code'], amt['avg_amt']))
    except Exception as e:
        print(f"  [WARN] 流动性查询失败，跳过过滤: {e}")
        return pool_df

    keep, dropped = [], 0
    for _, row in pool_df.iterrows():
        t = ts_code(row['code'])
        avg_yuan = avg_map.get(t)
        if avg_yuan is None or avg_yuan * 1000 < min_avg_amount:
            dropped += 1
            continue
        keep.append(row)
    out = pd.DataFrame(keep).reset_index(drop=True) if keep else pool_df.iloc[0:0]
    print(f"  [流动性] 日均成交额≥{min_avg_amount/1e4:.0f}万({lookback}日) "
          f"过滤：保留 {len(out)} 只 / 剔除 {dropped} 只")
    return out


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
    c = str(c).strip()
    """智能处理已有后缀"""
    if '.' in c:
        c = c.split('.')[0]
    c = c.zfill(6)
    return c + (".SH" if c.startswith("6") else ".SZ")


def load_stock_prices(code, start, end, conn, lookback_days=250):
    """
    从本地数据库加载日线行情数据（使用外部传入的连接）
    并返回前复权价格（adj_open, adj_high, adj_low, adj_close）
    参数：
        lookback_days: 回溯天数（用于计算MA200等指标）
    """
    # 计算起始日期（减去 lookback_days）
    start_dt = pd.Timestamp(start)
    lookback_start = (start_dt - pd.Timedelta(days=lookback_days)).strftime("%Y%m%d")
    
    # 加载行情数据 + 复权因子（通过 JOIN）
    query = """
        SELECT d.trade_date, d.open, d.high, d.low, d.close, d.vol,
               af.adj_factor
        FROM daily d
        LEFT JOIN adj_factor af ON d.ts_code = af.ts_code 
                             AND d.trade_date = af.trade_date
        WHERE d.ts_code = ? AND d.trade_date BETWEEN ? AND ?
        ORDER BY d.trade_date
    """
    df = pd.read_sql_query(query, conn, params=(ts_code(code), lookback_start, end))
    if df.empty:
        return None
    
    # 处理缺失的 adj_factor（LEFT JOIN 可能导致 NaN）
    if df['adj_factor'].isna().any():
        na_count = df['adj_factor'].isna().sum()
        print(f"    [WARN] {code} 的 adj_factor 缺失 {na_count} 条，正在填充...")
        # 前向填充 + 后向填充 + 用1.0填充剩余的
        df['adj_factor'] = df['adj_factor'].ffill().bfill().fillna(1.0)
    
    # 确保数值列为 float
    for col in ['open', 'high', 'low', 'close', 'vol', 'adj_factor']:
        if col in df.columns:
            df[col] = df[col].astype(float)
    
    # 计算前复权价格（以最新价格为基准）
    latest_adj = df['adj_factor'].iloc[-1]
    df['adj_open'] = df['open'] * (df['adj_factor'] / latest_adj)
    df['adj_high'] = df['high'] * (df['adj_factor'] / latest_adj)
    df['adj_low'] = df['low'] * (df['adj_factor'] / latest_adj)
    df['adj_close'] = df['close'] * (df['adj_factor'] / latest_adj)
    
    # 统一列名：vol → volume（兼容策略代码）
    if 'vol' in df.columns and 'volume' not in df.columns:
        df = df.rename(columns={'vol': 'volume'})
    
    # 丢弃辅助列，保持 DataFrame 干净
    df = df.drop(columns=['adj_factor'])
    
    return df


_last_bench_meta = {}

def load_benchmark(code, start, end, conn, mode=None):
    """加载基准指数（委托 bench_index，优先全收益口径）。
    返回 DataFrame[trade_date, close]；实际使用的口径记录到 _last_bench_meta[code] 供报告标注。"""
    global _last_bench_meta
    # 本引擎 NAV 口径由 --price-mode 决定（dual/hfq/raw）：
    #   hfq  → 含分红，基准应对齐到全收益
    #   raw  → 不含分红，基准应对齐到价格指数
    #   dual → 同时出两套净值，基准按全收益（与 hfq 轨可比）
    _pm = globals().get("PRICE_MODE")
    _nav_mode = {"hfq": "hfq", "raw": "raw"}.get(str(_pm).lower()) if _pm else None
    df, meta = bi.load_benchmark(code, start, end, conn=conn, mode=mode,
                                 nav_price_mode=_nav_mode)
    _last_bench_meta[code] = meta
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


def calculate_var(return_series, conf_levels=(0.95, 0.99), capital=None, method="both"):
    """
    计算风险价值 VaR（参数法 + 历史法），用于「前瞻风险」报告。

    return_series : 周期收益率序列（小数，0.01=1%）；也可直接传资产净值序列
                    （值常 >>1，会自动差分转收益率）。
    conf_levels   : 置信水平，默认 (0.95, 0.99)
    capital       : 当前组合市值（换算金额用）；None 则只返回百分比
    method        : 'param' | 'hist' | 'both'

    返回 dict：{
        0.95: {'param_loss','hist_loss','param_amt','hist_amt','param_ret','hist_ret'},
        0.99: {...}
    }
    说明：
      - 参数法（正态假设）：分位收益率 = μ - z·σ，z=1.645(95%)/2.326(99%)；
        损失幅度 = max(0, -(μ - z·σ))。
      - 历史法（经验分位，抗肥尾）：直接取 (1-c) 分位收益率，损失=其负值。
      注意：VaR 只覆盖「正常行情」，尾部极端风险(黑天鹅)需另配硬止损。
    """
    rs = np.asarray(return_series, dtype=float)
    if rs.ndim == 1 and len(rs) > 1 and np.max(np.abs(rs)) > 5:
        # 传入的是净值序列 → 转收益率
        rs = np.diff(rs) / rs[:-1]
    rs = rs[np.isfinite(rs)]
    if len(rs) < 5:
        return {c: {"param_loss": 0.0, "hist_loss": 0.0, "param_amt": 0.0,
                   "hist_amt": 0.0, "param_ret": 0.0, "hist_ret": 0.0} for c in conf_levels}
    Z = {0.90: 1.282, 0.95: 1.645, 0.99: 2.326}
    mu = float(np.mean(rs))
    sigma = float(np.std(rs, ddof=1))
    out = {}
    for c in conf_levels:
        z = Z.get(c, 1.645)
        param_ret = mu - z * sigma
        param_loss = max(0.0, -param_ret)
        if method in ("hist", "both"):
            q = max(0.0, min(1.0, 1.0 - c))
            hist_ret = float(np.quantile(rs, q))
        else:
            hist_ret = param_ret
        hist_loss = max(0.0, -hist_ret)
        out[c] = {
            "param_ret": param_ret, "hist_ret": hist_ret,
            "param_loss": param_loss, "hist_loss": hist_loss,
            "param_amt": (param_loss * capital) if capital else 0.0,
            "hist_amt": (hist_loss * capital) if capital else 0.0,
        }
    return out


def _norm_date(d):
    """把各种日期格式（int 20220104 / str '2022-01-04' / Timestamp）归一化为 YYYY-MM-DD"""
    if d is None:
        return None
    s = str(d).strip().split(" ")[0]  # 去掉可能的时分秒
    digits = "".join(ch for ch in s if ch.isdigit())
    if len(digits) >= 8:
        return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"
    return s


def max_drawdown_with_dates(values, dates=None, start_idx=0):
    """
    计算最大回撤，并返回其发生的时间区间（峰值日 → 谷值日）。
    values : 权益曲线（按时间顺序）
    dates  : 与 values 对齐的日期列表（可选；缺省则无法定位时间点）
    start_idx : 实际回测起点在 values 中的下标。权益曲线常包含回测开始前
               的「回溯期」平曲线（value=初始资金），其起点会被误当成峰值日，
               导致回撤区间早于回测开始日。传入 start_idx 后，峰值/谷值只在
               回测窗口内求解，回撤区间必然落在 [start_idx, 末尾] 之内。
    返回 (max_dd%, peak_date, trough_date)
    注意：峰值日必须早于谷值日。若全局最高点出现在谷值之后，
    记录的是「谷值时刻对应的运行峰值」而非更晚的全局最高点。
    """
    if not values or len(values) < 2:
        return 0.0, None, None
    # 回测窗口起点（防御越界）
    if start_idx < 0 or start_idx >= len(values) - 1:
        start_idx = 0
    peak = values[start_idx]
    peak_idx = start_idx
    max_dd = 0.0
    trough_idx = start_idx
    peak_idx_at_trough = start_idx
    for i in range(start_idx, len(values)):
        val = values[i]
        if val > peak:
            peak = val
            peak_idx = i
        dd = (peak - val) / peak * 100
        if dd > max_dd:
            max_dd = dd
            trough_idx = i
            peak_idx_at_trough = peak_idx  # 记录谷值时刻对应的峰值位置（必 ≤ trough_idx）
    pk = dates[peak_idx_at_trough] if (dates and peak_idx_at_trough < len(dates)) else None
    tr = dates[trough_idx] if (dates and trough_idx < len(dates)) else None
    return max_dd, pk, tr


def _dd_period_str(dd_period):
    """把 (peak_date, trough_date) 格式化为 'YYYY-MM~YYYY-MM'，无数据返回 '--'"""
    pk, tr = (dd_period or (None, None))
    pk = _norm_date(pk)
    tr = _norm_date(tr)
    if pk and tr:
        return f"{pk}~{tr}"
    return pk or tr or "--"


def _aggregate_dd_years(results):
    """统计各股票最大回撤峰值所在年份分布，返回 {年份: 数量}（按年份升序）"""
    counts = {}
    for r in results:
        period = r.get("dd_period")
        if not period or period[0] is None:
            continue
        y = str(period[0])[:4]
        if y.isdigit():
            counts[y] = counts.get(y, 0) + 1
    return dict(sorted(counts.items()))


def backtest_buy_hold(df, capital, cfg, start_idx=0):
    """
    买入持有（模拟普通股民操作：买入后一直持有，可启用ATR动态止损）
    
    使用复权价格计算收益，确保与插件策略的价格基准一致。
    
    ATR动态止损（可选）：
    - 启用后，当收盘价跌破ATR止损价时触发止损
    - 止损后不再重新买入（保持"买入持有"语义）
    """
    # 使用复权价格（与插件策略一致）
    close_col = "adj_close" if "adj_close" in df.columns else "close"
    high_col = "adj_high" if "adj_high" in df.columns else "high"
    low_col = "adj_low" if "adj_low" in df.columns else "low"
    
    close = df[close_col].values
    high = df[high_col].values
    low = df[low_col].values
    n = len(close)
    
    # 买入持有：全部资金在 start_idx 买入，一直持有
    p0 = close[start_idx]
    shares = int(capital / p0 / 100) * 100
    if shares == 0:
        return 0.0, 0, 0.0
    
    cash = capital - shares * p0 - calc_fee_rb("buy", p0, shares, trade_date=df.iloc[start_idx]["trade_date"])  # 买入成本（佣金+滑点）
    
    # 检查是否启用ATR止损
    use_atr_stop = cfg.get("use_atr_stop", False)
    
    if use_atr_stop:
        # 导入ATR止损模块
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from backtest.atr_stop_loss import ATRStopLoss
        
        # 初始化ATR止损
        atr_period = cfg.get("atr_period", 14)
        atr_mult = cfg.get("atr_mult", 3.0)
        trail_mult = cfg.get("trail_mult", 3.0)
        
        sl = ATRStopLoss(atr_period=atr_period, atr_mult=atr_mult, trail_mult=trail_mult)
        atr_values = sl.calc_atr(high, low, close)
        
        # 建仓
        entry_atr = atr_values[start_idx] if start_idx < len(atr_values) else 0.0
        if entry_atr > 0:
            sl.on_entry(entry_price=p0, atr_val=entry_atr)
        
        # 追踪止损
        portfolio_values = []
        dates = []
        exit_idx = n - 1  # 默认持有到最后
        
        for i in range(n):
            if i < start_idx:
                portfolio_values.append(capital)  # 未买入前，保持现金
                dates.append(df.iloc[i]["trade_date"])
            else:
                # 检查是否触发止损（从 start_idx+1 开始）
                if i > start_idx and entry_atr > 0:
                    curr_atr = atr_values[i] if i < len(atr_values) else entry_atr
                    sl.update(high_price=high[i], atr_val=curr_atr)
                    
                    should_stop, stop_price, reason = sl.check_stop(close_price=close[i])
                    if should_stop:
                        exit_idx = i
                        break
                
                portfolio_values.append(cash + shares * close[i])
                dates.append(df.iloc[i]["trade_date"])
        
        # 如果止损触发，后续日期保持现金（卖出后）
        if exit_idx < n - 1:
            sell_price = close[exit_idx]
            cash_after_sell = cash + shares * sell_price - calc_fee_rb("sell", sell_price, shares, trade_date=df.iloc[exit_idx]["trade_date"])  # 卖出成本（佣金+印花税+滑点）
            for i in range(exit_idx + 1, n):
                portfolio_values.append(cash_after_sell)
                dates.append(df.iloc[i]["trade_date"])
        
        # 计算最终收益
        final_value = portfolio_values[-1]
        ret = (final_value / capital - 1) * 100
        max_dd, pk, tr = max_drawdown_with_dates(portfolio_values, dates, start_idx)

        return ret, 1, max_dd, (pk, tr)
    
    else:
        # 原始逻辑：一直持有
        portfolio_values = []
        dates = []
        for i in range(n):
            if i < start_idx:
                portfolio_values.append(capital)  # 未买入前，保持现金
                dates.append(df.iloc[i]["trade_date"])
            else:
                portfolio_values.append(cash + shares * close[i])
                dates.append(df.iloc[i]["trade_date"])
        
        final = portfolio_values[-1] - calc_fee_rb("sell", close[n - 1], shares, trade_date=df.iloc[n - 1]["trade_date"])  # 期末按收盘价卖出，扣除卖出成本
        ret = (final / capital - 1) * 100
        max_dd, pk, tr = max_drawdown_with_dates(portfolio_values, dates, start_idx)

        return ret, 0, max_dd, (pk, tr)


def backtest_rsi(df, capital, cfg, start_idx=0):
    """RSI超卖买入 / 超买卖出"""
    period = cfg.get("rsi_period", 14)
    kelly_cap = _make_kelly_cap(cfg)  # 凯利总持仓封顶（None=不封顶）
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
    trade_records = []
    portfolio_values = []
    dates = []
    
    for i in range(n):
        if i < start_idx:
            portfolio_values.append(capital)
            dates.append(df.iloc[i]["trade_date"])
            continue
        p = close[i]
        prev_rsi = rsi[i-1] if i > 0 else 50
        
        if pos == 0 and prev_rsi < ovs:
            intended = int(cash * 0.5 / p / 100) * 100
            pos = BaseStrategy.cap_by_kelly(cfg.get("total_capital", capital), 0, cash, kelly_cap, p, intended)
            if pos > 0:
                cash -= pos * p * 1.0002
                cost = p
                trade_records.append({"action": "BUY", "price": p, "shares": pos})
        elif pos > 0:
            if prev_rsi > ovb or (p > cost * (1+tp)) or (p < cost * (1-sl)):
                cash += pos * p * 0.9988
                trade_records.append({"action": "SELL", "price": p, "shares": pos})
                pos, cost = 0, 0.0
        
        portfolio_values.append(cash + pos * p)
        dates.append(df.iloc[i]["trade_date"])
    
    # 循环结束后计算最终收益率和最大回撤
    final = portfolio_values[-1] if portfolio_values else capital
    ret = (final / capital - 1) * 100
    max_dd, pk, tr = max_drawdown_with_dates(portfolio_values, dates, start_idx)

    return ret, trade_records, max_dd, (pk, tr)


def backtest_bollinger(df, capital, cfg, start_idx=0):
    """布林带：跌破下轨买，突破上轨卖"""
    period = cfg.get("period", 20)
    kelly_cap = _make_kelly_cap(cfg)  # 凯利总持仓封顶（None=不封顶）
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
    trade_records = []
    portfolio_values = []
    dates = []
    
    for i in range(n):
        if i < start_idx:
            portfolio_values.append(capital)
            dates.append(df.iloc[i]["trade_date"])
            continue
        p = close[i]
        if i >= period and pos == 0 and p <= lower[i] > 0:
            intended = int(cash * 0.5 / p / 100) * 100
            pos = BaseStrategy.cap_by_kelly(cfg.get("total_capital", capital), 0, cash, kelly_cap, p, intended)
            if pos > 0:
                cash -= pos * p * 1.0002
                cost = p
                trade_records.append({"action": "BUY", "price": p, "shares": pos})
        elif pos > 0 and i >= period:
            if p >= upper[i] or p > cost * (1+tp) or p < cost * (1-sl):
                cash += pos * p * 0.9988
                trade_records.append({"action": "SELL", "price": p, "shares": pos})
                pos, cost = 0, 0.0
        portfolio_values.append(cash + pos * p)
        dates.append(df.iloc[i]["trade_date"])
    
    final = portfolio_values[-1] if portfolio_values else capital
    ret = (final / capital - 1) * 100
    max_dd, pk, tr = max_drawdown_with_dates(portfolio_values, dates, start_idx)

    return ret, trade_records, max_dd, (pk, tr)


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

    kelly_cap = _make_kelly_cap(cfg)  # 凯利总持仓封顶（None=不封顶）

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
            return None, [], 0.0
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    n = len(close)
    min_n = max(long_period, atr_period, short_exit, long_exit) + 1
    if n < min_n:
        return None, [], 0.0

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

    trade_records = []
    portfolio_values = []  # 记录每日总资产（用于计算收益）
    dates = []  # 与 portfolio_values 对齐的交易日（用于定位回撤区间）

    # 计算策略实际起始位置（跳过回溯期）
    loop_start = max(start_idx, long_period, atr_period, long_exit)
    if loop_start >= n:
        return 0.0, [], 0.0

    for i in range(n):
        # 跳过回溯期（只记录资产，不执行交易逻辑）
        if i < loop_start:
            portfolio_values.append(cash + s1_pos * close[i] + s2_pos * close[i])
            dates.append(df.iloc[i]["trade_date"])
            continue

        p = close[i]
        if p <= 0:
            portfolio_values.append(cash + s1_pos * p + s2_pos * p)
            dates.append(df.iloc[i]["trade_date"])
            continue

        atr_i = atr[i] if i >= atr_period else atr[atr_period] if atr_period < n else 0
        if atr_i <= 0:
            portfolio_values.append(cash + s1_pos * p + s2_pos * p)
            dates.append(df.iloc[i]["trade_date"])
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
                    trade_records.append({"action": "SELL", "price": p, "shares": s1_pos})
                    s1_pos = 0; s1_adds = 0
                if s2_pos > 0:
                    cash += s2_pos * p * sell_cost
                    trade_records.append({"action": "SELL", "price": p, "shares": s2_pos})
                    s2_pos = 0; s2_adds = 0

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
                    unit_shares = BaseStrategy.cap_by_kelly(cfg.get("total_capital", capital), s1_pos + s2_pos, cash, kelly_cap, p, unit_shares)
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
                            trade_records.append({"action": "BUY", "price": p, "shares": unit_shares})

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
                    add_shares = BaseStrategy.cap_by_kelly(cfg.get("total_capital", capital), s1_pos + s2_pos, cash, kelly_cap, p, add_shares)
                    cost = add_shares * p * buy_cost
                    if add_shares > 0 and cost <= cash:
                        cash -= cost
                        s1_cost = (s1_cost * s1_pos + p * add_shares) / (s1_pos + add_shares)
                        s1_pos += add_shares
                        s1_last_add_price = p
                        s1_adds += 1
                        s1_highest = max(s1_highest, p)
                        trade_records.append({"action": "BUY", "price": p, "shares": add_shares})

            # 止损：跌破止损价 或 跌破10日低点
            if s1_pos > 0:
                s1_highest = max(s1_highest, p)
                # 追踪止损价（随着最高价上移）
                trail_stop = s1_highest - trail_atr_mult * atr_i
                s1_stop_price = max(s1_stop_price, trail_stop)
                if p <= s1_stop_price or (i >= short_exit and p < s1_exit[i]):
                    cash += s1_pos * p * sell_cost
                    trade_records.append({"action": "SELL", "price": p, "shares": s1_pos})
                    s1_pos = 0; s1_adds = 0; s1_stop_price = 0

        # ── 系统2（长期）────────────────────────────────
        if use_long:
            if s2_pos == 0 and i >= long_period:
                if p > s2_entry[i] and volume_ok[i] and (not trend_filter or p > trend_ma[i]):
                    # 计算建仓单位（1% 风险原则）
                    risk_capital = capital * risk_pct  # 用初始总资金
                    unit_shares = int(risk_capital / atr_i) if atr_i > 0 else 0
                    unit_shares = max((unit_shares // 100) * 100, 100)  # 取整到100股，最少100股
                    unit_shares = BaseStrategy.cap_by_kelly(cfg.get("total_capital", capital), s1_pos + s2_pos, cash, kelly_cap, p, unit_shares)
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
                            trade_records.append({"action": "BUY", "price": p, "shares": unit_shares})

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
                    add_shares = BaseStrategy.cap_by_kelly(cfg.get("total_capital", capital), s1_pos + s2_pos, cash, kelly_cap, p, add_shares)
                    cost = add_shares * p * buy_cost
                    if add_shares > 0 and cost <= cash:
                        cash -= cost
                        s2_cost = (s2_cost * s2_pos + p * add_shares) / (s2_pos + add_shares)
                        s2_pos += add_shares
                        s2_last_add_price = p
                        s2_adds += 1
                        s2_highest = max(s2_highest, p)
                        trade_records.append({"action": "BUY", "price": p, "shares": add_shares})

            if s2_pos > 0:
                s2_highest = max(s2_highest, p)
                trail_stop = s2_highest - trail_atr_mult * atr_i
                s2_stop_price = max(s2_stop_price, trail_stop)
                if p <= s2_stop_price or (i >= long_exit and p < s2_exit[i]):
                    cash += s2_pos * p * sell_cost
                    trade_records.append({"action": "SELL", "price": p, "shares": s2_pos})
                    s2_pos = 0; s2_adds = 0; s2_stop_price = 0

        # 记录当日总资产
        total_val = cash + s1_pos * p + s2_pos * p
        portfolio_values.append(total_val)
        dates.append(df.iloc[i]["trade_date"])
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
    max_dd, pk, tr = max_drawdown_with_dates(portfolio_values, dates, start_idx)
    return ret, trade_records, max_dd, (pk, tr)


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
        return 0.0, trades, 0.0

    # 计算最终资产
    final_value = daily_values[-1]["portfolio_value"] if daily_values else capital
    ret = (final_value / capital - 1) * 100
    
    # 计算最大回撤
    portfolio_values = [v["portfolio_value"] for v in daily_values]
    max_dd = calculate_max_drawdown(portfolio_values)

    return ret, trades, max_dd


def _get_all_stocks_from_db():
    """
    从本地数据库 stock_basic 表获取所有 A 股股票列表

    Returns:
        DataFrame with columns: ts_code, name, industry
    """
    import sqlite3
    conn = sqlite3.connect(DATA["local_db_path"])
    try:
        # 存活股（当前上市）
        alive = pd.read_sql_query(
            "SELECT ts_code, name, COALESCE(industry, '未知') AS industry "
            "FROM stock_basic WHERE ts_code NOT LIKE '%.BJ'",
            conn,
        )
        # 退市股（在 daily 有历史成交，但已不在 stock_basic）→ 纳入以消除幸存者偏差
        delisted = pd.read_sql_query(
            "SELECT DISTINCT d.ts_code, COALESCE(sb.name, d.ts_code) AS name, '未知' AS industry "
            "FROM daily d LEFT JOIN stock_basic sb ON d.ts_code = sb.ts_code "
            "WHERE sb.ts_code IS NULL AND d.ts_code NOT LIKE '%.BJ'",
            conn,
        )
        df = pd.concat([alive, delisted], ignore_index=True)
        df = df.drop_duplicates(subset=["ts_code"]).sort_values("ts_code").reset_index(drop=True)
        # 添加 code 列（6位数字代码，与 get_hs300_components 返回格式一致）
        df['code'] = df['ts_code'].str.extract(r'(\d{6})', expand=False)
        print(f"  ✓ 从 stock_basic 获取 {len(alive)} 只 + 退市股 {len(delisted)} 只 = {len(df)} 只 A 股（含退市，修正幸存者偏差）")
        return df
    except Exception as e:
        print(f"  [ERR] 获取全市场股票失败: {e}，降级使用中证800+中证500并集")
        df = pd.read_sql_query(
            "SELECT DISTINCT d.ts_code, COALESCE(sb.name, d.ts_code) AS name, '未知' AS industry "
            "FROM index_constituent d LEFT JOIN stock_basic sb ON d.ts_code = sb.ts_code "
            "WHERE d.index_code IN ('000300.SH','000905.SH') "
            "AND d.trade_date = (SELECT MAX(trade_date) FROM index_constituent)",
            conn,
        )
        # 添加 code 列（6位数字代码）
        df['code'] = df['ts_code'].str.extract(r'(\d{6})', expand=False)
        print(f"  ✓ 降级获取 {len(df)} 只股票（沪深300+中证500）")
        return df
    finally:
        conn.close()


def _get_prev_trading_day(date_str):
    """
    从数据库查询指定日期之前的最近一个交易日
    
    Args:
        date_str: YYYYMMDD 格式的日期
    
    Returns:
        str: YYYYMMDD 格式的交易日，如查询失败返回 date_str 本身
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.execute(
            "SELECT MAX(trade_date) FROM daily WHERE trade_date < ?",
            (date_str,),
        )
        row = cur.fetchone()
        conn.close()
        if row and row[0]:
            return str(row[0])
        print(f"  [WARN] 未找到 {date_str} 前的交易日，使用原日期")
        return date_str
    except Exception as e:
        print(f"  [WARN] 查询前交易日失败: {e}，使用原日期")
        return date_str


def _get_index_constituents_from_db(index_code: str, as_of_date: str = None) -> pd.DataFrame:
    """
    直接从本地DB查询指数成分股。
    as_of_date: YYYYMMDD，返回该日期（含）之前最新快照的成分股；
                为 None 时返回全局最新快照（保留旧行为，含幸存者偏差）。
    传入 as_of_date 可消除幸存者偏差（含曾成分但已退市的股票）。
    """
    try:
        conn = sqlite3.connect(DB_PATH)

        if as_of_date:
            snap = pd.read_sql_query(
                "SELECT MAX(REPLACE(trade_date,'-','')) AS d FROM index_constituent "
                "WHERE index_code=? AND REPLACE(trade_date,'-','') <= ?",
                conn, params=(index_code, str(as_of_date)))
            query_date = str(snap.iloc[0]['d']) if len(snap) and snap.iloc[0]['d'] else None
            # ── 数据边界保护 ──────────────────────────────────────────────
            # 本地成分快照有左边界：例如沪深300最早仅 20160129、中证500最早 20150130。
            # 当回测起点很早(选股日早于成分数据起始)时，上面查不到快照会直接返回空池→崩溃。
            # 此时回退到该指数「最早可用快照」，让长周期回测能跑起来；
            # 下游 list_date 过滤会剔除选股日之后才 IPO 的新股，部分抵消前视偏差。
            if not query_date:
                earliest = pd.read_sql_query(
                    "SELECT MIN(REPLACE(trade_date,'-','')) AS d FROM index_constituent "
                    "WHERE index_code=?",
                    conn, params=(index_code,))
                query_date = str(earliest.iloc[0]['d']) if len(earliest) and earliest.iloc[0]['d'] else None
                if query_date:
                    print(f"  [WARN] 指数 {index_code} 在 {as_of_date} 及之前无成分快照，"
                          f"回退到最早可用快照 {query_date}（轻微前视·仅用于长周期回测，"
                          f"已用上市日期过滤部分抵消）")
        else:
            cur = conn.execute(
                "SELECT MAX(trade_date) FROM index_constituent WHERE index_code=?",
                (index_code,))
            row = cur.fetchone()
            query_date = str(row[0]) if row and row[0] else None

        if not query_date:
            conn.close()
            logger.warning(f"未找到指数 {index_code} 的任何数据")
            return pd.DataFrame(columns=['code', 'name'])

        # 查询该快照日成分股（用 REPLACE 兼容两种日期格式）
        df = pd.read_sql_query(
            "SELECT ts_code FROM index_constituent WHERE index_code=? AND REPLACE(trade_date, '-', '')=?",
            conn,
            params=(index_code, query_date)
        )

        if len(df) > 0:
            # 提取6位数字代码（去掉交易所后缀）
            df['code'] = df['ts_code'].str.extract(r'(\d{6})', expand=False)
            df['name'] = ''
            conn.close()
            result = df[['code', 'name']].reset_index(drop=True)
            logger.info(f"从本地DB获取 {index_code} 成分股: {len(result)} 只 (快照日期:{query_date})")
            return result
        else:
            logger.warning(f"查询指数 {index_code} 返回0行")
            conn.close()
    except Exception as e:
        logger.warning(f"查询指数成分股失败 ({index_code}): {e}")
    # 查询失败，返回空DataFrame
    return pd.DataFrame(columns=['code', 'name'])


def _get_hs300_from_db(as_of_date=None) -> pd.DataFrame:
    """直接从本地DB查询沪深300成分股（as_of_date 可消除幸存者偏差）"""
    return _get_index_constituents_from_db('000300.SH', as_of_date)


def _get_zz500_from_db(as_of_date=None) -> pd.DataFrame:
    """直接从本地DB查询中证500成分股（as_of_date 可消除幸存者偏差）"""
    return _get_index_constituents_from_db('000905.SH', as_of_date)


def _get_zz800_from_db(as_of_date=None) -> pd.DataFrame:
    """直接从本地DB查询中证800成分股（沪深300+中证500）"""
    hs300 = _get_index_constituents_from_db('000300.SH', as_of_date)
    zz500 = _get_index_constituents_from_db('000905.SH', as_of_date)
    # 合并去重
    combined = pd.concat([hs300, zz500], ignore_index=True)
    combined = combined.drop_duplicates(subset=['code']).reset_index(drop=True)
    logger.info(f"中证800成分股: 沪深300({len(hs300)}) + 中证500({len(zz500)}) = {len(combined)}")
    return combined


def _get_zz1000_from_db(as_of_date=None) -> pd.DataFrame:
    """直接从本地DB查询中证1000成分股（as_of_date 可消除幸存者偏差）"""
    return _get_index_constituents_from_db('000852.SH', as_of_date)


# [已移除] _get_kcb_cyb_from_db：科创板+创业板(高风险) 股票池已按需求删除（对散户不友好）


def run_selection():
    """执行多因子选股，返回 TOP N 股票列表"""
    
    # ── 动态计算选股日：回测开始日 T 的前一个交易日 T-1 ──
    backtest_start = BACKTEST["start_date"]
    prev_day = _get_prev_trading_day(backtest_start)
    SELECTION["date"] = prev_day
    print(f"  选股日自动计算: 回测开始 {backtest_start} → 前交易日 {prev_day}")

    # 选股日统一格式 YYYYMMDD（供指数成分股"历史时点快照"查询，消除幸存者偏差）
    sel_date = SELECTION["date"]
    selection_date_fmt = sel_date.replace("-", "") if (len(sel_date) == 10 and "-" in sel_date) else sel_date
    
    from src.data_fetcher import DataFetcher
    from src.factor_calculator import FactorCalculator
    from src.factor_processor import FactorProcessor
    from src.stock_selector import StockSelector

    print(f"\n{'='*60}")
    print(f"  选股阶段 — {SELECTION['date']} | 池:{SELECTION['stock_pool']} | TOP {SELECTION['top_n']}")
    print(f"{'='*60}")

    # 本地数据库已有完整的 fina_indicator（25万+行）和 daily_basic 数据，
    # 关闭 Tushare 备份，避免全市场选股时 Tushare API 卡死
    DATA["use_tushare_backup"] = False

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
    # 多选一倍候选，供回测时递补（如 top_n=5 则选10只）
    _candidate_n = max(SELECTION["top_n"] * 2, SELECTION["top_n"] + 10)
    selector = StockSelector(config={"top_n": _candidate_n})

    # 根据 stock_pool 配置动态选择股票池
    pool_map = {
        "hs300": lambda: _get_hs300_from_db(selection_date_fmt),
        "zz500": lambda: _get_zz500_from_db(selection_date_fmt),
        "zz800": lambda: _get_zz800_from_db(selection_date_fmt),
        "zz1000": lambda: _get_zz1000_from_db(selection_date_fmt),
        "all":   lambda: _get_all_stocks_from_db(),
    }
    get_pool = pool_map.get(SELECTION["stock_pool"], fetcher.get_hs300_components)
    
    pool = get_pool()
    
    # ── 新增：过滤掉回测起始日还没上市的股票 ────────────────
    print(f"\n  [过滤] 检查股票上市日期...")
    original_len = len(pool)  # 保存过滤前的股票数量
    
    try:
        import sqlite3
        conn = sqlite3.connect(DB_PATH)
        
        # 获取选股基准日期（格式: YYYYMMDD，如 20220103）
        selection_date = SELECTION['date']
        
        # 确保 selection_date 是 YYYYMMDD 格式（无横杠）
        if len(selection_date) == 10 and '-' in selection_date:
            # 如果是 YYYY-MM-DD 格式，去掉横杠
            selection_date_fmt = selection_date.replace('-', '')
        else:
            selection_date_fmt = selection_date
        
        print(f"    选股日期: {selection_date} => 比较用: {selection_date_fmt}")
        
        # 过滤：只保留 list_date <= selection_date 的股票
        filtered_pool = []
        skip_count = 0
        for _, row in pool.iterrows():
            code = row['code']
            ts_code_str = ts_code(code)  # 转换为 Tushare 格式（避免覆盖函数名）
            
            try:
                result = conn.execute(
                    "SELECT list_date FROM stock_basic WHERE ts_code = ?", 
                    (ts_code_str,)
                ).fetchone()
                
                if result is None:
                    # 退市股不在 stock_basic（无 list_date），但历史上已上市 → 保留（修正幸存者偏差）
                    filtered_pool.append(row)
                    continue
                
                list_date = result[0]
                
                # 确保 list_date 是 YYYYMMDD 格式（无横杠）
                if len(list_date) == 10 and '-' in list_date:
                    list_date_fmt = list_date.replace('-', '')
                else:
                    list_date_fmt = list_date
                
                # 比较日期：如果 list_date <= selection_date，则保留
                if list_date_fmt <= selection_date_fmt:
                    filtered_pool.append(row)
                    
            except Exception as e:
                # 出错则跳过
                skip_count += 1
                continue
        
        conn.close()
        
        # 更新 pool
        pool = pd.DataFrame(filtered_pool).reset_index(drop=True)
        
        print(f"    过滤前: {original_len} 只")
        print(f"    过滤后: {len(pool)} 只")
        print(f"    过滤掉: {original_len - len(pool)} 只（上市晚于 {selection_date}）")
        if skip_count > 0:
            print(f"    [注] 跳过（无上市日期）: {skip_count} 只")
        
    except Exception as e:
        print(f"  [WARN] 过滤上市日期失败: {e}")
        print(f"  [WARN] 将继续使用全部股票池（可能包含未上市股票）")
    
    # 验证股票池数量
    expected_max = {"hs300": 300, "zz500": 500, "zz800": 800, "zz1000": 1000}.get(SELECTION["stock_pool"], 10000)
    if len(pool) > expected_max * 1.5:  # 允许50%误差（如有重复）
        print(f"  [WARN] 警告: 股票池数量异常！配置={SELECTION['stock_pool']}, 预期≤{expected_max}, 实际={len(pool)}")
        print(f"  [WARN] 可能误用了全市场数据，请检查 get_zz800_components 实现")
    print(f"  股票池 [{SELECTION['stock_pool']}]: {len(pool)} 只")

    # ── 流动性预过滤（保守阈值，主要跑大盘股时几乎不剔除成分股）────
    try:
        conn = sqlite3.connect(DB_PATH)
        pool = prefilter_by_liquidity(conn, pool, selection_date_fmt)
        conn.close()
    except Exception as e:
        print(f"  [WARN] 流动性过滤异常，沿用原股票池: {e}")

    # 防御：确保 pool 有 code 列
    if pool.empty or "code" not in pool.columns:
        print(f"  [ERROR] 股票池为空或列名错误！请检查 _get_{SELECTION['stock_pool']}_from_db 实现")
        print(f"  [DEBUG] pool.empty={pool.empty}, columns={list(pool.columns)}")
        return pd.DataFrame()

    factors = calculator.calculate_all_factors(
        pool["code"].tolist(), fetcher,
        start_date=None,          # 自动计算（end_date 前 2 年）
        end_date=SELECTION["date"],
        max_workers=5
    )
    print(f"  因子计算: {len(factors)} 只 × {len([c for c in factors.columns if c.startswith(('VF','GF','QF','MF','TF','LVF','MWF'))])} 因子")

    processed = processor.process(factors)
    selected = selector.select(processed, top_n=_candidate_n)

    # 补全股票名称（直接从 stock_basic 表读取，不依赖 ts_code 函数）
    import sqlite3
    DB = DB_PATH
    names = {}
    try:
        conn = sqlite3.connect(DB)
        codes_to_query = selected["code"].tolist()
        
        for code in codes_to_query:
            # 直接拼 Tushare 格式：6开头.SH，0/3开头.SZ
            tsc = code + (".SH" if code.startswith("6") else ".SZ")
            
            r = conn.execute("SELECT name FROM stock_basic WHERE ts_code=?", (tsc,)).fetchone()
            if r and r[0]:
                names[code] = r[0]
            else:
                # 调试：打印找不到名称的股票（已禁用）
                pass
        
        conn.close()
        
    except Exception as e:
        print(f"  [WARN] 补全股票名称失败: {e}")
    
    # 填充 name 列
    selected["name"] = selected["code"].map(names).fillna("")
    # 检查是否有名称仍为空的
    empty_names = (selected["name"] == "").sum()
    if empty_names > 0:
        print(f"  [WARN] 仍有 {empty_names} 只股票名称为空")
        # 打印前3只名称为空的股票代码
        empty_codes = selected[selected["name"] == ""]["code"].head(3).tolist()
        print(f"  [WARN] 示例: {empty_codes}")

    selector.print_top_stocks(selected, n=min(20, len(selected)))

    # 保存选股结果 CSV（命名规则: selection/multi_YYYYMM_selection.csv）
    if OUTPUT.get("save_csv"):
        sel_dir = OUTPUT.get("selection_dir", os.path.join(OUTPUT["dir"], "selection"))
        os.makedirs(sel_dir, exist_ok=True)
        from datetime import datetime
        ym = SELECTION["date"][:6]  # "20220103" -> "202201"
        csv_path = os.path.join(sel_dir, f"multi_{ym}_selection.csv")
        selected[["code", "name"]].to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"  选股结果已保存 → {csv_path}")

    return selected
    
    
# ══════════════════════════════════════════════════════
# 价值投资选股
# ══════════════════════════════════════════════════════

def run_value_selection():
    """
    执行价值投资选股，返回选股结果DataFrame
    选股失败时自动后移交易日重试，直到选出股票或达到最大尝试次数
    """
    from src.value_stock_selector import ValueStockSelector
    from src.data_fetcher import DataFetcher

    backtest_start = BACKTEST["start_date"]
    VALUE_STRATEGY["top_n"] = SELECTION["top_n"]
    VALUE_STRATEGY["stock_pool"] = SELECTION["stock_pool"]

    # 获取交易日列表（用于后移）
    import sqlite3
    conn = sqlite3.connect(DB_PATH)
    trade_dates = pd.read_sql_query(
        "SELECT DISTINCT trade_date FROM daily ORDER BY trade_date",
        conn
    )["trade_date"].tolist()
    conn.close()

    # 找到回测开始日前一个交易日的索引
    prev_day = _get_prev_trading_day(backtest_start)
    if prev_day not in trade_dates:
        print(f"  [ERROR] 选股日 {prev_day} 不在交易日列表中！")
        return pd.DataFrame()
    date_idx = trade_dates.index(prev_day)

    MAX_ATTEMPTS = 20  # 最多尝试20个交易日

    for attempt in range(MAX_ATTEMPTS):
        current_date = trade_dates[date_idx]
        VALUE_STRATEGY["date"] = current_date

        if attempt == 0:
            print(f"  价值投资选股日: 回测开始 {backtest_start} → 前交易日 {current_date}")
        else:
            print(f"  [重试] 选股日后移 {attempt} 天 → {current_date}")

        # 配置数据源
        df_config = {
            "primary_source": DATA["primary_source"],
            "tushare_token": DATA.get("tushare_token", ""),
            "local_db_path": DATA["local_db_path"],
            "use_akshare_backup": DATA["use_akshare_backup"],
            "use_tushare_backup": False,
        }

        fetcher = DataFetcher(**df_config)
        _orig_top_n = VALUE_STRATEGY["top_n"]
        VALUE_STRATEGY["top_n"] = max(_orig_top_n * 2, _orig_top_n + 10)
        selector = ValueStockSelector(VALUE_STRATEGY, fetcher)
        VALUE_STRATEGY["top_n"] = _orig_top_n

        print(f"\n{'='*60}")
        print(f"  价值投资选股 — {current_date} | 池:{VALUE_STRATEGY['stock_pool']}")
        print(f"{'='*60}")

        selected = selector.select_stocks(date=current_date)

        if selected is not None and len(selected) > 0:
            # 选股成功，保存结果
            output_dir = VALUE_STRATEGY.get("output_dir", "data/results/value_strategy")
            output_file = VALUE_STRATEGY.get("output_file", "value_selection_{date}.csv")
            filepath = selector.export_to_csv(selected, filename=output_file, output_dir=output_dir)

            print(f"\n{'='*60}")
            print(f"  价值投资选股完成！共找到 {len(selected)} 只股票（选股日: {current_date}）")
            print(f"  结果已保存 → {filepath}")
            print(f"{'='*60}\n")

            return selected[["code", "name"]] if "name" in selected.columns else selected[["code"]]

        # 选股失败，后移一天
        print(f"  [WARN] 选股失败，后移到下一交易日重试...")
        date_idx += 1
        if date_idx >= len(trade_dates):
            print(f"\n  [ERROR] 已尝试 {attempt+1} 个交易日，仍无法选出股票！")
            break

    return pd.DataFrame()
    

# ══════════════════════════════════════════════════════
# 回测函数
# ══════════════════════════════════════════════════════

def run_dividend_low_vol_selection() -> pd.DataFrame:
    """
    执行红利低波选股，返回选股结果DataFrame
    选股失败时自动后移交易日重试，直到选出股票或达到最大尝试次数
    """
    from src.dividend_low_vol_selector import DividendLowVolSelector
    from src.data_fetcher import DataFetcher

    backtest_start = BACKTEST["start_date"]
    DIVIDEND_LOW_VOL["top_n"] = SELECTION["top_n"]
    DIVIDEND_LOW_VOL["stock_pool"] = SELECTION["stock_pool"]

    # 获取交易日列表（用于后移）
    import sqlite3
    conn = sqlite3.connect(DB_PATH)
    trade_dates = pd.read_sql_query(
        "SELECT DISTINCT trade_date FROM daily ORDER BY trade_date",
        conn
    )["trade_date"].tolist()
    conn.close()

    # 找到回测开始日前一个交易日的索引
    prev_day = _get_prev_trading_day(backtest_start)
    if prev_day not in trade_dates:
        print(f"  [ERROR] 选股日 {prev_day} 不在交易日列表中！")
        return pd.DataFrame()
    date_idx = trade_dates.index(prev_day)

    MAX_ATTEMPTS = 20  # 最多尝试20个交易日

    for attempt in range(MAX_ATTEMPTS):
        current_date = trade_dates[date_idx]
        DIVIDEND_LOW_VOL["date"] = current_date

        if attempt == 0:
            print(f"  红利低波选股日: 回测开始 {backtest_start} → 前交易日 {current_date}")
        else:
            print(f"  [重试] 选股日后移 {attempt} 天 → {current_date}")

        # 多选一倍候选，供回测时递补
        DIVIDEND_LOW_VOL["_orig_top_n"] = DIVIDEND_LOW_VOL["top_n"]
        DIVIDEND_LOW_VOL["top_n"] = max(DIVIDEND_LOW_VOL["top_n"] * 2, DIVIDEND_LOW_VOL["top_n"] + 10)

        df_config = {
            "primary_source": DATA["primary_source"],
            "tushare_token": DATA.get("tushare_token", ""),
            "local_db_path": DATA["local_db_path"],
            "use_akshare_backup": DATA["use_akshare_backup"],
            "use_tushare_backup": False,
        }
        fetcher = DataFetcher(**df_config)
        selector = DividendLowVolSelector(DIVIDEND_LOW_VOL, fetcher)
        DIVIDEND_LOW_VOL["top_n"] = DIVIDEND_LOW_VOL["_orig_top_n"]
        del DIVIDEND_LOW_VOL["_orig_top_n"]

        print(f"\n{'='*60}")
        print(f"  红利低波选股 — {current_date} | 池:{DIVIDEND_LOW_VOL['stock_pool']}")
        print(f"{'='*60}")

        selected = selector.select_stocks(date=current_date)

        if selected is not None and len(selected) > 0:
            # 选股成功，保存结果
            output_dir = DIVIDEND_LOW_VOL.get("output_dir", "data/results/dividend_low_vol")
            output_file = DIVIDEND_LOW_VOL.get("output_file", "dividend_low_vol_{date}.csv")
            filepath = selector.export_to_csv(selected, filename=output_file, output_dir=output_dir)

            print(f"\n{'='*60}")
            print(f"  红利低波选股完成！共找到 {len(selected)} 只股票（选股日: {current_date}）")
            print(f"  结果已保存 → {filepath}")
            print(f"{'='*60}\n")

            return selected[["code", "name"]] if "name" in selected.columns else selected[["code"]]

        # 选股失败，后移一天
        print(f"  [WARN] 选股失败，后移到下一交易日重试...")
        date_idx += 1
        if date_idx >= len(trade_dates):
            print(f"\n  [ERROR] 已尝试 {attempt+1} 个交易日，仍无法选出股票！")
            break

    return pd.DataFrame()


# ══════════════════════════════════════════════════════
# 狗股策略选股（Dogs of the Market）
# ══════════════════════════════════════════════════════

def run_dogs_of_market_selection():
    """
    执行狗股策略选股，返回选股结果DataFrame
    选股失败时自动后移交易日重试，直到选出股票或达到最大尝试次数
    """
    from src.dogs_of_market_selector import DogsOfMarketSelector
    from src.data_fetcher import DataFetcher

    backtest_start = BACKTEST["start_date"]
    DOGS_OF_MARKET["top_n"] = SELECTION["top_n"]
    DOGS_OF_MARKET["stock_pool"] = SELECTION["stock_pool"]

    # 获取交易日列表（用于后移）
    import sqlite3
    conn = sqlite3.connect(DB_PATH)
    trade_dates = pd.read_sql_query(
        "SELECT DISTINCT trade_date FROM daily ORDER BY trade_date",
        conn
    )["trade_date"].tolist()
    conn.close()

    # 找到回测开始日前一个交易日的索引
    prev_day = _get_prev_trading_day(backtest_start)
    if prev_day not in trade_dates:
        print(f"  [ERROR] 选股日 {prev_day} 不在交易日列表中！")
        return pd.DataFrame()
    date_idx = trade_dates.index(prev_day)

    MAX_ATTEMPTS = 20  # 最多尝试20个交易日

    for attempt in range(MAX_ATTEMPTS):
        current_date = trade_dates[date_idx]
        DOGS_OF_MARKET["date"] = current_date

        if attempt == 0:
            print(f"  狗股策略选股日: 回测开始 {backtest_start} → 前交易日 {current_date}")
        else:
            print(f"  [重试] 选股日后移 {attempt} 天 → {current_date}")

        df_config = {
            "primary_source": DATA["primary_source"],
            "tushare_token": DATA.get("tushare_token", ""),
            "local_db_path": DATA["local_db_path"],
            "use_akshare_backup": DATA["use_akshare_backup"],
            "use_tushare_backup": False,
        }
        fetcher = DataFetcher(**df_config)
        selector = DogsOfMarketSelector(DOGS_OF_MARKET, fetcher)

        print(f"\n{'='*60}")
        print(f"  狗股策略选股 — {current_date} | 池:{DOGS_OF_MARKET['stock_pool']}")
        print(f"{'='*60}")

        selected = selector.select_stocks(date=current_date)

        if selected is not None and len(selected) > 0:
            # 选股成功，保存结果
            output_dir = DOGS_OF_MARKET.get("output_dir", "data/results/dogs_of_market")
            output_file = DOGS_OF_MARKET.get("output_file", "dogs_of_market_{date}.csv")
            filepath = selector.export_to_csv(selected, filename=output_file, output_dir=output_dir)

            print(f"\n{'='*60}")
            print(f"  狗股策略选股完成！共找到 {len(selected)} 只股票（选股日: {current_date}）")
            print(f"  结果已保存 → {filepath}")
            print(f"{'='*60}\n")

            return selected[["code", "name"]] if "name" in selected.columns else selected[["code"]]

        # 选股失败，后移一天
        print(f"  [WARN] 选股失败，后移到下一交易日重试...")
        date_idx += 1
        if date_idx >= len(trade_dates):
            print(f"\n  [ERROR] 已尝试 {attempt+1} 个交易日，仍无法选出股票！")
            break

    return pd.DataFrame()


def run_backtest(stocks):
    """对股票列表执行所有已启用的策略"""
    # 由「总初始资金 / 选股数量」推导每支资金（向下取整到整百股）
    _top_n = max(int(SELECTION.get("top_n", 5)), 1)
    _total_capital = int(BACKTEST.get("total_capital", BACKTEST["per_stock_capital"] * _top_n))
    if _total_capital < 100000:
        _total_capital = 100000
    capital = int(_total_capital // _top_n // 100) * 100
    if capital < 100:
        capital = 100
    start, end = BACKTEST["start_date"], BACKTEST["end_date"]

    # 自动根据股票池设置基准指数
    stock_pool_to_benchmark = {
        "hs300": "000300.SH",   # 沪深300
        "zz500": "000905.SH",   # 中证500
        "zz800": "000906.SH",   # 中证800
        "zz1000": "000852.SH",   # 中证1000
        "all":   "000300.SH",   # 全A股用沪深300作为基准
    }
    if SELECTION["stock_pool"] in stock_pool_to_benchmark:
        BACKTEST["benchmark"] = stock_pool_to_benchmark[SELECTION["stock_pool"]]

    benchmark = BACKTEST["benchmark"]
    # 指数代码 → 中文显示名称
    _BENCHMARK_NAME = {
        "000300.SH": "沪深300",
        "000905.SH": "中证500",
        "000906.SH": "中证800",
        "000852.SH": "中证1000",
        "399006.SZ": "创业板指",
    }
    benchmark_name = _BENCHMARK_NAME.get(benchmark, benchmark)

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
    # 递补逻辑：候选股票可能数据不足，遍历候选列表直到凑满 top_n 只有效股票
    target_n = SELECTION["top_n"]
    print(f"\n  加载股票数据 ({start} → {end})...  [目标: {target_n}只，候选: {len(stocks)}只]")
    stock_data = {}
    bh_results = {}
    skipped = []
    valid_count = 0

    for _, row in stocks.iterrows():
        if valid_count >= target_n:
            break  # 已凑满，停止
        code, name = row["code"], row.get("name", "")
        df = load_stock_prices(code, start, end, conn, lookback_days=250)
        if df is not None and len(df) >= 30:
            # 找到第一个 >= start 的交易日索引
            start_idx = df[df["trade_date"] >= start].index.min()
            if pd.isna(start_idx):
                skipped.append(f"{code}({name}): 回测起始日不在数据范围内")
                continue
            start_idx = int(start_idx)
            stock_data[code] = (name, df, start_idx)
            # BH 收益 = 回测起始日买入 → 结束日卖出（使用复权价格，与策略一致）
            bh_close_col = "adj_close" if "adj_close" in df.columns else "close"
            bh_ret = (df[bh_close_col].iloc[-1] / df[bh_close_col].iloc[start_idx] - 1) * 100
            bh_results[code] = bh_ret
            valid_count += 1
        else:
            reason = "价格数据为空" if df is None else f"数据不足({len(df)}天, 需≥30天)"
            skipped.append(f"{code}({name}): {reason}")

    conn.close()
    if skipped:
        print(f"  [递补] {valid_count}/{target_n} 只有效，以下候选股票数据不足，已从排名中递补:")
        for s in skipped[:5]:  # 只显示前5条
            print(f"    - {s}")
        if len(skipped) > 5:
            print(f"    ... 还有 {len(skipped) - 5} 只")
    if valid_count < target_n:
        print(f"  [WARN] 警告: 有效股票只有 {valid_count}/{target_n} 只，候选已用尽！")
    print(f"  有效股票: {valid_count}/{target_n}  [递补完成]")

    # ═─ 内置策略函数（实盘有效，已对齐成本模型；原插件加载器已移除）───
    strategy_funcs = {
        "buy_hold": (backtest_buy_hold, 0),
        "rsi": (backtest_rsi, 0),
        "bollinger": (backtest_bollinger, 0),
        "turtle": (backtest_turtle, 0),
        "rsi_trend": (backtest_rsi_trend, 0),
    }

    # ═══ 加载策略插件（自动发现 backtest/*.py 中的 BaseStrategy 子类）═══
    plugins = load_strategy_plugins()
    if plugins:
        print(f"  已发现 {len(plugins)} 个策略插件: {', '.join(plugins.keys())}")

    enabled = [(k, v) for k, v in STRATEGIES.items() if v.get("enabled")]
    print(f"\n{'='*100}")
    print(f"  回测阶段 — {start} → {end} | 基准{benchmark}{bi.benchmark_meta_label(_last_bench_meta.get(benchmark))}: {idx_ret:+.2f}% | 总资金{_total_capital/10000:.0f}万 ÷ {_top_n}只 = 每支{capital/10000:.0f}万")
    print(f"  启用策略: {', '.join([s['name'] for _, s in enabled])}")
    print(f"{'='*100}")

    all_summaries = {}

    # ── 回测区间年数（用于年化收益率）──
    try:
        _sd = datetime.strptime(start, "%Y%m%d")
        _ed = datetime.strptime(end, "%Y%m%d")
    except ValueError:
        _sd = datetime.strptime(start, "%Y-%m-%d")
        _ed = datetime.strptime(end, "%Y-%m-%d")
    _years = max((_ed - _sd).days / 365.25, 1e-9)

    def _annualized(ret_pct, years):
        """把总收益率(百分比)按区间年数年化"""
        base = 1.0 + ret_pct / 100.0
        if base <= 0 or years <= 0:
            return -100.0 if base <= 0 else ret_pct
        return (base ** (1.0 / years) - 1.0) * 100.0

    for skey, scfg in enabled:
        sname = scfg["name"]
        print(f"\n{'─'*100}")
        print(f"  【{sname}】")
        print(f"  {'代码':<8} {'名称':<8} {'初始本金':>9} {'期末资产':>9} {'盈亏金额':>9} {'收益率':>8} {'超额':>8} {'交易':>6} {'最大回撤':>10} {'回撤区间':>24} {'胜率':>6}")
        print(f"  {'─'*80}")

        results = []
        for code, (name, df, start_idx) in stock_data.items():
            try:
                # ── 优先使用插件（类方式，自动发现 backtest/*_plugin.py）───
                if skey in plugins:
                    strategy_class = plugins[skey]
                    scfg = dict(scfg)
                    scfg["total_capital"] = _total_capital  # 凯利封顶按组合总资金算，避免高价股被禁仓
                    strategy = strategy_class(capital, scfg)
                    result = strategy.run(df, start_idx)
                    ret = result.get("returns", 0.0)
                    trades = len(result.get("trades", []))
                    # 计算胜率
                    win_rate, win_cnt, total_trades = calc_win_rate_from_trades(result.get("trades", []))
                    # 计算最大回撤
                    daily_values = result.get("daily_values", [])
                    if daily_values:
                        portfolio_values = [v["portfolio_value"] for v in daily_values]
                        dates = [v.get("date") for v in daily_values]
                        max_dd, pk, tr = max_drawdown_with_dates(portfolio_values, dates)
                    else:
                        max_dd, pk, tr = 0.0, None, None
                    dd_period = (pk, tr)
                # ── 回退到内置函数（兼容旧策略）───
                elif skey in strategy_funcs:
                    func = strategy_funcs[skey][0]
                    out = func(df, capital, scfg, start_idx)
                    # 兼容旧3元组 (ret, trades, max_dd) 与新4元组 (..., (peak, trough))
                    if len(out) == 4:
                        _r, _t, _dd, _period = out
                    else:
                        _r, _t, _dd = out
                        _period = (None, None)
                    if skey == "buy_hold":
                        # 买入持有：只有1次买卖，盈利=100%胜率，亏损=0%胜率
                        ret, trades, max_dd, dd_period = _r, _t, _dd, _period
                        win_rate = 100.0 if ret > 0 else 0.0
                    else:
                        # 其余内置函数返回 (ret, trade_records, max_dd[, dd_period])，
                        # 用 calc_win_rate_from_trades 计算胜率（与插件策略一致）
                        ret, trade_records, max_dd, dd_period = _r, _t, _dd, _period
                        trades = len(trade_records)
                        win_rate, win_cnt, total_trades = calc_win_rate_from_trades(trade_records)
                else:
                    print(f"  [ERR] 未找到策略: {skey}")
                    continue
                
                if ret is None:
                    continue
                exc = ret - idx_ret
                bh = bh_results.get(code, 0)
                vs_bh = ret - bh if skey != "buy_hold" else 0.0
                beat = "[OK]" if ret > idx_ret else "[ERR]"
                # 计算盈亏金额
                final_val = capital * (1 + ret / 100)
                profit = final_val - capital
                print(f"  {code:<8} {name:<8} {capital:>9,} {final_val:>9,.0f} {profit:>+9,.0f} {ret:>+7.2f}% {exc:>+7.2f}% {trades:>6} {max_dd:>6.2f}% {_dd_period_str(dd_period):<24} {win_rate:>6.1f}%")
                results.append({"ret": ret, "ann_ret": _annualized(ret, _years), "exc": exc, "vs_bh": vs_bh, "trades": trades, "beat": ret > idx_ret, "max_dd": max_dd, "dd_period": dd_period, "profit": profit, "final_val": final_val, "win_rate": win_rate})
            except Exception as e:
                print(f"  {code:<8} {name:<8} {'ERR':>8} ({e})")

        if results:
            rets = [r["ret"] for r in results]
            # 买入持有策略不计算"优于BH"（自己不需要比自己）
            n_better_bh = 0 if skey == "buy_hold" else sum(1 for r in results if r["vs_bh"] > 0)
            s = all_summaries[sname] = {
                "mean": np.mean(rets), "median": np.median(rets),
                "annual_mean": np.mean([r["ann_ret"] for r in results]),
                "best": np.max(rets), "worst": np.min(rets),
                "n_pos": sum(1 for r in rets if r > 0),
                "n_beat": sum(1 for r in results if r["beat"]),
                "n_better_bh": n_better_bh,
                "n": len(results),
                "trades_mean": np.mean([r["trades"] for r in results]),
                "max_dd_mean": np.mean([r["max_dd"] for r in results]),
                "win_rate_mean": np.mean([r["win_rate"] for r in results]),
                "dd_year_counts": _aggregate_dd_years(results),
            }
            # 买入持有不显示"优于BH"
            if skey == "buy_hold":
                better_bh_str = "优于BH N/A"
            else:
                better_bh_str = f"优于BH {s['n_better_bh']}/{len(results)}"
            print(f"  {'─'*80}")
            print(f"  汇总: 均值{s['mean']:+.2f}% 中位数{s['median']:+.2f}% "
                  f"正收益{s['n_pos']}/{len(results)} 跑赢{s['n_beat']}/{len(results)} "
                  f"{better_bh_str} 均交易{s['trades_mean']:.1f}次 "
                  f"均最大回撤{s['max_dd_mean']:.2f}% 胜率{s['win_rate_mean']:.1f}%")
            if s.get("dd_year_counts"):
                yr_desc = " ".join(f"{y}年({c})" for y, c in s["dd_year_counts"].items())
                print(f"  最大回撤高发期: {yr_desc}")
            # 资金汇总
            total_initial = capital * len(results)
            total_final = sum(r["final_val"] for r in results)
            total_profit = sum(r["profit"] for r in results)
            total_ret = (total_final / total_initial - 1) * 100 if total_initial > 0 else 0
            print(f"  资金汇总: 总投入{total_initial:>9,.0f} (=每支{capital:,.0f}元 × {len(results)}只)  总资产{total_final:>9,.0f}  "
                  f"总盈亏{total_profit:>+9,.0f}  总收益率{total_ret:+.2f}%  年化收益率均值{s['annual_mean']:+.2f}%")

    # ══ 总表 ══
    print(f"\n\n{'='*100}")
    print(f"  【策略对比总表】")
    print(f"{'='*100}")
    print(f"  {'策略':<16} {'均值':>8} {'中位数':>8} {'正收益':>8} {'跑赢指数':>8} {'优于BH':>8} {'均交易':>6} {'均最大回撤':>12} {'胜率':>6} {'最佳':>10} {'最差':>10} {'回撤高发期':>22}")
    print(f"  {'─'*90}")
    for skey, scfg in enabled:
        s = all_summaries.get(scfg["name"], {})
        if s:
            better_bh_display = f"{s['n_better_bh']}/{s['n']}" if skey != "buy_hold" else "N/A"
            yr_desc = " ".join(f"{y}({c})" for y, c in s.get("dd_year_counts", {}).items()) if s.get("dd_year_counts") else "--"
            print(f"  {scfg['name']:<16} {s['mean']:>+7.2f}% {s['median']:>+7.2f}% "
                  f"{s['n_pos']}/{s['n']:>4}  {s['n_beat']}/{s['n']:>4}  {better_bh_display:>6}  "
                  f"{s['trades_mean']:>5.1f}  {s['max_dd_mean']:>11.2f}%  {s['win_rate_mean']:>5.1f}%  {s['best']:>+9.2f}% {s['worst']:>+9.2f}%  {yr_desc:>20}")
    bh_mean = np.mean(list(bh_results.values()))
    print(f"  {'─'*90}")
    print(f"  {'买入持有(等权)':<16} {bh_mean:>+7.2f}%")
    print(f"  {benchmark_name + '基准':<16} {idx_ret:>+7.2f}%  {bi.benchmark_meta_label(_last_bench_meta.get(benchmark))}")
    _w = bi.check_consistency(globals().get("PRICE_MODE"), _last_bench_meta.get(benchmark))
    if _w:
        print(f"  {_w}")
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
                    "胜率%": round(s["win_rate_mean"], 2),
                    "最大回撤高发期": " ".join(f"{y}({c})" for y, c in s.get("dd_year_counts", {}).items()) if s.get("dd_year_counts") else "--",
                    "最佳%": round(s["best"], 2), "最差%": round(s["worst"], 2),
                })
        rows.append({"策略": "买入持有(等权)", "均值收益%": round(bh_mean, 2)})
        rows.append({"策略": f"{benchmark_name}基准", "均值收益%": round(idx_ret, 2)})
        pd.DataFrame(rows).to_csv(
            os.path.join(OUTPUT.get("backtest_dir", os.path.join(OUTPUT["dir"], "backtest")),
                        f"backtest_{BACKTEST['start_date'][:4]}.csv"),
            index=False, encoding="utf-8-sig")
        print(f"  报告已保存 → {OUTPUT.get('backtest_dir', os.path.join(OUTPUT['dir'], 'backtest'))}/backtest_{BACKTEST['start_date'][:4]}.csv")


# ════════════════════════════════════════════════════════
# 入口
# ════════════════════════════════════════════════════════
if __name__ == "__main__":
    # ── 命令行参数解析 ──────────────────────────────────────────────────
    # 用法:
    #   python run_backtest.py                # 使用 config.py 默认配置
    #   python run_backtest.py --source multi # 使用最新 multi_*.csv
    #   python run_backtest.py --source ml    # 使用最新 ml_*.csv
    # ─────────────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="多因子选股 + 回测系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例:
  python run_backtest.py                # 使用 config.py 默认配置
  python run_backtest.py --source multi # 使用最新 multi_*.csv 进行回测
  python run_backtest.py --source ml    # 使用最新 ml_*.csv 进行回测(默认v4模型)
  python run_backtest.py --source ml --model v5  # 使用 v5 模型选股并回测
  python run_backtest.py --list         # 列出所有可用的 CSV 文件
"""
    )
    parser.add_argument(
        "--source", "-s",
        type=str,
        choices=["multi", "value", "div_low_vol", "dogs", "dogs_annual", "csv", "manual", "monthly_rebalance", "sc_rotation", "sc_kara", "dca_etf", "macd_regime"],
        default=None,
        help="选股策略来源: multi(多因子) / value(价值投资) / div_low_vol(红利低波) / dogs(狗股策略) / dogs_annual(年度调仓) / csv(指定文件) / manual(手动列表) / monthly_rebalance(月度调仓) / sc_rotation(小市值轮动·周频止损版) / sc_kara(Kara小市值轮动·纯最小市值月频零过滤) / dca_etf(ETF定投·单产品/宽基篮子·月/周)"
    )
    parser.add_argument(
        "--start-date", type=str, default=None,
        help="回测开始日期 (YYYYMMDD)，覆盖 config.py 的 BACKTEST['start_date']"
    )
    parser.add_argument(
        "--end-date", type=str, default=None,
        help="回测结束日期 (YYYYMMDD)，覆盖 config.py 的 BACKTEST['end_date']"
    )
    parser.add_argument(
        "--top-n", type=int, default=None,
        help="选股数量，覆盖 config.py 的 SELECTION['top_n']"
    )
    parser.add_argument(
        "--capital", type=int, default=None,
        help="选股族回测总初始资金（元），覆盖 config.py 的 BACKTEST['total_capital']；"
             "每支资金 = 总资金 // 选股数量（向下取整到整百股）"
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
        help="列出 data/results/selection/ 下所有可用的 CSV 文件"
    )
    parser.add_argument(
        "--auto", action="store_true",
        help="自动匹配最新的 CSV 文件（--source csv 时可用）"
    )
    parser.add_argument(
        "--select-only", action="store_true",
        help="只选股，不回测"
    )
    parser.add_argument(
        "--ma-short", type=int, default=None,
        help="双均线短期周期（覆盖 config.py 的 ma_short）"
    )
    parser.add_argument(
        "--ma-long", type=int, default=None,
        help="双均线长期周期（覆盖 config.py 的 ma_long）"
    )
    parser.add_argument(
        "--stock-pool", type=str, default=None,
        help="股票池: hs300 | zz500 | zz800 | zz1000 | all"
    )
    parser.add_argument(
        "--benchmark", type=str, default=None,
        help="基准指数代码（如 000300.SH 或 000906.SH）"
    )
    parser.add_argument(
        "--selection-method", type=str, default="value",
        choices=["value", "div_low_vol", "momentum", "div_low_vol_quality", "div_growth"],
        help="月度调仓的选股策略: value(价值) / div_low_vol(红利低波) / momentum(动量追涨) / div_low_vol_quality(红利低波质量复合·季度调仓) / div_growth(高股息+基本面成长)"
    )
    parser.add_argument(
        "--dlvq-mode", type=str, default="official_compact",
        choices=["official", "official_improved", "official_compact"],
        help="div_low_vol_quality 的模式: official(月频/等权/TOP5) / official_improved(季频/股息率加权/TOP25) / official_compact(季频/股息率加权/TOP12/行业≤2·落地版)"
    )
    parser.add_argument(
        "--dlvq-rebal", type=str, default=None,
        choices=["month", "quarter", "half", "year"],
        help="div_low_vol_quality 调仓频率覆盖(默认None=沿用模式默认季度); 传 year/half 可降频压换手(详见divlow_b7_demystify.md §3.3/§5.6); month/quarter 显式指定"
    )
    parser.add_argument(
        "--div-channel-overlay", action="store_true",
        help="红利通道仓位 overlay（000922通道位置→权益仓位系数，贵减仓/便宜满仓；已验证正向）。默认即开启，此 flag 仅显式打开"
    )
    parser.add_argument(
        "--no-div-channel-overlay", action="store_true",
        help="关闭红利通道仓位 overlay，回归满仓普通红利低波基线（对照用）"
    )
    parser.add_argument(
        "--div-channel-mode", default="rolling", choices=["rolling", "fixed"],
        help="通道位置算法: rolling(前N日min/max, 默认) / fixed(固定上下轨)"
    )
    parser.add_argument(
        "--div-channel-window", type=int, default=756,
        help="rolling 模式回看窗口(交易日, 默认756≈3年)"
    )
    parser.add_argument(
        "--div-channel-bottom", type=float, default=None,
        help="fixed 模式通道下轨(000922 点位)"
    )
    parser.add_argument(
        "--div-channel-top", type=float, default=None,
        help="fixed 模式通道上轨(000922 点位)"
    )
    parser.add_argument(
        "--k-min", type=float, default=0.5,
        help="通道最贵时的权益仓位系数下限(默认0.5；小账户档用0.3)"
    )
    parser.add_argument(
        "--k-max", type=float, default=1.0,
        help="通道最便宜时的权益仓位系数上限(默认1.0满仓)"
    )
    parser.add_argument(
        "--live", action="store_true",
        help="回测结束后打印上一期已选买列表(历史对照)"
    )
    parser.add_argument(
        "--live-forward", action="store_true",
        help="回测结束后以今日为选股日重跑 selector + 今日通道k，打印今日该买清单"
    )
    parser.add_argument(
        "--var-stop", action="store_true",
        help="启用 VAR 动态止损（动量月度同款 ATR 追踪：跌破[最高收-倍数×ATR]次日开盘卖；默认关）"
    )
    parser.add_argument(
        "--atr-mult", type=float, default=2.0,
        help="VAR动态止损的 ATR 倍数（默认2.0，与动量月度一致）"
    )
    parser.add_argument(
        "--atr-cooling", type=int, default=5,
        help="VAR动态止损的买入后冷静期交易日数（默认5，期内不触发止损）"
    )
    parser.add_argument(
        "--leverage-filter", action="store_true",
        help="启用杠杆因子风控过滤（产权比率一票否决+利息保障倍数；默认关）"
    )
    parser.add_argument(
        "--de-ratio-exclude-pct", type=float, default=5.0,
        help="杠杆过滤：产权比率最高的百分之多少被剔除（默认5%%）"
    )
    parser.add_argument(
        "--icover-min", type=float, default=3.0,
        help="杠杆过滤：利息保障倍数最小值（默认3倍；<=0不启用）"
    )
    parser.add_argument(
        "--div-quality-filter", action="store_true",
        help="启用红利质量三因子过滤（连续分红年数+分红现金覆盖+分红增长；默认关，仅红利低波生效）"
    )
    parser.add_argument(
        "--div-years-min", type=int, default=3,
        help="红利质量：要求连续分红年数下限（默认3年；<=0不启用）"
    )
    parser.add_argument(
        "--div-growth-min", type=float, default=None,
        help="红利质量：分红增长CAGR下限（小数，如0.05=5%%；默认None=随--div-quality-filter启用后按0%%要求不萎缩）"
    )
    parser.add_argument(
        "--dq-ocf", action="store_true",
        help="红利质量：仅启用「稳② 经营现金流覆盖分红」单维度（自动关闭连续分红年数/分红增长），用于隔离验证分红现金流维度本身是否有用"
    )
    parser.add_argument(
        "--dogs-strategy", type=str, default="dogs",
        choices=["dogs", "value", "magic", "ep", "ep_obv"],
        help="年度调仓的子策略: dogs(狗股·高股息+低PB) / value(价值选股·破净+ROE+现金流) / magic(神奇公式·ROC+EY双排名) / ep(EP行业中性·低PE) / ep_obv(EP+OBV吸筹过滤)"
    )
    parser.add_argument(
        "--value-mode", type=str, default="pobreak",
        choices=["pobreak", "pure_bm"],
        help="价值选股模式(配合 --source value / monthly_rebalance value / dogs_annual value): "
             "pobreak=破净价值(PB<1+ROE质量) / pure_bm=放宽破净·全市场BM前N%%门槛(让中性化/分位真正区分)"
    )
    parser.add_argument(
        "--price-mode", type=str, default="dual",
        choices=["dual", "hfq", "raw"],
        help="年度调仓(dogs_annual)价格口径: dual=双轨同时算后复权+原始价(默认,一次出两套结果) / "
             "hfq=仅后复权(含分红再投) / raw=仅原始价(不含分红,用于看纯选股α)"
    )
    parser.add_argument(
        "--value-size-neutral", action="store_true",
        help="价值选股: 市值中性化(对BM回归掉市值,取残差作纯价值得分)"
    )
    parser.add_argument(
        "--value-pct", type=float, default=None,
        help="价值选股: BM分位筛选(如0.3=全市场BM前30%%,Fama-French前20-30%%口径)"
    )
    parser.add_argument(
        "--value-quality-gates", type=str, default="on",
        choices=["on", "off"],
        help="价值选股: 四道质量门槛(②盈余质量ocfps/eps ③杠杆debt+ocf_to_debt ④应收周转 ⑤估值纵向分位) "
             "on=按config阈值启用(默认) / off=全部关闭(回退到仅破净+ROE, 用于对照)"
    )
    parser.add_argument(
        "--hold-count", type=int, default=None,
        help="小市值轮动持仓只数(流通市值最小N只)，覆盖默认7"
    )
    parser.add_argument(
        "--empty-jan-apr", action="store_true",
        help="小市值轮动: 1月/4月空仓(年报/一季报窗口)"
    )
    parser.add_argument(
        "--stop-loss", action="store_true",
        help="小市值轮动: 开启三层止损(单票-12%% / 中证2000单日-6.6%% / 昨涨停今炸板)"
    )
    parser.add_argument(
        "--sc-fundamental", action="store_true",
        help="小市值轮动: 开启基本面过滤(最近年报 净利润>0 且 营收>5亿)"
    )
    parser.add_argument(
        "--sc-quality-filter", action="store_true",
        help="小市值轮动(A档): 质量门禁 roe>0 & 净资产>0 & 负债率<70%% & 经营现金流>0，避雷提质"
    )
    parser.add_argument(
        "--sc-growth-tilt", action="store_true",
        help="小市值轮动(B档): 成长倾斜，最小市值桶内优先 净利润同比>0 且 高roe，size为底+成长增强"
    )
    parser.add_argument(
        "--sc-vol-filter", action="store_true",
        help="小市值轮动(维度3): 极端波动过滤，剔除近60日收益率方差最高5%%的票(波动特别极端→伪超额)"
    )
    parser.add_argument(
        "--sc-style-switch", action="store_true",
        help="小市值轮动(维度5): 风格切换，沪深300连续20个周二跑赢中证1000→当周空仓(小市值阶段性失效避险)"
    )
    parser.add_argument(
        "--exclude-delisted", action="store_true",
        help="小市值轮动: 剔除已退市股(INNER JOIN), 用于幸存者偏差对照"
    )
    parser.add_argument(
        "--min-avg-amount-k", type=float, default=None,
        help="小市值轮动: 流动性门槛(日均成交额,千元), 默认30000(=3000万)"
    )
    parser.add_argument(
        "--sc-mode", type=str, default="single",
        choices=["single", "compare", "sensitivity", "size_quintile"],
        help="小市值轮动运行模式: single(单次) / compare(含退市vs剔除退市对照) / sensitivity(持仓数×流动性网格) / size_quintile(维度2·市值分位小/中/大桶对照)"
    )
    parser.add_argument(
        "--hold-grid", default="5,7,10,15",
        help="小市值轮动 sensitivity 模式: 持仓数网格(逗号分隔)"
    )
    parser.add_argument(
        "--liq-grid", default="30000,50000,80000,100000",
        help="小市值轮动 sensitivity 模式: 流动性门槛网格(千元)"
    )
    parser.add_argument(
        "--sc-pool-mode", type=str, default="zz2000",
        choices=["cyb", "zz2000", "zz1000"],
        help="小市值轮动选股宇宙: cyb(纯创业板) / zz2000(中证2000风格·含微盘尾) / zz1000(中证1000风格·剔除微盘尾)"
    )
    parser.add_argument(
        "--sc-bucket", type=str, default=None,
        choices=["small", "mid", "large"],
        help="小市值轮动·市值分位桶(单独跑某一档): small(最小N只) / mid(宇宙40%%分位档) / large(宇宙最大N只)。仅 single 模式生效，size_quintile 模式忽略(自身跑三桶)"
    )
    parser.add_argument(
        "--sc-no-html", action="store_true",
        help="小市值轮动(single模式)·不生成HTML净值曲线报告(默认会同时生成明细+HTML)"
    )
    parser.add_argument(
        "--sc-no-detail", action="store_true",
        help="小市值轮动(single模式)·不导出文本+CSV回测明细(默认会同时生成明细+HTML)"
    )
    parser.add_argument(
        "--kara-exclude-688", action="store_true",
        help="Kara小市值轮动(sc_kara): 剔除科创板(688)，用于对照加科创板选股的贡献"
    )
    # ── 定投 ETF（dca_etf）专用参数 ──
    parser.add_argument(
        "--dca-freq", type=str, default="both",
        choices=["monthly", "weekly", "both"],
        help="定投频率: monthly(每月首交易日) / weekly(每周首交易日) / both(两者+对比)"
    )
    parser.add_argument(
        "--dca-code", type=str, default="510300.SH",
        help="定投标的 ETF：单代码 / 逗号分隔篮子(如 510300.SH,510500.SH,159915.SZ) / 预设名(core6,core4,core3,large3,all_legacy,a500_4)"
    )
    parser.add_argument(
        "--dca-preset", type=str, default=None,
        help="预设篮子名(等效于 --dca-code 设同名)：core6/core4/core3/large3/all_legacy/a500_4"
    )
    parser.add_argument(
        "--dca-monthly", type=float, default=4000.0,
        help="月度定投每期金额(元)，默认 4000"
    )
    parser.add_argument(
        "--dca-weekly", type=float, default=1000.0,
        help="周度定投每期金额(元)，默认 1000"
    )
    parser.add_argument(
        "--dca-mode", type=str, default="plain", choices=["plain", "smart"],
        help="定投模式: plain(普通纪律定投) / smart(均线增强·5周/20周线操作法)"
    )
    # ── VaR 仓位缩放（动量轮动专用，复现「设计即锁回撤」）──
    parser.add_argument(
        "--var-control", type=int, default=0, choices=[0, 90, 95, 99],
        help="VaR仓位缩放置信水平: 0=关闭 | 90/95/99=启用(对应分位)"
    )
    parser.add_argument(
        "--var-maxdd", type=float, default=15.0,
        help="目标最大回撤上限(%%)，用于反解每期风险预算（默认15）"
    )
    parser.add_argument(
        "--var-n", type=int, default=5,
        help="连续下跌周期数 N（趋势类=5，反转类=3），用于分摊回撤预算（默认5）"
    )
    parser.add_argument(
        "--value-area", type=int, default=0,
        help="价值区过滤回看天数: 0=关闭 | >0=启用(动量/反转生效，默认0)"
    )
    parser.add_argument(
        "--va-pct", type=float, default=70.0,
        help="价值区覆盖成交量比例(%%)（默认70）"
    )
    parser.add_argument(
        "--macd-filter", type=str, default=None,
        choices=["golden", "regime"],
        help="MACD信号模式(monthly_rebalance div_low_vol择时共用): golden=旧金叉死叉当按钮 | regime=金叉须叠加指数>MA200且非盘整(语境感知)。不指定则按策略默认：逆转=regime，红利低波=golden"
    )
    args = parser.parse_args()

    # ── 用命令行参数覆盖 config（不修改 config.py 文件）──
    if args.ma_short is not None:
        STRATEGIES["dual_ma"]["ma_short"] = args.ma_short
    if args.ma_long is not None:
        STRATEGIES["dual_ma"]["ma_long"] = args.ma_long
    if args.stock_pool is not None:
        SELECTION["stock_pool"] = args.stock_pool
    if args.benchmark is not None:
        BACKTEST["benchmark"] = args.benchmark
    
    # 新增：覆盖回测区间和选股数量
    if args.start_date is not None:
        BACKTEST["start_date"] = args.start_date
        print(f"  [参数] 回测开始日期: {args.start_date}")
    if args.end_date is not None:
        BACKTEST["end_date"] = args.end_date
        print(f"  [参数] 回测结束日期: {args.end_date}")
    if args.top_n is not None:
        SELECTION["top_n"] = args.top_n
        print(f"  [参数] 选股数量: {args.top_n}")
    if args.capital is not None:
        if args.capital < 100000:
            print(f"  [警告] 总初始资金 {args.capital} 低于最小限制 100000，已强制使用 100000")
            args.capital = 100000
        BACKTEST["total_capital"] = args.capital
        print(f"  [参数] 总初始资金: {args.capital}")

    # ── 价值选股四道质量门槛开关（仅价值相关路径生效）──
    _value_related = (
        args.source in ("value", "monthly_rebalance")
        or (args.source == "dogs_annual" and getattr(args, "dogs_strategy", "") == "value")
    )
    if _value_related:
        if getattr(args, "value_quality_gates", "on") == "off":
            VALUE_STRATEGY["eq_ocf_eps_min"] = 0
            VALUE_STRATEGY["lev_debt_to_assets_max"] = 0
            VALUE_STRATEGY["lev_require_ocf_to_debt_pos"] = False
            VALUE_STRATEGY["ar_turn_yoy_drop_max"] = 0
            VALUE_STRATEGY["val_hist_years"] = 0
            print("  [参数] 价值四道质量门槛: 关闭(仅破净+ROE)")
        else:
            print("  [参数] 价值四道质量门槛: 启用"
                  f"(盈余质量ocfps/eps>={VALUE_STRATEGY.get('eq_ocf_eps_min')}, "
                  f"负债率<={VALUE_STRATEGY.get('lev_debt_to_assets_max')}%, "
                  f"应收降幅<={VALUE_STRATEGY.get('ar_turn_yoy_drop_max')}, "
                  f"PE<=自身{VALUE_STRATEGY.get('val_hist_years')}年"
                  f"{int(VALUE_STRATEGY.get('val_hist_pe_pct_max',0.5)*100)}%分位)")


    # 列出可用 CSV 文件
    if args.list:
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT.get("selection_dir", os.path.join(OUTPUT["dir"], "selection")))
        csv_files = sorted(glob.glob(os.path.join(results_dir, "*.csv")))
        print(f"\n  可用 CSV 文件 ({results_dir}):")
        print("  " + "-" * 50)
        for f in csv_files:
            mtime = os.path.getmtime(f)
            mtime_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
            size_kb = os.path.getsize(f) / 1024
            fname = os.path.basename(f)
            # 标记类型
            tag = ""
            if fname.startswith("ml_"):
                tag = "  [ML机器学习]"
            elif fname.startswith("multi_"):
                tag = "  [多因子选股]"
            elif "backtest_" in fname:
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
        if args.select_only:
            # 只选股，不回测
            print(f"\n  只选股模式 (多因子)...")
            stocks = run_selection()
            print(f"\n{'='*60}")
            print(f"  选股结果（共 {len(stocks)} 只）:")
            for i, (_, row) in enumerate(stocks.iterrows(), 1):
                print(f"    {i}. {row['code']}  {row.get('name', '')}")
            print(f"\n{'='*60}")
            sel_dir = OUTPUT.get('selection_dir', os.path.join(OUTPUT['dir'], 'selection'))
            print(f"  结果已保存到 {sel_dir}/")
            print(f"{'='*60}\n")
            sys.exit(0)
        else:
            # 如果传入了 --top-n，先重新选股（覆盖已有 CSV）
            if args.top_n is not None:
                print(f"\n  [参数] --top-n 已设置({args.top_n})，先重新选股...")
                run_selection()
            
            # 自动匹配最新的 multi_*.csv
            results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT.get("selection_dir", os.path.join(OUTPUT["dir"], "selection")))
            os.makedirs(results_dir, exist_ok=True)
            multi_files = sorted(glob.glob(os.path.join(results_dir, "multi_*.csv")), key=os.path.getmtime, reverse=True)
            if not multi_files:
                print(f"\n  [ERROR] 未找到 multi_*.csv 文件在 {results_dir}")
                print(f"  请先运行多因子选股 (src/stock_selector.py)\n")
                sys.exit(1)
            csv_path = multi_files[0]
            stocks = pd.read_csv(csv_path, dtype={"code": str})
            stocks["name"] = stocks.get("name", "")
            print(f"\n  使用多因子选股结果: {os.path.basename(csv_path)} ({len(stocks)} 只)")
            run_backtest(stocks)
            sys.exit(0)

    elif args.source == "value":
        # 价值投资选股
        VALUE_STRATEGY["value_mode"] = args.value_mode
        VALUE_STRATEGY["value_size_neutral"] = args.value_size_neutral
        VALUE_STRATEGY["value_pct"] = args.value_pct
        print(f"\n  价值投资选股模式 (模式={args.value_mode}"
              f"{'·放宽破净·BM分位门槛' if args.value_mode=='pure_bm' else '·破净+ROE质量'}"
              f" | 市值中性化={'开' if args.value_size_neutral else '关'}"
              f" | BM分位={('前%.0f%%'%(args.value_pct*100)) if args.value_pct else '关'})...")
        stocks = run_value_selection()
        if stocks is None or len(stocks) == 0:
            print(f"\n  [ERROR] 价值投资选股失败！")
            sys.exit(1)
        
        if args.select_only:
            # 只选股，不回测
            print(f"\n{'='*60}")
            print(f"  选股结果（共 {len(stocks)} 只）:")
            for i, (_, row) in enumerate(stocks.iterrows(), 1):
                print(f"    {i}. {row['code']}  {row.get('name', '')}")
            print(f"\n{'='*60}")
            print(f"  结果已保存到 data/results/value_strategy/")
            print(f"{'='*60}\n")
            sys.exit(0)
        
        run_backtest(stocks)
        sys.exit(0)
        
    elif args.source == "div_low_vol":
        # 红利低波选股
        print(f"\n  红利低波选股模式...")
        stocks = run_dividend_low_vol_selection()
        if stocks is None or len(stocks) == 0:
            print(f"\n  [ERROR] 红利低波选股失败！")
            sys.exit(1)
        
        if args.select_only:
            # 只选股，不回测
            print(f"\n{'='*60}")
            print(f"  选股结果（共 {len(stocks)} 只）:")
            for i, (_, row) in enumerate(stocks.iterrows(), 1):
                print(f"    {i}. {row['code']}  {row.get('name', '')}")
            print(f"\n{'='*60}")
            output_dir = DIVIDEND_LOW_VOL.get("output_dir", "data/results/dividend_low_vol")
            print(f"  结果已保存到 {output_dir}/")
            print(f"{'='*60}\n")
            sys.exit(0)
        
        run_backtest(stocks)
        sys.exit(0)
        
    elif args.source == "dogs":
        # 狗股策略选股
        print(f"\n  狗股策略选股模式...")
        stocks = run_dogs_of_market_selection()
        if stocks is None or len(stocks) == 0:
            print(f"\n  [ERROR] 狗股策略选股失败！")
            sys.exit(1)
        
        if args.select_only:
            print(f"\n{'='*60}")
            print(f"  选股结果（共 {len(stocks)} 只）:")
            for i, (_, row) in enumerate(stocks.iterrows(), 1):
                print(f"    {i}. {row['code']}  {row.get('name', '')}")
            print(f"\n{'='*60}")
            output_dir = DOGS_OF_MARKET.get("output_dir", "data/results/dogs_of_market")
            print(f"  结果已保存到 {output_dir}/")
            print(f"{'='*60}\n")
            sys.exit(0)
        
        run_backtest(stocks)
        sys.exit(0)
        
    elif args.source == "dogs_annual":
        # 狗股/价值 年度调仓回测
        dogs_strat = args.dogs_strategy or "dogs"
        if dogs_strat == "magic":
            dogs_strat_name = "神奇公式(Magic Formula·ROC+EY双排名)"
        elif dogs_strat == "value":
            dogs_strat_name = "价值选股(破净+ROE+现金流)"
        elif dogs_strat == "ep":
            dogs_strat_name = "EP行业中性(低PE·行业内五分位)"
        elif dogs_strat == "ep_obv":
            dogs_strat_name = "EP+OBV吸筹过滤(低PE·OBV净流入)"
        else:
            dogs_strat_name = "狗股策略(高股息+低PB)"
        print(f"\n  {dogs_strat_name} · 年度调仓回测模式...")
        print(f"  回测区间: {BACKTEST['start_date']} ~ {BACKTEST['end_date']}")
        print(f"  选股数量: {SELECTION.get('top_n', 5)}")
        print(f"  总资金: {BACKTEST.get('total_capital', 500000):,} 元 (将均分到每只)")
        print(f"  {'='*60}")

        from run_dogs_annual import run_backtest as run_dogs_annual_bt
        run_dogs_annual_bt(
            start_date=BACKTEST["start_date"],
            end_date=BACKTEST["end_date"],
            top_n=SELECTION.get("top_n", 5),
            capital=BACKTEST["total_capital"],
            strategy=dogs_strat,
            value_mode=args.value_mode,
            price_mode=args.price_mode,
        )
        sys.exit(0)
        
    elif args.source == "monthly_rebalance":
        # 月度调仓回测（直接导入调用，避免子进程输出混乱）
        sel_method = args.selection_method or "value"
        method_names = {"value": "价值选股", "div_low_vol": "红利低波选股", "momentum": "动量效应追涨", "div_low_vol_quality": "红利低波质量复合(季度调仓)"}
        print(f"\n  月度调仓回测模式...")
        print(f"  回测区间: {BACKTEST['start_date']} ~ {BACKTEST['end_date']}")
        print(f"  选股策略: {method_names.get(sel_method, sel_method)}")
        print(f"  选股数量: {SELECTION.get('top_n', 5)}")

        if sel_method == "momentum":
            from run_monthly_rebalance import run_momentum_backtest
            print(f"  {'='*60}")
            # 动量12个月 + 月度调仓 + 2×ATR止损 + 不跳近期 + MA200熊市空仓
            run_momentum_backtest(
                start_date=BACKTEST["start_date"],
                end_date=BACKTEST["end_date"],
                top_n=SELECTION.get("top_n", 5),
                lookback_months=12,
                stock_pool={
                    "hs300": "000300.SH", "zz500": "000905.SH",
                    "zz800": "000906.SH", "zz1000": "000852.SH",
                }.get(SELECTION.get("stock_pool", "zz800"), None),
                rebalance_freq_months=1,
                atr_stop_multiple=2.0,
                skip_recent_months=0,
                trend_filter_ma=200,
                var_control=args.var_control,
                var_maxdd=args.var_maxdd,
                var_n=args.var_n,
                value_area=args.value_area,
                va_pct=args.va_pct,
            )
        elif sel_method == "div_low_vol_quality":
            # 红利低波「质量复合」策略：官方编制法实战三档（季度调仓）
            import run_dividend_low_vol_quality_bt as dlq
            dlq.START = BACKTEST["start_date"]
            dlq.END = BACKTEST["end_date"]
            _dlq_mode = args.dlvq_mode
            _dlq_rebal = args.dlvq_rebal or "quarter"
            _freq_cn = {"month": "月频", "quarter": "季频", "half": "半年频", "year": "年频"}.get(_dlq_rebal, _dlq_rebal)
            print(f"  {'='*60}")
            print(f"  ※ 本策略调仓频率:【{_freq_cn}】(rebal={_dlq_rebal})：官方编制法(中证红利低波930955口径)，"
                  f"{_freq_cn} / 股息率加权；模式={_dlq_mode}")
            print(f"  ※ 红利通道仓位 overlay: "
                  f"{'关闭(满仓基线)' if args.no_div_channel_overlay else '开启'}"
                  f"{'' if args.no_div_channel_overlay else f' · {args.div_channel_mode}/w{args.div_channel_window}/k{args.k_min}'}")
            print(f"  ※ 股票池: 全A(all·锁定) — 本策略候选宇宙需全市场，zz800会饿死候选导致失真")
            dlq.run_official_backtest(
                _dlq_mode,
                rebal=args.dlvq_rebal,  # None=沿用模式默认(季度); year/half 降频压换手
                pool="all",  # 本策略锁定全A池（zz800候选被饿死、质量复合失真；验证基线均在 all 池）
                capital=args.capital,
                overlay=not args.no_div_channel_overlay,
                channel_mode=args.div_channel_mode,
                channel_window=args.div_channel_window,
                channel_bottom=args.div_channel_bottom,
                channel_top=args.div_channel_top,
                k_min=args.k_min,
                k_max=args.k_max,
                live=args.live,
                live_forward=args.live_forward,
            )
        else:
            from run_monthly_rebalance import run_backtest as run_monthly_rebalance_bt
            print(f"  {'='*60}")
            # 红利质量过滤档位解析：
            #  --dq-ocf             → 仅稳②(现金流覆盖分红)，关年数/增长
            #  --div-quality-filter → 全开(年数默认3 / 增长默认0%)；--div-years-min/--div-growth-min 可覆盖
            #  均未指定             → 全关
            if args.dq_ocf:
                _dq_filter = True
                _dq_years = 0
                _dq_ocf = True
                _dq_growth = None
            elif args.div_quality_filter:
                _dq_filter = True
                _dq_years = args.div_years_min
                _dq_ocf = True
                _dq_growth = 0.0 if args.div_growth_min is None else args.div_growth_min
            else:
                _dq_filter = False
                _dq_years = args.div_years_min
                _dq_ocf = True
                _dq_growth = args.div_growth_min
            run_monthly_rebalance_bt(
                start_date=BACKTEST["start_date"],
                end_date=BACKTEST["end_date"],
                top_n=SELECTION.get("top_n", 5),
                selection_method=sel_method,
                value_mode=args.value_mode,
                value_size_neutral=args.value_size_neutral,
                value_pct=args.value_pct,
                # 因子类策略(EP行业中性)月度调仓即退出，关闭个股15%止损；
                # 价值/红利低波沿用默认 STOP_LOSS 行为。
                stop_loss_pct=0 if sel_method in ("ep_neutral", "ep_obv") else None,
                # VAR动态止损（动量月度同款 ATR 追踪）；默认关，仅 --var-stop 时启用
                var_stop=args.var_stop,
                atr_mult=args.atr_mult,
                atr_cooling=args.atr_cooling,
                # 杠杆因子风控过滤（默认关）
                leverage_filter=args.leverage_filter,
                de_ratio_exclude_pct=args.de_ratio_exclude_pct,
                icover_min=args.icover_min,
                # 红利质量过滤（档位见调用前解析块）
                div_quality_filter=_dq_filter,
                div_years_min=_dq_years,
                require_ocf_cover=_dq_ocf,
                div_growth_min=_dq_growth,
                macd_filter_mode=args.macd_filter,
            )
        sys.exit(0)
        
    elif args.source == "macd_regime":
        # MACD 背离感知策略（池选股 + 月度调仓，无 KDJ）
        # 复用 run_macd_regime.run_strategy（已修复选股 set 哈希随机顺序导致的非确定性 bug；
        # 消融证明 KDJ-J 确认门反而拖累净值 ~22pp，故 kdj_gate 默认关）。
        # 与旧的 macd_kdj 择时插件（金叉/KDJ极值当按钮，已退休）无关——本策略是池选股形态。
        from run_macd_regime import run_strategy as _run_macd_regime
        # 优先用命令行 --stock-pool（000XXX.SH 或 hs300/zz800 键均可），否则回退 config SELECTION
        _SP_MAP = {"000300.SH": "hs300", "000905.SH": "zz500",
                   "000906.SH": "zz800", "000852.SH": "zz1000"}
        if getattr(args, "stock_pool", None):
            _pool = _SP_MAP.get(args.stock_pool, args.stock_pool)
        else:
            _pool = SELECTION.get("stock_pool", "zz800")
        _topn = args.top_n if args.top_n else SELECTION.get("top_n", 10)
        _cap = args.capital if args.capital else BACKTEST.get("total_capital", 100000)
        print(f"\n  MACD背离感知策略（池选股 + 月度调仓）")
        print(f"  回测区间: {BACKTEST['start_date']} ~ {BACKTEST['end_date']}")
        print(f"  股票池: {_pool} | 持仓: {_topn} | 资金: {_cap:,}")
        _run_macd_regime(
            BACKTEST["start_date"], BACKTEST["end_date"],
            pool=_pool, capital=_cap, top_n=_topn,
            regime_filter=True, kdj_gate=False,
        )
        sys.exit(0)

    elif args.source == "sc_rotation":
        # 小市值轮动回测（全市场最小流通市值·中证2000风格宇宙）
        from backtest_small_cap_rotation import (
            run_backtest as sc_run,
            run_survivor_bias_comparison,
            run_sensitivity,
            run_size_quintile_comparison,
        )
        _capital = args.capital if args.capital is not None else BACKTEST.get("total_capital", 500000)
        _hold = args.hold_count if args.hold_count is not None else 7
        _mode = args.sc_mode
        print(f"\n  小市值轮动策略 · 回测")
        print(f"  区间: {BACKTEST['start_date']} ~ {BACKTEST['end_date']}  | 持仓: {_hold} 只")
        print(f"  总资金: {_capital:,} 元")
        print(f"  空仓1/4月: {'开' if args.empty_jan_apr else '关'}  | 三层止损: {'开' if args.stop_loss else '关'}"
              f"  | A档质量门禁: {'开' if args.sc_quality_filter else '关'}  | B档成长倾斜: {'开' if args.sc_growth_tilt else '关'}"
              f"  | 维度3波动过滤: {'开' if args.sc_vol_filter else '关'}  | 维度5风格切换: {'开' if args.sc_style_switch else '关'}")
        print(f"  模式: {_mode}  | 选股宇宙: {args.sc_pool_mode}")
        print(f"  {'='*60}")
        if _mode == "compare":
            run_survivor_bias_comparison(
                BACKTEST["start_date"], BACKTEST["end_date"], hold_count=_hold,
                capital=_capital, empty_jan_apr=args.empty_jan_apr,
                enable_stop_loss=args.stop_loss, fundamental_filter=args.sc_fundamental,
                pool_mode=args.sc_pool_mode,
                quality_filter=args.sc_quality_filter, growth_tilt=args.sc_growth_tilt,
            )
        elif _mode == "sensitivity":
            run_sensitivity(
                BACKTEST["start_date"], BACKTEST["end_date"], capital=_capital,
                empty_jan_apr=args.empty_jan_apr, enable_stop_loss=args.stop_loss,
                fundamental_filter=args.sc_fundamental,
                hold_grid=[int(x) for x in args.hold_grid.split(",")],
                liq_grid=[float(x) for x in args.liq_grid.split(",")],
                pool_mode=args.sc_pool_mode,
                quality_filter=args.sc_quality_filter, growth_tilt=args.sc_growth_tilt,
            )
        elif _mode == "size_quintile":
            run_size_quintile_comparison(
                BACKTEST["start_date"], BACKTEST["end_date"], hold_count=_hold,
                capital=_capital, pool_mode=args.sc_pool_mode,
            )
        else:
            # 市值分位桶：small/mid/large → (order, offset, label)
            _BUCKET_MAP = {
                "small": ("ASC", 0, "小市值桶(最小N只)"),
                "mid":   ("ASC", 800, "中市值桶(宇宙40%分位档)"),
                "large": ("DESC", 0, "大市值桶(宇宙最大N只)"),
            }
            _b_order, _b_offset, _b_label = ("ASC", 0, None)
            if args.sc_bucket:
                _b_order, _b_offset, _b_label = _BUCKET_MAP[args.sc_bucket]
            _sc_detail_path = None
            if not args.sc_no_detail:
                _bn = f"_{args.sc_bucket}" if args.sc_bucket else ""
                _sc_detail_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "outputs",
                    f"sc_detail{_bn}_{args.sc_pool_mode}_{_hold}_{BACKTEST['start_date']}_{BACKTEST['end_date']}")
            sc_run(
                BACKTEST["start_date"], BACKTEST["end_date"], hold_count=_hold,
                capital=_capital, empty_jan_apr=args.empty_jan_apr,
                enable_stop_loss=args.stop_loss, fundamental_filter=args.sc_fundamental,
                exclude_delisted=args.exclude_delisted, min_avg_amount_k=args.min_avg_amount_k,
                pool_mode=args.sc_pool_mode,
                quality_filter=args.sc_quality_filter, growth_tilt=args.sc_growth_tilt,
                vol_filter=args.sc_vol_filter, style_switch=args.sc_style_switch,
                pool_order=_b_order, pool_offset=_b_offset, bucket_label=_b_label,
                detail_path=_sc_detail_path, no_html=args.sc_no_html,
                var_control=args.var_control, var_maxdd=args.var_maxdd, var_n=args.var_n,
            )
        sys.exit(0)

    elif args.source == "sc_kara":
        # Kara 小市值轮动（纯最小市值·月频·等权·零过滤器·无止损）— 平台集成入口
        # 引擎复用 backtest_kara_small_cap.run_backtest（与 sc_rotation 周频止损版是两类东西）
        from backtest_kara_small_cap import run_backtest as kara_run, _print_report as kara_print
        _hold = args.hold_count if args.hold_count is not None else 20
        _pool = args.sc_pool_mode  # cyb / zz2000 / zz1000
        print(f"\n  Kara 小市值轮动策略 · 回测（纯最小市值·月频·等权·零过滤器）")
        print(f"  区间: {BACKTEST['start_date']} ~ {BACKTEST['end_date']}  | 持仓: {_hold} 只")
        print(f"  选股宇宙: {_pool}（屏蔽老三板/北交所，含科创板）")
        print(f"  剔除科创板(688): {'是' if args.kara_exclude_688 else '否'}")
        print(f"  流动性门槛: {args.min_avg_amount_k if args.min_avg_amount_k is not None else '默认(3000万)'} 千元")
        print(f"  {'='*60}")
        _kara_res = kara_run(
            start=BACKTEST["start_date"], end=BACKTEST["end_date"],
            hold_count=_hold, pool_mode=_pool,
            min_avg_amount_k=args.min_avg_amount_k,
            exclude_688=args.kara_exclude_688,
        )
        kara_print(_kara_res)
        sys.exit(0)

    elif args.source == "csv":
        # 自动匹配最新的选股 CSV 文件（只匹配 selection/ 子目录）
        if args.auto or not args.file:
            results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT.get("selection_dir", os.path.join(OUTPUT["dir"], "selection")))
            os.makedirs(results_dir, exist_ok=True)
            csv_files = sorted(glob.glob(os.path.join(results_dir, "*.csv")), key=os.path.getmtime, reverse=True)
            if not csv_files:
                print(f"\n  [ERROR] 未找到 CSV 文件在 {results_dir}")
                print(f"  请先运行选股\n")
                sys.exit(1)
            csv_path = csv_files[0]
            print(f"\n  自动匹配最新 CSV: {os.path.basename(csv_path)}")
        else:
            csv_path = args.file
        
        if not os.path.exists(csv_path):
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

    elif args.source == "dca_etf":
        # 宽基 ETF 定投（单标的 / 篮子等权，月度 / 周度）+ 一次性投入对比
        from run_dca_etf import run_both, run_backtest as run_dca_one
        _sd = BACKTEST["start_date"]
        _ed = BACKTEST["end_date"]
        _code = args.dca_preset if args.dca_preset else args.dca_code
        print(f"\n  宽基 ETF 定投策略（DCA）")
        print(f"  标的: {_code} | 区间: {_sd} ~ {_ed}")
        print(f"  月投: ¥{args.dca_monthly:,.0f} | 周投: ¥{args.dca_weekly:,.0f} | 频率: {args.dca_freq} | 模式: {args.dca_mode}")
        print(f"  {'='*60}")
        if args.dca_freq == "both":
            run_both(start_date=_sd, end_date=_ed, codes=_code,
                     monthly_amount=args.dca_monthly, weekly_amount=args.dca_weekly,
                     mode=args.dca_mode)
        else:
            run_dca_one(start_date=_sd, end_date=_ed, freq=args.dca_freq, codes=_code,
                        monthly_amount=args.dca_monthly, weekly_amount=args.dca_weekly,
                        mode=args.dca_mode)
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
        
        if args.select_only:
            # 只选股，不回测
            print(f"\n{'='*60}")
            print(f"  选股结果（共 {len(stocks)} 只）：")
            for i, (_, row) in enumerate(stocks.iterrows(), 1):
                print(f"    {i}. {row['code']}  {row.get('name', '')}")
            print(f"\n{'='*60}")
            sel_dir = OUTPUT.get("selection_dir", os.path.join(OUTPUT["dir"], "selection"))
            print(f"  结果已保存到 {sel_dir}/")
            print(f"{'='*60}\n")
            sys.exit(0)
        
        run_backtest(stocks)

    print(f"\n{'='*60}")
    print("  完成！修改 config.py 参数后可再次运行。")
    print(f"{'='*60}\n")
