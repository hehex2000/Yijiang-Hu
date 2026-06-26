"""
月度调仓策略回测脚本（v3 - 参考 run_dividend_lowvol_opt_v2 重写）
=============================================
- 每月第5个交易日调仓（T日T-1日数据选股，T日开盘价执行）
- 选股策略：价值选股 / 红利低波（双重排序 + MACD择时）
- MACD择时（红利低波专用）：
  - 金叉（DIF > DEA）：选股 + 调仓 + 买回减仓股票
  - 死叉（DIF < DEA）：减仓50%（不清仓）
- 止损：-15%（T日收盘价触发，T+1日开盘价卖出）
- 止盈（价值选股专用）：PB > 1.2（T日触发，T+1日开盘价卖出）
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from config import DATA, BACKTEST, SELECTION, FACTOR_CALCULATOR, FACTOR_PROCESSOR
    DB_PATH = DATA.get("local_db_path", "D:/tu-shareData/astock_daily.db")
    INIT_CAPITAL = BACKTEST.get("initial_capital", 50000)
    # 不再使用模块级 TOP_N，改为动态读取 SELECTION["top_n"]
except (ImportError, KeyError, AttributeError):
    DB_PATH = "D:/tu-shareData/astock_daily.db"
    INIT_CAPITAL = 50000
    FACTOR_CALCULATOR = {}
    FACTOR_PROCESSOR = {}

def get_top_n():
    """动态获取选股数量（优先从 config.SELECTION 读取）"""
    try:
        from config import SELECTION
        return SELECTION.get("top_n", 5)
    except (ImportError, KeyError, AttributeError):
        return 5

STOP_LOSS = 0.15           # 止损线 -15%
PB_SELL_THRESHOLD = 1.2    # 止盈 PB 阈值
BEAR_REDUCE = 0.50          # MACD死叉减仓比例 50%
COMMISSION_RATE = 0.00025  # 佣金率
COMMISSION_MIN = 5.0       # 最低佣金
STAMP_DUTY_RATE = 0.001    # 印花税率（卖出收取）

# ---- MACD参数 ----
MACD_FAST   = 12
MACD_SLOW   = 26
MACD_SIGNAL = 9

# ---- 股票池 → 指数成分股映射 ----
STOCK_POOL_INDEX = {
    "hs300": "000300.SH",  # 沪深300
    "zz500": "000905.SH",  # 中证500
    "zz800": "000906.SH",  # 中证800
    "zz1000": "000852.SH",  # 中证1000
    "all":    None,          # 全A股，不过滤
}

# 指数代码 → 显示名称
INDEX_DISPLAY_NAME = {
    "000300.SH": "沪深300",
    "000905.SH": "中证500",
    "000906.SH": "中证800",
    "000852.SH": "中证1000",
    None:         "全A股",
}

def get_stock_pool_index():
    """从配置读取股票池对应的指数代码"""
    try:
        from config import SELECTION
        pool = SELECTION.get("stock_pool", "zz800")
    except (ImportError, KeyError, AttributeError):
        pool = "zz800"
    return STOCK_POOL_INDEX.get(pool, "000906.SH")

# ---- 低波因子 ----
VOL_WINDOW = 120  # 波动率计算窗口（交易日）


def get_conn():
    return sqlite3.connect(DB_PATH)


def get_stock_name(ts_code):
    conn = get_conn()
    row = pd.read_sql_query(
        "SELECT name FROM stock_basic WHERE ts_code = ? LIMIT 1",
        conn, params=(ts_code,)
    )
    conn.close()
    if len(row) > 0:
        return row.iloc[0]['name']
    return ts_code


def calc_fee(buy_or_sell, price, shares):
    amount = price * shares
    commission = max(amount * COMMISSION_RATE, COMMISSION_MIN)
    if buy_or_sell == 'buy':
        return commission
    else:
        stamp_duty = amount * STAMP_DUTY_RATE
        return commission + stamp_duty


# ============================================================
#  价格获取（复权）
# ============================================================

def get_price(ts_code, trade_date):
    """获取不复权收盘价（实际使用价格）"""
    if ts_code in ("000906.SH",):
        conn = get_conn()
        row = pd.read_sql_query(
            "SELECT close FROM index_daily WHERE ts_code = ? AND trade_date = ?",
            conn, params=(ts_code, trade_date)
        )
        if len(row) > 0:
            price = float(row.iloc[0]["close"])
            conn.close()
            return price
        row2 = pd.read_sql_query(
            "SELECT close FROM index_daily WHERE ts_code = ? AND trade_date < ? ORDER BY trade_date DESC LIMIT 1",
            conn, params=(ts_code, trade_date)
        )
        conn.close()
        if len(row2) > 0:
            return float(row2.iloc[0]["close"])
        return None

    conn = get_conn()
    row = pd.read_sql_query("""
        SELECT d.close AS raw_close
        FROM daily d
        WHERE d.ts_code = ? AND d.trade_date = ?
    """, conn, params=(ts_code, trade_date))

    if len(row) > 0:
        price = float(row.iloc[0]["raw_close"])
        conn.close()
        return price

    row2 = pd.read_sql_query("""
        SELECT d.close AS raw_close
        FROM daily d
        WHERE d.ts_code = ? AND d.trade_date < ?
        ORDER BY d.trade_date DESC LIMIT 1
    """, conn, params=(ts_code, trade_date))

    if len(row2) > 0:
        price = float(row2.iloc[0]["raw_close"])
        conn.close()
        return price

    conn.close()
    return None


def get_open_price(ts_code, trade_date):
    """获取不复权开盘价（实际使用价格）"""
    if ts_code in ("000906.SH",):
        return get_price(ts_code, trade_date)

    conn = get_conn()
    row = pd.read_sql_query("""
        SELECT d.open AS raw_open
        FROM daily d
        WHERE d.ts_code = ? AND d.trade_date = ?
    """, conn, params=(ts_code, trade_date))

    if len(row) > 0:
        price = float(row.iloc[0]["raw_open"])
        conn.close()
        return price

    row2 = pd.read_sql_query("""
        SELECT d.open AS raw_open
        FROM daily d
        WHERE d.ts_code = ? AND d.trade_date < ?
        ORDER BY d.trade_date DESC LIMIT 1
    """, conn, params=(ts_code, trade_date))

    if len(row2) > 0:
        price = float(row2.iloc[0]["raw_open"])
        conn.close()
        return price

    conn.close()
    return None


def get_pb(ts_code, trade_date):
    """获取某日PB值（用于价值选股的止盈判断）"""
    conn = get_conn()
    row = pd.read_sql_query(
        "SELECT pb FROM daily_basic WHERE ts_code = ? AND trade_date <= ? ORDER BY trade_date DESC LIMIT 1",
        conn, params=(ts_code, trade_date)
    )
    conn.close()
    if len(row) > 0 and pd.notna(row.iloc[0]["pb"]):
        return float(row.iloc[0]["pb"])
    return None


# ============================================================
#  交易日 / 调仓日
# ============================================================

def get_trade_dates(start_date, end_date):
    conn = get_conn()
    rows = pd.read_sql_query(
        "SELECT DISTINCT trade_date FROM daily WHERE trade_date BETWEEN ? AND ? ORDER BY trade_date",
        conn, params=(start_date, end_date)
    )
    conn.close()
    return rows["trade_date"].tolist()


def get_monthly_5th_trading_days(trade_dates):
    """每月第5个交易日"""
    df = pd.DataFrame({"trade_date": trade_dates})
    df["ym"] = df["trade_date"].astype(str).str[:6]
    _days = set()
    for _, g in df.groupby("ym"):
        dates = g["trade_date"].tolist()
        _days.add(dates[4] if len(dates) >= 5 else dates[-1])
    return _days


# ============================================================
#  中证800成分股
# ============================================================

ZZ800_CACHE = None
ZZ800_INDEX_CODE = None  # 缓存对应的指数代码

def get_index_constituents(index_code=None):
    """
    获取指数成分股（支持动态指数）
    
    Args:
        index_code: 指数代码（如 "000906.SH"），为None时从配置读取
    """
    global ZZ800_CACHE, ZZ800_INDEX_CODE
    
    # 如果未指定指数代码，从配置读取
    if index_code is None:
        index_code = get_stock_pool_index()
    
    # 全A股模式：不过滤
    if index_code is None:
        return None  # 调用方需要检查返回值
    
    # 如果指数代码没变，使用缓存
    if ZZ800_CACHE is not None and ZZ800_INDEX_CODE == index_code:
        return ZZ800_CACHE
    
    conn = get_conn()
    rows = pd.read_sql_query(
        "SELECT ts_code FROM index_constituent WHERE index_code = ?",
        conn, params=(index_code,)
    )
    conn.close()
    
    ZZ800_CACHE = set(rows["ts_code"].tolist()) if len(rows) > 0 else set()
    ZZ800_INDEX_CODE = index_code
    return ZZ800_CACHE


# ============================================================
#  MACD 计算
# ============================================================

def calc_macd(ts_code, trade_date, is_index=False):
    """计算MACD指标，返回 (dif, dea, macd_hist)"""
    table = "index_daily" if is_index else "daily"
    conn = get_conn()
    rows = pd.read_sql_query(f"""
        SELECT trade_date, close FROM {table}
        WHERE ts_code = ? AND trade_date <= ?
        ORDER BY trade_date DESC
        LIMIT 200
    """, conn, params=(ts_code, trade_date))
    conn.close()

    if len(rows) < MACD_SLOW + MACD_SIGNAL:
        return None, None, None

    closes = rows["close"].values[::-1]
    ema_fast = pd.Series(closes).ewm(span=MACD_FAST, adjust=False).mean().values
    ema_slow = pd.Series(closes).ewm(span=MACD_SLOW, adjust=False).mean().values
    dif = ema_fast - ema_slow
    dea = pd.Series(dif).ewm(span=MACD_SIGNAL, adjust=False).mean().values
    macd_hist = 2 * (dif - dea)
    return float(dif[-1]), float(dea[-1]), float(macd_hist[-1])


def is_macd_golden(ts_code, trade_date, is_index=False):
    dif, dea, _ = calc_macd(ts_code, trade_date, is_index)
    if dif is None or dea is None:
        return False
    return dif > dea


# ============================================================
#  波动率计算
# ============================================================

def calc_volatility(ts_code, trade_date, window=VOL_WINDOW):
    """年化波动率：过去N日收益率标准差 × sqrt(252)"""
    conn = get_conn()
    rows = pd.read_sql_query("""
        SELECT close FROM daily
        WHERE ts_code = ? AND trade_date <= ?
        ORDER BY trade_date DESC
        LIMIT ?
    """, conn, params=(ts_code, trade_date, window + 1))
    conn.close()

    if len(rows) < max(window * 0.6, 60):
        return None

    closes = rows["close"].values[::-1]
    returns = (closes[1:] - closes[:-1]) / np.where(closes[:-1] == 0, 1, closes[:-1])
    return float(np.std(returns) * np.sqrt(252))


# ============================================================
#  选股函数
# ============================================================

def select_stocks(trade_date, top_n=None):
    """
    价值选股：
    - PB < 1.0（破净）
    - ROE > 8%
    - 流动比率 >= 1.2
    - 股票池成分股（从配置读取）
    - PE > 0 且 < 30
    """
    if top_n is None:
        top_n = get_top_n()
    index_code = get_stock_pool_index()
    zz_set = get_index_constituents(index_code)  # None 表示全A股
    conn = get_conn()

    actual_date = trade_date
    while True:
        cnt = pd.read_sql_query(
            "SELECT COUNT(*) AS n FROM daily_basic WHERE trade_date = ?",
            conn, params=(actual_date,)
        ).iloc[0]['n']
        if cnt > 0:
            break
        prev = pd.read_sql_query(
            "SELECT MAX(trade_date) AS max_date FROM daily_basic WHERE trade_date < ?",
            conn, params=(actual_date,)
        )
        actual_date = prev.iloc[0, 0]
        if actual_date is None:
            conn.close()
            return pd.DataFrame()

    df = pd.read_sql_query("""
        SELECT DISTINCT ts_code, pe_ttm, pb, close, total_mv
        FROM daily_basic
        WHERE trade_date = ?
          AND pe_ttm > 0 AND pe_ttm < 30
          AND pb > 0 AND pb < 1.0
          AND total_mv > 0
    """, conn, params=(actual_date,))
    conn.close()
    
    if df.empty:
        return df
    
    # 股票池过滤（None表示全A股，不过滤）
    if zz_set is not None:
        df = df[df['ts_code'].isin(zz_set)]
        if df.empty:
            return df
    
    # 过滤 ROE > 8% 且 流动比率 >= 1.2
    if 'roe' in df.columns:
        df = df[df['roe'].notna() & (df['roe'] > 8)]
    if 'current_ratio' in df.columns:
        df = df[df['current_ratio'].notna() & (df['current_ratio'] >= 1.2)]

    result = df.sort_values('pb', ascending=True).head(top_n)
    print(f"  [选股] 筛选后{len(df)}只，取前{top_n}只，实际返回{len(result)}只")
    return result[['ts_code']]


def select_dividend_low_vol_stocks(trade_date, top_n=None):
    """
    红利低波双重排序选股：
    1. 股票池成分股（从配置读取）
    2. 估值过滤：PE/PB合理 + 有分红
    3. 个股MACD金叉过滤
    4. 股息率 + 波动率双重排序
    """
    if top_n is None:
        top_n = get_top_n()
    
    index_code = get_stock_pool_index()
    zz_set = get_index_constituents(index_code)  # None 表示全A股
    conn = get_conn()

    actual_date = trade_date
    while True:
        cnt = pd.read_sql_query(
            "SELECT COUNT(*) AS n FROM daily_basic WHERE trade_date = ?",
            conn, params=(actual_date,)
        ).iloc[0]['n']
        if cnt > 0:
            break
        prev = pd.read_sql_query(
            "SELECT MAX(trade_date) AS max_date FROM daily_basic WHERE trade_date < ?",
            conn, params=(actual_date,)
        )
        actual_date = prev.iloc[0, 0]
        if actual_date is None:
            conn.close()
            return pd.DataFrame()

    df = pd.read_sql_query("""
        SELECT ts_code, pe_ttm, pb, dv_ttm, total_mv
        FROM daily_basic
        WHERE trade_date = ?
          AND pe_ttm > 0 AND pe_ttm < 50
          AND pb > 0 AND pb < 10
          AND dv_ttm > 0
          AND total_mv > 0
    """, conn, params=(actual_date,))
    conn.close()

    if df.empty:
        return df

    # 股票池过滤（None表示全A股，不过滤）
    if zz_set is not None:
        df = df[df['ts_code'].isin(zz_set)]
        if df.empty:
            return df

    # 排除ST
    conn = get_conn()
    st_codes = pd.read_sql_query(
        "SELECT ts_code FROM stock_basic WHERE name LIKE '%ST%' OR name LIKE '%*%'",
        conn
    )
    conn.close()
    if len(st_codes) > 0:
        st_set = set(st_codes["ts_code"].tolist())
        df = df[~df['ts_code'].isin(st_set)]

    if df.empty:
        return df

    # 个股MACD金叉过滤
    print(f"  MACD过滤个股中...")
    macd_ok = []
    for ts_code in df['ts_code']:
        if is_macd_golden(ts_code, actual_date, is_index=False):
            macd_ok.append(ts_code)
    df = df[df['ts_code'].isin(macd_ok)]
    print(f"  MACD金叉过滤后：{len(df)}只")

    if df.empty:
        return df

    # 计算波动率
    print(f"  计算 {len(df)} 只股票波动率...")
    volatilities = {}
    for idx, row in df.iterrows():
        code = row['ts_code']
        vol = calc_volatility(code, actual_date)
        if vol is not None:
            volatilities[code] = vol

    vol_codes = set(volatilities.keys())
    df = df[df['ts_code'].isin(vol_codes)]
    if df.empty:
        return df

    df['volatility'] = df['ts_code'].map(volatilities)

    # 双重排序
    df['dv_rank'] = df['dv_ttm'].rank(pct=True)
    df['vol_rank'] = df['volatility'].rank(pct=True, ascending=False)
    df['score'] = (df['dv_rank'] + df['vol_rank']) / 2

    result = df.sort_values('score', ascending=False).head(top_n)
    codes = result['ts_code'].tolist()
    name_str = ', '.join([f"{c}({get_stock_name(c)})" for c in codes])
    print(f"  最终选出：{name_str}")
    return result[['ts_code']]


def select_by_method(method, trade_date, top_n=None):
    """调度选股函数"""
    if top_n is None:
        top_n = get_top_n()
    
    if method == "div_low_vol":
        return select_dividend_low_vol_stocks(trade_date, top_n)
    else:  # value
        return select_stocks(trade_date, top_n)


# ============================================================
#  主回测
# ============================================================

def run_backtest(start_date="20200102", end_date="20251231", top_n=None, selection_method="value", select_only=False):
    if top_n is None:
        top_n = get_top_n()

    # 获取股票池显示名称
    _pool_idx = get_stock_pool_index()
    if _pool_idx is None:
        _pool_name = "全A股"
    else:
        _pool_name = INDEX_DISPLAY_NAME.get(_pool_idx, _pool_idx)
    
    print("=" * 70)
    print(f"月度调仓回测：{start_date} ~ {end_date}")
    print("=" * 70)
    print(f"  选股池：{_pool_name}成分股")
    print(f"  选股策略：{'红利低波' if selection_method == 'div_low_vol' else '价值选股'}")
    print(f"  持仓数量：{top_n}只（等权重）")
    print(f"  止损：-{STOP_LOSS*100:.0f}%")
    print(f"  MACD死叉：减仓{BEAR_REDUCE*100:.0f}%（不清仓），金叉买回\n")

    trade_dates = get_trade_dates(start_date, end_date)
    rebalance_set = get_monthly_5th_trading_days(trade_dates)

    # 仅选股模式：执行第一次选股后退出，不回测
    if select_only:
        if len(rebalance_set) == 0:
            print(f"\n  [ERROR] 没有找到调仓日！")
            return
        first_rb = sorted(rebalance_set)[0]
        selected_codes = select_stocks(first_rb, top_n)
        print(f"\n{'='*60}")
        if selected_codes and len(selected_codes) > 0:
            print(f"  选股结果（共 {len(selected_codes)} 只）:")
            for c in selected_codes:
                print(f"    {c}({get_stock_name(c)})")
        else:
            print(f"  [ERROR] 选股失败！")
        print(f"\n{'='*60}\n")
        return

    positions    = {}
    cash         = INIT_CAPITAL
    stop_count   = 0
    reduce_count = 0
    daily_vals   = []
    trades       = []
    pending_orders = []

    print(f"交易日总数：{len(trade_dates)}")

    for i, td in enumerate(trade_dates):
        # ═══ 步骤1：执行待执行订单（止损卖出、减仓卖出）═══
        if len(pending_orders) > 0:
            remaining = []
            for order in pending_orders:
                ts_code = order["ts_code"]
                open_price = get_open_price(ts_code, td)
                if open_price is None:
                    remaining.append(order)
                    continue
                if ts_code not in positions:
                    continue
                pos = positions[ts_code]
                sell_shares = min(order.get("shares", pos["shares"]), pos["shares"])
                if sell_shares <= 0:
                    continue
                proceeds = sell_shares * open_price
                fee = calc_fee('sell', open_price, sell_shares)
                cash += proceeds - fee
                pos["shares"] -= sell_shares
                reason = order.get("reason", "")
                trades.append({
                    "date": td, "action": "SELL", "code": ts_code,
                    "name": get_stock_name(ts_code),
                    "price": open_price, "shares": sell_shares, "reason": reason
                })
                if pos["shares"] == 0:
                    del positions[ts_code]
            pending_orders = remaining

        # ═══ 步骤2：记录当日市值 ═══
        total_value = cash
        for code, pos in positions.items():
            price = get_price(code, td)
            if price is not None:
                total_value += pos["shares"] * price
        daily_vals.append({"date": td, "value": total_value})

        # ═══ 步骤3：检查止损（T日收盘价触发，创建T+1日挂单）═══
        for code in list(positions.keys()):
            price = get_price(code, td)
            if price is None:
                continue
            pos = positions[code]
            if price < pos["buy_price"] * (1 - STOP_LOSS):
                name = get_stock_name(code)
                pending_orders.append({
                    "type": "sell", "ts_code": code,
                    "shares": pos["shares"], "reason": "stop_loss"
                })
                print(f"  🔴 止损 {code}({name})：{td} 收盘{price:.2f} < 买入价{pos['buy_price']:.2f}×({1-STOP_LOSS:.0%})")
                stop_count += 1

        # ═══ 步骤3b：PB止盈（仅价值选股，T日收盘触发，T+1日开盘执行）═══
        if selection_method == "value":
            for code in list(positions.keys()):
                if any(o["ts_code"] == code and o.get("reason") == "stop_loss" for o in pending_orders):
                    continue  # 已触发止损，不重复卖出
                pb_val = get_pb(code, td)
                if pb_val is None or pb_val < PB_SELL_THRESHOLD:
                    continue
                pos = positions[code]
                name = get_stock_name(code)
                pending_orders.append({
                    "type": "sell", "ts_code": code,
                    "shares": pos["shares"], "reason": "pb_take_profit"
                })
                print(f"  🟢 止盈 {code}({name})：{td} PB={pb_val:.2f} > 阈值{PB_SELL_THRESHOLD}")

        # ═══ 步骤4：调仓日决策（仅调仓日执行）═══
        if td not in rebalance_set:
            continue

        prev_td = trade_dates[i-1] if i > 0 else td

        # ═══ 红利低波：MACD大盘择时 ═══
        if selection_method == "div_low_vol":
            # 获取基准指数代码（全A股模式用中证800作为MACD基准）
            benchmark_idx = get_stock_pool_index()
            if benchmark_idx is None:
                benchmark_idx = "000906.SH"
            benchmark_name = INDEX_DISPLAY_NAME.get(benchmark_idx, benchmark_idx)
            
            dif, dea, _ = calc_macd(benchmark_idx, prev_td, is_index=True)
            
            if dif is not None and dea is not None and dif > dea:
                # ── MACD金叉：选股 + 调仓 + 买回减仓股票 ──
                print(f"\nMACD金叉：{benchmark_name} DIF {dif:.2f} > DEA {dea:.2f}")
                print(f"调仓日：{td}")

                stocks = select_by_method(selection_method, prev_td, top_n=top_n)
                new_codes = stocks['ts_code'].tolist() if not stocks.empty else []

                # 买回之前减仓的股票（不超过当前持仓量，防止止损后超买）
                for code in list(positions.keys()):
                    if code in new_codes and positions[code].get("reduced_shares", 0) > 0:
                        name = get_stock_name(code)
                        open_price = get_open_price(code, td)
                        if open_price is None:
                            continue
                        buy_back = min(positions[code]["reduced_shares"], positions[code]["shares"])
                        if buy_back <= 0:
                            continue
                        cost = buy_back * open_price
                        fee = calc_fee('buy', open_price, buy_back)
                        if cost + fee <= cash:
                            cash -= cost + fee
                            # 加权平均买入价（用于止损判断）
                            old_value = positions[code]["shares"] * positions[code]["buy_price"]
                            positions[code]["shares"] += buy_back
                            positions[code]["buy_price"] = (old_value + buy_back * open_price) / positions[code]["shares"]
                            positions[code].pop("reduced_shares", None)
                            print(f"  🔷 买回 {code}({name})：{buy_back}股 @ {open_price:.2f}")
                            trades.append({
                                "date": td, "action": "BUY", "code": code, "name": name,
                                "price": open_price, "shares": buy_back, "reason": "buy_back"
                            })

                if not stocks.empty:
                    # 卖出不在新池中的旧持仓
                    for code in list(positions.keys()):
                        if code not in new_codes:
                            name = get_stock_name(code)
                            open_price = get_open_price(code, td)
                            if open_price is None:
                                continue
                            pos = positions[code]
                            if "reduced_shares" in pos:
                                del pos["reduced_shares"]
                            proceeds = pos["shares"] * open_price
                            fee = calc_fee('sell', open_price, pos["shares"])
                            cash += proceeds - fee
                            print(f"  ✅ 卖出 {code}({name})：{pos['shares']}股 @ {open_price:.2f}")
                            trades.append({
                                "date": td, "action": "SELL", "code": code, "name": name,
                                "price": open_price, "shares": pos["shares"], "reason": "rebalance"
                            })
                            del positions[code]

                # 买入新股票
                new_to_buy = [c for c in new_codes if c not in positions]
                if len(new_to_buy) > 0:
                    cash_per_stock = cash / len(new_to_buy)
                    skipped_stocks = []  # 记录因资金不足跳过的股票
                    
                    for ts_code in new_to_buy:
                        name = get_stock_name(ts_code)
                        open_price = get_open_price(ts_code, td)
                        if open_price is None:
                            print(f"  ⚠️ 跳过 {ts_code}({name})：无开盘价数据")
                            skipped_stocks.append(f"{ts_code}({name})：无开盘价数据")
                            continue
                        
                        max_shares = int(cash_per_stock / open_price / 100) * 100
                        if max_shares >= 100:
                            cost = max_shares * open_price
                            fee = calc_fee('buy', open_price, max_shares)
                            if cost + fee <= cash:
                                cash -= cost + fee
                                positions[ts_code] = {"shares": max_shares, "buy_price": open_price}
                                print(f"  ✅ 买入 {ts_code}({name})：{max_shares}股 @ {open_price:.2f}")
                                trades.append({
                                    "date": td, "action": "BUY", "code": ts_code, "name": name,
                                    "price": open_price, "shares": max_shares, "reason": "rebalance"
                                })
                            else:
                                skip_msg = f"{ts_code}({name})：资金不足（需要{cost+fee:.2f}元，可用{cash:.2f}元）"
                                print(f"  ⚠️ 跳过 {skip_msg}")
                                skipped_stocks.append(skip_msg)
                        else:
                            skip_msg = f"{ts_code}({name})：价格{open_price:.2f}元过高，分配资金{cash_per_stock:.2f}元不足买100股"
                            print(f"  ⚠️ 跳过 {skip_msg}")
                            skipped_stocks.append(skip_msg)
                    
                    # 打印跳过汇总
                    if skipped_stocks:
                        print(f"\n  ⚠️ 资金不足汇总：本次调仓跳过 {len(skipped_stocks)} 只股票")
                        for i, skip_msg in enumerate(skipped_stocks, 1):
                            print(f"    {i}. {skip_msg}")
                else:
                    print(f"  选股为空，保持现有仓位")

            elif dif is not None and dea is not None:
                # ── MACD死叉：减仓50%（不清仓）──
                print(f"\nMACD死叉：{benchmark_name} DIF {dif:.2f} < DEA {dea:.2f}，减仓{BEAR_REDUCE*100:.0f}%")
                print(f"调仓日：{td}")

                for code in list(positions.keys()):
                    name = get_stock_name(code)
                    pos = positions[code]
                    if pos.get("reduced_shares", 0) > 0:
                        continue
                    open_price = get_open_price(code, td)
                    if open_price is None:
                        continue
                    sell_shares = (int(pos["shares"] * BEAR_REDUCE) // 100) * 100
                    if sell_shares == 0:
                        continue
                    if sell_shares > 0:
                        proceeds = sell_shares * open_price
                        fee = calc_fee('sell', open_price, sell_shares)
                        cash += proceeds - fee
                        pos["shares"] -= sell_shares
                        positions[code]["reduced_shares"] = positions[code].get("reduced_shares", 0) + sell_shares
                        reduce_count += 1
                        print(f"  🔶 减仓 {code}({name})：{sell_shares}股 @ {open_price:.2f}")
                        trades.append({
                            "date": td, "action": "SELL", "code": code, "name": name,
                            "price": open_price, "shares": sell_shares, "reason": "macd_death"
                        })
                        if pos["shares"] == 0:
                            del positions[code]
            else:
                print(f"\n调仓日 {td}：MACD数据不足，保持现有仓位")

        else:
            # ═══ 价值选股：无MACD过滤，直接调仓 ═══
            stocks = select_by_method(selection_method, prev_td, top_n=top_n)
            new_codes = stocks['ts_code'].tolist() if not stocks.empty else []

            if not stocks.empty:
                # 判断是否需要调仓
                current_codes = set(positions.keys())
                new_code_set = set(new_codes)
                if current_codes == new_code_set:
                    print(f"\n调仓日 {td}：选股相同，持仓不变")
                    continue
                
                print(f"\n调仓日 {td}：选股{len(new_codes)}只")
                print(f"  本次选股：{[f'{c}({get_stock_name(c)})' for c in new_codes]}")
                print(f"  当前持仓：{[f'{c}({get_stock_name(c)})' for c in positions.keys()]}")
                
                # 卖出不在新池中的旧持仓
                for code in list(positions.keys()):
                    if code not in new_codes:
                        name = get_stock_name(code)
                        open_price = get_open_price(code, td)
                        if open_price is None:
                            continue
                        pos = positions[code]
                        proceeds = pos["shares"] * open_price
                        fee = calc_fee('sell', open_price, pos["shares"])
                        cash += proceeds - fee
                        print(f"  ✅ 卖出 {code}({name})：{pos['shares']}股 @ {open_price:.2f}")
                        trades.append({
                            "date": td, "action": "SELL", "code": code, "name": name,
                            "price": open_price, "shares": pos["shares"], "reason": "rebalance"
                        })
                        del positions[code]

                # 买入新股票
                new_to_buy = [c for c in new_codes if c not in positions]
                if len(new_to_buy) > 0:
                    cash_per_stock = cash / len(new_to_buy)
                    skipped_stocks = []  # 记录因资金不足跳过的股票
                    
                    for ts_code in new_to_buy:
                        name = get_stock_name(ts_code)
                        open_price = get_open_price(ts_code, td)
                        if open_price is None:
                            print(f"  ⚠️ 跳过 {ts_code}({name})：无开盘价数据")
                            skipped_stocks.append(f"{ts_code}({name})：无开盘价数据")
                            continue
                        
                        max_shares = int(cash_per_stock / open_price / 100) * 100
                        if max_shares >= 100:
                            cost = max_shares * open_price
                            fee = calc_fee('buy', open_price, max_shares)
                            if cost + fee <= cash:
                                cash -= cost + fee
                                positions[ts_code] = {"shares": max_shares, "buy_price": open_price}
                                print(f"  ✅ 买入 {ts_code}({name})：{max_shares}股 @ {open_price:.2f}")
                                trades.append({
                                    "date": td, "action": "BUY", "code": ts_code, "name": name,
                                    "price": open_price, "shares": max_shares, "reason": "rebalance"
                                })
                            else:
                                skip_msg = f"{ts_code}({name})：资金不足（需要{cost+fee:.2f}元，可用{cash:.2f}元）"
                                print(f"  ⚠️ 跳过 {skip_msg}")
                                skipped_stocks.append(skip_msg)
                        else:
                            skip_msg = f"{ts_code}({name})：价格{open_price:.2f}元过高，分配资金{cash_per_stock:.2f}元不足买100股"
                            print(f"  ⚠️ 跳过 {skip_msg}")
                            skipped_stocks.append(skip_msg)
                    
                    # 打印跳过汇总
                    if skipped_stocks:
                        print(f"\n  ⚠️ 资金不足汇总：本次调仓跳过 {len(skipped_stocks)} 只股票")
                        for i, skip_msg in enumerate(skipped_stocks, 1):
                            print(f"    {i}. {skip_msg}")
            else:
                print(f"\n调仓日 {td}：选股为空，保持现有仓位")

    # ═══ 回测结束：用最后一天收盘价平仓所有持仓 ═══
    if len(trade_dates) > 0:
        last_date = trade_dates[-1]
        for code in list(positions.keys()):
            name = get_stock_name(code)
            price = get_price(code, last_date)
            if price is not None:
                pos = positions[code]
                proceeds = pos["shares"] * price
                fee = calc_fee('sell', price, pos["shares"])
                cash += proceeds - fee
                trades.append({
                    "date": last_date, "action": "SELL", "code": code, "name": name,
                    "price": price, "shares": pos["shares"], "reason": "backtest_end"
                })
                del positions[code]

    # 最终资产（正常情况positions已清空，cash即为最终值；此处为安全兜底）
    final_value = cash
    if len(positions) > 0:
        for code, pos in positions.items():
            price = get_price(code, trade_dates[-1])
            if price is not None:
                final_value += pos["shares"] * price

    total_return = (final_value / INIT_CAPITAL - 1) * 100
    days = len(trade_dates)
    years = days / 252
    annual_return = ((final_value / INIT_CAPITAL) ** (1 / years) - 1) * 100 if years > 0 else 0

    vals = np.array([d["value"] for d in daily_vals])
    cummax = np.maximum.accumulate(vals)
    safe_cummax = np.where(cummax == 0, 1, np.array(cummax, dtype=float))
    drawdowns = (vals - cummax) / safe_cummax
    max_dd = float(np.min(drawdowns)) * 100

    # 夏普比率
    rets = np.diff(vals) / np.where(vals[:-1] == 0, 1, vals[:-1])
    if len(rets) > 1 and np.std(rets) > 0:
        sharpe = (np.mean(rets) * 252 - 0.025) / (np.std(rets) * np.sqrt(252))
    else:
        sharpe = 0.0

    print(f"\n{'='*70}")
    print("  回测结果")
    print(f"{'='*70}")
    print(f"  初始资金：{INIT_CAPITAL:,.2f}")
    print(f"  最终资产：{final_value:,.2f}")
    print(f"  总收益率：{total_return:+.2f}%")
    print(f"  年化收益率：{annual_return:+.2f}%")
    print(f"  最大回撤：{max_dd:.2f}%")
    print(f"  夏普比率：{sharpe:.2f}")
    print(f"  交易次数：{len(trades)}")
    print(f"  止损触发：{stop_count} 次")
    print(f"  减仓次数：{reduce_count} 次")

    # 动态基准指数对比
    benchmark_idx = get_stock_pool_index()
    if benchmark_idx is None:
        benchmark_idx = "000906.SH"  # 全A股模式用中证800作为基准
    benchmark_name = INDEX_DISPLAY_NAME.get(benchmark_idx, benchmark_idx)
    
    conn = get_conn()
    b_start = pd.read_sql_query(
        "SELECT close FROM index_daily WHERE ts_code = ? AND trade_date >= ? ORDER BY trade_date ASC LIMIT 1",
        conn, params=(benchmark_idx, trade_dates[0])
    )
    b_end = pd.read_sql_query(
        "SELECT close FROM index_daily WHERE ts_code = ? AND trade_date <= ? ORDER BY trade_date DESC LIMIT 1",
        conn, params=(benchmark_idx, trade_dates[-1])
    )
    conn.close()

    if len(b_start) > 0 and len(b_end) > 0:
        idx_start = float(b_start.iloc[0]['close'])
        idx_end = float(b_end.iloc[0]['close'])
        idx_return = (idx_end / idx_start - 1) * 100
        outperf = total_return - idx_return
        print(f"\n{'='*70}")
        print(f"  {benchmark_name}涨幅：{idx_return:+.2f}%")
        print(f"  策略{'跑赢' if outperf>0 else '跑输'}指数：{outperf:+.2f}%")

    # 保存结果
    csv_dir = "data/results/monthly_rebalance"
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = f"{csv_dir}/backtest_{start_date}_{end_date}.csv"
    pd.DataFrame(daily_vals).to_csv(csv_path, index=False)
    print(f"\n  结果已保存：{csv_path}")

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "trades": len(trades),
        "daily_values": daily_vals,
    }


# ══════════════════════════════════════════
#  入口
# ══════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="月度调仓回测")
    parser.add_argument("start_date", nargs="?", default="20200102", help="开始日期 YYYYMMDD")
    parser.add_argument("end_date", nargs="?", default="20251231", help="结束日期 YYYYMMDD")
    parser.add_argument("--top-n", type=int, default=None, help="选股数量")
    parser.add_argument("--selection-method", type=str, default="value",
                        choices=["value", "div_low_vol"], help="选股策略")
    parser.add_argument("--select-only", action="store_true",
                        help="只选股，不回测")
    args = parser.parse_args()

    print(f"回测周期：{args.start_date} ~ {args.end_date}")
    print(f"选股策略：{args.selection_method}")
    run_backtest(args.start_date, args.end_date, top_n=args.top_n, selection_method=args.selection_method, select_only=args.select_only)
