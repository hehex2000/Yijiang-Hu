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
SLIPPAGE_RATE = 0.001  # 滑点率（0.1%，模拟实盘买卖价差和冲击成本）


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
    """计算含滑点的总交易成本"""
    amount = price * shares
    commission = max(amount * COMMISSION_RATE, COMMISSION_MIN)
    slippage = amount * SLIPPAGE_RATE  # 买/卖均含滑点
    if buy_or_sell == 'buy':
        return commission + slippage
    else:
        stamp_duty = amount * STAMP_DUTY_RATE
        return commission + stamp_duty + slippage


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

def get_index_constituents(index_code=None, trade_date=None):
    """
    获取指数成分股（支持动态指数 + 历史成分股）

    优先取 trade_date 当天或之前最近的成分股快照；
    若表中无历史快照，则 fallback 到全量最新成分股。
    
    Args:
        index_code: 指数代码（如 "000906.SH"），为None时从配置读取
        trade_date: 调仓日（YYYYMMDD），用于取历史成分股快照
    """
    global ZZ800_CACHE, ZZ800_INDEX_CODE
    
    # 如果未指定指数代码，从配置读取
    if index_code is None:
        index_code = get_stock_pool_index()
    
    # 全A股模式：不过滤
    if index_code is None:
        return None  # 调用方需要检查返回值

    # 构建缓存键（含日期以区分历史快照）
    cache_key = (index_code, trade_date) if trade_date else (index_code, "latest")
    if ZZ800_CACHE is not None and ZZ800_INDEX_CODE == cache_key:
        return ZZ800_CACHE
    
    conn = get_conn()
    if trade_date:
        # 取调仓日当天或之前最近的成分股快照
        rows = pd.read_sql_query("""
            SELECT ts_code FROM index_constituent
            WHERE index_code = ? AND trade_date <= ?
            AND trade_date = (
                SELECT MAX(trade_date) FROM index_constituent
                WHERE index_code = ? AND trade_date <= ?
            )
        """, conn, params=(index_code, trade_date, index_code, trade_date))
    else:
        rows = pd.read_sql_query(
            "SELECT ts_code FROM index_constituent WHERE index_code = ?",
            conn, params=(index_code,)
        )
    conn.close()

    if len(rows) == 0 and trade_date:
        # fallback：若历史快照为空，使用全量最新成分股
        conn2 = get_conn()
        rows = pd.read_sql_query(
            "SELECT ts_code FROM index_constituent WHERE index_code = ?",
            conn2, params=(index_code,)
        )
        conn2.close()
    
    ZZ800_CACHE = set(rows["ts_code"].tolist()) if len(rows) > 0 else set()
    ZZ800_INDEX_CODE = cache_key
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


def is_above_ma(code, trade_date, period=20, is_index=True):
    """检查指数收盘价是否在MA之上"""
    table = "index_daily" if is_index else "daily"
    conn = get_conn()
    rows = pd.read_sql_query(f"""
        SELECT close FROM {table}
        WHERE ts_code = ? AND trade_date <= ?
        ORDER BY trade_date DESC LIMIT ?
    """, conn, params=(code, trade_date, period + 5))
    conn.close()
    if len(rows) < period:
        return True  # 数据不足时默认允许交易
    closes = rows["close"].values[::-1]
    ma = np.mean(closes[-period:])
    return float(closes[-1]) > ma


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


# ============================================================
#  动量选股函数（新增）
# ============================================================

# 动量回看月数常量
MOMENTUM_LOOKBACK = 6          # 默认6个月
MOMENTUM_TOP_N = 5             # 默认选5只


def calc_momentum_return(ts_code, trade_date, lookback_months=6):
    """
    计算个股在指定区间内的累计收益率（动量因子）

    使用不复权收盘价计算：(最新收盘价 - N个月前的收盘价) / N个月前的收盘价
    对应 Jegadeesh & Titman (1993) 的动量形成期收益率

    Args:
        ts_code: 股票代码
        trade_date: 当前交易日期（YYYYMMDD）
        lookback_months: 回看月数（3/6/12）

    Returns:
        float: 区间收益率（百分比，如 0.15 表示15%），数据不足时返回 None
    """
    from datetime import datetime
    from dateutil.relativedelta import relativedelta

    # 计算回看起始日期（用日历月数往前推，再多加10个交易日缓冲区）
    dt = datetime.strptime(trade_date, "%Y%m%d")
    start_dt = dt - relativedelta(months=lookback_months) - relativedelta(days=15)
    start_date = start_dt.strftime("%Y%m%d")

    conn = get_conn()
    rows = pd.read_sql_query("""
        SELECT trade_date, close
        FROM daily
        WHERE ts_code = ? AND trade_date >= ? AND trade_date < ?
        ORDER BY trade_date
    """, conn, params=(ts_code, start_date, trade_date))
    conn.close()

    if len(rows) < 2:
        return None

    # 取最早和最晚的收盘价
    first_close = float(rows.iloc[0]['close'])
    last_close = float(rows.iloc[-1]['close'])

    if first_close <= 0:
        return None

    ret = (last_close - first_close) / first_close
    return ret


def select_momentum_stocks(trade_date, lookback_months=6, top_n=5, index_code=None,
                           skip_recent_months=1):
    """
    动量选股：按过去N个月收益率排名，取前top_n只

    策略逻辑：
    1. 全A股范围（或指定指数成分股）
    2. 排除ST股票
    3. 排除过去N个月数据不足的股票
    4. 计算每只股票的N个月收益率（跳过最近M个月，避免短期反转干扰）
    5. 按收益率从高到低排序
    6. 取前top_n只

    [论文依据：Jegadeesh & Titman (1993) 发现跳最近1个月可显著提高动量信号质量]

    Args:
        trade_date: 调仓日期（YYYYMMDD）
        lookback_months: 回看月数（3/6/12）
        top_n: 选股数量
        index_code: 指数代码（None=全A股）
        skip_recent_months: 跳过最近N个月（默认1，避免短期反转）

    Returns:
        DataFrame: 包含 ts_code 列的选中股票表
    """
    from datetime import datetime
    from dateutil.relativedelta import relativedelta

    conn = get_conn()

    # ===== 1. 确定回看日期范围（跳过最近N个月）=====
    dt = datetime.strptime(trade_date, "%Y%m%d")
    total_months = lookback_months + skip_recent_months
    end_dt = dt - relativedelta(months=skip_recent_months)  # 结束于 skip_recent_months 前
    start_dt = dt - relativedelta(months=total_months)      # 开始于 total 个月前
    buffer_dt = start_dt - relativedelta(days=20)
    start_date_str = buffer_dt.strftime("%Y%m%d")
    end_date_str = end_dt.strftime("%Y%m%d")

    # ===== 2. 获取股票池（用调仓日取当时成分股）=====
    if index_code:
        constituents = get_index_constituents(index_code, trade_date=trade_date)
        if constituents is None or len(constituents) == 0:
            conn.close()
            print(f"  [选股] ⚠️ 指数 {index_code} 无成分股数据")
            return pd.DataFrame()
        stock_set = constituents
    else:
        # 全A股：从 daily 表获取近期有交易的股票（性能优化：用trade_date限制）
        rows = pd.read_sql_query("""
            SELECT DISTINCT d.ts_code
            FROM daily d
            WHERE d.trade_date = (
                SELECT MAX(trade_date) FROM daily WHERE trade_date <= ?
            )
        """, conn, params=(trade_date,))
        stock_set = set(rows['ts_code'].tolist())

    # ===== 3. 排除ST股票 =====
    st_codes = pd.read_sql_query(
        "SELECT ts_code FROM stock_basic WHERE name LIKE '%ST%' OR name LIKE '%*%'",
        conn
    )
    st_set = set(st_codes['ts_code'].tolist()) if len(st_codes) > 0 else set()
    candidates = stock_set - st_set
    conn.close()

    print(f"  [选股] 股票池共 {len(stock_set)} 只，排除ST后 {len(candidates)} 只")

    if len(candidates) == 0:
        print(f"  [选股] ⚠️ 无候选股票")
        return pd.DataFrame()

    # ===== 4. 批量获取行情数据 =====
    conn2 = get_conn()
    all_data = pd.read_sql_query("""
        SELECT ts_code, trade_date, close
        FROM daily
        WHERE trade_date >= ? AND trade_date <= ?
    """, conn2, params=(start_date_str, end_date_str))
    conn2.close()

    if all_data.empty:
        print(f"  [选股] ⚠️ 区间无数据")
        return pd.DataFrame()

    # Python端过滤候选股
    all_data = all_data[all_data['ts_code'].isin(candidates)]
    if all_data.empty:
        print(f"  [选股] ⚠️ 候选股无数据")
        return pd.DataFrame()

    # ===== 5. 计算每股收益率 =====
    def calc_stock_return(group):
        closes = group['close'].values
        if len(closes) < 2:
            return None
        # 找到实际 lookback 区间内的首尾价格
        # 取 trade_date 前最近的 close 作为 end_price
        # 取 start_date 后最近的 close 作为 start_price
        first_c = float(closes[0])
        last_c = float(closes[-1])
        if first_c <= 0:
            return None
        return (last_c - first_c) / first_c

    returns = {}
    for code, group in all_data.groupby('ts_code'):
        ret = calc_stock_return(group)
        if ret is not None:
            returns[code] = ret

    if not returns:
        print(f"  [选股] ⚠️ 无有效动量数据")
        return pd.DataFrame()

    # ===== 6. 排序取前N =====
    sorted_codes = sorted(returns.items(), key=lambda x: x[1], reverse=True)
    selected = sorted_codes[:top_n]

    result_codes = [c[0] for c in selected]
    name_str = ', '.join([f"{c}({get_stock_name(c)})" for c in result_codes])
    ret_str = ', '.join([f"{c}:{r:+.2%}" for c, r in selected])
    skip_info = f"（跳{skip_recent_months}月）" if skip_recent_months > 0 else ""
    print(f"  [选股] 动量{lookback_months}个月{skip_info} → 前{top_n}只：{name_str}")
    print(f"  [选股] 动量收益率：{ret_str}")

    return pd.DataFrame({'ts_code': result_codes})


def select_by_method(method, trade_date, top_n=None, lookback_months=None):
    """调度选股函数"""
    if top_n is None:
        top_n = get_top_n()
    
    if method == "momentum":
        lb = lookback_months if lookback_months is not None else MOMENTUM_LOOKBACK
        index_code = get_stock_pool_index()  # 从配置读取
        return select_momentum_stocks(trade_date, lookback_months=lb, top_n=top_n, index_code=index_code)
    elif method == "div_low_vol":
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
    profit_amount = final_value - INIT_CAPITAL
    print(f"  初始资金：{INIT_CAPITAL:,.2f}")
    print(f"  最终资产：{final_value:,.2f}")
    print(f"  总盈亏：{profit_amount:+,.2f} 元")
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
#  动量回测运行器（增强版）
# ══════════════════════════════════════════

def get_atr(ts_code, trade_date, period=14):
    """
    计算指定日期下某只股票的ATR（平均真实波幅）

    ATR = SMA(TR, period)
    TR = max(H-L, |H-prev_C|, |L-prev_C|)

    Args:
        ts_code: 股票代码
        trade_date: 交易日（YYYYMMDD），含当天数据
        period: ATR周期（默认14）

    Returns:
        float: ATR值，数据不足返回 None
    """
    conn = get_conn()
    rows = pd.read_sql_query("""
        SELECT trade_date, high, low, close
        FROM daily
        WHERE ts_code = ? AND trade_date <= ?
        ORDER BY trade_date DESC
        LIMIT ?
    """, conn, params=(ts_code, trade_date, period + 30))
    conn.close()

    if len(rows) < period + 1:
        return None

    # 按日期升序排列
    rows = rows.iloc[::-1].reset_index(drop=True)

    tr_values = []
    for i in range(1, len(rows)):
        h = float(rows.iloc[i]['high'])
        l = float(rows.iloc[i]['low'])
        pc = float(rows.iloc[i - 1]['close'])
        tr = max(h - l, abs(h - pc), abs(l - pc))
        tr_values.append(tr)

    if len(tr_values) < period:
        return None

    atr = sum(tr_values[-period:]) / period
    return atr * 1.0  # 确保返回float


def get_close_price(ts_code, trade_date):
    """获取某只股票指定日期的收盘价"""
    conn = get_conn()
    rows = pd.read_sql_query(
        "SELECT close FROM daily WHERE ts_code = ? AND trade_date = ?",
        conn, params=(ts_code, trade_date)
    )
    conn.close()
    return float(rows.iloc[0]['close']) if len(rows) > 0 else None


def run_momentum_backtest(start_date="20200101", end_date="20251231",
                          top_n=5, lookback_months=6, stock_pool=None,
                          rebalance_freq_months=1, atr_stop_multiple=0,
                          atr_cooling_days=0, trailing_stop_pct=0,
                          skip_recent_months=1, trend_filter_ma=0):
    """
    动量效应轮动回测（支持灵活调仓频率 + ATR止损/固定比例止损 + 冷静期 + 跳近期反转）

    每月第5个交易日（或每N个月第5个交易日）调仓：
    1. T-1日按过去N个月收益率选出动量最强的top_n只
    2. T日开盘价卖出不在新池的旧持仓
    3. T日开盘价等权重买入新进入的股票
    4. 每日检查止损（ATR或固定比例，买入冷静期内不触发），触发则次日开盘卖出

    Args:
        start_date: 回测开始日期
        end_date: 回测结束日期
        top_n: 持仓数量
        lookback_months: 动量回看月数（3/6/12）
        stock_pool: 股票池代码（如 "000906.SH"），None=全A股
        rebalance_freq_months: 调仓频率月数（1=每月，3=每季度）
        atr_stop_multiple: ATR止损倍数（0=不启用，与trailing_stop_pct互斥）
        atr_cooling_days: 买入后冷静期交易日数（期内不触发止损）
        trailing_stop_pct: 固定比例trailing stop（0=不启用，如0.15=15%）
        skip_recent_months: 跳过最近N个月（默认1，避免短期反转干扰动量信号）

    Returns:
        dict: 绩效指标
    """
    if stock_pool is None:
        pool_display = "全A股"
    else:
        pool_display = INDEX_DISPLAY_NAME.get(stock_pool, stock_pool)

    freq_label = f"每{rebalance_freq_months}个月" if rebalance_freq_months > 1 else "每月"
    
    if atr_stop_multiple > 0:
        stop_label = f" | ATR止损{atr_stop_multiple}倍"
        stop_detail = f"  ATR止损：{atr_stop_multiple}倍ATR（跌破最高价-{atr_stop_multiple}×ATR即卖出）"
    elif trailing_stop_pct > 0:
        stop_label = f" | 固定{trailing_stop_pct:.0%}止损"
        stop_detail = f"  固定止损：最高价回撤{trailing_stop_pct:.0%}即卖出（trailing stop）"
    else:
        stop_label = ""
        stop_detail = ""
    
    cooling_label = f" | 冷静期{atr_cooling_days}日" if atr_cooling_days > 0 else ""

    trend_filter_label = f" | MA{trend_filter_ma}过滤" if trend_filter_ma > 0 else ""

    print("=" * 70)
    print(f"动量效应回测（动量{lookback_months}个月 × {freq_label}调仓{stop_label}{cooling_label}{trend_filter_label}）")
    print("=" * 70)
    print(f"  股票池：{pool_display}")
    print(f"  持仓数量：{top_n}只（等权重）")
    print(f"  回测区间：{start_date} ~ {end_date}")
    print(f"  形成期（J）：{lookback_months}个月" + (f"（跳过最近{skip_recent_months}个月）" if skip_recent_months > 0 else ""))
    print(f"  持有期（K）：{rebalance_freq_months}个月（{freq_label}调仓）")
    if trend_filter_ma > 0:
        print(f"  市场过滤：指数<{trend_filter_ma}日MA时空仓等待")
    if stop_detail:
        print(stop_detail)
        if atr_cooling_days > 0:
            print(f"  冷静期：买入后{atr_cooling_days}个交易日内不触发止损")
    print(f"  佣金：万2.5（最低5元）| 印花税：千1 | 滑点：0.1% | 成分股：按调仓日历史快照\n")

    # === 获取交易日期 ===
    trade_dates = get_trade_dates(start_date, end_date)
    monthly_rebalance = get_monthly_5th_trading_days(trade_dates)
    # 按指定频率采样调仓日
    rebalance_set = set(list(monthly_rebalance)[::rebalance_freq_months])
    print(f"交易日总数：{len(trade_dates)}，调仓日：{len(rebalance_set)}次")

    # === 初始化 ===
    positions = {}   # {code: {"shares": N, "buy_price": P, "highest_close": P, "stop_triggered": bool}}
    cash = INIT_CAPITAL
    daily_vals = []
    trades = []
    stop_count = 0  # 止损次数统计

    # 预加载股票名称缓存
    name_cache = {}

    def get_name(code):
        if code not in name_cache:
            name_cache[code] = get_stock_name(code)
        return name_cache[code]

    for i, td in enumerate(trade_dates):
        # ========== 止损卖出（开盘执行昨日标记的止损） ==========
        use_stop = atr_stop_multiple > 0 or trailing_stop_pct > 0
        if use_stop and positions:
            for code in list(positions.keys()):
                pos = positions[code]

                # 冷静期检查
                buy_idx = pos.get("buy_idx", 0)
                holding_days = i - buy_idx
                if atr_cooling_days > 0 and holding_days < atr_cooling_days:
                    continue

                # 上一交易日已标记止损 → 今日开盘卖出
                if pos.get("stop_triggered", False):
                    open_price = get_open_price(code, td)
                    if open_price is not None:
                        proceeds = pos["shares"] * open_price
                        fee = calc_fee('sell', open_price, pos["shares"])
                        cash += proceeds - fee
                        stop_count += 1
                        print(f"  🔴 止损卖出 {code}({get_name(code)})：{pos['shares']}股 @ {open_price:.2f}")
                        trades.append({
                            "date": td, "action": "SELL", "code": code, "name": get_name(code),
                            "price": open_price, "shares": pos["shares"], "reason": "stop_loss"
                        })
                        del positions[code]
                    continue

                # 检查当前收盘价是否跌破止损线（次日开盘卖出）
                close_price = get_price(code, td)
                if close_price is None:
                    continue

                if close_price > pos.get("highest_close", 0):
                    pos["highest_close"] = close_price

                if atr_stop_multiple > 0:
                    atr = get_atr(code, td, period=14)
                    if atr is None or atr <= 0:
                        continue
                    stop_price = pos["highest_close"] - atr_stop_multiple * atr
                elif trailing_stop_pct > 0:
                    stop_price = pos["highest_close"] * (1 - trailing_stop_pct)
                else:
                    continue

                if close_price < stop_price:
                    positions[code]["stop_triggered"] = True
                    mode = "ATR" if atr_stop_multiple > 0 else "固定比例"
                    print(f"  ⚠️ {mode}止损触发 {code}({get_name(code)})：收盘{close_price:.2f} < 止损{stop_price:.2f}（持有{holding_days}日）")

        # ========== 调仓日：卖出旧仓 + 买入新仓（均按当日开盘价）==========
        if td in rebalance_set:
            # ===== 市场趋势过滤：指数<MA200时只卖不买 =====
            benchmark_idx = stock_pool if stock_pool else "000906.SH"
            market_ok = True
            if trend_filter_ma > 0:
                market_ok = is_above_ma(benchmark_idx, td, period=trend_filter_ma, is_index=True)
                if not market_ok:
                    print(f"\n  ⏸️ {td} 指数<{trend_filter_ma}日MA，空仓等待")

            prev_td = trade_dates[i - 1] if i > 0 else td

            # 熊市时：卖出所有持仓，不做选股
            if not market_ok and positions:
                for code in list(positions.keys()):
                    open_price = get_open_price(code, td)
                    if open_price is None:
                        continue
                    pos = positions[code]
                    proceeds = pos["shares"] * open_price
                    fee = calc_fee('sell', open_price, pos["shares"])
                    cash += proceeds - fee
                    print(f"  ✅ 卖出 {code}({get_name(code)})：{pos['shares']}股 @ {open_price:.2f}")
                    trades.append({
                        "date": td, "action": "SELL", "code": code, "name": get_name(code),
                        "price": open_price, "shares": pos["shares"], "reason": "trend_filter_sell"
                    })
                    del positions[code]
                print(f"  💤 空仓等待中证800重回{trend_filter_ma}日MA上方")
                continue  # 跳过选股和买入

            # 选股（T-1日收盘数据）
            stocks = select_momentum_stocks(
                prev_td,
                lookback_months=lookback_months,
                top_n=top_n,
                index_code=stock_pool,
                skip_recent_months=skip_recent_months,
            )
            new_codes = stocks['ts_code'].tolist() if not stocks.empty else []

            if not new_codes:
                print(f"\n调仓日 {td}：选股为空，保持现有仓位")
            else:
                current_codes = set(positions.keys())
                new_set = set(new_codes)

                if current_codes == new_set:
                    print(f"\n调仓日 {td}：持仓不变")
                else:
                    print(f"\n调仓日 {td}：动量组合变更")
                    print(f"  新选：{[f'{c}({get_name(c)})' for c in new_codes]}")
                    print(f"  旧仓：{[f'{c}({get_name(c)})' for c in positions.keys()]}")

                    # 卖出不再选中的旧持仓（按当日开盘价）
                    for code in list(positions.keys()):
                        if code not in new_set:
                            open_price = get_open_price(code, td)
                            if open_price is None:
                                continue
                            pos = positions[code]
                            proceeds = pos["shares"] * open_price
                            fee = calc_fee('sell', open_price, pos["shares"])
                            cash += proceeds - fee
                            print(f"  ✅ 卖出 {code}({get_name(code)})：{pos['shares']}股 @ {open_price:.2f}")
                            trades.append({
                                "date": td, "action": "SELL", "code": code, "name": get_name(code),
                                "price": open_price, "shares": pos["shares"], "reason": "momentum_rebalance"
                            })
                            del positions[code]

                    # 买入新选中的股票（按new_to_buy均分现金，买不起跳过）
                    new_to_buy = [c for c in new_codes if c not in positions]
                    if new_to_buy:
                        cash_per_stock = cash / len(new_to_buy)
                        for ts_code in new_to_buy:
                            open_price = get_open_price(ts_code, td)
                            if open_price is None:
                                continue
                            max_shares = int(cash_per_stock / open_price / 100) * 100
                            if max_shares < 100:
                                continue
                            cost = max_shares * open_price
                            fee = calc_fee('buy', open_price, max_shares)
                            if cost + fee <= cash:
                                cash -= cost + fee
                                positions[ts_code] = {
                                    "shares": max_shares,
                                    "buy_price": open_price,
                                    "buy_idx": i,
                                    "highest_close": open_price,
                                    "stop_triggered": False,
                                }
                                print(f"  ✅ 买入 {ts_code}({get_name(ts_code)})：{max_shares}股 @ {open_price:.2f}")
                                trades.append({
                                    "date": td, "action": "BUY", "code": ts_code, "name": get_name(ts_code),
                                    "price": open_price, "shares": max_shares, "reason": "momentum_rebalance"
                                })

        # ========== 每日市值记录（Bug #2修复：调仓日后记录，反映当日实际持仓）==========
        total_value = cash
        for code, pos in list(positions.items()):
            price = get_price(code, td)
            if price is not None:
                total_value += pos["shares"] * price
        daily_vals.append({"date": td, "value": total_value})

    # === 回测结束：平仓 ===
    if trade_dates:
        last_date = trade_dates[-1]
        for code in list(positions.keys()):
            price = get_price(code, last_date)
            if price is not None:
                pos = positions[code]
                proceeds = pos["shares"] * price
                fee = calc_fee('sell', price, pos["shares"])
                cash += proceeds - fee
                trades.append({
                    "date": last_date, "action": "SELL", "code": code, "name": get_name(code),
                    "price": price, "shares": pos["shares"], "reason": "backtest_end"
                })
                del positions[code]

    # === 计算绩效 ===
    final_value = cash
    total_return = (final_value / INIT_CAPITAL - 1) * 100
    days = len(trade_dates)
    years = days / 252
    annual_return = ((final_value / INIT_CAPITAL) ** (1 / years) - 1) * 100 if years > 0 else 0

    vals = np.array([d["value"] for d in daily_vals])
    cummax = np.maximum.accumulate(vals)
    safe_cummax = np.where(cummax == 0, 1, np.array(cummax, dtype=float))
    drawdowns = (vals - cummax) / safe_cummax
    max_dd = float(np.min(drawdowns)) * 100

    rets = np.diff(vals) / np.where(vals[:-1] == 0, 1, vals[:-1])
    if len(rets) > 1 and np.std(rets) > 0:
        sharpe = (np.mean(rets) * 252 - 0.025) / (np.std(rets) * np.sqrt(252))
    else:
        sharpe = 0.0

    # === 基准指数 ===
    benchmark_idx = stock_pool if stock_pool else "000906.SH"
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

    idx_return = 0
    if len(b_start) > 0 and len(b_end) > 0:
        idx_start = float(b_start.iloc[0]['close'])
        idx_end = float(b_end.iloc[0]['close'])
        idx_return = (idx_end / idx_start - 1) * 100

    # === 输出 ===
    print(f"\n{'=' * 70}")
    print(f"  动量{lookback_months}个月 × {freq_label}调仓 回测结果")
    print(f"{'=' * 70}")
    profit_amount = final_value - INIT_CAPITAL
    print(f"  初始资金：{INIT_CAPITAL:,.2f}")
    print(f"  最终资产：{final_value:,.2f}")
    print(f"  总盈亏：{profit_amount:+,.2f} 元")
    print(f"  总收益率：{total_return:+.2f}%")
    print(f"  年化收益率：{annual_return:+.2f}%")
    print(f"  最大回撤：{max_dd:.2f}%")
    print(f"  夏普比率：{sharpe:.2f}")
    print(f"  交易次数：{len(trades)}")
    if atr_stop_multiple > 0 or trailing_stop_pct > 0:
        print(f"  止损次数：{stop_count}")
    print(f"  中证800涨幅：{idx_return:+.2f}%")
    print(f"  超额收益：{total_return - idx_return:+.2f}%")

    # 保存结果
    csv_dir = "data/results/momentum_rebalance"
    os.makedirs(csv_dir, exist_ok=True)
    freq_suffix = f"_{rebalance_freq_months}m_rebal"
    if atr_stop_multiple > 0:
        stop_suffix = f"_atr{atr_stop_multiple}"
    elif trailing_stop_pct > 0:
        stop_suffix = f"_trail{int(trailing_stop_pct*100)}"
    else:
        stop_suffix = ""
    cooling_suffix = f"_cool{atr_cooling_days}" if atr_cooling_days > 0 else ""
    csv_path = f"{csv_dir}/momentum_{lookback_months}m{freq_suffix}{stop_suffix}{cooling_suffix}_{start_date}_{end_date}.csv"
    pd.DataFrame(daily_vals).to_csv(csv_path, index=False)
    print(f"\n  结果已保存：{csv_path}")

    return {
        "lookback_months": lookback_months,
        "rebalance_freq_months": rebalance_freq_months,
        "atr_stop_multiple": atr_stop_multiple,
        "trailing_stop_pct": trailing_stop_pct,
        "skip_recent_months": skip_recent_months,
        "atr_cooling_days": atr_cooling_days,
        "total_return": total_return,
        "annual_return": annual_return,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "trades": len(trades),
        "idx_return": idx_return,
        "stop_count": stop_count,
        "daily_values": daily_vals,
    }


def compare_momentum_periods(start_date="20200101", end_date="20251231",
                             top_n=5, stock_pool=None):
    """
    对比3/6/12个月动量回看窗口效果

    Args:
        start_date: 回测开始日期
        end_date: 回测结束日期
        top_n: 持仓数量
        stock_pool: 股票池代码，None=全A股

    Returns:
        dict: 各周期回测结果
    """
    lookbacks = [3, 6, 12]
    results = {}

    for lb in lookbacks:
        print(f"\n\n{'#' * 70}")
        print(f"#  开始回测：动量{lb}个月")
        print(f"{'#' * 70}\n")
        result = run_momentum_backtest(
            start_date=start_date,
            end_date=end_date,
            top_n=top_n,
            lookback_months=lb,
            stock_pool=stock_pool,
        )
        results[lb] = result

    # === 输出对比表格 ===
    print(f"\n\n{'=' * 70}")
    print(f"  动量效应轮动策略对比（{'全A股' if stock_pool is None else stock_pool}·持有{top_n}只·月调仓）")
    print(f"  回测区间：{start_date} ~ {end_date}")
    print(f"{'=' * 70}")

    # 找基准收益（取最后一次回测的基准）
    idx_ret = results.get(12, {}).get("idx_return", 0)

    header = f"{'指标':<16} {'3个月':>10} {'6个月':>10} {'12个月':>10} {'中证800':>10}"
    print(f"\n{header}")
    print("-" * 60)

    rows = [
        ("总收益率(%)",    [results[lb]["total_return"]    for lb in lookbacks] + [idx_ret]),
        ("年化收益率(%)",   [results[lb]["annual_return"]   for lb in lookbacks] + ["-"]),
        ("最大回撤(%)",     [results[lb]["max_drawdown"]    for lb in lookbacks] + ["-"]),
        ("夏普比率",       [results[lb]["sharpe"]          for lb in lookbacks] + ["-"]),
        ("交易次数",       [results[lb]["trades"]          for lb in lookbacks] + ["-"]),
    ]

    for label, vals in rows:
        vals_str = [f"{v:>+8.2f}" if isinstance(v, (int, float)) and abs(v) > 0.01 else str(v).rjust(10) for v in vals]
        print(f"{label:<16} {'  '.join(vals_str)}")

    print(f"\n{'=' * 70}")
    print(f"  💡 结论：")
    
    # 找出最佳周期
    best_lb = max(lookbacks, key=lambda lb: results[lb]["total_return"])
    best_ret = results[best_lb]["total_return"]
    print(f"    最佳动量周期：{best_lb}个月（总收益率 {best_ret:+.2f}%）")

    for lb in lookbacks:
        r = results[lb]
        outperf = r["total_return"] - idx_ret
        print(f"    动量{lb}个月：收益 {r['total_return']:+.2f}% | "
              f"年化 {r['annual_return']:+.2f}% | "
              f"回撤 {r['max_drawdown']:.2f}% | "
              f"夏普 {r['sharpe']:.2f} | "
              f"超额 {outperf:+.2f}%")

    # 保存对比结果
    csv_dir = "data/results/momentum_rebalance"
    os.makedirs(csv_dir, exist_ok=True)
    report_path = f"{csv_dir}/momentum_compare_{start_date}_{end_date}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"动量效应轮动策略对比报告\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"股票池：{'全A股' if stock_pool is None else stock_pool}\n")
        f.write(f"持仓数量：{top_n}只\n")
        f.write(f"回测区间：{start_date} ~ {end_date}\n")
        f.write(f"{'=' * 60}\n\n")
        f.write(f"{'指标':<16} {'3个月':>10} {'6个月':>10} {'12个月':>10} {'中证800':>10}\n")
        f.write("-" * 60 + "\n")
        for label, vals in rows:
            vals_str = [f"{v:>+8.2f}" if isinstance(v, (int, float)) and abs(v) > 0.01 else str(v).rjust(10) for v in vals]
            f.write(f"{label:<16} {'  '.join(vals_str)}\n")
        f.write(f"\n结论：最佳动量周期 = {best_lb}个月（总收益率 {best_ret:+.2f}%）\n")
    print(f"\n  对比报告已保存：{report_path}")

    return results

# ══════════════════════════════════════════
#  短期逆转效应策略（新增）
# ══════════════════════════════════════════

def select_reversal_stocks(trade_date, lookback_days=5, top_n=5, index_code=None):
    """
    短期逆转选股：按过去N日收益率从低到高排名，取跌幅最大的top_n只
    过滤：ST、一字跌停、上市<60天
    """
    from datetime import datetime, timedelta
    conn = get_conn()

    if index_code:
        constituents = get_index_constituents(index_code, trade_date=trade_date)
        if constituents is None or len(constituents) == 0:
            conn.close()
            return pd.DataFrame()
        stock_set = constituents
    else:
        rows = pd.read_sql_query("""
            SELECT DISTINCT d.ts_code FROM daily d
            WHERE d.trade_date = (SELECT MAX(trade_date) FROM daily WHERE trade_date <= ?)
        """, conn, params=(trade_date,))
        stock_set = set(rows['ts_code'].tolist())

    st_codes = pd.read_sql_query(
        "SELECT ts_code FROM stock_basic WHERE name LIKE '%ST%' OR name LIKE '%*%'", conn)
    st_set = set(st_codes['ts_code'].tolist()) if len(st_codes) > 0 else set()
    candidates = stock_set - st_set

    limdown = pd.read_sql_query("""
        SELECT ts_code FROM daily WHERE trade_date = ? AND close = low AND pct_chg <= -9.5
    """, conn, params=(trade_date,))
    candidates -= set(limdown['ts_code'].tolist()) if len(limdown) > 0 else set()

    dt = datetime.strptime(trade_date, "%Y%m%d")
    cutoff = (dt - timedelta(days=60)).strftime("%Y%m%d")
    new_ipo = pd.read_sql_query("SELECT ts_code FROM stock_basic WHERE list_date > ?", conn, params=(cutoff,))
    candidates -= set(new_ipo['ts_code'].tolist()) if len(new_ipo) > 0 else set()
    conn.close()

    print(f"  [逆转] {len(stock_set)}只 → 过滤后 {len(candidates)}只")
    if len(candidates) == 0:
        return pd.DataFrame()

    conn2 = get_conn()
    start_str = (dt - timedelta(days=lookback_days + 15)).strftime("%Y%m%d")
    all_data = pd.read_sql_query("""
        SELECT ts_code, trade_date, close FROM daily
        WHERE trade_date >= ? AND trade_date <= ?
    """, conn2, params=(start_str, trade_date))
    conn2.close()

    all_data = all_data[all_data['ts_code'].isin(candidates)]
    if all_data.empty:
        return pd.DataFrame()

    returns = {}
    for code, group in all_data.groupby('ts_code'):
        closes = group['close'].values
        if len(closes) < 2: continue
        lc, fc = float(closes[-1]), float(closes[0])
        if fc <= 0 or lc <= 0: continue
        returns[code] = (lc - fc) / fc

    if not returns:
        return pd.DataFrame()

    sorted_codes = sorted(returns.items(), key=lambda x: x[1])[:top_n]
    ret_str = ', '.join([f"{c}:{r:+.2%}" for c, r in sorted_codes])
    print(f"  [逆转] {lookback_days}日跌幅最大：{ret_str}")
    return pd.DataFrame({'ts_code': [c[0] for c in sorted_codes]})


def run_reversal_backtest(start_date="20251201", end_date="20251231",
                          lookback_days=5, top_n=5, stock_pool=None,
                          holding_days=1, market_filter="none",
                          stop_loss_pct=0):
    """短期逆转效应——轮动回测
    market_filter: "none" | "ma20" | "macd" 市场趋势过滤
    stop_loss_pct: 个股止损比例（0=不启用，如0.08=跌破买价8%止损）
    """
    pool_display = INDEX_DISPLAY_NAME.get(stock_pool, "全A股") if stock_pool else "全A股"
    freq_label = f"每{holding_days}日" if holding_days > 1 else "每日"
    filter_label = {"none":"无过滤", "ma20":"价格>MA20", "macd":"MACD金叉"}.get(market_filter, "无过滤")
    stop_label = f" | 止损-{stop_loss_pct:.0%}" if stop_loss_pct > 0 else ""
    benchmark_idx = stock_pool if stock_pool else "000906.SH"

    print("=" * 70)
    print(f"短期逆转效应回测（{lookback_days}日跌幅 × {freq_label}轮动 × {filter_label}{stop_label}）")
    print("=" * 70)
    print(f"  股票池：{pool_display} | 持仓：{top_n}只 | 市场过滤：{filter_label}")
    if stop_loss_pct > 0:
        print(f"  个股止损：跌破买入价{stop_loss_pct:.0%}即卖出")
    print(f"  区间：{start_date} ~ {end_date}")
    print(f"  佣金：万2.5（最低5元）| 印花税：千1 | 滑点：0.1% | 成分股：按调仓日历史快照\n")

    trade_dates = get_trade_dates(start_date, end_date)
    if len(trade_dates) < 2:
        print("⚠️ 交易日不足")
        return None

    positions = {}
    cash = INIT_CAPITAL
    daily_vals = []
    trades = []
    prev_held = set()
    stop_count = 0
    gname = get_stock_name
    day_count = 0  # 持有天数计数器

    for i, td in enumerate(trade_dates):
        day_count += 1

        # ===== 执行昨日标记的止损卖出（开盘执行） =====
        to_execute = []
        for code, pos in list(positions.items()):
            if pos.get("stop_now", False):
                to_execute.append(code)
        for code in to_execute:
            op = get_open_price(code, td)
            if op is not None:
                pos = positions[code]
                cash += pos["shares"] * op - calc_fee('sell', op, pos["shares"])
                stop_count += 1
                print(f"  🔴 止损卖出 {code}({gname(code)})：{pos['shares']}股 @ {op:.2f}")
                trades.append({"date": td, "action": "SELL", "code": code, "name": gname(code),
                              "price": op, "shares": pos["shares"], "reason": "stop_loss"})
                del positions[code]

        # ===== 每日止损检查（Bug #4修复：所有交易日都检查） =====
        if stop_loss_pct > 0 and positions:
            for code, pos in list(positions.items()):
                close_p = get_price(code, td)
                if close_p is None:
                    continue
                buy_p = pos.get("buy_price", close_p)
                if close_p <= buy_p * (1 - stop_loss_pct):
                    # Bug #1修复：真正标记 stop_now
                    positions[code]["stop_now"] = True
                    print(f"  ⚠️ 止损触发 {code}({gname(code)})：收盘{close_p:.2f} ≤ {buy_p*(1-stop_loss_pct):.2f}（买入价{buy_p:.2f}）")

        # 只在轮动日（每holding_days天）或首次建仓时交易
        is_rotation_day = (i == 0) or (day_count >= holding_days)
        if not is_rotation_day:
            # 非轮动日：只记录市值
            total_value = cash
            for code, pos in positions.items():
                p = get_price(code, td)
                if p is not None: total_value += pos["shares"] * p
            daily_vals.append({"date": td, "value": total_value})
            continue

        day_count = 0

        # ===== 市场过滤检查（Bug #2修复：用prev_td而非td，避免未来函数） =====
        prev_td = trade_dates[i - 1] if i > 0 else td
        allow_buy = True
        if market_filter == "ma20":
            allow_buy = is_above_ma(benchmark_idx, prev_td)  # ← prev_td
            if not allow_buy:
                print(f"  ⏸️ {td} 指数<MA20，空仓等待（基于{prev_td}数据）")
        elif market_filter == "macd":
            allow_buy = is_macd_golden(benchmark_idx, prev_td, is_index=True)  # ← prev_td
            if not allow_buy:
                print(f"  ⏸️ {td} MACD死叉，空仓等待（基于{prev_td}数据）")

        # 记录今天要卖出的代码（用于禁止重复）
        today_sold = set(positions.keys())

        # 开盘卖出所有（无论市场状态，熊市也卖）
        for code in list(positions.keys()):
            op = get_open_price(code, td)
            if op is None: continue
            p = positions[code]
            cash += p["shares"] * op - calc_fee('sell', op, p["shares"])
            trades.append({"date": td, "action": "SELL", "code": code, "name": gname(code),
                          "price": op, "shares": p["shares"], "reason": "reversal"})
            del positions[code]

        # 市场不允许买入 → 跳过选股
        if not allow_buy:
            prev_held = set(today_sold)
            total_value = cash
            daily_vals.append({"date": td, "value": total_value})
            continue

        # 选股（基于 prev_td 数据）
        stocks = select_reversal_stocks(prev_td, lookback_days=lookback_days, top_n=top_n, index_code=stock_pool)
        codes = stocks['ts_code'].tolist() if not stocks.empty else []

        # 过滤重复（不买昨天持仓过的）
        if prev_held and codes:
            filtered = [c for c in codes if c not in prev_held]
            skipped = [c for c in codes if c in prev_held]
            if skipped:
                print(f"  🚫 禁止重复：{skipped}")
            if len(filtered) < top_n:
                extra = select_reversal_stocks(prev_td, lookback_days=lookback_days,
                                               top_n=top_n + len(skipped), index_code=stock_pool)
                extra_codes = [c for c in extra['ts_code'].tolist()
                              if c not in prev_held and c not in filtered]
                filtered += extra_codes
            codes = filtered[:top_n]
        prev_held = set(today_sold)

        # 等权重买入（顺延+集中买一手）
        if codes:
            cps = cash / len(codes)
            bought_count = 0
            skipped_codes = []
            for tc in codes:
                op = get_open_price(tc, td)
                if op is None: continue
                s = int(cps / op / 100) * 100
                if s < 100:
                    print(f"  ⚠️ 跳过 {tc}({gname(tc)})：股价{op:.2f}过高，{cps:.0f}元不足以买1手")
                    skipped_codes.append(tc)
                    continue
                cost = s * op + calc_fee('buy', op, s)
                if cost <= cash:
                    cash -= cost
                    positions[tc] = {"shares": s, "buy_price": op, "stop_now": False}
                    bought_count += 1
                    trades.append({"date": td, "action": "BUY", "code": tc, "name": gname(tc),
                                  "price": op, "shares": s, "reason": "reversal"})
                else:
                    print(f"  ⚠️ 跳过 {tc}({gname(tc)})：资金不足（需要{cost:.0f}，可用{cash:.0f}）")

            # ——— 顺延：买不起就往下多选几只替补 ———
            if skipped_codes and cash > 0 and bought_count < top_n:
                extra_needed = len(skipped_codes) + top_n
                extra_stocks = select_reversal_stocks(prev_td, lookback_days=lookback_days,
                                                      top_n=extra_needed, index_code=stock_pool)
                already_held = set(positions.keys())
                extra_codes = [c for c in extra_stocks['ts_code'].tolist()
                              if c not in already_held and c not in codes]
                if extra_codes:
                    bought_extra = 0
                    remaining_slots = top_n - len(positions)
                    for tc in extra_codes:
                        if bought_extra >= len(skipped_codes) or remaining_slots <= 0:
                            break
                        op = get_open_price(tc, td)
                        if op is None: continue
                        cash_per_extra = cash / max(remaining_slots, 1)
                        s = int(cash_per_extra / op / 100) * 100
                        if s < 100: continue
                        cost = s * op + calc_fee('buy', op, s)
                        if cost <= cash:
                            cash -= cost
                            positions[tc] = {"shares": s, "buy_price": op, "stop_now": False}
                            bought_extra += 1
                            remaining_slots -= 1
                            print(f"  🔄 替补买入 {tc}({gname(tc)})：{s}股 @ {op:.2f}")
                            trades.append({"date": td, "action": "BUY", "code": tc, "name": gname(tc),
                                          "price": op, "shares": s, "reason": "reversal_fallback"})

                # ——— 集中剩余资金买一手 ———
                if cash > 0 and len(positions) < top_n:
                    all_candidates = list(set(codes + extra_codes))
                    cheapest = None
                    cheapest_cost = float('inf')
                    for tc in all_candidates:
                        if tc in positions: continue
                        op = get_open_price(tc, td)
                        if op is None: continue
                        c1h = 100 * op + calc_fee('buy', op, 100)
                        if c1h <= cash and op < cheapest_cost:
                            cheapest = tc
                            cheapest_cost = c1h
                    if cheapest is not None:
                        op = get_open_price(cheapest, td)
                        fee = calc_fee('buy', op, 100)
                        cash -= 100 * op + fee
                        positions[cheapest] = {"shares": 100, "buy_price": op, "stop_now": False}
                        print(f"  💰 集中余款买1手 {cheapest}({gname(cheapest)})：100股 @ {op:.2f}")
                        trades.append({"date": td, "action": "BUY", "code": cheapest, "name": gname(cheapest),
                                      "price": op, "shares": 100, "reason": "reversal_1hand"})

            if bought_count == 0 and len(positions) == 0:
                print(f"  ⚠️ 调仓日 {td}：全部候选均买不起！")

        # 收盘市值
        tv = cash
        for code, pos in positions.items():
            p = get_price(code, td)
            if p is not None: tv += pos["shares"] * p
        daily_vals.append({"date": td, "value": tv})
        print(f"  {td} 市值 {tv:,.0f} | 持仓 {list(positions.keys())}")

        # 收盘市值
        tv = cash
        for code, pos in positions.items():
            p = get_price(code, td)
            if p is not None: tv += pos["shares"] * p
        daily_vals.append({"date": td, "value": tv})
        print(f"  {td} 市值 {tv:,.0f} | 持仓 {list(positions.keys())}")

    # 平仓
    if trade_dates:
        for code in list(positions.keys()):
            p = get_price(code, trade_dates[-1])
            if p is not None:
                cash += positions[code]["shares"] * p - calc_fee('sell', p, positions[code]["shares"])
                del positions[code]
        daily_vals.append({"date": trade_dates[-1], "value": cash})

    # 绩效
    fv = cash
    vals = np.array([d["value"] for d in daily_vals])
    tr = (fv / INIT_CAPITAL - 1) * 100
    days = len(trade_dates)
    ar = ((fv / INIT_CAPITAL) ** (1 / max(days/252, 0.01)) - 1) * 100
    cm = np.maximum.accumulate(vals)
    safe = np.where(cm == 0, 1, np.array(cm, dtype=float))
    dd = float(np.min((vals - cm) / safe)) * 100
    rets = np.diff(vals) / np.where(vals[:-1] == 0, 1, vals[:-1])
    sp = (np.mean(rets) * 252 - 0.025) / (np.std(rets) * np.sqrt(252)) if len(rets) > 1 and np.std(rets) > 0 else 0

    print(f"\n{'=' * 70}")
    print(f"  短期逆转效应回测结果")
    print(f"{'=' * 70}")
    profit_amount = fv - INIT_CAPITAL
    print(f"  回测天数：{days}个交易日 | 初始资金：{INIT_CAPITAL:,.0f}")
    print(f"  最终资产：{fv:,.2f} | 总盈亏：{profit_amount:+,.2f} 元")
    print(f"  总收益率：{tr:+.2f}% | 年化：{ar:+.2f}%")
    stxt = f" | 止损 {stop_count}次" if stop_loss_pct > 0 and stop_count > 0 else ""
    print(f"  最大回撤：{dd:.2f}% | 夏普比率：{sp:.2f} | 交易：{len(trades)}{stxt}")
    print(f"\n  逐日净值：")
    for dv in daily_vals:
        chg = (dv["value"] / INIT_CAPITAL - 1) * 100
        bar = "█" * max(0, int(chg)) if chg > 0 else "░" * min(0, int(-chg))
        print(f"    {dv['date']}  {dv['value']:>10,.0f} ({chg:+.2f}%) {bar}")

    return {"total_return": tr, "annual_return": ar, "max_drawdown": dd,
            "sharpe": sp, "trades": len(trades), "daily_values": daily_vals}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="月度调仓回测")
    parser.add_argument("start_date", nargs="?", default="20200102", help="开始日期 YYYYMMDD")
    parser.add_argument("end_date", nargs="?", default="20251231", help="结束日期 YYYYMMDD")
    parser.add_argument("--top-n", type=int, default=None, help="选股数量")
    parser.add_argument("--selection-method", type=str, default="value",
                        choices=["value", "div_low_vol", "momentum", "reversal"], help="选股策略")
    parser.add_argument("--select-only", action="store_true",
                        help="只选股，不回测")
    parser.add_argument("--lookback", type=int, default=6,
                        choices=[3, 6, 12], help="动量回看月数（仅 momentum 模式）")
    parser.add_argument("--compare", action="store_true",
                        help="对比模式：依次跑3/6/12个月动量回测")
    parser.add_argument("--stock-pool", type=str, default=None,
                        help="股票池指数代码（如 000300.SH），默认全A股")
    parser.add_argument("--rebalance-freq", type=int, default=1,
                        choices=[1, 3], help="调仓频率月数（1=每月，3=每季度，仅 momentum 模式）")
    parser.add_argument("--atr-stop", type=float, default=0,
                        help="ATR止损倍数（0=不启用，建议2~3，与--trailing-stop互斥，仅 momentum 模式）")
    parser.add_argument("--trailing-stop", type=float, default=0,
                        help="固定比例trailing stop（0=不启用，如0.15=15%，与--atr-stop互斥，仅 momentum 模式）")
    parser.add_argument("--atr-cooling", type=int, default=0,
                        help="买入后冷静期交易日数（期内不触发止损，仅 momentum 模式）")
    parser.add_argument("--skip-recent", type=int, default=1,
                        choices=[0, 1, 2], help="跳过最近N个月（默认1，避免短期反转，仅 momentum 模式）")
    parser.add_argument("--trend-filter", type=int, default=0,
                        help="市场趋势过滤MA周期（0=不启用，200=指数<200日MA时空仓，仅 momentum 模式）")
    parser.add_argument("--reversal-lookback", type=int, default=5,
                        help="逆转策略回看天数（仅 reversal 模式，默认5）")
    parser.add_argument("--reversal-hold", type=int, default=1,
                        help="逆转策略持有天数（仅 reversal 模式，默认1=每日轮动）")
    parser.add_argument("--reversal-stop", type=float, default=0,
                        help="逆转策略个股止损比例（0=不启用，0.08=8%，仅 reversal 模式）")
    parser.add_argument("--market-filter", type=str, default="none",
                        choices=["none", "ma20", "macd"], help="市场趋势过滤（仅 reversal 模式）")
    args = parser.parse_args()

    if args.selection_method == "momentum":
        if args.compare:
            # 对比模式：跑3/6/12三个周期
            print(f"动量效应轮动策略对比回测")
            print(f"{'=' * 60}")
            compare_momentum_periods(
                start_date=args.start_date,
                end_date=args.end_date,
                top_n=args.top_n if args.top_n is not None else MOMENTUM_TOP_N,
                stock_pool=args.stock_pool,
            )
        else:
            # 单次动量回测
            top_n = args.top_n if args.top_n is not None else MOMENTUM_TOP_N
            run_momentum_backtest(
                start_date=args.start_date,
                end_date=args.end_date,
                top_n=top_n,
                lookback_months=args.lookback,
                stock_pool=args.stock_pool,
                rebalance_freq_months=args.rebalance_freq,
                atr_stop_multiple=args.atr_stop,
                atr_cooling_days=args.atr_cooling,
                trailing_stop_pct=args.trailing_stop,
                skip_recent_months=args.skip_recent,
                trend_filter_ma=args.trend_filter,
            )
    elif args.selection_method == "reversal":
        # 短期逆转策略
        print(f"短期逆转效应回测")
        print(f"{'=' * 60}")
        run_reversal_backtest(
            start_date=args.start_date,
            end_date=args.end_date,
            lookback_days=args.reversal_lookback,
            top_n=args.top_n if args.top_n is not None else 5,
            stock_pool=args.stock_pool,
            holding_days=args.reversal_hold,
            market_filter=args.market_filter,
            stop_loss_pct=args.reversal_stop,
        )
    else:
        print(f"回测周期：{args.start_date} ~ {args.end_date}")
        print(f"选股策略：{args.selection_method}")
        run_backtest(args.start_date, args.end_date, top_n=args.top_n, selection_method=args.selection_method, select_only=args.select_only)
