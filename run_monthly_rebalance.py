"""
月度调仓策略回测脚本（v3 - 参考 run_dividend_lowvol_opt_v2 重写）
=============================================
- 每月第5个交易日调仓（T日T-1日数据选股，T日开盘价执行）
- 选股策略：价值选股 / 红利低波（双重排序 + MACD择时）
- MACD择时（红利低波专用，regime 语境感知·默认开启）：
  - 金叉（DIF > DEA）且 指数>MA200 且 非盘整(布林带宽分位>=25%)：选股 + 调仓 + 买回减仓股票
  - 死叉（DIF < DEA）：减仓50%（不清仓）
  - 金叉但语境未确认(盘整/下跌)：保持现有仓位（不买不卖）
  （旧 pure golden-cross 仍可经 --macd-filter golden 回退，用于 A/B 对照）
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
    _BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DB_PATH = DATA.get("local_db_path", "")
    if not DB_PATH or not os.path.exists(DB_PATH):
        # 优先使用项目目录下的数据文件
        DB_PATH = os.path.join(_BASE_DIR, "data", "tu-sharedata", "astock_daily.db")
    INIT_CAPITAL = BACKTEST.get("monthly_rebalance_capital", 100000)
    # 不再使用模块级 TOP_N，改为动态读取 SELECTION["top_n"]
except (ImportError, KeyError, AttributeError):
    # fallback: 使用项目相对路径
    _BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DB_PATH = os.path.join(_BASE_DIR, "data", "tu-sharedata", "astock_daily.db")
    INIT_CAPITAL = 100000
    FACTOR_CALCULATOR = {}
    FACTOR_PROCESSOR = {}

# 成交量分布 / 价值区因子（视频微结构灵感；纯本地DB取数）
try:
    from volume_profile import value_area_pass, fakeout_reclaim as vp_fakeout_reclaim
except (ImportError, Exception):
    value_area_pass = None
    vp_fakeout_reclaim = None

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
STAMP_DUTY_RATE = 0.001    # 印花税率（卖出收取）· 2023-08-28 前的旧税率
# ── 印花税历史分段（2023-08-28 起证券交易印花税由 0.1% 减半至 0.05%）──
STAMP_DUTY_RATE_OLD  = 0.001    # 2023-08-27 及以前
STAMP_DUTY_RATE_NEW  = 0.0005   # 2023-08-28 起
STAMP_DUTY_CUT_DATE  = 20230828


### ── 成交日上下文（让全平台策略自动用上真实历史税率）────────────
# 设计：所有交易流程必然是「先取价(get_price/get_open_price) → 紧接着算费(calc_fee)」，
# 因此取价函数会自动把当前成交日登记到上下文，calc_fee 无需显式传日期即可用对税率。
# 优先级：calc_fee 显式 trade_date > 上下文日期 > 兜底旧税率(并计数告警)。
_CTX_TRADE_DATE = None
_CTX_MISS_COUNT = 0          # 无任何日期信息的计费次数（审计用）
_CTX_WARNED = False


def set_trade_date_ctx(trade_date):
    """显式登记当前成交日（新策略可主动调用；老策略靠取价函数自动登记）。"""
    global _CTX_TRADE_DATE
    if trade_date is not None:
        _CTX_TRADE_DATE = int(trade_date)


def get_trade_date_ctx():
    return _CTX_TRADE_DATE


def reset_fee_ctx():
    """回测开始前清空上下文与审计计数。"""
    global _CTX_TRADE_DATE, _CTX_MISS_COUNT, _CTX_WARNED
    _CTX_TRADE_DATE = None
    _CTX_MISS_COUNT = 0
    _CTX_WARNED = False


def fee_ctx_audit():
    """返回 {"last_date":..., "miss_count":...}，供报告核对税率口径是否全程生效。"""
    return {"last_date": _CTX_TRADE_DATE, "miss_count": _CTX_MISS_COUNT}


def stamp_duty_rate(trade_date=None):
    """按成交日返回真实印花税率（2023-08-28 起 0.1% → 0.05%）。

    trade_date 为空时回退到成交日上下文；上下文也为空则用旧税率并记一次 miss。
    """
    global _CTX_MISS_COUNT, _CTX_WARNED
    td = trade_date if trade_date is not None else _CTX_TRADE_DATE
    if td is None:
        _CTX_MISS_COUNT += 1
        if not _CTX_WARNED:
            _CTX_WARNED = True
            print("  ⚠️ [费用] 计费时无成交日上下文，印花税暂按旧税率 0.1%；"
                  "如为新写策略请调用 set_trade_date_ctx(td)")
        return STAMP_DUTY_RATE_OLD
    td = int(td)
    return STAMP_DUTY_RATE_NEW if td >= STAMP_DUTY_CUT_DATE else STAMP_DUTY_RATE_OLD

# ── 流动性过滤（保守阈值，跑大盘股时几乎不剔除成分股）────
# daily.amount 单位为"千元"，故阈值以元传入时需 ×1000 换算
LIQUIDITY_MIN_AVG_AMOUNT = 50_000_000   # 日均成交额下限（元）：5000万
LIQUIDITY_LOOKBACK       = 20           # 滚动窗口（交易日）

# ---- 辅助函数 ----

def calc_win_rate(trades):
    """从交易记录列表计算胜率（FIFO匹配买卖对）"""
    if not trades:
        return 0.0, 0, 0
    pending = {}  # code -> [{"price": p, "shares": s}]
    win = 0
    total = 0
    for t in trades:
        code = t["code"]
        action = t["action"]
        shares = t["shares"]
        price = t["price"]
        if action.startswith("BUY"):
            if code not in pending:
                pending[code] = []
            pending[code].append({"price": price, "shares": shares})
        elif action.startswith("SELL"):
            remaining = shares
            while remaining > 0 and code in pending and pending[code]:
                first = pending[code][0]
                pnl_shares = min(first["shares"], remaining)
                pnl = (price - first["price"]) * pnl_shares
                total += 1
                if pnl > 0:
                    win += 1
                first["shares"] -= pnl_shares
                remaining -= pnl_shares
                if first["shares"] <= 0:
                    pending[code].pop(0)
    wr = (win / total * 100) if total > 0 else 0.0
    return wr, win, total

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
    "000001.SH": "上证指数",
    "000016.SH": "上证50",
    "000300.SH": "沪深300",
    "000688.SH": "科创50",
    "000698.SH": "科创100",
    "000852.SH": "中证1000",
    "000905.SH": "中证500",
    "000906.SH": "中证800",
    "000985.SH": "中证全指",
    "399001.SZ": "深证成指",
    "399006.SZ": "创业板指",
    "399673.SZ": "创业板50",
    "932000.SH": "中证2000",
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


def calc_fee(buy_or_sell, price, shares, trade_date=None):
    """计算含滑点的总交易成本

    印花税按成交日历史分段：2023-08-27 及以前 0.1%，2023-08-28 起 0.05%。
    trade_date 不传时自动回退到成交日上下文（由 get_price/get_open_price 登记），
    因此全平台既有策略无需改调用即可用上真实税率。
    """
    amount = price * shares
    commission = max(amount * COMMISSION_RATE, COMMISSION_MIN)
    slippage = amount * SLIPPAGE_RATE  # 买/卖均含滑点
    if buy_or_sell == 'buy':
        return commission + slippage
    else:
        stamp_duty = amount * stamp_duty_rate(trade_date)
        return commission + stamp_duty + slippage


def calc_fee_breakdown(buy_or_sell, price, shares, trade_date=None):
    """同 calc_fee，但返回分项 dict，供成本审计报告使用。

    return: {"commission":x, "slippage":x, "stamp_duty":x, "total":x}
    """
    amount = price * shares
    commission = max(amount * COMMISSION_RATE, COMMISSION_MIN)
    slippage = amount * SLIPPAGE_RATE
    duty = amount * stamp_duty_rate(trade_date) if buy_or_sell != 'buy' else 0.0
    return {"commission": commission, "slippage": slippage,
            "stamp_duty": duty, "total": commission + slippage + duty}


# ============================================================
#  价格获取（复权）
# ============================================================

def get_price(ts_code, trade_date):
    """获取不复权收盘价（实际使用价格）"""
    set_trade_date_ctx(trade_date)   # 登记成交日 → calc_fee 自动用对当期印花税率
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
    """获取交易执行价格

    2026-07-06前：开盘价（正常盘中交易）
    2026-07-06后：收盘价（盘后30分钟定价交易，非未来函数）
    """
    set_trade_date_ctx(trade_date)   # 登记成交日 → calc_fee 自动用对当期印花税率
    td = int(trade_date) if isinstance(trade_date, str) else trade_date
    if td >= 20260706:
        # 盘后定价交易 → 使用当日收盘价
        return get_price(ts_code, trade_date)

    # 以下为原有逻辑：开盘价（2026-07-06前）
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


# ════════════════════════════════════════════════════════════
#  VAR/ATR 动态止损用的按交易日批量 OHLC 缓存（仅 var_stop 启用时填充）
# ════════════════════════════════════════════════════════════
_DAY_OHLC = {}

def _get_ohlc(ts_code, trade_date):
    """返回 (high, low, close)；批量取当日所有股票 OHLC 并缓存，每天仅 1 次查询。缺失返回 None。

    用于 VAR/ATR 动态止损的滚动真实波幅(TR)计算，避免逐笔查库拖垮性能。
    """
    d = _DAY_OHLC.get(trade_date)
    if d is None:
        conn = get_conn()
        df = pd.read_sql_query(
            "SELECT ts_code, high, low, close FROM daily WHERE trade_date = ?",
            conn, params=(trade_date,))
        conn.close()
        d = {}
        for _, r in df.iterrows():
            h = float(r["high"]) if pd.notna(r["high"]) else None
            l = float(r["low"]) if pd.notna(r["low"]) else None
            c = float(r["close"]) if pd.notna(r["close"]) else None
            d[str(r["ts_code"])] = (h, l, c)
        _DAY_OHLC[trade_date] = d
    return d.get(ts_code)


# ════════════════════════════════════════════════════════════
#  按交易日批量涨跌幅缓存（仅 div_growth「涨停跑路」日规则启用时填充）
# ════════════════════════════════════════════════════════════
_DAY_PCT = {}
_PCT_CONN = None  # 复用连接：get_price 每次开连接 ~100ms，日规则每天全市场取数不能重复开

def _get_day_pct(ts_code, trade_date):
    """返回 code 在 trade_date 的涨跌幅%（(close-pre_close)/pre_close×100），
    当日无行情（停牌）返回 None。批量取当日所有股票并缓存，每天仅 1 次查询。"""
    global _PCT_CONN
    d = _DAY_PCT.get(trade_date)
    if d is None:
        if _PCT_CONN is None:
            _PCT_CONN = get_conn()
        df = pd.read_sql_query(
            "SELECT ts_code, close, pre_close FROM daily WHERE trade_date = ?",
            _PCT_CONN, params=(trade_date,))
        d = {}
        for _, r in df.iterrows():
            pc = float(r["pre_close"]) if pd.notna(r["pre_close"]) and r["pre_close"] else None
            cl = float(r["close"]) if pd.notna(r["close"]) else None
            d[str(r["ts_code"])] = (cl - pc) / pc * 100 if (pc and cl) else None
        _DAY_PCT[trade_date] = d
    return d.get(ts_code)


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
    """每月第5个交易日（按时间升序返回，保证跨进程可复现）

    注意：必须返回**有序 list**。历史上这里用 set 返回，
    调用方 list(...)[::N] 采样时受 PYTHONHASHSEED 影响，
    集合转 list 的顺序每个进程都不同 → 调仓日程漂移 → 回测结果不可复现。
    """
    df = pd.DataFrame({"trade_date": trade_dates})
    df["ym"] = df["trade_date"].astype(str).str[:6]
    _days = []
    for _, g in df.groupby("ym"):
        dates = g["trade_date"].tolist()
        _days.append(dates[4] if len(dates) >= 5 else dates[-1])
    return sorted(_days)


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
            WHERE index_code = ? AND ts_code NOT LIKE '688%' AND trade_date <= ?
            AND trade_date = (
                SELECT MAX(trade_date) FROM index_constituent
                WHERE index_code = ? AND ts_code NOT LIKE '688%' AND trade_date <= ?
            )
        """, conn, params=(index_code, trade_date, index_code, trade_date))
    else:
        rows = pd.read_sql_query(
            "SELECT ts_code FROM index_constituent WHERE index_code = ? AND ts_code NOT LIKE '688%'",
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


def _bb_width_pct(ts_code, trade_date, is_index=False, win=20, lookback=120):
    """布林带宽(标准差/均值)在最近 lookback 日的滚动分位(0~1)。
    分位低 = 波动率收缩 = 中枢/盘整期(regime 视角，对应视频《缠论中枢》)。"""
    table = "index_daily" if is_index else "daily"
    conn = get_conn()
    rows = pd.read_sql_query(f"""
        SELECT close FROM {table}
        WHERE ts_code = ? AND trade_date <= ?
        ORDER BY trade_date DESC LIMIT ?
    """, conn, params=(ts_code, trade_date, lookback + win))
    conn.close()
    if len(rows) < lookback + win:
        return 0.5  # 数据不足 → 中性(非中枢)
    closes = rows["close"].values[::-1]
    widths = []
    for j in range(len(closes) - win + 1):
        seg = closes[j:j + win]
        m = np.mean(seg)
        widths.append((np.std(seg) / m) if m != 0 else 0.0)
    widths = np.asarray(widths, dtype=float)
    cur = widths[-1]
    return float((widths < cur).mean())


def macd_state(ts_code, trade_date, is_index_signal=False,
               regime_code=None, regime_is_index=True,
               mode="regime", ma_period=200, bb_win=20,
               bb_lookback=120, bb_pct_th=0.25):
    """把 MACD 金叉/死叉升级为『语境感知』信号(落实视频《MACD金叉死叉》核心：
    金叉/死叉不是买卖按钮，要看趋势与盘整语境)。返回 'golden'/'death'/'neutral'。
      mode='golden' : 旧逻辑，仅 DIF>DEA→golden，否则 death(退化为原 is_macd_golden)
      mode='regime' : 金叉须叠加 [指数>MA200 且 非中枢(带宽分位>=th)] 才算 golden；
                      死叉→death；金叉但语境未确认→neutral(不买/不减)
    signal 用 ts_code(个股或指数)；regime 语境默认同标的，可传 regime_code 指定大盘(看大作小)。
    """
    dif, dea, _ = calc_macd(ts_code, trade_date, is_index_signal)
    if dif is None or dea is None:
        return "neutral"
    if dif > dea:
        if mode == "golden":
            return "golden"
        rc = ts_code if regime_code is None else regime_code
        ri = is_index_signal if regime_code is None else regime_is_index
        above_ma = is_above_ma(rc, trade_date, period=ma_period, is_index=ri)
        bb_pct = _bb_width_pct(rc, trade_date, ri, win=bb_win, lookback=bb_lookback)
        if above_ma and bb_pct >= bb_pct_th:
            return "golden"
        return "neutral"
    return "death"


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

def select_stocks(trade_date, top_n=None, mode="pobreak", size_neutral=False, value_pct=None):
    """
    价值选股（统一逻辑，来自 src.value_stock_selector.select_value_stocks）：
    - mode="pobreak"（默认）：PB < 1.0（破净）+ ROE>8% + 流动比率>=1.2 + 0<PE_TTM<30
    - mode="pure_bm"：放宽破净约束（去掉 pb<1.0），改由全市场 BM 分位门槛筛选便宜标的
    说明：ROE/流动比率 从 fina_indicator 取真实财报数据，修正此前
    daily_basic 无此列导致两条件被静默跳过的 bug。
    """
    if top_n is None:
        top_n = get_top_n()
    try:
        from config import SELECTION as _SEL
        pool = _SEL.get("stock_pool", "zz800")
    except Exception:
        pool = "zz800"
    from src.value_stock_selector import select_value_stocks as _shared
    return _shared(trade_date, top_n, pool, size_neutral=size_neutral,
                   value_pct=value_pct, mode=mode)


def select_dividend_low_vol_stocks(trade_date, top_n=None,
                                   leverage_filter=False, de_ratio_exclude_pct=5, icover_min=3,
                                   div_quality_filter=False, div_years_min=3,
                                   require_ocf_cover=True, div_growth_min=None,
                                   macd_filter_mode="golden"):
    """
    红利低波双重排序选股：
    1. 股票池成分股（从配置读取）
    2. 估值过滤：PE/PB合理 + 有分红
    3. 杠杆因子风控过滤（可选）：剔除高杠杆/低偿付能力股
    4. 红利质量三因子过滤（可选）：连续分红年数 + 分红现金覆盖 + 分红增长
    5. 个股MACD金叉过滤
    6. 股息率 + 波动率双重排序
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

    # 杠杆因子风控过滤（可选）
    if leverage_filter:
        df = apply_leverage_filters(df, actual_date,
                                    de_ratio_exclude_pct=de_ratio_exclude_pct,
                                    icover_min=icover_min)

    # 红利质量三因子过滤（可选）：稳(连续分红年数+现金覆盖) + 增(分红增长)
    if div_quality_filter:
        df = apply_dividend_quality_filters(df, actual_date,
                                            div_years_min=div_years_min,
                                            require_ocf_cover=require_ocf_cover,
                                            div_growth_min=div_growth_min)

    # 个股MACD过滤（regime 模式：个股金叉 + 大盘语境[指数>MA200 且非盘整] 才保留）
    print(f"  MACD过滤个股中... (模式={macd_filter_mode})")
    _bench = get_stock_pool_index() or "000906.SH"
    macd_ok = []
    for ts_code in df['ts_code']:
        if macd_state(ts_code, actual_date, is_index_signal=False,
                      regime_code=_bench, regime_is_index=True,
                      mode=macd_filter_mode) == "golden":
            macd_ok.append(ts_code)
    df = df[df['ts_code'].isin(macd_ok)]
    print(f"  MACD过滤后：{len(df)}只")

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
#  高股息 + 基本面成长 双因子选股（B站视频策略 · 全A池）
#  对应 run_dividend_growth_monthly.py 的 select_picks 逻辑：
#   筛1 股息率排名 : dv_ttm(全A) 取前 top_pct（视频为"三年总分红/市值"，TTM代理）
#   筛2 基本面五关 : PE∈(0,pe_max] | PEG∈[peg_min,peg_max] | ROE>roe_min
#                   | 营收同比>rev_min | 净利同比>np_min
#                   （fina_indicator 最新报告，ann_date≤trade_date 防未来函数）
#   筛3 交易层     : 剔除停牌/昨涨停 → 取股息率前 top_n
# ============================================================
_A_SHARE_RE = None

def select_dividend_growth_stocks(trade_date, top_n=None,
                                  top_pct=0.10, pe_max=20.0,
                                  peg_min=0.08, peg_max=2.0,
                                  roe_min=3.0, rev_min=5.0, np_min=11.0):
    """高股息+基本面成长双因子选股（返回含 ts_code 的 DataFrame）。"""
    global _A_SHARE_RE
    if _A_SHARE_RE is None:
        import re as _re
        _A_SHARE_RE = _re.compile(r"^(60|00|30|68)\d{4}\.(SH|SZ)$")
    if top_n is None:
        top_n = get_top_n()

    conn = get_conn()
    # daily_basic 个别交易日整日缺失 → 回退最近可用日
    actual_date = trade_date
    while True:
        cnt = pd.read_sql_query(
            "SELECT COUNT(*) AS n FROM daily_basic WHERE trade_date = ?",
            conn, params=(actual_date,)).iloc[0]["n"]
        if cnt > 0:
            break
        prev = pd.read_sql_query(
            "SELECT MAX(trade_date) AS m FROM daily_basic WHERE trade_date < ?",
            conn, params=(actual_date,))
        actual_date = prev.iloc[0, 0]
        if actual_date is None:
            conn.close()
            return pd.DataFrame()
    d_int = int(str(actual_date))

    df = pd.read_sql_query(
        "SELECT ts_code, dv_ttm, pe_ttm FROM daily_basic WHERE trade_date = ?",
        conn, params=(actual_date,))
    if df.empty:
        conn.close()
        return pd.DataFrame()
    df = df[(df["dv_ttm"] > 0) & (df["pe_ttm"] > 0)].copy()
    df = df[df["ts_code"].str.match(_A_SHARE_RE)]
    if df.empty:
        conn.close()
        return pd.DataFrame()

    # 剔除 ST（证券简称含 ST）
    ph = ",".join("?" for _ in df["ts_code"].tolist())
    st = pd.read_sql_query(
        f"SELECT ts_code, name FROM stock_basic WHERE ts_code IN ({ph})",
        conn, params=df["ts_code"].tolist())
    st = st[~st["name"].str.contains("ST", case=False, na=False)]
    df = df[df["ts_code"].isin(st["ts_code"])]
    if df.empty:
        conn.close()
        return pd.DataFrame()

    # ── 筛1：股息率前 top_pct ──
    n_top = max(int(round(len(df) * top_pct)), 1)
    df = df.sort_values("dv_ttm", ascending=False).head(n_top).copy()

    # ── 筛2：基本面五关（最新财报 ann_date ≤ 选股日）──
    fina = pd.read_sql_query(
        "SELECT ts_code, ann_date, roe, netprofit_yoy, tr_yoy FROM fina_indicator "
        "WHERE ann_date IS NOT NULL AND CAST(ann_date AS INTEGER) > 0 "
        "AND CAST(ann_date AS INTEGER) <= ?",
        conn, params=(d_int,))
    conn.close()
    if not fina.empty:
        fina = fina.sort_values("ann_date").drop_duplicates("ts_code", keep="last")
        # ⚠️ .values 迭代产出 numpy.ndarray（isinstance(tuple) 为 False）→ 必须转 tuple
        fmap = dict(zip(fina["ts_code"], map(tuple, fina[["roe", "netprofit_yoy", "tr_yoy"]].values)))
    else:
        fmap = {}
    f = df["ts_code"].map(fmap)
    roe = f.map(lambda x: x[0] if isinstance(x, tuple) else float("nan"))
    np_yoy = f.map(lambda x: x[1] if isinstance(x, tuple) else float("nan"))
    tr_yoy = f.map(lambda x: x[2] if isinstance(x, tuple) else float("nan"))
    peg = df["pe_ttm"] / np_yoy
    m = (
        (df["pe_ttm"] > 0) & (df["pe_ttm"] <= pe_max) &
        (peg >= peg_min) & (peg <= peg_max) &
        (roe > roe_min) & (tr_yoy > rev_min) & (np_yoy > np_min)
    )
    df = df[m]
    if df.empty:
        return pd.DataFrame()

    # ── 筛3：交易层（T-1 停牌 / 昨涨停）──
    conn2 = get_conn()
    drow = pd.read_sql_query(
        "SELECT ts_code, close, pre_close FROM daily WHERE trade_date = ?",
        conn2, params=(actual_date,))
    conn2.close()
    pmap = {}
    for _, r in drow.iterrows():
        pc = float(r["pre_close"]) if pd.notna(r["pre_close"]) and r["pre_close"] else None
        cl = float(r["close"]) if pd.notna(r["close"]) else None
        pmap[str(r["ts_code"])] = (cl - pc) / pc * 100 if (pc and cl) else None

    def _lim(code):
        if code.startswith("688"):
            return 19.9
        if code.startswith("300"):
            return 19.9 if d_int >= 20200824 else 9.9
        return 9.9

    keep = []
    for c in df["ts_code"]:
        p = pmap.get(c)
        if p is None:          # T-1 无行情 → 停牌
            continue
        if p >= _lim(c) - 0.1:  # 昨涨停
            continue
        keep.append(c)
    df = df[df["ts_code"].isin(keep)]
    if df.empty:
        return pd.DataFrame()

    df = df.sort_values("dv_ttm", ascending=False).head(top_n).copy()
    df["name"] = df["ts_code"].map(dict(zip(st["ts_code"], st["name"])))
    df["roe"] = roe.reindex(df.index).round(2)
    df["np_yoy"] = np_yoy.reindex(df.index).round(1)
    df["tr_yoy"] = tr_yoy.reindex(df.index).round(1)
    df["peg"] = peg.reindex(df.index).round(3)
    df["score"] = df["dv_ttm"]
    codes = df["ts_code"].tolist()
    name_str = ', '.join([f"{c}({df.loc[df['ts_code']==c,'name'].iloc[0]})" for c in codes])
    print(f"  最终选出（高股息+成长）：{name_str}")
    return df.reset_index(drop=True)


# ============================================================
#  杠杆因子风控过滤器（可选预处理层）
#  对应视频《量化交易：杠杆因子拆解》四大因子的信号映射：
#    ① 资产负债率 → 风险惩罚（备选扣分项）
#    ② 产权比率 → 一票否决（剔除极端高杠杆）
#    ③ 长期负债/权益 → 宏观轮动（暂不作为过滤条件）
#    ④ 利息保障倍数 → 偿付能力过滤（可选启用）
# ============================================================

def apply_leverage_filters(df, trade_date, de_ratio_exclude_pct=5, icover_min=3):
    """
    对选股 DataFrame 执行杠杆因子过滤。

    参数
    ----
    df : DataFrame
        必须含 'ts_code' 列。
    trade_date : str
        格式 YYYYMMDD，选股日。
    de_ratio_exclude_pct : float
        产权比率最高的百分之多少被剔除（默认 5%）。<=0 时不启用。
    icover_min : float
        利息保障倍数最小值（默认 3 倍）。<=0 时不启用。

    返回
    ----
    DataFrame : 过滤后的 df（列不变，仅行减少），并打印过滤统计。
    """
    if de_ratio_exclude_pct <= 0 and icover_min <= 0:
        return df

    codes = df['ts_code'].tolist()
    if not codes:
        return df

    # 占位符
    ph = ','.join(['?'] * len(codes))

    # 各列在 SQL 中直接比对的基准日期
    ref_date = trade_date
    min_date = str(int(trade_date[:4]) - 2) + trade_date[4:]  # 2 年前

    conn = get_conn()

    # ── 因子② 产权比率 ──
    # fina_indicator 直接有 debt_to_eqt（负债/权益，%）
    df_de = pd.read_sql_query(f"""
        SELECT f.ts_code, f.debt_to_eqt AS de
        FROM fina_indicator f
        INNER JOIN (
            SELECT ts_code, MAX(end_date) AS md
            FROM fina_indicator
            WHERE ts_code IN ({ph})
              AND end_date >= ? AND end_date <= ?
              AND debt_to_eqt IS NOT NULL
            GROUP BY ts_code
        ) lt ON f.ts_code = lt.ts_code AND f.end_date = lt.md
    """, conn, params=list(codes) + [min_date, ref_date])

    # ── 因子④ 利息保障倍数 ──
    # EBIT / 利息费用（income.fin_exp_int_exp 为 TEXT，需 CAST）
    df_ic = pd.read_sql_query(f"""
        SELECT i.ts_code,
               i.ebit,
               CAST(i.fin_exp_int_exp AS REAL) AS fin_exp_int_exp
        FROM income i
        INNER JOIN (
            SELECT ts_code, MAX(end_date) AS md
            FROM income
            WHERE ts_code IN ({ph})
              AND end_date >= ? AND end_date <= ?
              AND ebit IS NOT NULL AND ebit > 0
              AND fin_exp_int_exp IS NOT NULL AND fin_exp_int_exp != ''
              AND CAST(fin_exp_int_exp AS REAL) > 0
            GROUP BY ts_code
        ) lt ON i.ts_code = lt.ts_code AND i.end_date = lt.md
    """, conn, params=list(codes) + [min_date, ref_date])

    conn.close()

    n0 = len(df)
    merged = df.merge(df_de, on='ts_code', how='left')

    # 利息保障倍数
    if len(df_ic) > 0:
        merged = merged.merge(df_ic, on='ts_code', how='left')
        merg_mask = merged['ebit'].notna() & merged['fin_exp_int_exp'].notna() & (merged['fin_exp_int_exp'] > 0)
        merged['interest_cover'] = None
        merged.loc[merg_mask, 'interest_cover'] = \
            merged.loc[merg_mask, 'ebit'] / merged.loc[merg_mask, 'fin_exp_int_exp']
    else:
        merged['interest_cover'] = None

    # ── 产权比率一票否决 ──
    if de_ratio_exclude_pct > 0:
        has_de = merged['de'].notna()
        n_has_de = has_de.sum()
        if n_has_de > 0:
            thr = merged.loc[has_de, 'de'].quantile(1 - de_ratio_exclude_pct / 100)
            drop_mask = has_de & (merged['de'] > thr)
            n_drop = drop_mask.sum()
            merged = merged[~drop_mask]
            if n_drop > 0:
                print(f"  ⛔ 杠杆过滤①：产权比率一票否决 — 剔除最高 {de_ratio_exclude_pct:.0f}%（阈值>{thr:.2f}），共 {n_drop} 只")

    # ── 利息保障倍数过滤 ──
    if icover_min > 0:
        has_ic = merged['interest_cover'].notna()
        n_has_ic = has_ic.sum()
        if n_has_ic > 0:
            drop_mask = has_ic & (merged['interest_cover'] < icover_min)
            n_drop = drop_mask.sum()
            merged = merged[~drop_mask]
            if n_drop > 0:
                print(f"  ⛔ 杠杆过滤②：利息保障倍数 — 剔除 <{icover_min} 倍共 {n_drop} 只（偿债能力不足）")

    # 清理辅助列
    drop_cols = [c for c in ['de', 'ebit', 'fin_exp_int_exp', 'interest_cover'] if c in merged.columns]
    merged = merged.drop(columns=drop_cols)

    n1 = len(merged)
    if n0 - n1 > 0:
        print(f"    = 杠杆过滤合计剔除 {n0 - n1} 只，保留 {n1}/{n0} 只")
    return merged


# ============================================================
#  红利质量三因子过滤（对标「高/稳/增」框架的 稳 + 增）
# ============================================================
def apply_dividend_quality_filters(df, trade_date,
                                   div_years_min=3, require_ocf_cover=True,
                                   div_growth_min=None):
    """
    红利质量三因子过滤（视频「高/稳/增」框架的 稳 + 增 两块）：
      稳① 连续分红年数：dividend_detail 中连续 cash_div>0 的年数 ≥ div_years_min
                         （对标中证红利「过去3年连续现金分红」硬门槛）
      稳② 分红现金覆盖：最新年 ocfps（每股经营现金流）≥ 最新年 cash_div
                         （经营现金流覆盖每股分红，防「借钱/透支分红」陷阱）
      增  分红增长：连续分红区间内 cash_div CAGR ≥ div_growth_min
                    （对标美股「分红贵族/连续增派」的质量维度；None=不要求）
    数据窗口：分红历史取 trade_date 当年及之前 6 年（div_proc='实施'）；
              ocfps 取 2 年内最新年报。
    缺失处理：无分红历史的票 consec_years=0 → 被 稳① 剔除（红利策略本就不要非分红票）；
              ocfps 缺失则不参与覆盖检查（不因此排除）。
    """
    if df.empty:
        return df

    codes = df['ts_code'].tolist()
    if not codes:
        return df
    ph = ','.join('?' * len(codes))

    ref_year = int(trade_date[:4])
    div_min_date = f"{ref_year - 6}1231"
    div_max_date = f"{ref_year}1231"
    ocf_min_date = f"{ref_year - 2}0101"
    ocf_max_date = f"{ref_year}1231"

    conn = get_conn()

    df_div = pd.read_sql_query(f"""
        SELECT ts_code, end_date, cash_div
        FROM dividend_detail
        WHERE ts_code IN ({ph})
          AND end_date >= ? AND end_date <= ?
          AND div_proc = '实施'
        ORDER BY ts_code, end_date
    """, conn, params=list(codes) + [div_min_date, div_max_date])

    df_ocf = pd.read_sql_query(f"""
        SELECT f.ts_code, f.ocfps
        FROM fina_indicator f
        INNER JOIN (
            SELECT ts_code, MAX(end_date) AS md
            FROM fina_indicator
            WHERE ts_code IN ({ph})
              AND end_date >= ? AND end_date <= ?
              AND ocfps IS NOT NULL
            GROUP BY ts_code
        ) lt ON f.ts_code = lt.ts_code AND f.end_date = lt.md
    """, conn, params=list(codes) + [ocf_min_date, ocf_max_date])

    conn.close()

    n0 = len(df)

    consec_years = {}
    div_growth = {}
    latest_cd = {}
    for ts_code, g in df_div.groupby('ts_code'):
        g = g.copy()
        g['yr'] = g['end_date'].str[:4].astype(int)
        g = g.sort_values('yr')
        cds = dict(zip(g['yr'], g['cash_div']))
        yrs = sorted(cds.keys(), reverse=True)
        consec = 0
        run_years = []
        for y in yrs:
            if cds[y] and cds[y] > 0:
                consec += 1
                run_years.append(y)
            else:
                break
        consec_years[ts_code] = consec
        latest_cd[ts_code] = g.iloc[-1]['cash_div']
        if len(run_years) >= 2:
            y0, y1 = min(run_years), max(run_years)
            d0, d1 = cds[y0], cds[y1]
            if d0 > 0 and d1 > 0 and (y1 - y0) > 0:
                div_growth[ts_code] = (d1 / d0) ** (1.0 / (y1 - y0)) - 1.0

    df['_consec_years'] = df['ts_code'].map(consec_years).fillna(0)
    df['_div_growth'] = df['ts_code'].map(div_growth)
    df['_latest_cd'] = df['ts_code'].map(latest_cd).fillna(0)
    df['_ocfps'] = df['ts_code'].map(
        dict(zip(df_ocf['ts_code'], df_ocf['ocfps'])) if len(df_ocf) else {})

    # ── 稳① 连续分红年数门槛 ──
    if div_years_min > 0:
        drop = df['_consec_years'] < div_years_min
        n_drop = int(drop.sum())
        if n_drop > 0:
            df = df[~drop]
            print(f"  ⛔ 红利质量①：连续分红年数<{div_years_min}年 — 剔除 {n_drop} 只")

    # ── 稳② 分红现金覆盖 ──
    if require_ocf_cover:
        has_ocf = df['_ocfps'].notna()
        mask = has_ocf & (df['_ocfps'] < df['_latest_cd'])
        n_drop = int(mask.sum())
        if n_drop > 0:
            df = df[~mask]
            print(f"  ⛔ 红利质量②：经营现金流未覆盖分红（ocfps<每股分红）— 剔除 {n_drop} 只（防借钱/透支分红）")

    # ── 增 分红增长 ──
    if div_growth_min is not None:
        has_g = df['_div_growth'].notna()
        mask = has_g & (df['_div_growth'] < div_growth_min)
        n_drop = int(mask.sum())
        if n_drop > 0:
            df = df[~mask]
            print(f"  ⛔ 红利质量③：分红增长CAGR<{div_growth_min*100:.1f}% — 剔除 {n_drop} 只")

    df = df.drop(columns=[c for c in ['_consec_years', '_div_growth', '_latest_cd', '_ocfps']
                          if c in df.columns])

    n1 = len(df)
    if n0 - n1 > 0:
        print(f"    = 红利质量过滤合计剔除 {n0 - n1} 只，保留 {n1}/{n0} 只")
    return df


# ============================================================
#  OBV 吸筹过滤（共享工具：run_ep_neutral.py 等外部策略导入使用）
#  注：EP 行业中性月度选股已迁往年度调仓（run_dogs_annual.py --strategy ep/ep_obv），
#      独立月度研究版见 run_ep_neutral.py。
# ============================================================

def obv_accumulation_filter(codes, end_date, lookback=20):
    """返回 codes 中 OBV 净流量 > 0（资金净流入 / 吸筹）的子集。

    OBV 净流量 = Σ_t sign(close_t − close_{t-1}) × vol_t，窗口为 end_date 前
    lookback 个交易日。净流量 > 0 表示近期内主动买盘占优（主力吸筹 / 趋势承接），
    净流量 < 0 表示抛压主导（派发）。用于 EP 选出的便宜股做"低相关过滤"——剔除
    便宜但无人接盘的票，只留"便宜且有人在买"的票。

    性能：批量取最后 lookback+1 个全局交易日，一次查询覆盖全部候选，避免逐票查库。
    数据不足（窗口<2日）时原样返回，不做过滤。
    """
    if not codes:
        return []
    conn = get_conn()
    dates = pd.read_sql_query(
        "SELECT DISTINCT trade_date FROM daily WHERE trade_date <= ? "
        "ORDER BY trade_date DESC LIMIT ?",
        conn, params=(end_date, lookback + 1))["trade_date"].tolist()
    if len(dates) < 2:
        conn.close()
        return list(codes)
    dates = sorted(dates)
    ph_codes = ",".join("?" * len(codes))
    ph_dates = ",".join("?" * len(dates))
    df = pd.read_sql_query(
        f"SELECT ts_code, trade_date, close, vol FROM daily "
        f"WHERE ts_code IN ({ph_codes}) AND trade_date IN ({ph_dates}) "
        f"ORDER BY ts_code, trade_date ASC",
        conn, params=list(codes) + dates)
    conn.close()
    keep = []
    for code, g in df.groupby("ts_code"):
        g = g.sort_values("trade_date")
        closes = g["close"].astype(float).to_numpy()
        vols = g["vol"].astype(float).to_numpy()
        if len(closes) < 2:
            keep.append(code)
            continue
        contrib = np.sign(np.diff(closes)) * vols[1:]
        if float(contrib.sum()) > 0:
            keep.append(code)
    return keep



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
        # 屏蔽科创板(688)与北交所(.BJ)：投资门槛对散户不友好，本平台统一剔除
        stock_set = {c for c in rows['ts_code'].tolist() if not c.startswith('688') and not c.endswith('.BJ')}

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
        ORDER BY ts_code, trade_date
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


def select_by_method(method, trade_date, top_n=None, lookback_months=None,
                     value_mode="pobreak", value_size_neutral=False, value_pct=None,
                     leverage_filter=False, de_ratio_exclude_pct=5, icover_min=3,
                     div_quality_filter=False, div_years_min=3,
                     require_ocf_cover=True, div_growth_min=None,
                     macd_filter_mode="golden"):
    """调度选股函数"""
    if top_n is None:
        top_n = get_top_n()
    
    if method == "momentum":
        lb = lookback_months if lookback_months is not None else MOMENTUM_LOOKBACK
        index_code = get_stock_pool_index()  # 从配置读取
        return select_momentum_stocks(trade_date, lookback_months=lb, top_n=top_n, index_code=index_code)
    elif method == "div_low_vol":
        return select_dividend_low_vol_stocks(trade_date, top_n,
                                              leverage_filter=leverage_filter,
                                              de_ratio_exclude_pct=de_ratio_exclude_pct,
                                              icover_min=icover_min,
                                              div_quality_filter=div_quality_filter,
                                              div_years_min=div_years_min,
                                              require_ocf_cover=require_ocf_cover,
                                              div_growth_min=div_growth_min,
                                              macd_filter_mode=macd_filter_mode)
    elif method == "div_growth":
        # 高股息+基本面成长（B站视频策略）：三筛 + 月调仓 + 涨停跑路日规则
        return select_dividend_growth_stocks(trade_date, top_n)
    else:  # value
        return select_stocks(trade_date, top_n, mode=value_mode,
                             size_neutral=value_size_neutral, value_pct=value_pct)


# ============================================================
#  主回测
# ============================================================

def run_backtest(start_date="20200102", end_date="20251231", top_n=None, selection_method="value",
                 select_only=False, value_mode="pobreak", value_size_neutral=False, value_pct=None,
                 stop_loss_pct=None, var_stop=False, atr_mult=2.0, atr_cooling=5,
                 leverage_filter=False, de_ratio_exclude_pct=5, icover_min=3,
                 div_quality_filter=False, div_years_min=3,
                 require_ocf_cover=True, div_growth_min=None,
                 macd_filter_mode=None,
                 interrupt_start=None, interrupt_months=0, interrupt_pct=0.0):
    # 止损：默认沿用模块常量 STOP_LOSS(15%) 以保持价值/红利低波既有行为；
    # 因子类策略月度调仓即退出，传 0 关闭个股止损。
    if stop_loss_pct is None:
        stop_loss_pct = STOP_LOSS
    # 高股息+基本面成长：视频策略无个股止损（月调仓即退出），强制关闭
    if selection_method == "div_growth":
        stop_loss_pct = 0.0
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
    _sm_name = {"div_low_vol": "红利低波", "momentum": "动量追涨",
                "div_growth": "高股息+基本面成长"}.get(selection_method, "价值选股")
    print(f"  选股策略：{_sm_name}")
    if selection_method == "value":
        print(f"  价值模式：{value_mode}"
              f"{' (放宽破净·BM分位门槛)' if value_mode=='pure_bm' else ' (破净+ROE质量)'}"
              f" | 市值中性化={'开' if value_size_neutral else '关'}"
              f" | BM分位={'前%.0f%%'%(value_pct*100) if value_pct else '关'}")
    print(f"  持仓数量：{top_n}只（等权重）")
    print(f"  个股止损：{'关闭' if stop_loss_pct == 0 else f'-{stop_loss_pct*100:.0f}%'}")
    if var_stop:
        print(f"  VAR动态止损（动量月度同款ATR追踪）：倍数{atr_mult} | 冷静期{atr_cooling}日 | 跌破[最高收-{atr_mult}×ATR]次日开盘卖")
    else:
        print(f"  VAR动态止损：关闭")
    if leverage_filter:
        print(f"  杠杆因子风控过滤（产权比率一票否决+利息保障倍数）：剔除产权比率最高{de_ratio_exclude_pct:.0f}% | 利息保障倍数>={icover_min}")
    if div_quality_filter:
        _growth_txt = f"≥{div_growth_min*100:.1f}%" if div_growth_min is not None else "不启用"
        print(f"  红利质量：连续分红≥{div_years_min if div_years_min > 0 else '不限制'}年 | 经营现金流覆盖分红={'是' if require_ocf_cover else '否'} | 分红增长CAGR{_growth_txt}")
    print(f"  MACD信号模式：{macd_filter_mode or 'golden'}（golden=旧金叉死叉 | regime=金叉须指数>MA200且非盘整；不指定则红利低波默认 golden）")
    print(f"  MACD死叉：减仓{BEAR_REDUCE*100:.0f}%（不清仓），金叉买回\n")

    trade_dates = get_trade_dates(start_date, end_date)
    rebalance_set = get_monthly_5th_trading_days(trade_dates)

    # 仅选股模式：执行第一次选股后退出，不回测
    if select_only:
        if len(rebalance_set) == 0:
            print(f"\n  [ERROR] 没有找到调仓日！")
            return
        first_rb = sorted(rebalance_set)[0]
        selected_codes = select_stocks(first_rb, top_n, mode=value_mode,
                                       size_neutral=value_size_neutral, value_pct=value_pct)
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
    _DAY_OHLC.clear()   # 清空 OHLC 日缓存（每次回测独立）
    _DAY_PCT.clear()    # 清空涨跌幅缓存（div_growth 日规则用）
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

        # ═══ 步骤3：检查止损（T日收盘价触发，创建T+1日挂单）
        #   含 VAR/ATR 动态止损（动量月度同款，仅 var_stop=True 时启用）═══
        for code in list(positions.keys()):
            ohlc = _get_ohlc(code, td) if var_stop else None
            price = ohlc[2] if (var_stop and ohlc is not None) else get_price(code, td)
            if price is None:
                continue
            pos = positions[code]
            # ── VAR/ATR 动态止损：更新最高收/滚动ATR/追踪止损线，过冷静期跌破即标记 ──
            if var_stop and ohlc is not None and pos.get("atr_stop_price") is not None:
                h, l, c = ohlc
                if h is not None and l is not None and c is not None:
                    if pos.get("highest_close") is None:
                        pos["highest_close"] = c
                    else:
                        pos["highest_close"] = max(pos["highest_close"], c)
                    prev_c = pos.get("last_close", c)
                    tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
                    tw = pos.setdefault("tr_window", [])
                    tw.append(tr)
                    if len(tw) > 14:
                        tw.pop(0)
                    if len(tw) >= 14:
                        atr = sum(tw) / len(tw)
                        new_stop = pos["highest_close"] - atr_mult * atr
                        if new_stop > pos["atr_stop_price"]:
                            pos["atr_stop_price"] = new_stop
                    pos["last_close"] = c
                    if (i - pos.get("entry_idx", -999)) > atr_cooling and c < pos["atr_stop_price"]:
                        pending_orders.append({
                            "type": "sell", "ts_code": code,
                            "shares": pos["shares"], "reason": "var_stop"
                        })
                        print(f"  🔴 VAR止损 {code}({get_stock_name(code)})：{td} 收盘{c:.2f} < 止损线{pos['atr_stop_price']:.2f}（最高收{pos['highest_close']:.2f}）")
                        stop_count += 1
            # ── 固定百分比止损（原有，价值/红利低波沿用）──
            if stop_loss_pct > 0 and price < pos["buy_price"] * (1 - stop_loss_pct):
                name = get_stock_name(code)
                pending_orders.append({
                    "type": "sell", "ts_code": code,
                    "shares": pos["shares"], "reason": "stop_loss"
                })
                print(f"  🔴 止损 {code}({name})：{td} 收盘{price:.2f} < 买入价{pos['buy_price']:.2f}×({1-stop_loss_pct:.0%})")
                stop_count += 1

        # ═══ 步骤3b：PB止盈（仅"破净价值"选股，T日收盘触发，T+1日开盘执行）═══
        #   pure_bm 模式放宽了破净约束，PB 普遍>=1，PB 修复止盈不再适用，故关闭。
        if selection_method == "value" and value_mode == "pobreak":
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

        # ═══ 步骤3c：高股息+基本面成长 · 涨停跑路日规则 ═══
        #   持仓昨涨停、今未封住涨停 → 当日收盘卖出（视频策略：短线资金抬轿后形态破坏就跑）
        #   与 run_dividend_growth_monthly.py 独立脚本口径一致（昨/今涨跌幅均由 close/pre_close 计算）
        if selection_method == "div_growth" and i > 0 and positions:
            for code in list(positions.keys()):
                if any(o["ts_code"] == code and o.get("reason") in ("stop_loss", "var_stop", "pb_take_profit")
                       for o in pending_orders):
                    continue
                p_prev = _get_day_pct(code, trade_dates[i - 1])
                p_cur = _get_day_pct(code, td)
                if p_prev is None or p_cur is None:
                    continue
                _lim = 19.9 if code.startswith("688") else (
                    19.9 if (code.startswith("300") and int(td) >= 20200824) else 9.9)
                if p_prev >= _lim - 0.1 and p_cur < _lim - 0.1:
                    px = get_price(code, td)
                    if px is None:
                        continue
                    pos = positions[code]
                    name = get_stock_name(code)
                    proceeds = pos["shares"] * px
                    fee = calc_fee("sell", px, pos["shares"])
                    cash += proceeds - fee
                    print(f"  🟡 涨停跑路 {code}({name})：{td} 昨涨停{p_prev:.1f}%今未封{p_cur:.1f}%，收盘{px:.2f}卖出")
                    trades.append({
                        "date": td, "action": "SELL", "code": code, "name": name,
                        "price": px, "shares": pos["shares"], "reason": "limit_up_break"
                    })
                    del positions[code]

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
            _mode = macd_filter_mode or "golden"
            macd_st = macd_state(benchmark_idx, prev_td, is_index_signal=True, mode=_mode)

            if macd_st == "golden":
                # ── MACD金叉(语境确认)：选股 + 调仓 + 买回减仓股票 ──
                print(f"\nMACD金叉：{benchmark_name} DIF {dif:.2f} > DEA {dea:.2f}"
                      + ("" if _mode == "golden" else " [regime确认·指数>MA200且非盘整]"))
                print(f"调仓日：{td}")

                stocks = select_by_method(selection_method, prev_td, top_n=top_n,
                                          value_mode=value_mode, value_size_neutral=value_size_neutral,
                                          value_pct=value_pct,
                                          leverage_filter=leverage_filter,
                                          de_ratio_exclude_pct=de_ratio_exclude_pct,
                                          icover_min=icover_min,
                                          div_quality_filter=div_quality_filter,
                                          div_years_min=div_years_min,
                                          require_ocf_cover=require_ocf_cover,
                                          div_growth_min=div_growth_min,
                                          macd_filter_mode=_mode)
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
                                if var_stop:
                                    _atr0 = get_atr(ts_code, prev_td, 14)
                                    if _atr0 and _atr0 > 0:
                                        positions[ts_code].update({
                                            "highest_close": open_price,
                                            "atr_stop_price": open_price - atr_mult * _atr0,
                                            "entry_idx": i,
                                            "last_close": open_price,
                                            "tr_window": [],
                                        })
                                    else:
                                        positions[ts_code]["atr_stop_price"] = None
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

            elif macd_st == "death":
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
                # neutral: 金叉但语境未确认(盘整/下跌) 或 数据不足
                _reason = "MACD数据不足" if (dif is None or dea is None) else "金叉但语境未确认(盘整/下跌)，保持现有仓位"
                print(f"\n调仓日 {td}：{_reason}")

        else:
            # ═══ 价值选股：无MACD过滤，直接调仓 ═══
            stocks = select_by_method(selection_method, prev_td, top_n=top_n,
                                      leverage_filter=leverage_filter,
                                      de_ratio_exclude_pct=de_ratio_exclude_pct,
                                      icover_min=icover_min,
                                      div_quality_filter=div_quality_filter,
                                      div_years_min=div_years_min,
                                      require_ocf_cover=require_ocf_cover,
                                      div_growth_min=div_growth_min)
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
                                if var_stop:
                                    _atr0 = get_atr(ts_code, prev_td, 14)
                                    if _atr0 and _atr0 > 0:
                                        positions[ts_code].update({
                                            "highest_close": open_price,
                                            "atr_stop_price": open_price - atr_mult * _atr0,
                                            "entry_idx": i,
                                            "last_close": open_price,
                                            "tr_window": [],
                                        })
                                    else:
                                        positions[ts_code]["atr_stop_price"] = None
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

    # ── 现实折扣三件套（扣通胀 / 定投拖累 / 中断模拟）──
    disc = compute_reality_discounts(
        daily_vals, INIT_CAPITAL,
        interrupt_start=interrupt_start,
        interrupt_months=interrupt_months,
        interrupt_pct=interrupt_pct,
    )
    if "real_total_return" in disc:
        print(f"  ── 现实折扣（预期管理，不改收益计算）──")
        print(f"  扣通胀真实总收益：{disc['real_total_return']:+.2f}% ｜ 真实年化：{disc['real_annual_return']:+.2f}%")
    if "dca_drag_pct" in disc:
        print(f"  定投对比(DCA)：一次性建仓较分12月定投 {disc['dca_drag_pct']:+.2f}%"
              f"（正=一次性占优·负=定投占优）｜ 终值 一次性 {disc['dca_lump_final']:,.0f} / 定投 {disc['dca_dca_final']:,.0f}")
    if "interrupt_loss_pct" in disc:
        print(f"  中断模拟：{interrupt_start}起撤{interrupt_pct*100:.0f}%持有{interrupt_months}月，"
              f"终值损失 {disc['interrupt_loss_pct']:+.2f}%（终值 {disc['interrupt_final']:,.0f}）")

    print(f"  交易次数：{len(trades)}")
    win_rate, win_cnt, tot_cnt = calc_win_rate(trades)
    if tot_cnt > 0:
        print(f"  胜率：{win_rate:.1f}%（{win_cnt}/{tot_cnt}）")
    print(f"  止损触发：{stop_count} 次")
    if var_stop:
        print(f"  VAR动态止损触发：{stop_count} 次（ATR追踪·动量月度同款）")
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
        "reality_discounts": disc,
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


def estimate_basket_var(codes, trade_date, conf=0.95, lookback=120, method="hist"):
    """
    估计等权篮子在 trade_date 时刻的「单期（日）VaR 损失比例」（正数，小数）。

    取各成分股截至 trade_date 的回看窗口日收益，等权合成篮子日收益序列，
    再取经验/参数分位得到损失比例。用于 VaR 仓位缩放（设计即锁回撤）。

    返回 损失比例(小数)；数据不足返回 None。
    注意：成交日期格式需与库中 trade_date 列一致（沿用 get_price 的入参类型）。
    """
    if not codes:
        return None
    try:
        conn = get_conn()
        series = []
        for code in codes:
            rows = pd.read_sql_query(
                "SELECT close FROM daily WHERE ts_code=? AND trade_date<=? "
                "ORDER BY trade_date DESC LIMIT ?",
                conn, params=(code, trade_date, int(lookback) + 1))
            closes = rows["close"].values
            if len(closes) >= 2:
                closes = closes[::-1]  # 升序
                r = np.diff(closes) / closes[:-1]
                series.append(r)
        conn.close()
    except Exception:
        return None
    if not series:
        return None
    n = min(len(s) for s in series)
    if n < 10:
        return None
    basket = np.mean(np.array([s[-n:] for s in series]), axis=0)
    if method == "param":
        mu = basket.mean()
        sigma = basket.std(ddof=1)
        z = {0.90: 1.282, 0.95: 1.645, 0.99: 2.326}.get(conf, 1.645)
        var_ret = mu - z * sigma
    else:
        q = max(0.0, min(1.0, 1.0 - conf))
        var_ret = float(np.quantile(basket, q))
    loss = -var_ret
    return float(loss) if loss > 0 else 0.0


def _var_invest_ratio(codes, trade_date, var_control, var_maxdd, var_n, var_lookback, var_method, freq_months=1):
    """
    反解 VaR 投入比例（0~1）。var_control<=0 时返回 1.0（不缩放，满仓）。
    逻辑：持有期VaR = 日VaR × √(持有交易日) ；风险预算 = 目标回撤/N ；
          投入比例 = min(1, 预算/持有期VaR)（封顶100%不杠杆，余下转现金）。
    """
    if not (var_control and var_control > 0):
        return 1.0
    if not codes:
        return 1.0
    bvar = estimate_basket_var(codes, trade_date, conf=var_control / 100.0,
                               lookback=var_lookback, method=var_method)
    if not bvar or bvar <= 0:
        return 1.0
    holding_days = max(1, int(round(freq_months * 21)))
    hold_var = bvar * (holding_days ** 0.5)
    risk_budget = (var_maxdd / 100.0) / max(1, var_n)
    return min(1.0, risk_budget / hold_var)


def equity_curve_var(values, capital=None, conf_levels=(0.95, 0.99), method="hist"):
    """
    由权益曲线（净值序列）计算 VaR 报告（参数法+历史法），供策略结果输出。
    返回 dict：{0.95: {'param_loss','hist_loss','param_amt','hist_amt'}, ...}
    """
    vals = np.asarray(values, dtype=float)
    rs = np.diff(vals) / np.where(vals[:-1] == 0, 1, vals[:-1])
    rs = rs[np.isfinite(rs)]
    Z = {0.90: 1.282, 0.95: 1.645, 0.99: 2.326}
    if len(rs) < 5:
        return {c: {"param_loss": 0.0, "hist_loss": 0.0, "param_amt": 0.0, "hist_amt": 0.0}
                for c in conf_levels}
    mu = float(np.mean(rs))
    sigma = float(np.std(rs, ddof=1))
    out = {}
    for c in conf_levels:
        z = Z.get(c, 1.645)
        param_loss = max(0.0, -(mu - z * sigma))
        if method in ("hist", "both"):
            q = max(0.0, min(1.0, 1.0 - c))
            hist_loss = max(0.0, -float(np.quantile(rs, q)))
        else:
            hist_loss = param_loss
        out[c] = {
            "param_loss": param_loss, "hist_loss": hist_loss,
            "param_amt": (param_loss * capital) if capital else 0.0,
            "hist_amt": (hist_loss * capital) if capital else 0.0,
        }
    return out


def run_momentum_backtest(start_date="20200101", end_date="20251231",
                          top_n=5, lookback_months=6, stock_pool=None,
                          rebalance_freq_months=1, atr_stop_multiple=0,
                          atr_cooling_days=0, trailing_stop_pct=0,
                          skip_recent_months=1, trend_filter_ma=0,
                          var_control=0, var_maxdd=15.0, var_n=5,
                          var_lookback=120, var_method="hist",
                          value_area=0, va_pct=70.0):
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

    # VaR 仓位缩放标签
    if var_control and var_control > 0:
        var_label = (f" | VaR仓位缩放({var_control}%·目标回撤{var_maxdd:.0f}%·"
                     f"N={var_n}·回看{var_lookback}d·{var_method})")
        var_detail = (f"  VaR仓位缩放：把目标最大回撤 {var_maxdd:.0f}% 分摊到 {var_n} 个连续"
                      f"下跌周期→每期风险预算；调仓时按篮子持有期VaR反解投入比例，"
                      f"余下转现金（不杠杆）。conf={var_control}%，方法={var_method}")
    else:
        var_label = ""
        var_detail = ""

    # 价值区过滤标签
    if value_area and value_area > 0:
        va_label = f" | 价值区过滤({value_area}d·{va_pct:.0f}%)"
        va_detail = (f"  价值区过滤：动量只接处于价值区内/上方的票（市场接受当前价），"
                     f"剔除价值区下方被拒绝的弱势票。回看{value_area}日。")
    else:
        va_label = ""
        va_detail = ""

    print("=" * 70)
    print(f"动量效应回测（动量{lookback_months}个月 × {freq_label}调仓{stop_label}{cooling_label}{trend_filter_label}{var_label}）")
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
    if var_detail:
        print(var_detail)
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

            # ===== 可选：价值区过滤（动量只接价值区内/上方）=====
            if value_area and value_area > 0 and new_codes:
                _passed = []
                _rejected = []
                for _c in new_codes:
                    _ok, _why = value_area_pass(_c, prev_td, lookback=value_area,
                                                va_pct=va_pct / 100.0, mode="momentum")
                    if _ok:
                        _passed.append(_c)
                    else:
                        _rejected.append(_c)
                if _passed:
                    print(f"  🏷️ 价值区过滤：{len(_passed)}/{len(new_codes)} 通过"
                          f"（剔除{len(_rejected)}只在价值区下方）")
                    new_codes = _passed
                else:
                    print(f"  🏷️ 价值区过滤：全被剔除，回退到原始动量组合")
                del _passed, _rejected

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
                        # === VaR 仓位缩放：反解投入比例 ===
                        invest_ratio = 1.0
                        bvar = None
                        hold_var = None
                        if var_control and var_control > 0:
                            bvar = estimate_basket_var(
                                new_to_buy, td, conf=var_control / 100.0,
                                lookback=var_lookback, method=var_method)
                            if bvar and bvar > 0:
                                holding_days = max(1, rebalance_freq_months * 21)
                                hold_var = bvar * (holding_days ** 0.5)  # 持有期VaR(sqrt-time)
                                risk_budget = (var_maxdd / 100.0) / max(1, var_n)
                                invest_ratio = min(1.0, risk_budget / hold_var)
                        cash_per_stock = (cash * invest_ratio) / len(new_to_buy)
                        if var_control and var_control > 0:
                            bv_pct = f"{bvar * 100:.2f}%" if bvar is not None else "n/a"
                            hv_pct = f"{hold_var * 100:.2f}%" if hold_var is not None else "n/a"
                            print(f"  🛡️ VaR缩放(var={var_control}%): 篮子日VaR损={bv_pct} → 持有期VaR={hv_pct} "
                                  f"→ 投入比例={invest_ratio * 100:.0f}%（预留现金{(1 - invest_ratio) * 100:.0f}%）")
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
    # === VaR 前瞻风险（基于权益曲线日收益）===
    eq_var = equity_curve_var(vals, capital=final_value, conf_levels=(0.95, 0.99), method="hist")
    print(f"  风险价值 VaR(95%)：单日最多亏 {eq_var[0.95]['hist_loss'] * 100:.2f}% "
          f"（≈{eq_var[0.95]['hist_amt']:,.0f}元，历史法）")
    print(f"  风险价值 VaR(99%)：单日最多亏 {eq_var[0.99]['hist_loss'] * 100:.2f}% "
          f"（≈{eq_var[0.99]['hist_amt']:,.0f}元，历史法）")
    print(f"  交易次数：{len(trades)}")
    win_rate, win_cnt, tot_cnt = calc_win_rate(trades)
    if tot_cnt > 0:
        print(f"  胜率：{win_rate:.1f}%（{win_cnt}/{tot_cnt}）")
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
        "var95_hist_loss": eq_var[0.95]["hist_loss"],
        "var99_hist_loss": eq_var[0.99]["hist_loss"],
        "var95_amt": eq_var[0.95]["hist_amt"],
        "var99_amt": eq_var[0.99]["hist_amt"],
        "var_control": var_control,
        "var_maxdd": var_maxdd,
        "trades": len(trades),
        "win_rate": calc_win_rate(trades)[0],
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
        ("胜率(%)",        [results[lb].get("win_rate", 0) for lb in lookbacks] + ["-"]),
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

def prefilter_by_liquidity(conn, codes, trade_date,
                           min_avg_amount=LIQUIDITY_MIN_AVG_AMOUNT,
                           lookback=LIQUIDITY_LOOKBACK):
    """
    流动性预过滤：剔除 trade_date 往前 lookback 个交易日日均成交额低于阈值的股票。
    daily.amount 单位为千元 → avg_amt * 1000 = 元。
    Args: codes: set/list of ts_code（如 000001.SZ）
    Returns: 通过过滤的 ts_code 集合
    """
    if not codes:
        return set()
    try:
        win = pd.read_sql_query(
            "SELECT trade_date FROM daily WHERE trade_date <= ? "
            "ORDER BY trade_date DESC LIMIT ?",
            conn, params=(trade_date, lookback))
        if len(win) == 0:
            return set(codes)
        win_start = win['trade_date'].min()
        ph = ",".join("?" * len(codes))
        amt = pd.read_sql_query(
            f"SELECT ts_code, AVG(amount) AS avg_amt FROM daily "
            f"WHERE ts_code IN ({ph}) AND trade_date >= ? AND trade_date <= ? "
            f"GROUP BY ts_code",
            conn, params=list(codes) + [win_start, trade_date])
        avg_map = dict(zip(amt['ts_code'], amt['avg_amt']))
    except Exception as e:
        print(f"  [WARN] 流动性查询失败，跳过过滤: {e}")
        return set(codes)
    kept = {c for c in codes if (avg_map.get(c) is not None
                                 and avg_map[c] * 1000 >= min_avg_amount)}
    dropped = len(codes) - len(kept)
    print(f"  [流动性] 日均成交额≥{min_avg_amount/1e4:.0f}万({lookback}日) "
          f"过滤：保留 {len(kept)} 只 / 剔除 {dropped} 只")
    return kept


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
        # 屏蔽科创板(688)与北交所(.BJ)：投资门槛对散户不友好，本平台统一剔除
        stock_set = {c for c in rows['ts_code'].tolist() if not c.startswith('688') and not c.endswith('.BJ')}

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

    # 流动性过滤（保守阈值，跑大盘股时几乎不剔除成分股）
    candidates = prefilter_by_liquidity(conn, candidates, trade_date)

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
                          macd_filter_mode=None,
                          stop_loss_pct=0,
                          var_control=0, var_maxdd=15.0, var_n=3, var_lookback=120, var_method="hist",
                          value_area=0, va_pct=70.0, fakeout_reclaim=False):
    """短期逆转效应——轮动回测
    market_filter: "none" | "ma20" | "macd" 市场趋势过滤
    stop_loss_pct: 个股止损比例（0=不启用，如0.08=跌破买价8%止损）
    value_area: 价值区过滤回看天数(0=关)；fakeout_reclaim: 是否优先"扫止损→收回"标的
    """
    pool_display = INDEX_DISPLAY_NAME.get(stock_pool, "全A股") if stock_pool else "全A股"
    freq_label = f"每{holding_days}日" if holding_days > 1 else "每日"
    filter_label = {"none":"无过滤", "ma20":"价格>MA20", "macd":"MACD金叉(regime)"}.get(market_filter, "无过滤")
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
    if var_control and var_control > 0:
        print(f"  VaR仓位缩放：目标回撤{var_maxdd:.0f}%·N={var_n}·回看{var_lookback}d·conf={var_control}%（设计即锁回撤，余下转现金不杠杆）")

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
            macd_st = macd_state(benchmark_idx, prev_td, is_index_signal=True, mode=macd_filter_mode or "regime")  # ← prev_td
            allow_buy = (macd_st == "golden")
            if not allow_buy:
                _r = "MACD死叉" if macd_st == "death" else "金叉但语境未确认(盘整/下跌)"
                print(f"  ⏸️ {td} {_r}，空仓等待（基于{prev_td}数据）")

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

        # ===== 可选：价值区过滤 + fakeout-reclaim 优先 =====
        if codes and (value_area and value_area > 0) or (codes and fakeout_reclaim):
            _va_pass, _fo_hit = [], []
            for _c in codes:
                if value_area and value_area > 0:
                    _ok, _ = value_area_pass(_c, prev_td, lookback=value_area,
                                             va_pct=va_pct / 100.0, mode="reversal")
                else:
                    _ok = True
                _hit, _ = vp_fakeout_reclaim(_c, prev_td, lookback=lookback_days) if fakeout_reclaim else (False, "")
                if _ok:
                    _va_pass.append(_c)
                if _hit:
                    _fo_hit.append(_c)
            # 价值区过滤：优先价值区下方(超跌)标的，但若全被剔除则回退
            if value_area and value_area > 0:
                if _va_pass:
                    codes = _va_pass
                    print(f"  🏷️ 价值区过滤：{len(_va_pass)}/{len(stocks) if not stocks.empty else 0} 在价值区下沿(超跌区)")
                else:
                    print(f"  🏷️ 价值区过滤：全被剔除，回退原始组合")
            # fakeout-reclaim 优先：把命中的排到前面
            if fakeout_reclaim and _fo_hit:
                _ordered = [c for c in _fo_hit if c in codes] + [c for c in codes if c not in _fo_hit]
                codes = _ordered
                print(f"  🎯 fakeout-reclaim 优先：{len(_fo_hit)}/{len(codes)} 命中(扫止损后收回)")
            del _va_pass, _fo_hit

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
        _ir = _var_invest_ratio(codes, prev_td, var_control, var_maxdd, var_n, var_lookback, var_method, holding_days)
        _saved_cash = cash
        cash = cash * _ir
        if var_control and var_control > 0 and _ir < 1.0:
            print(f"  🛡️ VaR缩放(var={var_control}%): 投入比例={_ir * 100:.0f}%（预留现金{(1 - _ir) * 100:.0f}%）")
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
        # 恢复现金（VaR 预留部分不被动用）
        cash = _saved_cash - (_saved_cash * _ir - cash)

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
    win_rate, win_cnt, tot_cnt = calc_win_rate(trades)
    if tot_cnt > 0:
        stxt += f" | 胜率 {win_rate:.1f}%"
    print(f"  最大回撤：{dd:.2f}% | 夏普比率：{sp:.2f} | 交易：{len(trades)}{stxt}")
    print(f"\n  逐日净值：")
    for dv in daily_vals:
        chg = (dv["value"] / INIT_CAPITAL - 1) * 100
        bar = "█" * max(0, int(chg)) if chg > 0 else "░" * min(0, int(-chg))
        print(f"    {dv['date']}  {dv['value']:>10,.0f} ({chg:+.2f}%) {bar}")

    return {"total_return": tr, "annual_return": ar, "max_drawdown": dd, "var_control": var_control,
            "sharpe": sp, "trades": len(trades), "daily_values": daily_vals}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="月度调仓回测")
    parser.add_argument("start_date", nargs="?", default="20200102", help="开始日期 YYYYMMDD")
    parser.add_argument("end_date", nargs="?", default="20251231", help="结束日期 YYYYMMDD")
    parser.add_argument("--top-n", type=int, default=None, help="选股数量")
    parser.add_argument("--selection-method", type=str, default="value",
                        choices=["value", "div_low_vol", "momentum", "reversal", "div_low_vol_quality"],
                        help="选股策略(div_low_vol_quality=红利低波质量复合·季度调仓)")
    parser.add_argument("--dlvq-mode", type=str, default="official_compact",
                        choices=["official", "official_improved", "official_compact"],
                        help="div_low_vol_quality 模式: official(月频/等权/TOP5) / official_improved(季频/股息率加权/TOP25) / official_compact(季频/股息率加权/TOP12/行业≤2·落地版)")
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
                        help="固定比例trailing stop（0=不启用，如0.15=15%%，与--atr-stop互斥，仅 momentum 模式）")
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
                        help="逆转策略个股止损比例（0=不启用，0.08=8%%，仅 reversal 模式）")
    parser.add_argument("--market-filter", type=str, default="none",
                        choices=["none", "ma20", "macd"], help="市场趋势过滤（仅 reversal 模式）")
    parser.add_argument("--macd-filter", type=str, default=None,
                        choices=["golden", "regime"],
                        help="MACD信号模式：golden=旧金叉死叉当按钮 | regime=金叉须叠加指数>MA200且非盘整(语境感知)。不指定则按策略默认：逆转=regime，红利低波=golden")
    parser.add_argument("--var-control", type=int, default=0, choices=[0, 90, 95, 99],
                        help="VaR仓位缩放置信水平: 0=关闭 | 90/95/99=启用（仅 reversal 模式）")
    parser.add_argument("--var-maxdd", type=float, default=15.0,
                        help="目标最大回撤上限(%%) ，反解每期风险预算（仅 reversal 模式，默认15）")
    parser.add_argument("--var-n", type=int, default=3,
                        help="连续下跌周期数 N（反转类=3），分摊回撤预算（仅 reversal 模式，默认3）")
    parser.add_argument("--value-area", type=int, default=0,
                        help="价值区过滤回看天数: 0=关闭 | >0=启用(对动量/反转生效，默认0)")
    parser.add_argument("--va-pct", type=float, default=70.0,
                        help="价值区覆盖成交量比例(%%)，默认70")
    parser.add_argument("--fakeout-reclaim", action="store_true", default=False,
                        help="反转优先'扫止损→快速收回'标的（仅 reversal 模式）")
    parser.add_argument("--interrupt-start", type=str, default=None,
                        help="中断模拟起点(YYYYMM)，配合--interrupt-pct使用")
    parser.add_argument("--interrupt-months", type=int, default=0,
                        help="中断模拟：撤出资金空仓月数（默认0=不模拟）")
    parser.add_argument("--interrupt-pct", type=float, default=0.0,
                        help="中断模拟：撤出资金比例(0~1，如0.5=撤一半)，默认0=不模拟")
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
                value_area=args.value_area,
                va_pct=args.va_pct,
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
            macd_filter_mode=args.macd_filter,
            stop_loss_pct=args.reversal_stop,
            var_control=args.var_control,
            var_maxdd=args.var_maxdd,
            var_n=args.var_n,
            value_area=args.value_area,
            va_pct=args.va_pct,
            fakeout_reclaim=args.fakeout_reclaim,
        )
    elif args.selection_method == "div_low_vol_quality":
        # 红利低波「质量复合」策略：官方编制法实战三档（季度调仓）
        import run_dividend_low_vol_quality_bt as dlq
        dlq.START = args.start_date
        dlq.END = args.end_date
        _dlq_mode = args.dlvq_mode
        print(f"红利低波质量复合【季度调仓】回测：{args.start_date} ~ {args.end_date}")
        print(f"  模式: {_dlq_mode}（官方编制法930955口径·季频/股息率加权）")
        dlq.run_official_backtest(_dlq_mode)
    else:
        print(f"回测周期：{args.start_date} ~ {args.end_date}")
        print(f"选股策略：{args.selection_method}")
        run_backtest(args.start_date, args.end_date, top_n=args.top_n, selection_method=args.selection_method, select_only=args.select_only,
                     interrupt_start=args.interrupt_start, interrupt_months=args.interrupt_months, interrupt_pct=args.interrupt_pct,
                     macd_filter_mode=args.macd_filter)


# ════════════════════════════════════════════════════════════════════
#  现实折扣三件套：扣通胀 / 定投拖累 / 中断模拟
#  灵感：B站《复利对普通人来说，远比想象中的更加困难》(BV1iyN16BE18)
#  说明：这三个指标不改引擎的收益数学，只在"输出层"给零售投资者做预期管理。
# ════════════════════════════════════════════════════════════════════

# 中国 CPI 同比（NBS，年度）。2025 为估计值，可自行替换。
# 用于把名义收益折算成"扣通胀真实收益"。
ANNUAL_CPI_YOY = {
    2010: 0.033, 2011: 0.054, 2012: 0.026, 2013: 0.026, 2014: 0.020,
    2015: 0.014, 2016: 0.020, 2017: 0.016, 2018: 0.021, 2019: 0.029,
    2020: 0.025, 2021: 0.009, 2022: 0.020, 2023: 0.002, 2024: 0.003,
    2025: 0.005,
}
DEFAULT_INFLATION_FALLBACK = 0.020  # 缺年份时回退假设


def _month_end_values(daily_vals):
    """从每日净值取每月最后一个值 → [(yyyymm, value), ...]"""
    if not daily_vals:
        return []
    buckets = {}
    for d in daily_vals:
        ym = str(d["date"])[:6]
        buckets[ym] = float(d["value"])
    return [(ym, buckets[ym]) for ym in sorted(buckets.keys())]


def compute_reality_discounts(daily_vals, init_capital,
                              inflation=None,
                              interrupt_start=None,
                              interrupt_months=None,
                              interrupt_pct=0.0):
    """计算'现实折扣三件套'，返回 dict（缺数据则跳过对应键）。

    1) 扣通胀真实收益 real_total_return / real_annual_return
    2) 定投拖累 dca_drag_pct / dca_lump_final / dca_dca_final
       —— 一次性建仓 vs 把同等本金分 12 个月定投，终值差异
    3) 中断模拟 interrupt_loss_pct / interrupt_final
       —— 某时点撤出 p% 持有 m 个月(0收益)后重新投入，错过的市场收益
    """
    out = {}
    if not daily_vals or len(daily_vals) < 2:
        return out

    final_value = float(daily_vals[-1]["value"])

    # ── 1) 扣通胀 ────────────────────────────────────────
    cpi = inflation if isinstance(inflation, dict) else ANNUAL_CPI_YOY
    sy = int(str(daily_vals[0]["date"])[:4])
    ey = int(str(daily_vals[-1]["date"])[:4])
    cum_cpi = 1.0
    for y in range(sy, ey + 1):
        cum_cpi *= (1.0 + cpi.get(y, DEFAULT_INFLATION_FALLBACK))
    real_final = final_value / cum_cpi
    out["real_total_return"] = (real_final / init_capital - 1) * 100
    # 用日期跨度估算年数（与采样频率无关），更稳健
    s0, s1 = str(daily_vals[0]["date"]), str(daily_vals[-1]["date"])
    months = (int(s1[:4]) * 12 + int(s1[4:6])) - (int(s0[:4]) * 12 + int(s0[4:6]))
    years = max(months, 1) / 12.0
    out["real_annual_return"] = (
        (real_final / init_capital) ** (1 / years) - 1) * 100 if years > 0 else 0.0

    # ── 2) 定投拖累 ─────────────────────────────────────
    mv = _month_end_values(daily_vals)
    if len(mv) >= 2:
        mret = []
        for i in range(1, len(mv)):
            prev, cur = mv[i - 1][1], mv[i][1]
            mret.append((cur / prev - 1) if prev > 0 else 0.0)
        n = min(12, len(mret))
        lump_final = init_capital * float(np.prod([1 + r for r in mret]))
        dca_final = 0.0
        for i in range(n):
            slice_ret = mret[i:]
            dca_final += (init_capital / n) * float(np.prod([1 + r for r in slice_ret]))
        out["dca_lump_final"] = lump_final
        out["dca_dca_final"] = dca_final
        out["dca_drag_pct"] = (
            (lump_final - dca_final) / lump_final * 100) if lump_final > 0 else 0.0

    # ── 3) 中断模拟（序列风险）─────────────────────────
    if (interrupt_pct and interrupt_pct > 0 and interrupt_start is not None
            and interrupt_months and interrupt_months > 0 and len(mv) >= 2):
        target = str(interrupt_start)[:6]
        mv_dates = [m[0] for m in mv]
        idx = next((k for k, ym in enumerate(mv_dates) if ym >= target), None)
        if idx is not None and idx + interrupt_months < len(mv):
            v_k = mv[idx][1]
            gap_growth = float(np.prod(
                [1 + mret[idx + j] for j in range(interrupt_months)])) - 1
            lost = interrupt_pct * v_k * gap_growth
            out["interrupt_final"] = final_value - lost
            out["interrupt_loss_pct"] = (
                lost / final_value * 100) if final_value > 0 else 0.0

    return out
