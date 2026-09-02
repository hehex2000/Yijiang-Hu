# -*- coding: utf-8 -*-
"""
全收益基准加载器（统一真相源）
================================

修复历史口径缺陷：策略净值用 hfq(含分红再投)，但平台基准长期用
index_daily 价格指数(不含分红) → 所有"超额收益"被系统性高估
（沪深300 约 3.02%/年、中证800 约 2.19%/年，见 plan_totalreturn_audit.md）。

查找链（默认 mode='tr_official'）：
  1. index_tr_official  (Tushare 官方全收益指数，最权威)
  2. index_total_return (build_tr_index.py 自建 bottom-up 全收益)
  3. index_daily        (价格指数，原口径，用于复现旧结论)

开关（优先级：函数参数 > 环境变量 MFS_BENCHMARK > config.BENCHMARK_MODE > 默认）：
  tr_official  全收益优先（官方→自建→价格）        ← 默认
  total_return 自建全收益优先（自建→价格）
  net          官方净收益（已扣红利税，仅 000922 有）
  price        价格指数（原口径，复现旧结论用）

所有加载函数都返回 meta 字典，记录"实际使用的口径"，便于报告透明标注。
"""

import os
import sqlite3
import pandas as pd

try:
    from config import DATA
except Exception:
    DATA = {"local_db_path": r"D:\tu-shareData\astock_daily.db"}

# 价格指数代码 → 官方全收益指数代码（映射依据 download_tr_index.py 探测结果）
BENCHMARK_TR_MAP = {
    "000300.SH": "H00300.CSI",   # 沪深300全收益
    "000906.SH": "H00906.CSI",   # 中证800全收益
    "000922.SH": "H00922.CSI",   # 中证红利全收益
    "930955.SH": "H20955.CSI",   # 红利低波100全收益
}
# 净收益(已扣红利税) 单独提供，按需显式使用
BENCHMARK_NET_MAP = {
    "000922.SH": "000922CNY020.CSI",   # 中证红利净收益
}
VALID_MODES = ("tr_official", "total_return", "price", "net")


def resolve_mode(mode=None, nav_price_mode=None):
    """解析实际使用的基准模式（开关解析）。

    nav_price_mode: 策略 NAV 的计价口径（"raw" 未复权 / "hfq" 后复权含分红）。
    当调用方**未显式指定** mode（无函数参数、无 MFS_BENCHMARK 环境变量、无 config 覆盖）时，
    自动把基准对齐到 NAV 口径，避免「raw 净值 vs 全收益基准」这类错配系统性低估超额：
        - nav=raw  → 基准用 price（价格指数，同样不含分红）
        - nav=hfq  → 基准用 tr_official（全收益，同样含分红再投）
    显式指定 mode 时以显式为准（保留用户覆盖能力），但可用 check_consistency() 取告警。
    """
    m = mode or os.environ.get("MFS_BENCHMARK")
    if not m:
        try:
            m = __import__("config").BENCHMARK_MODE
        except Exception:
            m = None
    # "auto"（推荐默认）/ 未配置 → 按 NAV 口径自动对齐，保证两端同含或同不含分红
    if m is None or str(m).lower() == "auto":
        return "price" if nav_price_mode == "raw" else "tr_official"
    if m in VALID_MODES:
        return m
    return "tr_official"


def check_consistency(nav_price_mode, meta):
    """校验 NAV 口径与基准口径是否匹配，返回告警串（无问题返回空串）。

    错配方向：
      - raw NAV + 全收益基准 → 超额被**系统性低估**（分红被当成策略亏损）
      - hfq NAV + 价格基准   → 超额被**系统性高估**
    """
    if not nav_price_mode or not meta:
        return ""
    # ⚠️ 必须以 source_table（**实际**来源）为准，不能用 mode（**请求**口径）。
    #    load_benchmark 回退到价格指数时 mode 仍是 tr_official → 只看 mode 会漏报。
    #    （2026-09-01 实测：中证全指 000985.SH 在 hfq 下回退价格指数却不告警，
    #     旧代码因读键名写错(source vs source_table)导致 src 恒为空，退化为只看 mode。）
    src = str(meta.get("source_table") or meta.get("source") or "").lower()
    mode = str(meta.get("mode") or "").lower()
    if not src:
        return ""
    if "index_daily" in src or src == "price":
        bench_div = False           # 价格指数：不含分红
    elif ("tr" in src or "total_return" in src or "net" in src):
        bench_div = True            # 全收益/净收益：含分红
    else:
        return ""
    nav_div = (nav_price_mode == "hfq")
    if nav_div == bench_div:
        return ""
    if nav_div and not bench_div:
        return "⚠️口径错配：NAV含分红(hfq) vs 基准价格指数(不含分红) → 超额被【高估】"
    return "⚠️口径错配：NAV不含分红(raw) vs 基准全收益(含分红) → 超额被【低估】"


def _conn(conn=None):
    if conn is not None:
        return conn, False
    c = sqlite3.connect(DATA.get("local_db_path", ""))
    return c, True


def _read_official(tr_code, start, end, conn):
    # 🔴 必须按 tr_code 精确查，禁止按 local_code 裸查：
    #    同一个 local_code 可能挂多条序列（口径不同、数值不等）。实测 000922.SH 挂了两条：
    #      H00922.CSI       全收益·不扣税
    #      000922CNY020.CSI 净收益·扣税
    #    4045 个交易日**全部不一致**，差 62~63 点（约 1.6%）。按 local_code 裸查会随机取到一条，
    #    使基准悄悄变成另一个口径（表主键是 (tr_code, trade_date)，故 tr_code 查询安全无重复）。
    df = pd.read_sql_query(
        "SELECT trade_date, close FROM index_tr_official WHERE tr_code=? AND trade_date BETWEEN ? AND ? ORDER BY trade_date",
        conn, params=(tr_code, start, end))
    if not df.empty:
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df


def _read_net(net_code, start, end, conn):
    df = pd.read_sql_query(
        "SELECT trade_date, close FROM index_tr_official WHERE tr_code=? AND trade_date BETWEEN ? AND ? ORDER BY trade_date",
        conn, params=(net_code, start, end))
    if not df.empty:
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df


def _read_total_return(code, start, end, conn):
    df = pd.read_sql_query(
        "SELECT trade_date, idx_tr AS close FROM index_total_return WHERE index_code=? AND trade_date BETWEEN ? AND ? ORDER BY trade_date",
        conn, params=(code, start, end))
    if not df.empty:
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df


def _read_price(code, start, end, conn):
    df = pd.read_sql_query(
        "SELECT trade_date, close FROM index_daily WHERE ts_code=? AND trade_date BETWEEN ? AND ? ORDER BY trade_date",
        conn, params=(code, start, end))
    if not df.empty:
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df


def load_benchmark(code, start, end, conn=None, mode=None, nav_price_mode=None):
    """加载基准指数序列（优先全收益口径）。

    Returns:
        (df, meta)
          df  : DataFrame[trade_date, close]；无数据时为 None
          meta: dict(mode, source_table, resolved_code, note)
    """
    mode = resolve_mode(mode, nav_price_mode=nav_price_mode)
    quoted_mode = mode          # 请求口径（回退时 mode 会被改写成实际口径，保留请求值用于 note）
    c, own = _conn(conn)
    tr_code = BENCHMARK_TR_MAP.get(code)
    net_code = BENCHMARK_NET_MAP.get(code)
    df, meta = None, {}
    try:
        # 净收益优先（已扣红利税），缺失时 fall through 到全收益链
        if mode == "net" and net_code:
            df = _read_net(net_code, start, end, c)
            if df is not None and len(df) >= 2:
                return df, dict(mode="net", source_table="index_tr_official",
                                resolved_code=net_code, note="官方净收益(已扣红利税)")
        if mode in ("tr_official", "net"):
            if tr_code:
                df = _read_official(tr_code, start, end, c)
                if df is not None and len(df) >= 2:
                    return df, dict(mode=mode, source_table="index_tr_official",
                                    resolved_code=tr_code, note="官方全收益")
            df = _read_total_return(code, start, end, c)
            if df is not None and len(df) >= 2:
                return df, dict(mode=mode, source_table="index_total_return",
                                resolved_code=code, note="自建全收益(bottom-up)")
        elif mode == "total_return":
            df = _read_total_return(code, start, end, c)
            if df is not None and len(df) >= 2:
                return df, dict(mode=mode, source_table="index_total_return",
                                resolved_code=code, note="自建全收益(bottom-up)")
        # 最终回退：价格指数
        # ⚠️ 回退时 mode 必须改写为 "price"（**实际**口径），不能保留请求的 mode。
        #    否则 meta 会自称 tr_official 却来自 index_daily，下游（报告标签/告警）全部失真。
        df = _read_price(code, start, end, c)
        # 措辞：请求的本就是 price → 直接使用，不算"回退"（否则 raw 那跑会误报口径缺失）
        _suffix = ("" if quoted_mode == "price"
                   else f"·请求{quoted_mode}但无全收益，已回退")
        if df is not None and len(df) >= 2:
            return df, dict(mode="price", source_table="index_daily",
                            resolved_code=code,
                            note=f"价格指数(不含分红){_suffix}")
        return None, dict(mode="price", source_table=None, resolved_code=code,
                          note=f"无数据·请求{quoted_mode}")
    finally:
        if own:
            c.close()


def benchmark_return_between(code, start, end, conn=None, mode=None, nav_price_mode=None):
    """返回 (ret_pct, meta)；无数据返回 (None, meta)"""
    df, meta = load_benchmark(code, start, end, conn=conn, mode=mode, nav_price_mode=nav_price_mode)
    if df is None or len(df) < 2:
        return None, meta
    ret = (float(df["close"].iloc[-1]) / float(df["close"].iloc[0]) - 1) * 100
    return ret, meta


def benchmark_year_endpoints(code, year, conn=None, mode=None, nav_price_mode=None):
    """返回某年首末收盘价 (first_close, last_close, meta)；无数据 (None, None, meta)"""
    s, e = f"{year}0101", f"{year}1231"
    df, meta = load_benchmark(code, s, e, conn=conn, mode=mode, nav_price_mode=nav_price_mode)
    if df is None or len(df) < 1:
        return None, None, meta
    return float(df["close"].iloc[0]), float(df["close"].iloc[-1]), meta


def benchmark_meta_label(meta):
    """把 meta 渲染成可读标签，用于报告透明标注"""
    if not meta:
        return ""
    return f"[{meta.get('source_table')}|{meta.get('note')}]"
