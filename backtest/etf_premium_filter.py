# -*- coding: utf-8 -*-
"""
ETF 折溢价率 风险过滤器
========================
来源：B站「跟着Jim学量化」《ETF溢价，就等于后面还会涨吗？》(BV1JS316dETi)
      —— 经人工校验为正确知识，萃取为 ETF轮动 策略的买入前风险过滤器。

【视频核心结论（已验证正确）】
  1. 折溢价率 = (市价 − 参考值) / 参考值。参考值用 IOPV(盘中估算) 或 正式净值(收盘确认)，二者不可混用。
  2. 溢价 ≠ 会涨；折价 ≠ 无风险套利（涉及成组份额/费用/执行条件）。
  3. 跨境ETF(恒生/纳指) 因「境内交易时境外休市 + 汇率变动 + 参考值更新慢」，
     折溢价常被"虚假拉宽"；但高溢价买入后收敛 = 真实亏损 —— 散户最高频坑。
  4. 三步检查：① 对齐参考值 ② 对齐时间(尤其跨境,NAV有滞后) ③ 看偏离宽度 + 持续天数。

【为什么本过滤器用 NAV(收盘确认净值) 而非 IOPV(盘中参考净值)】
  来源：同 UP 主《一只ETF同时出现两个价格，你选对了吗？》(BV1YP326jE7S, 2026-07-29)
  - IOPV 是按申赎清单 + 篮子资产盘中价"估"出来的参考值，在标的停牌 / 交投清淡 /
    跨境休市(境外休市 + 汇率变动) 时会沿用旧价，看着平稳 ≠ 资产价值没变 → 失真。
  - 正式净值(NAV) 是收盘后按实际持仓与负债算出的确认值，不随盘中成交跳动，最可靠。
  - 本过滤器统一以 NAV 作为溢价基准，规避 IOPV 在停牌 / 盘整 / 跨境时段的假信号。
  - 跨境 ETF 额外用 staleness_days(NAV滞后 > 3日 → warn) 识别"参考值突然拉宽"的时滞坑。

【数据依赖】
  astock_daily.db 的 etf_daily 只有市价(OHLCV)，**没有 IOPV/NAV**。
  折溢价需要参考净值。已接入 tushare `fund_nav` → 本地表 `etf_nav`：
      ts_code TEXT, nav_date TEXT, unit_nav REAL
  load_nav() 自动建表 + 按标的按需回填（首次联网，之后纯本地）。
  补上后对本平台 ETF轮动 池即可实时过滤（见文末集成点）。

【集成点】
  在 run_etf_rotation.py 的选标步骤里，对动量入围标的调用 filter_etf_candidates()，
  把 caution=='block' 的剔除、'warn' 的降权，即可避免"高溢价追涨被收敛砸"。

阈值（可在 config 覆盖）：
  PREMIUM_WARN=0.03  溢价>3%  预警(不追高)
  PREMIUM_HARD=0.05  溢价>5%  禁止买入
  跨境ETF 阈值各减 CROSSBORDER_EXTRA=0.02（更严）
"""

from __future__ import annotations
import sqlite3
from datetime import date

# ── 阈值（可用 config 覆盖）──────────────────────────────
PREMIUM_WARN = 0.03
PREMIUM_HARD = 0.05
CROSSBORDER_EXTRA = 0.02   # 跨境ETF 阈值更严
PRICE_NAV_RATIO_CAP = 10.0  # 市价/净值 超过此倍数视为计价单位不一致
                          # （如货币ETF净值为1.0、市价≈100），跳过折溢价判断

# 跨境ETF（境内交易、参考境外资产，NAV滞后+溢价虚假拉宽）
CROSSBORDER_CODES = {
    "159920.SZ": "恒生ETF",
    "513100.SH": "纳指ETF",
    "513500.SH": "标普500ETF",
    "159941.SZ": "纳指ETF(广发)",
}


def is_crossborder(ts_code: str) -> bool:
    return ts_code in CROSSBORDER_CODES


def premium_rate(market_price: float, nav: float) -> float:
    """折溢价率 = (市价 − 净值) / 净值。nav<=0 返回 nan。"""
    if nav is None or nav <= 0 or market_price is None:
        return float("nan")
    return (market_price - nav) / nav


def staleness_days(nav_date: str, trade_date: str) -> int:
    """净值相对交易日的滞后天数（跨境ETF NAV常为T-1甚至更旧）。"""
    try:
        d1 = date(int(nav_date[:4]), int(nav_date[4:6]), int(nav_date[6:8]))
        d2 = date(int(trade_date[:4]), int(trade_date[4:6]), int(trade_date[6:8]))
        return (d2 - d1).days
    except Exception:
        return -1


def caution_level(prem: float, cross: bool, stale_days: int = 0) -> str:
    """
    返回 'ok' | 'warn' | 'block'
      · block: 溢价超硬阈值（跨境更严）→ 禁止买入
      · warn : 溢价超预警阈值，或跨境ETF净值滞后>3日(溢价可能虚假拉宽)
      · ok   : 其余
    """
    if prem != prem:  # nan
        return "ok"
    warn = PREMIUM_WARN - (CROSSBORDER_EXTRA if cross else 0)
    hard = PREMIUM_HARD - (CROSSBORDER_EXTRA if cross else 0)
    if prem >= hard:
        return "block"
    if prem >= warn:
        return "warn"
    # 跨境ETF：净值滞后过久 → 折溢价读数不可信，谨慎
    if cross and stale_days > 3:
        return "warn"
    return "ok"


def filter_one(ts_code: str, market_price: float, nav: float, nav_date: str, trade_date: str) -> dict:
    """单只ETF的过滤结果字典。"""
    cross = is_crossborder(ts_code)
    # 计价单位不一致保护：货币ETF 等净值按1.0计、市价≈100，折溢价无意义 → 直接放行
    if nav and market_price and market_price / nav > PRICE_NAV_RATIO_CAP:
        return {
            "ts_code": ts_code, "cross_border": cross,
            "market_price": market_price, "nav": nav,
            "premium_rate": float("nan"), "stale_days": 0,
            "caution": "ok",
        }
    prem = premium_rate(market_price, nav)
    level = caution_level(prem, cross, staleness_days(nav_date, trade_date))
    return {
        "ts_code": ts_code,
        "cross_border": cross,
        "market_price": market_price,
        "nav": nav,
        "premium_rate": prem,
        "stale_days": staleness_days(nav_date, trade_date),
        "caution": level,
    }


def filter_etf_candidates(rows: list[dict]) -> list[dict]:
    """
    批量过滤。rows: [{ts_code, market_price, nav, nav_date, trade_date}, ...]
    返回带 caution 字段的结果列表（原顺序）。
    """
    return [filter_one(r["ts_code"], r["market_price"], r.get("nav"),
                        r.get("nav_date", ""), r.get("trade_date", "")) for r in rows]


# ── 数据加载（tushare fund_nav → 本地 etf_nav）─────────────
import sys as _sys
import os as _os
import time as _time

_NAV_TABLE = "etf_nav"
_TUSHARE_INTERVAL = 0.4          # 请求间隔（秒），避免频率限制
_NAV_BACKFILL_START = "20100101"  # 首次回填起点（ETF 成立后才有数据）


def _ensure_nav_table(conn: sqlite3.Connection) -> None:
    """建 etf_nav 表（如不存在）。"""
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {_NAV_TABLE} (
            ts_code   TEXT NOT NULL,
            nav_date  TEXT NOT NULL,
            unit_nav  REAL,
            PRIMARY KEY (ts_code, nav_date)
        )
    """)
    conn.commit()


def _get_tushare_token() -> "str | None":
    """从 config / config_tushare 取 token（项目根目录需在 sys.path）。"""
    root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if root not in _sys.path:
        _sys.path.insert(0, root)
    try:
        import config  # type: ignore
        tok = config.DATA.get("tushare_token")
        if tok:
            return tok
    except Exception:
        pass
    try:
        import config_tushare  # type: ignore
        return config_tushare.TUSHARE_TOKEN
    except Exception:
        return None


def _fetch_nav_tushare(conn: sqlite3.Connection, codes: list[str]) -> None:
    """对本地缺失净值的 ETF，从 tushare `fund_nav` 拉取并落库。"""
    import tushare as ts
    token = _get_tushare_token()
    if not token:
        print("[etf_premium_filter] 未找到 tushare token，跳过 NAV 拉取。")
        return
    ts.set_token(token)
    pro = ts.pro_api()
    inserted_total = 0
    for code in codes:
        try:
            df = pro.fund_nav(
                ts_code=code,
                start_date=_NAV_BACKFILL_START,
                fields="ts_code,nav_date,unit_nav",
            )
        except Exception as e:
            print(f"[etf_premium_filter] fund_nav 拉取失败 {code}: {e}")
            _time.sleep(_TUSHARE_INTERVAL)
            continue
        if df is None or len(df) == 0:
            print(f"[etf_premium_filter] fund_nav 无数据 {code}")
            _time.sleep(_TUSHARE_INTERVAL)
            continue
        # 清洗：去除无效净值（NaN / 非正）
        df = df.dropna(subset=["unit_nav"])
        df = df[df["unit_nav"] > 0]
        rows = [
            (str(r["ts_code"]), str(r["nav_date"]), float(r["unit_nav"]))
            for _, r in df.iterrows()
        ]
        conn.executemany(
            f"INSERT OR IGNORE INTO {_NAV_TABLE} (ts_code, nav_date, unit_nav) "
            "VALUES (?, ?, ?)",
            rows,
        )
        conn.commit()
        inserted_total += len(rows)
        print(f"[etf_premium_filter] fund_nav {code}: +{len(rows)} 条")
        _time.sleep(_TUSHARE_INTERVAL)
    if inserted_total:
        print(f"[etf_premium_filter] 共写入 {inserted_total} 条 NAV。")


def load_nav(conn: sqlite3.Connection, codes: list[str],
             as_of: "str | None" = None) -> dict:
    """
    取每只 ETF 的净值（来自本地 etf_nav 表）。

    as_of: 交易日 YYYYMMDD。若提供，返回该日或之前最近的一个净值
           （用于历史回测逐日对齐，避免“最新净值 vs 陈旧市价”错配）；
           若省略，返回截至当前的最新净值（用于实时/近期过滤）。

    表不存在时自动创建；若某只 ETF 本地无数据，则按需从 tushare
    `fund_nav` 拉取并落库（一次性回填，之后只走本地，不再联网）。

    返回 {ts_code: (nav_date, unit_nav)}，取该只满足条件的最新净值。
    若某只无可用净值（本地缺失且拉取失败，或 as_of 早于其上市日），
    则不在结果中 —— 调用方据此传入 nav=None，过滤器退化 'ok'（不拦截）。
    """
    if not codes:
        return {}
    codes = list(dict.fromkeys(codes))  # 去重保序
    _ensure_nav_table(conn)
    ph = ",".join("?" * len(codes))
    params: list = list(codes)
    date_filter = ""
    if as_of:
        date_filter = " AND nav_date <= ?"
        params.append(as_of)
    try:
        rows = conn.execute(
            f"SELECT ts_code, nav_date, unit_nav FROM {_NAV_TABLE} "
            f"WHERE ts_code IN ({ph}){date_filter}",
            params,
        ).fetchall()
    except sqlite3.OperationalError:
        # 极端情况：建表失败
        return {}
    existing = {c: set() for c in codes}
    data = {}
    for code, nav_date, nav in rows:
        if code not in existing:
            continue
        existing[code].add(nav_date)
        data.setdefault(code, []).append((nav_date, nav))
    # 本地缺失的标的 → 联网拉取后重读
    need = [c for c in codes if not existing.get(c)]
    if need:
        print(f"[etf_premium_filter] 本地缺失 {len(need)} 只，联网拉取 NAV...")
        _fetch_nav_tushare(conn, need)
        rows = conn.execute(
            f"SELECT ts_code, nav_date, unit_nav FROM {_NAV_TABLE} "
            f"WHERE ts_code IN ({ph}){date_filter}",
            params,
        ).fetchall()
        data = {}
        for code, nav_date, nav in rows:
            data.setdefault(code, []).append((nav_date, nav))
    # 每只取满足条件的最新净值
    return {c: max(v, key=lambda x: x[0]) for c, v in data.items()}


# ── 演示：用合成数据验证过滤逻辑（无需外部数据）──────────
def _demo():
    samples = [
        # 沪深300ETF，轻度折价 → ok
        {"ts_code": "510300.SH", "market_price": 3.95, "nav": 4.00,
         "nav_date": "20260803", "trade_date": "20260803"},
        # 纳指ETF，溢价 6% → block（跨境更严）
        {"ts_code": "513100.SH", "market_price": 1.59, "nav": 1.50,
         "nav_date": "20260803", "trade_date": "20260803"},
        # 恒生ETF，溢价 2.5%（跨境 warn 阈值=3%）→ warn
        {"ts_code": "159920.SZ", "market_price": 1.025, "nav": 1.00,
         "nav_date": "20260803", "trade_date": "20260803"},
        # 纳指ETF，溢价 1% 但净值滞后 5 日 → warn（读数不可信）
        {"ts_code": "513100.SH", "market_price": 1.01, "nav": 1.00,
         "nav_date": "20260729", "trade_date": "20260803"},
    ]
    print("=== ETF 折溢价过滤器 demo ===")
    for r in filter_etf_candidates(samples):
        print(f"  {r['ts_code']:10s} 溢价={r['premium_rate']*100:+.2f}% "
              f"跨境={r['cross_border']} 滞后={r['stale_days']}天 "
              f"→ {r['caution'].upper()}")


if __name__ == "__main__":
    _demo()
