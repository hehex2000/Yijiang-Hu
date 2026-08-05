# -*- coding: utf-8 -*-
"""
宽基 ETF 定投策略（DCA, Dollar-Cost Averaging）回测
====================================================
支持两种频率：
  - 月度定投：每月首个交易日投入固定金额
  - 周度定投：每周首个交易日（周一）投入固定金额
并与「一次性投入（lump-sum）」对比。

支持「宽基篮子」：传入多只 ETF 代码（等权），每期固定金额按 N 只均分，
各自按前复权收盘价买入、累加份额；组合收益为各 ETF 市值之和。

数据：etf_daily（不复权行情）+ etf_adj_factor（复权因子）→ 前复权价。
费用：佣金万2.5（最低5元）+ 滑点0.1%，ETF 免印花税（与平台其他策略一致）。

年化收益采用 XIRR（资金加权内部收益率），避免「全部本金从第1天起一次性投入」
带来的高估——这是 DCA 正确的年化口径。同时给出「简易年化」作为对照，并标注其
口径缺陷，以便透明比较。

用法：
  python run_dca_etf.py --freq both
  python run_dca_etf.py --freq monthly --start 20180101 --end 20260715 --monthly 4000
  python run_dca_etf.py --freq weekly  --code 510300.SH --weekly 1000
  python run_dca_etf.py --code 510300.SH,510500.SH,159915.SZ   # 篮子(逗号分隔)
  python run_dca_etf.py --preset core6 --freq both             # 预设篮子
  python run_dca_etf.py --catalog                              # 仅生成可选清单 HTML
也可通过 run_backtest.py --source dca_etf 调用（--dca-code 支持逗号列表/预设名）。
"""
import sqlite3
import os
import sys
import math
import json
import argparse
import statistics
from datetime import datetime, date

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA  # noqa: E402

DB_PATH = DATA.get("local_db_path", "")

# ── 费用模型（与平台 run_monthly_rebalance / run_etf_rotation 一致）────────────
COMMISSION_RATE = 0.00025   # 佣金率 万2.5
COMMISSION_MIN = 5.0        # 最低佣金 5 元
SLIPPAGE_RATE = 0.001       # 滑点 0.1%

# ── 均线增强定投(smart) 默认参数（文章：《玩红利ETF的实战思路》5周线/20周线操作法）──
# 仅 smart 模式使用；plain 模式行为与改前完全一致。
SMART_DIP_BAND = 0.03      # 价格 <= 20周线*(1+3%) 视为"接近/跌破20周线"（中长期底部）→ 补仓，动用预留现金
SMART_OVERBAND = 0.10      # 价格 >= 5周线*(1+10%) 视为"远高于5周线"→ 等一等（仅投部分，余下转预留现金）
SMART_EXTRA_MULT = 1.0     # 底部补仓时最多额外部署 = 当期预算的 1 倍（即预留的那部分现金）
SMART_HOLD_RATIO = 0.5     # 高估(远高于5周线)时当期仅投预算 50%，余下转预留现金
SMART_EXTRA_SELL = 0.34    # 涨回20周线后每期赎回 34% 的补仓份额（逐步把补仓资金撤回）

# ── 宽基被动指数 ETF 可选清单（DB 已确认在库；无名称表，硬编码）───────────────
# code -> (名称, 类别)
BROAD_ETF = {
    "510300.SH": ("沪深300ETF(华泰柏瑞)", "沪深300"),
    "510330.SH": ("沪深300ETF(华夏)", "沪深300"),
    "159919.SZ": ("沪深300ETF(嘉实)", "沪深300"),
    "510050.SH": ("上证50ETF", "上证50"),
    "510500.SH": ("中证500ETF(南方)", "中证500"),
    "512500.SH": ("中证500ETF(华夏)", "中证500"),
    "512100.SH": ("中证1000ETF(南方)", "中证1000"),
    "159915.SZ": ("创业板ETF", "创业板指"),
    "588000.SH": ("科创50ETF(华夏)", "科创50"),
    "588080.SH": ("科创50ETF(易方达)", "科创50"),
    "515800.SH": ("中证800ETF", "中证800"),
    "159901.SZ": ("深证100ETF", "深证100"),
    "563800.SH": ("中证A500ETF(易方达)", "中证A500"),
    "159338.SZ": ("中证A500ETF(国泰)", "中证A500"),
    "560530.SH": ("中证A500ETF(摩根)", "中证A500"),
    "159339.SZ": ("中证A500ETF(银华)", "中证A500"),
    "563000.SH": ("中证A50ETF(易方达)", "中证A50"),
    "159591.SZ": ("中证A50ETF(嘉实)", "中证A50"),
    "560050.SH": ("MSCI中国A50ETF(汇添富)", "MSCI A50"),
    "159601.SZ": ("MSCI中国A50ETF(招商)", "MSCI A50"),
    "159783.SZ": ("双创50ETF(华夏)", "双创50"),
    "588380.SH": ("双创50ETF(科创板)", "双创50"),
    "563300.SH": ("中证2000ETF(华泰柏瑞)", "中证2000"),
    "159531.SZ": ("中证2000ETF(易方达)", "中证2000"),
    "159628.SZ": ("国证2000ETF(万家)", "国证2000"),
    "159633.SZ": ("北证50ETF(易方达)", "北证50"),
    # ── 红利 / 红利低波（文章《玩红利ETF的实战思路》核心标的）──
    "510880.SH": ("红利ETF(华泰柏瑞·上证红利)", "红利"),
    "512890.SH": ("红利低波ETF(华泰柏瑞·中证红利低波)", "红利低波"),
    "515080.SH": ("中证红利ETF(招商)", "红利"),
    "515100.SH": ("红利低波100ETF(景顺长城)", "红利低波"),
}

# ── 预设篮子（均为宽基被动指数 ETF）──────────────────────────────────────────
# all_legacy：2018 前已上市、可覆盖全历史区间的宽基
_PRESET_ALL_LEGACY = ["510300.SH", "510330.SH", "159919.SZ", "510050.SH",
                      "510500.SH", "512500.SH", "512100.SH", "159915.SZ", "159901.SZ"]
PRESETS = {
    "core6":   ["510300.SH", "510500.SH", "159915.SZ", "510050.SH", "512100.SH", "159901.SZ"],
    "core4":   ["510300.SH", "510500.SH", "159915.SZ", "510050.SH"],
    "core3":   ["510300.SH", "510500.SH", "159915.SZ"],
    "large3":  ["510300.SH", "510050.SH", "159901.SH"],
    "all_legacy": _PRESET_ALL_LEGACY,
    "a500_4":  ["563800.SH", "159338.SZ", "560530.SH", "159339.SZ"],
    # ── 红利系（文章精华）──
    # dividend：红利 + 红利低波 核心双子星（长期死拿已赢大多数，叠加均线增强做超额）
    "dividend": ["510880.SH", "512890.SH"],
    # div_tech：红利核心 + 创业板（文章提醒"别只买红利，行情悲观时挪一点到科技放大弹性"）
    "div_tech": ["510880.SH", "512890.SH", "159915.SZ"],
}


def resolve_codes(arg):
    """把 --code/--preset 参数解析为代码列表。
    - 若为预设名 → 展开
    - 若含逗号 → 拆分去空白
    - 否则单只
    仅保留 BROAD_ETF 中已知代码（容错未知代码）。"""
    if arg in PRESETS:
        return list(PRESETS[arg])
    if "," in arg:
        codes = [c.strip() for c in arg.split(",") if c.strip()]
    else:
        codes = [arg.strip()]
    known = [c for c in codes if c in BROAD_ETF]
    if not known:
        print(f"[!] 未识别的标的/预设: {arg}（不在宽基清单中）")
        return []
    if len(known) != len(codes):
        unknown = [c for c in codes if c not in BROAD_ETF]
        print(f"[!] 已忽略未知代码: {unknown}")
    return known


def get_conn():
    return sqlite3.connect(DB_PATH)


def load_series(code):
    """读取某 ETF 的全部日线 + 复权因子，返回含前复权价的 DataFrame。"""
    conn = get_conn()
    df = pd.read_sql(
        "SELECT trade_date, open, high, low, close, pre_close FROM etf_daily "
        "WHERE ts_code=? ORDER BY trade_date",
        conn, params=(code,),
    )
    af = pd.read_sql(
        "SELECT trade_date, adj_factor FROM etf_adj_factor WHERE ts_code=? ORDER BY trade_date",
        conn, params=(code,),
    )
    conn.close()
    if df.empty:
        return None
    df["trade_date"] = df["trade_date"].astype(str)
    af["trade_date"] = af["trade_date"].astype(str)
    m = df.merge(af, on="trade_date", how="left")
    m["adj_factor"] = m["adj_factor"].ffill().bfill().fillna(1.0)
    last_af = m["adj_factor"].iloc[-1]
    # 前复权：最新交易日价格 = 真实可交易价
    m["qfq_close"] = m["close"] * m["adj_factor"] / last_af
    m["qfq_open"] = m["open"] * m["adj_factor"] / last_af
    m["trade_date"] = m["trade_date"].astype(str)
    return m


def build_schedule(freq, dates):
    """从交易日序列生成定投日。
    monthly: 每月首个交易日；weekly: 每周首个交易日（按 ISO 周，周一）。"""
    if freq == "monthly":
        seen, sched = set(), []
        for d in dates:
            key = d[:6]  # YYYYMM
            if key not in seen:
                seen.add(key)
                sched.append(d)
        return sched
    elif freq == "weekly":
        seen, sched = set(), []
        for d in dates:
            dt = date(int(d[:4]), int(d[4:6]), int(d[6:8]))
            y, w, _ = dt.isocalendar()
            key = (y, w)
            if key not in seen:
                seen.add(key)
                sched.append(d)
        return sched
    else:
        raise ValueError(f"未知频率: {freq}")


def _weekly_ma_map(m):
    """由 ETF 日线(含前复权价)计算 5周/20周 均线映射：date -> (w5, w20)。

    仅用「过往周收盘」计算，绝对无未来函数：
      - 按 ISO 周聚合，取每周最后一个交易日的收盘作为该周收盘；
      - 对任意日期 d，取 ≤d 的最近一周的均线值（避免"当日非周收盘"带来的微未来函数）。
    返回的字典键为该 ETF 的全部交易日。
    """
    import bisect
    s = m[["trade_date", "qfq_close"]].copy()
    s = s[s["qfq_close"].notna()]
    if s.empty:
        return {}
    # 按 ISO 周聚合：取每周最后一个交易日的收盘
    wk = {}  # (y, w) -> (last_date, close)
    for _, row in s.iterrows():
        d = row["trade_date"]
        dt = date(int(d[:4]), int(d[4:6]), int(d[6:8]))
        y, w, _ = dt.isocalendar()
        if (y, w) not in wk or d > wk[(y, w)][0]:
            wk[(y, w)] = (d, float(row["qfq_close"]))
    wk_list = sorted(wk.values(), key=lambda x: x[0])
    wd = [x[0] for x in wk_list]
    wc = [x[1] for x in wk_list]
    n = len(wk_list)
    # 预计算各周均线
    week_ma = []
    for i in range(n):
        w5 = sum(wc[max(0, i - 4):i + 1]) / 5.0 if i >= 4 else None
        w20 = sum(wc[max(0, i - 19):i + 1]) / 20.0 if i >= 19 else None
        week_ma.append((w5, w20))
    # 映射到每个交易日（取 ≤d 的最近周）
    result = {}
    for d in s["trade_date"].tolist():
        i = bisect.bisect_right(wd, d) - 1
        result[d] = week_ma[i] if i >= 0 else (None, None)
    return result


def xirr(cf_dates, cf_amounts):
    """资金加权年化收益率（XIRR）。
    cf_dates: list[date]；cf_amounts: list[float]，流出为负、流入为正。
    用扫描 + 二分法求根，避免牛顿法对初值敏感。无解返回 None。"""
    if not cf_dates:
        return None
    d0 = min(cf_dates)

    def npv(r):
        s = 0.0
        for d, a in zip(cf_dates, cf_amounts):
            y = (d - d0).days / 365.25
            s += a / ((1.0 + r) ** y)
        return s

    prev = None
    for i in range(-990, 10001):  # rate 从 -0.99 到 10.0，步长 0.001
        r = i / 1000.0
        v = npv(r)
        if prev is not None and prev[1] != 0 and prev[1] * v < 0:
            lo, hi = prev[0], r
            for _ in range(200):
                mid = (lo + hi) / 2.0
                fm = npv(mid)
                if abs(fm) < 1e-9:
                    return mid
                if npv(lo) * fm <= 0:
                    hi = mid
                else:
                    lo = mid
            return (lo + hi) / 2.0
        prev = (r, v)
    return None


def _to_date(s):
    return date(int(s[:4]), int(s[4:6]), int(s[6:8]))


def _equity_var(returns, conf_levels=(0.95, 0.99), capital=None, method="hist"):
    """
    由逐期收益率序列计算 VaR 报告（参数法+历史法）。
    returns : 逐期增长率列表（小数）
    capital : 当前净值（换算金额用）
    返回 dict：{0.95: {'param_loss','hist_loss','param_amt','hist_amt'}, ...}
    """
    rs = [r for r in returns if math.isfinite(r)]
    Z = {0.90: 1.282, 0.95: 1.645, 0.99: 2.326}
    if len(rs) < 5:
        return {c: {"param_loss": 0.0, "hist_loss": 0.0, "param_amt": 0.0, "hist_amt": 0.0}
                for c in conf_levels}
    mu = statistics.mean(rs)
    sd = statistics.pstdev(rs)
    out = {}
    for c in conf_levels:
        z = Z.get(c, 1.645)
        param_loss = max(0.0, -(mu - z * sd))
        if method in ("hist", "both"):
            q = max(0.0, min(1.0, 1.0 - c))
            srt = sorted(rs)
            idx = max(0, min(len(srt) - 1, int(q * (len(srt) - 1))))
            hist_loss = max(0.0, -srt[idx])
        else:
            hist_loss = param_loss
        out[c] = {
            "param_loss": param_loss, "hist_loss": hist_loss,
            "param_amt": (param_loss * capital) if capital else 0.0,
            "hist_amt": (hist_loss * capital) if capital else 0.0,
        }
    return out


def run_backtest(codes, freq="monthly", monthly_amount=4000, weekly_amount=1000,
                 start_date="20180101", end_date="20260715",
                 save_csv=True, make_html=False, mode="plain",
                 dip_band=SMART_DIP_BAND, over_band=SMART_OVERBAND,
                 extra_mult=SMART_EXTRA_MULT):
    """执行单次定投回测（单标的或篮子），返回指标字典（含 equity 序列供绘图）。
    codes 可为: 单代码字符串 / 逗号分隔字符串 / 预设名 / 代码列表。

    mode:
      "plain" — 原版固定金额定投（行为与改前一致，仅纪律定投）。
      "smart" — 均线增强定投（文章《玩红利ETF的实战思路》5周/20周线操作法）：
                底部(价格≤20周线)动用预留现金补仓；远高于5周线(高估)少投、余下转预留现金；
                涨回20周线后逐步撤回补仓份额。详见 SMART_* 常量与执行循环。
    """
    if isinstance(codes, (list, tuple)):
        codes = resolve_codes(",".join(codes))
    else:
        codes = resolve_codes(codes)
    if not codes:
        print("[!] 无有效标的")
        return None

    # 载入各 ETF 序列
    series = {}
    for code in codes:
        m = load_series(code)
        if m is None:
            print(f"[!] 无数据: {code}（已跳过）")
            continue
        series[code] = m
    if not series:
        return None

    is_basket = len(series) > 1
    per = monthly_amount if freq == "monthly" else weekly_amount
    N = len(series)  # 等权只数

    # 组合交易日轴 = 各 ETF 在 [start,end] 内交易日的并集（升序）
    alld = set()
    for m in series.values():
        alld |= set(d for d in m["trade_date"] if start_date <= d <= end_date)
    alld = sorted(alld)
    if not alld:
        print(f"[!] 区间内无交易日")
        return None

    # 各 ETF 价格对齐到组合轴（ffill；未上市前为 None）
    price = {}
    for code, m in series.items():
        d2p = dict(zip(m["trade_date"], m["qfq_close"]))
        arr = []
        last = None
        for d in alld:
            if d in d2p:
                last = d2p[d]
            arr.append(last)
        price[code] = arr
    n = len(alld)

    sched = [d for d in build_schedule(freq, alld) if start_date <= d <= end_date]
    if not sched:
        print(f"[!] 区间内无定投日")
        return None
    sched_set = set(sched)

    # ── 均线增强模式：预计算每只 ETF 的 5周/20周 均线映射（无未来函数）──
    ma_map = {}
    if mode == "smart":
        for code, m in series.items():
            ma_map[code] = _weekly_ma_map(m)

    # 预留现金账户（smart 模式）：未投入市场的现金，期末计入权益；
    # 底部补仓额外买入的份额（涨回20周线后逐步撤回）。
    reserve = {c: 0.0 for c in series}
    extra_shares = {c: 0.0 for c in series}
    dip_periods = 0
    overbought_periods = 0

    # ── 执行定投：每期 per 元按 N 只均分，各 ETF 前复权收盘价买入累加份额 ──
    shares = {c: 0.0 for c in series}
    comp_invested = {c: 0.0 for c in series}
    invested_total = 0.0
    trades = []        # (date, budget_total, fee_total)
    fee_total_all = 0.0
    equity = [0.0] * n
    for i, d in enumerate(alld):
        if d in sched_set:
            day_fee = 0.0
            day_dip = False
            day_over = False
            day_budget = 0.0   # 本期实际可投预算（仅计已上市成分，未上市成分预算不计入本金）
            for c in series:
                px = price[c][i]
                if px is None:
                    continue  # 尚未上市：其预算不投入、也不计入本金（按实际投入口径）
                budget = per / N                      # 当期该 ETF 预算
                day_budget += budget
                w5 = w20 = None
                sell_net = 0.0
                # ── smart：涨回20周线后，先逐步撤回前期补仓份额 ──
                if mode == "smart":
                    w5, w20 = ma_map[c].get(d, (None, None))
                    if w5 is not None and w20 is not None and px > w20 and extra_shares[c] > 1e-9:
                        sell_n = extra_shares[c] * SMART_EXTRA_SELL
                        proceeds = sell_n * px
                        fee2 = max(proceeds * COMMISSION_RATE, COMMISSION_MIN) + proceeds * SLIPPAGE_RATE
                        sell_net = proceeds - fee2
                        reserve[c] += sell_net        # 撤回的现金回到预留账户
                        extra_shares[c] -= sell_n
                        shares[c] -= sell_n
                        day_fee += fee2
                # 注意：sell_net 已并入 reserve[c]，此处 avail 不要再重复加 sell_net
                avail = reserve[c] + budget
                invest = budget
                is_dip = False
                if mode == "smart":
                    if w5 is not None and w20 is not None:
                        if px <= w20 * (1 + dip_band):
                            # 接近/跌破20周线（中长期底部）：补仓，动用预留现金
                            invest = min(avail, budget * (1 + extra_mult))
                            is_dip = True
                            day_dip = True
                        elif px >= w5 * (1 + over_band):
                            # 远高于5周线（高估）：等一等，仅投部分，余下转预留现金
                            invest = budget * SMART_HOLD_RATIO
                            day_over = True
                if invest > 0:
                    fee = max(invest * COMMISSION_RATE, COMMISSION_MIN) + invest * SLIPPAGE_RATE
                    inv = invest - fee
                    sh = inv / px
                    shares[c] += sh
                    # comp_invested 记"当期预算"而非 deploy 金额，避免重复计预留现金（与 invested_total 对齐）
                    comp_invested[c] += budget
                    day_fee += fee
                    if is_dip:
                        extra_shares[c] += sh   # 记录补仓份额，供后续涨回时撤回
                reserve[c] = avail - invest
            if mode == "smart":
                if day_dip:
                    dip_periods += 1
                if day_over:
                    overbought_periods += 1
            # 每期预算：仅计已上市成分（按实际投入口径，避免未上市成分的幽灵本金稀释收益）
            invested_total += day_budget
            fee_total_all += day_fee
            trades.append((d, day_budget, day_fee))
        # 当日组合市值（含预留现金）
        e = 0.0
        for c in series:
            if price[c][i] is not None:
                e += shares[c] * price[c][i]
            e += reserve[c]
        equity[i] = e

    final_equity = equity[-1]
    last_d = _to_date(alld[-1])
    first_d = _to_date(sched[0])
    yrs = (last_d - first_d).days / 365.25

    # ── XIRR（资金加权）──
    cf_d = [_to_date(t[0]) for t in trades] + [last_d]
    cf_a = [-t[1] for t in trades] + [final_equity]
    x = xirr(cf_d, cf_a)

    # ── 简易年化（口径有缺陷：假设全部本金第1天一次性投入）──
    simple_ann = (final_equity / invested_total) ** (1 / yrs) - 1 if invested_total > 0 else None
    total_return = (final_equity - invested_total) / invested_total * 100 if invested_total > 0 else 0.0

    # ── 最大回撤 ──
    peak = equity[0]
    mdd = 0.0
    for e in equity:
        if e > peak:
            peak = e
        if peak > 0:
            dd = (e - peak) / peak
            if dd < mdd:
                mdd = dd
    mdd *= 100.0

    # ── 夏普（日收益，无风险利率≈0）──
    rets = []
    for i in range(1, n):
        if equity[i - 1] > 0:
            rets.append(equity[i] / equity[i - 1] - 1)
    sharpe = statistics.mean(rets) / statistics.pstdev(rets) * math.sqrt(252) if len(rets) > 2 and statistics.pstdev(rets) > 0 else 0.0

    # ── VaR 前瞻风险（基于账户净值逐期增长率）──
    var_info = _equity_var(rets, conf_levels=(0.95, 0.99), capital=final_equity)

    # ── 一次性投入基准（lump-sum）：相同总额在首个定投日等权买入并持有 ──
    # 注意：篮子成分若首日尚未上市（如 512890 晚于回测起点），无法买入。
    # 按「实际投入」口径，把总额等权分给【首日已上市】的成分并全额投满，
    # 避免未上市成分那份资金凭空蒸发（否则一次性基准会被系统性低估）。
    ls_total = invested_total
    ls_shares = {c: 0.0 for c in series}
    idx0 = alld.index(sched[0])
    ls_fee_all = 0.0
    ls_avail = [c for c in series if price[c][idx0] is not None]
    n_avail = len(ls_avail) or 1
    for c in ls_avail:
        px = price[c][idx0]
        amt = ls_total / n_avail
        fee = max(amt * COMMISSION_RATE, COMMISSION_MIN) + amt * SLIPPAGE_RATE
        ls_fee_all += fee
        ls_shares[c] = (amt - fee) / px
    ls_equity = [sum(ls_shares[c] * (price[c][i] or 0) for c in series) for i in range(n)]
    ls_final = ls_equity[-1]
    ls_x = xirr([first_d, last_d], [-ls_total, ls_final])
    ls_total_return = (ls_final - ls_total) / ls_total * 100 if ls_total > 0 else 0.0

    # ── 成分明细（篮子）──
    components = []
    if is_basket:
        for c in series:
            px_last = price[c][-1] or 0.0
            fv = ls_shares[c]  # placeholder
        # recompute component final value from DCA shares
        comp_final = {}
        for c in series:
            comp_final[c] = shares[c] * (price[c][-1] or 0.0)
        tot_fv = sum(comp_final.values()) or 1.0
        for c in series:
            components.append({
                "code": c,
                "name": BROAD_ETF[c][0],
                "cat": BROAD_ETF[c][1],
                "invested": comp_invested[c],
                "final_value": comp_final[c],
                "weight": comp_final[c] / tot_fv * 100,
                "shares": shares[c],
            })
        components.sort(key=lambda r: -r["final_value"])

    label_codes = codes if not is_basket else codes
    if is_basket:
        name = f"宽基篮子({N}只等权)"
        code_key = "basket_" + "+".join(c[:6] for c in codes)
    else:
        name = BROAD_ETF[codes[0]][0]
        code_key = codes[0]

    metrics = {
        "name": name, "codes": codes, "code_key": code_key, "freq": freq,
        "per_period": per, "is_basket": is_basket, "n_etf": N,
        "start_date": start_date, "end_date": end_date,
        "n_periods": len(sched),
        "total_invested": invested_total,
        "fee_total": fee_total_all,
        "final_equity": final_equity,
        "profit": final_equity - invested_total,
        "total_return_pct": total_return,
        "xirr_pct": (x * 100 if x is not None else None),
        "simple_ann_pct": (simple_ann * 100 if simple_ann is not None else None),
        "max_drawdown_pct": mdd,
        "sharpe": sharpe,
        "var95_loss": var_info[0.95]["hist_loss"],
        "var99_loss": var_info[0.99]["hist_loss"],
        "var95_amt": var_info[0.95]["hist_amt"],
        "var99_amt": var_info[0.99]["hist_amt"],
        "ls_final": ls_final,
        "ls_total_return_pct": ls_total_return,
        "ls_xirr_pct": (ls_x * 100 if ls_x is not None else None),
        "equity": equity,
        "ls_equity": ls_equity,
        "dates": alld,
        "sched": sched,
        "components": components,
        # ── 均线增强(smart) 相关 ──
        "mode": mode,
        "reserve_end": (sum(reserve.values()) if mode == "smart" else 0.0),
        "dip_periods": (dip_periods if mode == "smart" else 0),
        "overbought_periods": (overbought_periods if mode == "smart" else 0),
        "smart_params": {"dip_band": dip_band, "over_band": over_band, "extra_mult": extra_mult}
        if mode == "smart" else None,
    }

    _print_result(metrics)
    if save_csv:
        _save_csv(metrics)
    return metrics


def _print_result(mt):
    f = mt["freq"]
    label = "月度定投" if f == "monthly" else "周度定投"
    if mt["is_basket"]:
        hdr = f"  宽基篮子({mt['n_etf']}只等权) · {label}"
        sub = "  标的: " + " + ".join(f"{c}({BROAD_ETF[c][0]})" for c in mt["codes"])
    else:
        hdr = f"  {mt['name']}({mt['codes'][0]}) · {label}"
        sub = ""
    print(f"\n{'='*72}")
    print(hdr)
    if sub:
        print(sub)
    print(f"  区间 {mt['start_date']} ~ {mt['end_date']}  | 每期总投入 ¥{mt['per_period']:,.0f} (均分{mt['n_etf']}只) | 定投 {mt['n_periods']} 期")
    if mt["mode"] == "smart":
        sp = mt["smart_params"]
        print(f"  模式 : 均线增强定投(smart) | 底部补仓带={sp['dip_band']*100:.0f}% 高估带={sp['over_band']*100:.0f}% 补仓倍数={sp['extra_mult']:.1f}x")
    else:
        print(f"  模式 : 普通定投(plain，仅纪律定投)")
    print(f"{'='*72}")
    print(f"  累计投入本金 : {mt['total_invested']:,.2f} 元  (手续费合计 {mt['fee_total']:,.2f})")
    print(f"  期末市值     : {mt['final_equity']:,.2f} 元")
    print(f"  累计盈亏     : {mt['profit']:+,.2f} 元")
    print(f"  总收益率     : {mt['total_return_pct']:+.2f}%  (对本金)")
    print(f"  资金加权年化 : {mt['xirr_pct']:+.2f}%  (XIRR，正确口径)")
    print(f"  简易年化※    : {mt['simple_ann_pct']:+.2f}%  (※口径有偏，仅供对照)")
    print(f"  最大回撤     : {mt['max_drawdown_pct']:.2f}%")
    print(f"  夏普比率     : {mt['sharpe']:.2f}")
    print(f"  风险价值VaR  : 95%单期最多亏 {mt['var95_loss']*100:.2f}%（≈¥{mt['var95_amt']:,.0f}）"
          f" | 99%最多亏 {mt['var99_loss']*100:.2f}%（≈¥{mt['var99_amt']:,.0f}）[历史法·前瞻]")
    print(f"  ── 一次性投入对比（同总额、首期日买入持有）──")
    print(f"  一次性期末   : {mt['ls_final']:,.2f} 元 | 总收益 {mt['ls_total_return_pct']:+.2f}% | 年化 {mt['ls_xirr_pct']:+.2f}%")
    if mt["mode"] == "smart":
        print(f"  ── 均线增强专属 ──")
        print(f"  期末预留现金 : {mt['reserve_end']:,.2f} 元（未投入市场的弹药，已计入期末市值）")
        print(f"  底部补仓期数 : {mt['dip_periods']} 期 | 高估少投期数 : {mt['overbought_periods']} 期（共 {mt['n_periods']} 期）")
        print(f"  ※ 逻辑：价格≤20周线→动用预留现金补仓；远高于5周线→少投、余下转预留；涨回20周线→逐步撤回补仓份额")
    if mt["is_basket"]:
        print(f"  ── 成分期末市值 ──")
        for c in mt["components"]:
            print(f"    {c['code']:<11}{c['name']:<20} 投入¥{c['invested']:>9,.0f}  市值¥{c['final_value']:>9,.0f}  权重{c['weight']:5.1f}%")
    print(f"{'='*72}")


def _print_comparison(monthly, weekly):
    """run_both 末尾打印月/周/一次性并列表，避免控制台只看到末尾一段。"""
    m, w = monthly, weekly
    ls_final = m["ls_final"]; ls_tr = m["ls_total_return_pct"]; ls_x = m["ls_xirr_pct"]
    def f2(x): return f"{x:,.2f}"
    def pct(x): return f"{x:+.2f}%"
    print(f"\n{'='*72}")
    print("  月 / 周 / 一次性 投入 对比汇总")
    print(f"  {'指标':<14}{'月度定投':>16}{'周度定投':>16}{'一次性投入':>16}")
    print(f"  {'-'*62}")
    print(f"  {'累计投入':<14}{f2(m['total_invested']):>16}{f2(w['total_invested']):>16}{f2(m['total_invested']):>16}")
    print(f"  {'期末市值':<14}{f2(m['final_equity']):>16}{f2(w['final_equity']):>16}{f2(ls_final):>16}")
    print(f"  {'累计盈亏':<14}{f2(m['profit']):>16}{f2(w['profit']):>16}{f2(ls_final - m['total_invested']):>16}")
    print(f"  {'总收益率':<14}{pct(m['total_return_pct']):>16}{pct(w['total_return_pct']):>16}{pct(ls_tr):>16}")
    print(f"  {'资金加权年化':<14}{pct(m['xirr_pct']):>16}{pct(w['xirr_pct']):>16}{pct(ls_x):>16}")
    print(f"  {'简易年化※':<14}{pct(m['simple_ann_pct']):>16}{pct(w['simple_ann_pct']):>16}{'—':>16}")
    print(f"  {'最大回撤':<14}{m['max_drawdown_pct']:>15.2f}%{w['max_drawdown_pct']:>15.2f}%{'—':>16}")
    print(f"  {'夏普比率':<14}{m['sharpe']:>16.2f}{w['sharpe']:>16.2f}{'—':>16}")
    print(f"{'='*72}")
    print("  ※ 资金加权年化(XIRR)为正确口径；简易年化仅对照。一次性=首期日同总额买入持有。")


def _save_csv(mt):
    out_dir = os.path.join("data", "results", "dca_etf")
    os.makedirs(out_dir, exist_ok=True)
    f = mt["freq"]
    path = os.path.join(out_dir, f"dca_{f}_{mt['code_key']}_{mt['start_date']}_{mt['end_date']}.csv")
    rows = []
    equity = mt["equity"]
    dates = mt["dates"]
    ls = mt["ls_equity"]
    sched_set = set(mt["sched"])
    for i, d in enumerate(dates):
        rows.append({
            "trade_date": d,
            "is_invest_day": 1 if d in sched_set else 0,
            "dca_equity": round(equity[i], 2),
            "lumpsum_equity": round(ls[i], 2),
        })
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  结果已保存：{path}")


def _build_svg(dates, series_dict, title):
    """手绘 SVG 多折线图（无第三方依赖），series_dict: {label: [values]}。"""
    W, H = 900, 360
    pad_l, pad_r, pad_t, pad_b = 60, 20, 30, 30
    allv = [v for vals in series_dict.values() for v in vals if v is not None]
    if not allv:
        return ""
    vmin, vmax = min(allv), max(allv)
    span = (vmax - vmin) or 1.0
    vmin -= span * 0.05
    vmax += span * 0.05
    span = vmax - vmin
    n = len(dates)
    colors = {"月度定投": "#c0392b", "周度定投": "#2980b9", "一次性投入": "#27ae60"}

    def x(i):
        return pad_l + (W - pad_l - pad_r) * (i / (n - 1)) if n > 1 else pad_l

    def y(v):
        return pad_t + (H - pad_t - pad_b) * (1 - (v - vmin) / span)

    parts = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" font-family="sans-serif">']
    parts.append(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')
    parts.append(f'<text x="{W/2}" y="18" text-anchor="middle" font-size="15" fill="#222">{title}</text>')
    for k in range(5):
        val = vmin + span * k / 4
        yy = y(val)
        parts.append(f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{W-pad_r}" y2="{yy:.1f}" stroke="#eee"/>')
        parts.append(f'<text x="{pad_l-6}" y="{yy+4:.1f}" text-anchor="end" font-size="11" fill="#888">{val:,.0f}</text>')
    for label, vals in series_dict.items():
        col = colors.get(label, "#555")
        pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(vals) if v is not None)
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="1.6" opacity="0.9"/>')
    lx = pad_l + 10
    ly = pad_t + 12
    for label in series_dict:
        col = colors.get(label, "#555")
        parts.append(f'<rect x="{lx}" y="{ly-9}" width="14" height="4" fill="{col}"/>')
        parts.append(f'<text x="{lx+20}" y="{ly-4}" font-size="12" fill="#333">{label}</text>')
        ly += 18
    parts.append('</svg>')
    return "\n".join(parts)


def _catalog_rows_html():
    """生成可选清单（按类别分组）HTML 片段。"""
    cats_order = ["沪深300", "上证50", "中证500", "中证1000", "创业板指", "科创50",
                  "中证800", "深证100", "中证A500", "中证A50", "MSCI A50", "双创50",
                  "中证2000", "国证2000", "北证50", "红利", "红利低波"]
    by_cat = {}
    for code, (nm, cat) in BROAD_ETF.items():
        by_cat.setdefault(cat, []).append((code, nm))
    rows = []
    for cat in cats_order:
        if cat not in by_cat:
            continue
        rows.append(f'<tr><td rowspan="{len(by_cat[cat])+0}" style="background:#f0f4f8;font-weight:700;vertical-align:top">{cat}</td></tr>' if False else "")
        first = True
        for code, nm in by_cat[cat]:
            if first:
                rows.append(f'<tr><td style="background:#f0f4f8;font-weight:700;vertical-align:top" rowspan="{len(by_cat[cat])}">{cat}</td>'
                            f'<td>{code}</td><td style="text-align:left">{nm}</td>'
                            f'<td>✓ 在库</td></tr>')
                first = False
            else:
                rows.append(f'<tr><td>{code}</td><td style="text-align:left">{nm}</td><td>✓ 在库</td></tr>')
    return "\n".join(rows)


def _build_html(monthly, weekly, out_path):
    is_basket = monthly["is_basket"]
    sd, ed = monthly["start_date"], monthly["end_date"]
    dates = monthly["dates"]
    title_label = monthly["name"] if is_basket else f"{monthly['name']}（{monthly['codes'][0]}）"
    svg = _build_svg(dates, {
        "月度定投": monthly["equity"],
        "周度定投": weekly["equity"],
        "一次性投入": monthly["ls_equity"],
    }, f"{title_label} 定投权益曲线对比  {sd}~{ed}")

    def row(label, m):
        return (f"<tr><td>{label}</td>"
                f"<td>{m['total_invested']:,.0f}</td>"
                f"<td>{m['final_equity']:,.0f}</td>"
                f"<td>{m['profit']:+,.0f}</td>"
                f"<td>{m['total_return_pct']:+.2f}%</td>"
                f"<td>{m['xirr_pct']:+.2f}%</td>"
                f"<td>{m['max_drawdown_pct']:.2f}%</td>"
                f"<td>{m['sharpe']:.2f}</td>"
                f"<td>{m['n_periods']}</td></tr>")

    comp_rows = ""
    if is_basket:
        comp_rows = "<h2>篮子成分期末市值</h2><table><tr><th>代码</th><th>名称</th><th>类别</th><th>累计投入</th><th>期末市值</th><th>权重</th></tr>"
        for c in monthly["components"]:
            comp_rows += (f"<tr><td>{c['code']}</td><td style='text-align:left'>{c['name']}</td>"
                          f"<td>{c['cat']}</td><td>{c['invested']:,.0f}</td>"
                          f"<td>{c['final_value']:,.0f}</td><td>{c['weight']:.1f}%</td></tr>")
        comp_rows += "</table>"

    catalog_html = _catalog_rows_html()

    basket_note = ""
    if is_basket:
        basket_note = (f"<br>4. <b>篮子等权</b>：每期 ¥{monthly['per_period']:,.0f} 均分 {monthly['n_etf']} 只，各按前复权收盘价买入。"
                       f"注意——小额分散会放大<b>最低佣金¥5</b>的拖累：单只仅约 ¥{monthly['per_period']/monthly['n_etf']:,.0f}/期，"
                       f"佣金占比≈{COMMISSION_MIN/(monthly['per_period']/monthly['n_etf'])*100:.2f}%，远高于单标的定投。若想降低拖累，可整体提高每期金额。")

    # ── 文章精华（非量化信号，仅供参考；与 smart 模式/红利标的呼应）──
    _div_codes = {"510880.SH", "512890.SH", "515080.SH", "515100.SH"}
    is_div = any(c in _div_codes for c in monthly["codes"])
    is_smart = monthly["mode"] == "smart"
    _smart_echo = ""
    if is_smart:
        _smart_echo = ("本报告的 <b>smart（均线增强）</b> 模式即落地了文中的「5周线/20周线操作法」："
                       "价≤20周线→动用预留现金补仓；远高于5周线→少投、余下转预留；涨回20周线→逐步撤回补仓份额。")
    _div_echo = ""
    if is_div:
        _div_echo = ("标的含<b>红利/红利低波ETF</b>：文中强调红利长期「死拿」已赢大多数人，"
                     "任何策略都难跑赢死拿；网格对红利不划算（熊市慢牛），故红利系不与网格策略混用。")
    essence_block = f"""
<div class="note" style="background:#eafaf1;border-left-color:#27ae60">
<b>📈 红利ETF 实战思路（文章精华 · 非量化信号，仅供参考）</b><br>
· <b>大前提</b>：红利ETF长期「死拿不动」已赢大多数人；任何策略长期难跑赢死拿。但短期市场情绪会制造「不属于它的涨跌」，可从中赚超额。<br>
· <b>避坑</b>：红利做<b>网格不划算</b>（熊市反而慢牛，特别不适合网格）。{_div_echo}<br>
· <b>技术面（均线·smart 模式已落地）</b>：仅看 5周线/20周线——
  当前价&lt;5周线→定投；远高于5周线→等一等；价格≤20周线（中长期底部）→用预留约50%现金补仓；涨回20周线→逐步减仓、把补仓资金撤回；常态保留约一半现金做T/补仓。{_smart_echo}<br>
· <b>基本面（股息率安心法）</b>：红利股息率 vs 余额宝七日年化——&gt;余额宝✅值得长期持有；&lt;余额宝⚠️投机泡沫危险。
  （本回测未接入股息率数据，仅作「心里安慰」参考，不驱动交易。）<br>
· <b>别只买红利</b>：每次牛市科技股弹性远大于红利；行情最悲观时把红利仓位逐渐挪一点到创业板/科技ETF放大弹性（<code>--preset div_tech</code> 即此意）。<br>
· <b>费率坑</b>：优先 A 类（管理费低于 C 类，适合长期）；能场内就场内。场内默认「万三+5元起收」对小额定投极不友好——
  一万块手续费本应3元，因5元起收实收5元→实际费率被抬高。争取券商「万1.5、免5」账户可降10倍以上。
</div>"""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>{title_label} 定投策略对比报告</title>
<style>
 body{{font-family:-apple-system,'Microsoft YaHei',sans-serif;margin:24px;color:#222;}}
 h1{{font-size:22px;}} h2{{font-size:16px;margin-top:28px;color:#333;}}
 table{{border-collapse:collapse;width:100%;margin-top:10px;font-size:13px;}}
 th,td{{border:1px solid #ddd;padding:7px 9px;text-align:right;}}
 th{{background:#f5f5f5;}} td:first-child,th:first-child{{text-align:left;}}
 .note{{background:#fff8e1;padding:12px 14px;border-left:4px solid #f1c40f;font-size:13px;line-height:1.7;}}
 .svgbox{{margin-top:12px;border:1px solid #eee;}}
 .pos{{color:#c0392b;}} .neg{{color:#27ae60;}}
 .cat{{background:#f0f4f8;font-weight:700;}}
</style></head><body>
<h1>{title_label} 定投 ETF 策略对比报告</h1>
<p>回测区间 <b>{sd} ~ {ed}</b> ｜ {'篮子' if is_basket else '单标的'} ｜ 每期总投入 ¥{monthly['per_period']:,.0f} ｜ 周度每期 ¥{weekly['per_period']:,.0f}（年化投入近似相等，公平对比）</p>
{('<p>篮子标的：' + ' + '.join(f"{c}({BROAD_ETF[c][0]})" for c in monthly['codes']) + '</p>') if is_basket else ''}

<div class="svgbox">{svg}</div>

<h2>核心指标对比</h2>
<table>
<tr><th>策略</th><th>累计投入</th><th>期末市值</th><th>累计盈亏</th><th>总收益(对本金)</th><th>资金加权年化(XIRR)</th><th>最大回撤</th><th>夏普</th><th>期数</th></tr>
{row('月度定投', monthly)}
{row('周度定投', weekly)}
</table>

<h2>与一次性投入（lump-sum）对比</h2>
<table>
<tr><th>方案</th><th>投入</th><th>期末市值</th><th>总收益</th><th>资金加权年化</th></tr>
<tr><td>月度定投</td><td>{monthly['total_invested']:,.0f}</td><td>{monthly['final_equity']:,.0f}</td><td class="{'pos' if monthly['profit']>=0 else 'neg'}">{monthly['total_return_pct']:+.2f}%</td><td class="{'pos' if monthly['xirr_pct']>=0 else 'neg'}">{monthly['xirr_pct']:+.2f}%</td></tr>
<tr><td>周度定投</td><td>{weekly['total_invested']:,.0f}</td><td>{weekly['final_equity']:,.0f}</td><td class="{'pos' if weekly['profit']>=0 else 'neg'}">{weekly['total_return_pct']:+.2f}%</td><td class="{'pos' if weekly['xirr_pct']>=0 else 'neg'}">{weekly['xirr_pct']:+.2f}%</td></tr>
<tr><td>一次性投入(同总额)</td><td>{monthly['total_invested']:,.0f}</td><td>{monthly['ls_final']:,.0f}</td><td class="{'pos' if monthly['ls_total_return_pct']>=0 else 'neg'}">{monthly['ls_total_return_pct']:+.2f}%</td><td class="{'pos' if monthly['ls_xirr_pct']>=0 else 'neg'}">{monthly['ls_xirr_pct']:+.2f}%</td></tr>
</table>

{comp_rows}

<div class="note">
<b>口径说明（重要）：</b><br>
1. <b>年化用 XIRR（资金加权）</b>，而非「期末/本金 开年方」的简易年化。DCA 资金是逐期流入的，简易年化会把后期才投入的钱也当成从第1天起全额在账，系统性高估收益——本报告的「简易年化」仅作对照，结论以 XIRR 为准。<br>
2. <b>月 vs 周 为公平对比</b>：周度每期 ¥{weekly['per_period']:,.0f} ≈ 月度 ¥{monthly['per_period']:,.0f} ÷ 4.33，年化投入总额近似相等。两者差异主要来自买入时点密度与手续费结构。<br>
3. <b>一次性投入基准</b>：以与各定投方案相同的总额，在首个定投日一次性买入并持有至期末。在整体上行市场中，一次性投入通常因更早占用全部资金而年化占优；定投的价值在于平滑择时风险、强制执行纪律，而非追求更高收益。
{basket_note}
</div>

{essence_block}

<h2>可选宽基 ETF 清单（DB 已确认在库，供挑选）</h2>
<table>
<tr><th>类别</th><th>代码</th><th>名称</th><th>状态</th></tr>
{catalog_html}
</table>
<p style="font-size:12px;color:#888">使用：<code>--code 510300.SH,510500.SH,159915.SZ</code> 或预设 <code>--preset core6/core4/core3/large3/all_legacy/a500_4</code></p>
</body></html>"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


def build_catalog_html(out_path):
    """独立生成「可挑选宽基 ETF 清单」页面。"""
    cats_order = ["沪深300", "上证50", "中证500", "中证1000", "创业板指", "科创50",
                  "中证800", "深证100", "中证A500", "中证A50", "MSCI A50", "双创50",
                  "中证2000", "国证2000", "北证50", "红利", "红利低波"]
    by_cat = {}
    for code, (nm, cat) in BROAD_ETF.items():
        by_cat.setdefault(cat, []).append((code, nm))
    rows = []
    for cat in cats_order:
        if cat not in by_cat:
            continue
        items = by_cat[cat]
        for k, (code, nm) in enumerate(items):
            in_preset = [p for p, lst in PRESETS.items() if code in lst]
            preset_tag = (" 预设: " + ",".join(in_preset)) if in_preset else ""
            cell_cat = f'<td class="cat" rowspan="{len(items)}">{cat}</td>' if k == 0 else ""
            rows.append(f"<tr>{cell_cat}<td>{code}</td><td style='text-align:left'>{nm}</td><td>✓ 在库{preset_tag}</td></tr>")
    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>宽基 ETF 可选清单</title>
<style>
 body{{font-family:-apple-system,'Microsoft YaHei',sans-serif;margin:24px;color:#222;}}
 h1{{font-size:22px;}}
 table{{border-collapse:collapse;width:100%;margin-top:10px;font-size:13px;}}
 th,td{{border:1px solid #ddd;padding:7px 9px;text-align:left;}}
 th{{background:#f5f5f5;}}
 .cat{{background:#f0f4f8;font-weight:700;}}
 .tip{{background:#eef7ff;padding:12px 14px;border-left:4px solid #3498db;font-size:13px;line-height:1.7;}}
</style></head><body>
<h1>宽基被动指数 ETF 可选清单（{len(BROAD_ETF)} 只，DB 已确认在库）</h1>
<div class="tip">
<b>怎么用：</b><br>
· 单标的：<code>--code 510300.SH</code><br>
· 自定义篮子：<code>--code 510300.SH,510500.SH,159915.SZ,510050.SH,512100.SH,159901.SH</code><br>
· 预设篮子：<code>--preset core6</code>（沪深300+中证500+创业板+上证50+中证1000+深证100，全历史可用）<br>
  其他预设：core4 / core3 / large3 / all_legacy（9只全历史）/ a500_4（中证A500四只，仅2024后区间）<br>
  红利系：<code>--preset dividend</code>（红利ETF+红利低波ETF 双子星）/ <code>--preset div_tech</code>（红利双子星+创业板，行情悲观时挪一点到科技放大弹性）<br>
  <b>均线增强</b>：加 <code>--mode smart</code> 启用「5周线/20周线操作法」（底部补仓、高估少投、涨回撤回）；详见《玩红利ETF的实战思路》精华。<br>
<b>注意成立时间：</b>中证A500 / 中证A50 / 双创50 / 中证2000 / 北证50 多为 2021 年后成立，跑 2018 全历史区间时它们尚未上市，会自动跳过（不买入），仅成立后参与。红利ETF(510880) 自2010年上市、红利低波ETF(512890) 自2019年上市，历史较长。
</div>
<table>
<tr><th>类别</th><th>代码</th><th>名称</th><th>状态</th></tr>
{''.join(rows)}
</table>
</body></html>"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  可选清单已生成：{out_path}")


def run_both(codes, start_date="20180101", end_date="20260715",
             monthly_amount=4000, weekly_amount=1000, save_csv=True, make_html=True,
             mode="plain", dip_band=SMART_DIP_BAND, over_band=SMART_OVERBAND,
             extra_mult=SMART_EXTRA_MULT):
    """跑月度+周度+一次性对比，生成 HTML 报告。"""
    print(">>> 月度定投 ...")
    monthly = run_backtest(codes, "monthly", monthly_amount, weekly_amount,
                           start_date, end_date, save_csv=save_csv, make_html=False,
                           mode=mode, dip_band=dip_band, over_band=over_band, extra_mult=extra_mult)
    print("\n>>> 周度定投 ...")
    weekly = run_backtest(codes, "weekly", monthly_amount, weekly_amount,
                          start_date, end_date, save_csv=save_csv, make_html=False,
                          mode=mode, dip_band=dip_band, over_band=over_band, extra_mult=extra_mult)
    if monthly and weekly:
        _print_comparison(monthly, weekly)
    if monthly and weekly and make_html:
        out_dir = os.path.join("data", "results", "dca_etf")
        os.makedirs(out_dir, exist_ok=True)
        key = monthly["code_key"]
        out_path = os.path.join(out_dir, f"dca_report_{key}_{start_date}_{end_date}.html")
        _build_html(monthly, weekly, out_path)
        print(f"\n  对比报告已生成：{out_path}")
        return out_path
    return None


def main():
    ap = argparse.ArgumentParser(description="宽基 ETF 定投(DCA)回测（支持篮子）")
    ap.add_argument("--freq", choices=["monthly", "weekly", "both"], default="both")
    ap.add_argument("--code", default="510300.SH",
                    help="单标的代码，或逗号分隔的多标的(篮子)，或预设名(core6等)")
    ap.add_argument("--preset", default=None, help="预设篮子名: core6/core4/core3/large3/all_legacy/a500_4")
    ap.add_argument("--start", default="20180101")
    ap.add_argument("--end", default="20260715")
    ap.add_argument("--monthly", type=float, default=4000)
    ap.add_argument("--weekly", type=float, default=1000)
    ap.add_argument("--no-csv", action="store_true")
    ap.add_argument("--no-html", action="store_true")
    ap.add_argument("--catalog", action="store_true", help="仅生成可选宽基 ETF 清单 HTML")
    ap.add_argument("--mode", choices=["plain", "smart"], default="plain",
                    help="定投模式: plain(普通纪律定投) / smart(均线增强·5周/20周线操作法)")
    ap.add_argument("--dip-band", type=float, default=SMART_DIP_BAND,
                    help="smart 专用：价格≤20周线*(1+该比例)视为底部补仓带（默认0.03）")
    ap.add_argument("--over-band", type=float, default=SMART_OVERBAND,
                    help="smart 专用：价格≥5周线*(1+该比例)视为高估少投带（默认0.10）")
    ap.add_argument("--extra-mult", type=float, default=SMART_EXTRA_MULT,
                    help="smart 专用：底部补仓最多额外部署=当期预算的倍数（默认1.0）")
    args = ap.parse_args()

    if args.catalog:
        out_dir = os.path.join("data", "results", "dca_etf")
        os.makedirs(out_dir, exist_ok=True)
        build_catalog_html(os.path.join(out_dir, "dca_etf_catalog.html"))
        return

    codes = resolve_codes(args.preset if args.preset else args.code)
    if not codes:
        return
    print(f"# 标的: {codes}  模式: {args.mode}")
    if args.freq == "both":
        run_both(codes, args.start, args.end, args.monthly, args.weekly,
                 save_csv=not args.no_csv, make_html=not args.no_html,
                 mode=args.mode, dip_band=args.dip_band, over_band=args.over_band,
                 extra_mult=args.extra_mult)
    else:
        run_backtest(codes, args.freq, args.monthly, args.weekly,
                     args.start, args.end,
                     save_csv=not args.no_csv, make_html=False,
                     mode=args.mode, dip_band=args.dip_band, over_band=args.over_band,
                     extra_mult=args.extra_mult)


if __name__ == "__main__":
    main()
