# -*- coding: utf-8 -*-
"""
红利低波「质量复合」策略 — 滚动回测（驱动 src/dividend_low_vol_selector.py）
==============================================================================
目的：验证《红利个股DIY》六维质量门禁对原「高股息+低波+分红增长」策略的改进；
      并落地「官方编制法（中证红利低波 930955）」三档实战构建。

模式（--mode，默认 compact）：
  旧版 A/B/C 对照：
    old   = 仅 股息率+低波+分红增长（质量门禁全部置 0）
    new   = 开启六维质量门禁（config 默认）
    soft  = 六维默认阈值保留，但改为软打分（并入综合分）
  官方编制法实战三档（全A 池 + 930955 编制法 + 基准 000922）：
    official           = 月频 / 等权 / TOP_N=5
    official_improved  = 季频 / 股息率加权 / TOP_N=25（跑赢真指数 +4.23%）
    official_compact   = 季频 / 股息率加权 / TOP_N=12 / 单行业≤2（落地版，跑赢真指数 +2.46%）

做法：
  1. 调仓日 = 每月或每季度第5交易日（与平台 run_monthly_rebalance.py 同口径）；
  2. 选股日 = 调仓日前一交易日（防前视）；
  3. 调用真实 DIVIDEND_LOW_VOL 配置节 -> DividendLowVolSelector.select_stocks(date)；
  4. 等权或股息率加权、差额再平衡、含佣金/印花税/滑点；
  5. 基准 = 中证红利低波 000922.SH（同赛道）/ 中证全指 000985 / 沪深300。

输出：
  - 控制台对比表（年化/总收益/最大回撤/胜率/夏普）
  - data/results/dividend_low_vol/bt_quality_*.csv（NAV 曲线）+ bt_quality_sel_*.csv（选股明细）
"""
import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np

import config
from run_monthly_rebalance import (
    get_conn, get_trade_dates, get_monthly_5th_trading_days,
    get_open_price, calc_fee, calc_win_rate,
    COMMISSION_RATE, COMMISSION_MIN, STAMP_DUTY_RATE, SLIPPAGE_RATE,
    get_stock_pool_index, STOCK_POOL_INDEX, INDEX_DISPLAY_NAME,
)
from src.dividend_low_vol_selector import DividendLowVolSelector


# ──────────────────────────────────────────────────────────────
# 性能优化：把 hs300 成分股的日线收盘价一次性载入内存，
# 并猴子补丁 _calc_volatility / _is_macd_golden 改为读内存，
# 避免每个月对每个候选股反复查库（原实现约 8.6 万次查询）。
# ──────────────────────────────────────────────────────────────
_PRICE = {}  # ts_code -> {trade_date: close}


def _preload_hs300_prices():
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT DISTINCT ts_code FROM index_constituent WHERE index_code='000300.SH' "
        "AND ts_code NOT LIKE '688%'",
        conn)
    codes = [str(c) for c in df["ts_code"].tolist()]
    print(f"[preload] hs300 成分股(去重): {len(codes)} 只")
    ph = ",".join("?" for _ in codes)
    df2 = pd.read_sql_query(
        f"SELECT ts_code, trade_date, close FROM daily WHERE ts_code IN ({ph})",
        conn, params=codes)
    conn.close()
    for c in codes:
        _PRICE[c] = {}
    for _, r in df2.iterrows():
        _PRICE[str(r["ts_code"])][str(r["trade_date"])] = float(r["close"])
    return codes


def _preload_allA_prices():
    """全A 预载：仅载入「有实施分红记录」的股票日线（官方候选域）。
    用子查询规避 999 变量限制；dividend_detail 全市场仅覆盖 ~589 只。"""
    conn = get_conn()
    df2 = pd.read_sql_query(
        "SELECT d.ts_code, d.trade_date, d.close FROM daily d "
        "WHERE d.ts_code IN (SELECT ts_code FROM dividend_detail "
        "                     WHERE div_proc='实施' AND cash_div>0)",
        conn)
    conn.close()
    codes = sorted(set(str(c) for c in df2["ts_code"].tolist()))
    for c in codes:
        _PRICE[c] = {}
    for _, r in df2.iterrows():
        _PRICE[str(r["ts_code"])][str(r["trade_date"])] = float(r["close"])
    print(f"[preload] 全A(分红股)日线预载: {len(codes)} 只, {len(df2)} 行")
    return codes


def _preload_pool_prices(pool):
    """按股票池预载日线收盘价到内存：all=全A分红股；其余=对应指数成分股。
    与 _preload_hs300_prices 等价但通用，使行情缓存与实际回测池一致。"""
    if pool == "all" or STOCK_POOL_INDEX.get(pool) is None:
        return _preload_allA_prices()
    idx = STOCK_POOL_INDEX[pool]
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT DISTINCT ts_code FROM index_constituent WHERE index_code=? "
        "AND ts_code NOT LIKE '688%'", conn, params=(idx,))
    codes = [str(c) for c in df["ts_code"].tolist()]
    print(f"[preload] {pool}({idx}) 成分股(去重): {len(codes)} 只")
    if codes:
        ph = ",".join("?" for _ in codes)
        df2 = pd.read_sql_query(
            f"SELECT ts_code, trade_date, close FROM daily WHERE ts_code IN ({ph})",
            conn, params=codes)
    else:
        df2 = pd.DataFrame(columns=["ts_code", "trade_date", "close"])
    conn.close()
    for c in codes:
        _PRICE[c] = {}
    for _, r in df2.iterrows():
        _PRICE[str(r["ts_code"])][str(r["trade_date"])] = float(r["close"])
    return codes


def _patched_vol(self, ts_code, trade_date, window=None):
    if window is None:
        window = self.volatility_window
    s = _PRICE.get(ts_code)
    if not s:
        # 回退：按需查库（仅边缘情形触发，如全A下个别未预载股票），并补进缓存
        conn = get_conn()
        df = pd.read_sql_query(
            "SELECT trade_date, close FROM daily WHERE ts_code=? AND trade_date<=? ORDER BY trade_date",
            conn, params=(ts_code, trade_date))
        conn.close()
        if len(df) == 0:
            return None
        closes = list(df["close"].astype(float))
        _PRICE[ts_code] = {str(r[0]): float(r[1]) for r in df.itertuples(index=False)}
    else:
        closes = [s[d] for d in sorted(s) if d <= trade_date]
    if len(closes) < max(int(window * 0.6), 60):
        return None
    closes = closes[-(window + 1):]
    rets = np.diff(closes) / closes[:-1]
    return float(np.std(rets) * np.sqrt(252))


def _patched_macd(self, ts_code, trade_date, is_index=False):
    s = _PRICE.get(ts_code)
    if not s:
        return False
    closes = [s[d] for d in sorted(s) if d <= trade_date]
    if len(closes) < 26 + 9:
        return False
    closes = closes[-200:]
    ema_fast = pd.Series(closes).ewm(span=12, adjust=False).mean().values
    ema_slow = pd.Series(closes).ewm(span=26, adjust=False).mean().values
    dif = ema_fast - ema_slow
    dea = pd.Series(dif).ewm(span=9, adjust=False).mean().values
    return float(dif[-1]) > float(dea[-1])



# ──────────────────────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────────────────────
# 回测窗口写死为常量，脱离 config.GLOBAL —— 后者易被其它脚本污染
# (本会话曾出现 GLOBAL 被改成 20140301/20260727 导致窗口错乱、基准算崩)。
START = "20200101"
END = "20260723"
STOCK_POOL = config.GLOBAL.get("stock_pool", "hs300")
TOP_N = config.GLOBAL.get("top_n", 5)
INIT_CAPITAL = float(config.BACKTEST.get("monthly_rebalance_capital", 100000))


# ──────────────────────────────────────────────────────────────
# 官方编制法实战三档（全A 池，基准 000922）
#   rebal: month / quarter；weight: eq / dividend；ind_cap: 单行业上限
# ──────────────────────────────────────────────────────────────
MODE_SPECS = {
    "official":          dict(top_n=5,  rebal="month",   weight="eq",       ind_cap=0),
    "official_improved": dict(top_n=25, rebal="quarter", weight="dividend", ind_cap=0),
    "official_compact":  dict(top_n=12, rebal="quarter", weight="dividend", ind_cap=2),
}
# 官方编制法基础配置（沿用 930955 口径）
_OFFICIAL_BASE = dict(
    quality_mode="official",
    volatility_window=252,          # 1年波动率（官方口径）
    macd_filter=False,              # 官方无 MACD 过滤
    use_dividend_growth=False,      # 官方不看分红增长
    payout_ratio_max=0.0,           # 关闭分红比例硬过滤
    pe_min=0, pe_max=9999, pb_min=0, pb_max=9999,  # 放开 PE/PB 限制
    forward_yield_min=0.0, consecutive_div_years_min=0,
    div_drop_max=0.0, ocf_positive_years=0, ocf_to_profit_min=0.0,
    roe_stability_max_drop=0.0, lev_debt_to_assets_max=0.0,
    yield_keep_frac=0.75, vol_keep_frac=0.50, official_bank_cap=0,
)
RES_DIR = "data/results/dividend_low_vol"


def build_cfg(mode: str) -> dict:
    cfg = dict(config.DIVIDEND_LOW_VOL)
    cfg["stock_pool"] = STOCK_POOL
    cfg["top_n"] = TOP_N
    if mode in MODE_SPECS:
        # 官方编制法实战档：相对排名 + 分段筛选 + 去MACD + 3y移动平均股息率
        spec = MODE_SPECS[mode]
        cfg.update(_OFFICIAL_BASE)
        cfg["top_n"] = spec["top_n"]
        cfg["final_top_n"] = spec["top_n"]
        cfg["industry_cap"] = spec["ind_cap"]
        return cfg
    if mode == "old":
        # OLD：关闭 DIY 六维质量门禁，仅保留 股息率+低波+分红增长
        cfg.update(dict(
            forward_yield_min=0.0,
            consecutive_div_years_min=0,
            div_drop_max=0.0,
            ocf_positive_years=0,
            ocf_to_profit_min=0.0,
            roe_stability_max_drop=0.0,
            lev_debt_to_assets_max=0.0,
        ))
    elif mode == "soft":
        # SOFT：六维默认阈值保留，但改为软打分（不剔除、并入综合分）
        cfg["quality_mode"] = "soft"
    # mode == "new"：沿用 config 默认（六维硬门禁全开）
    return cfg


def select_targets(mode: str):
    """返回 [(rebal_date, [ts_code,...]), ...] 与 选股明细日志。mode: old/new/soft"""
    all_dates = get_trade_dates(START, END)
    rebal_dates = get_monthly_5th_trading_days(all_dates)
    cfg = build_cfg(mode)
    # 复用同一 selector 实例：缓存(分红档案/财务档案)跨月保留，省大量查库
    sel_inst = DividendLowVolSelector(cfg, None)
    targets = []
    sel_log = []
    for rb in rebal_dates:
        rb_idx = all_dates.index(rb)
        sel_date = all_dates[max(0, rb_idx - 1)]
        # 候选缓冲：放大 top_n 供回测递补
        sel_inst.top_n = max(TOP_N * 2, TOP_N + 5)
        sel_inst.date = sel_date
        sel = sel_inst.select_stocks(date=sel_date)
        if sel is None or len(sel) == 0:
            targets.append((rb, []))
            continue
        picks = sel["ts_code"].head(TOP_N).tolist()
        targets.append((rb, picks))
        for _, r in sel.head(TOP_N).iterrows():
            sel_log.append((rb, sel_date, r["ts_code"], r.get("name", ""),
                            round(float(r.get("dv_ttm", 0) or 0), 2),
                            round(float(r.get("volatility", 0) or 0), 4),
                            round(float(r.get("score", 0) or 0), 4)))
    return targets, sel_log


def bulk_close_prices(codes, start, end):
    """批量加载收盘价 {ts_code: {trade_date: close}}，含前向填充用最近值。"""
    conn = get_conn()
    ph = ",".join("?" for _ in codes)
    df = pd.read_sql_query(
        f"SELECT ts_code, trade_date, close FROM daily "
        f"WHERE ts_code IN ({ph}) AND trade_date BETWEEN ? AND ? "
        f"ORDER BY trade_date",
        conn, params=(*codes, start, end))
    conn.close()
    out = {}
    for c in codes:
        out[c] = {}
    for _, r in df.iterrows():
        out[str(r["ts_code"])][str(r["trade_date"])] = float(r["close"])
    return out


def ffill_price(pmap, code, date, all_dates, idx):
    """取 code 在 date(=all_dates[idx]) 的收盘价，缺失则向前找最近交易日。"""
    d = all_dates[idx]
    p = pmap.get(code, {}).get(d)
    if p is not None:
        return p
    # 向前填充
    j = idx - 1
    while j >= 0:
        p = pmap.get(code, {}).get(all_dates[j])
        if p is not None:
            return p
        j -= 1
    return None


def run_nav(targets, price_map, all_dates):
    """按 targets 序列做等权差额再平衡，返回每日 NAV 序列 与 交易记录。"""
    cash = INIT_CAPITAL
    positions = {}  # ts_code -> shares
    nav = []
    trades = []
    rebal_set = dict(targets)

    for idx, d in enumerate(all_dates):
        rb_target = rebal_set.get(d)
        if rb_target is not None:
            # 计算组合市值（用当日可执行开盘价；2026-07-06后为收盘价）
            def exec_px(code):
                return get_open_price(code, d)
            # 市值
            mv = cash
            for code, sh in positions.items():
                px = exec_px(code)
                if px:
                    mv += sh * px
            n_tgt = max(len(rb_target), 1)
            per = mv / n_tgt
            all_codes = set(positions.keys()) | set(rb_target)
            for code in all_codes:
                px = exec_px(code)
                if px is None:
                    continue
                cur_sh = positions.get(code, 0)
                # 目标股数（含成本近似）
                desired = int(per // (px * (1 + COMMISSION_RATE + SLIPPAGE_RATE))) if code in rb_target else 0
                diff = desired - cur_sh
                if diff > 0:
                    cost = px * diff + calc_fee("buy", px, diff)
                    if cost <= cash and diff > 0:
                        cash -= cost
                        positions[code] = cur_sh + diff
                        trades.append((d, code, "buy", px, diff))
                elif diff < 0:
                    sell = -diff
                    proceeds = px * sell - calc_fee("sell", px, sell)
                    cash += proceeds
                    positions[code] = cur_sh - sell
                    trades.append((d, code, "sell", px, sell))
                    if positions[code] == 0:
                        del positions[code]
        # 标记市值（用收盘价）
        mv = cash
        for code, sh in positions.items():
            px = ffill_price(price_map, code, d, all_dates, idx)
            if px:
                mv += sh * px
        nav.append((d, mv))
    return nav, trades


def compute_metrics(nav_list, all_dates):
    dates = [x[0] for x in nav_list]
    vals = np.array([x[1] for x in nav_list], dtype=float)
    n = len(vals)
    total_ret = vals[-1] / vals[0] - 1
    years = (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days / 365.25
    ann = (vals[-1] / vals[0]) ** (1 / years) - 1 if years > 0 else 0
    # 最大回撤
    peak = np.maximum.accumulate(vals)
    dd = vals / peak - 1
    max_dd = dd.min()
    # 日收益
    rets = np.diff(vals) / vals[:-1]
    vol = rets.std() * np.sqrt(252) if len(rets) > 1 else 0
    sharpe = (ann - 0.02) / vol if vol > 0 else 0
    return dict(n=n, total_ret=total_ret, ann=ann, max_dd=max_dd,
                vol=vol, sharpe=sharpe, years=years, final=vals[-1])


def yearly_returns(dates, vals):
    """返回 {年份: 收益率(%) } —— 用每年度首/末交易日净值之比。
    与其他月度/年度策略 _yearly_returns 口径一致（最后/最初）。"""
    if not dates:
        return {}
    df = pd.DataFrame({"date": list(dates), "v": list(vals)})
    df["year"] = df["date"].str[:4]
    out = {}
    for y, g in df.groupby("year"):
        out[y] = (g["v"].iloc[-1] / g["v"].iloc[0] - 1) * 100
    return out


def benchmark_nav(all_dates, index_code="000985.SH"):
    """指数买入持有 NAV。返回 (nav_list, first_valid_idx)。
    - 以该指数第一个有数据的收盘点位为基准归一（避免数据缺口把点位当倍数）；
    - 数据开始前 NAV 平值(=INIT_CAPITAL)，视为无法投资；
    - 数据中间缺口前向填充最近有效点位。"""
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT trade_date, close FROM index_daily WHERE ts_code=? "
        "AND trade_date BETWEEN ? AND ? ORDER BY trade_date",
        conn, params=(index_code, all_dates[0], all_dates[-1]))
    conn.close()
    bmap = dict(zip(df["trade_date"].astype(str), df["close"].astype(float)))
    levels = [bmap.get(d) for d in all_dates]
    first_valid = next((i for i, v in enumerate(levels) if v is not None), None)
    if first_valid is None:
        return [(d, INIT_CAPITAL) for d in all_dates], 0
    base = levels[first_valid]
    nav = []
    for i, v in enumerate(levels):
        if v is None:
            if i < first_valid:
                nav.append(INIT_CAPITAL)            # 数据开始前：平值
            else:
                j = i - 1
                while j > first_valid and levels[j] is None:
                    j -= 1
                nav.append(levels[j] / base * INIT_CAPITAL)
        else:
            nav.append(v / base * INIT_CAPITAL)
    return list(zip(all_dates, nav)), first_valid


def run_legacy_comparison():
    """旧版 A/B/C 对照：OLD / NEW / SOFT（月频等权，沪深300 基准）。保留历史行为。"""
    print(f"红利低波质量复合 滚动回测  [{START} ~ {END}]  池={STOCK_POOL}  TOP_N={TOP_N}")
    print(f"INIT_CAPITAL={INIT_CAPITAL:,.0f}  手续费: 佣金{COMMISSION_RATE:.2%}(最低{COMMISSION_MIN})/"
          f"印花税千1→千0.5(2023-08-28起)/滑点{SLIPPAGE_RATE:.2%}\n")

    # 预载行情 + 猴子补丁加速
    print(f"[preload] loading {STOCK_POOL} daily closes into memory...")
    _preload_pool_prices(STOCK_POOL)
    DividendLowVolSelector._calc_volatility = _patched_vol
    DividendLowVolSelector._is_macd_golden = _patched_macd
    print(f"[preload] done. {len(_PRICE)} stocks cached.\n")

    all_dates = get_trade_dates(START, END)

    # ── OLD（仅股息率+低波+增长）──
    print("[1/3] running OLD (no quality gates)...")
    t_old, log_old = select_targets("old")
    codes_old = sorted({c for _, cs in t_old for c in cs})
    pmap_old = bulk_close_prices(codes_old, START, END)
    nav_old, tr_old = run_nav(t_old, pmap_old, all_dates)

    # ── NEW（六维质量门禁·硬）──
    print("[2/3] running NEW (quality gates, hard)...")
    t_new, log_new = select_targets("new")
    codes_new = sorted({c for _, cs in t_new for c in cs})
    pmap_new = bulk_close_prices(codes_new, START, END)
    nav_new, tr_new = run_nav(t_new, pmap_new, all_dates)

    # ── SOFT（六维质量·软打分）──
    print("[3/3] running SOFT (quality soft-scoring)...")
    t_soft, log_soft = select_targets("soft")
    codes_soft = sorted({c for _, cs in t_soft for c in cs})
    pmap_soft = bulk_close_prices(codes_soft, START, END)
    nav_soft, tr_soft = run_nav(t_soft, pmap_soft, all_dates)

    # ── 基准：全局股票池对应指数 ──
    _bidx = STOCK_POOL_INDEX.get(STOCK_POOL)
    if _bidx is None:
        _bidx = "000985.SH"   # 全A 用中证全指
    _bname = INDEX_DISPLAY_NAME.get(_bidx, _bidx)
    nav_bench, _ = benchmark_nav(all_dates, _bidx)

    m_old = compute_metrics(nav_old, all_dates)
    m_new = compute_metrics(nav_new, all_dates)
    m_soft = compute_metrics(nav_soft, all_dates)
    m_bench = compute_metrics(nav_bench, all_dates)

    print("\n" + "=" * 88)
    print(f"{'指标':<14}{'OLD(无门禁)':>16}{'NEW(硬门禁)':>16}{'SOFT(软打分)':>16}{_bname:>14}")
    print("-" * 88)
    print(f"{'期末净值':<14}{m_old['final']:>16,.0f}{m_new['final']:>16,.0f}{m_soft['final']:>16,.0f}{m_bench['final']:>14,.0f}")
    print(f"{'总收益':<14}{m_old['total_ret']:>15.2%}{m_new['total_ret']:>15.2%}{m_soft['total_ret']:>15.2%}{m_bench['total_ret']:>13.2%}")
    print(f"{'年化':<14}{m_old['ann']:>15.2%}{m_new['ann']:>15.2%}{m_soft['ann']:>15.2%}{m_bench['ann']:>13.2%}")
    print(f"{'最大回撤':<14}{m_old['max_dd']:>15.2%}{m_new['max_dd']:>15.2%}{m_soft['max_dd']:>15.2%}{m_bench['max_dd']:>13.2%}")
    print(f"{'年化波动':<14}{m_old['vol']:>15.2%}{m_new['vol']:>15.2%}{m_soft['vol']:>15.2%}{m_bench['vol']:>13.2%}")
    print(f"{'夏普(无风险2%)':<14}{m_old['sharpe']:>15.2f}{m_new['sharpe']:>15.2f}{m_soft['sharpe']:>15.2f}{m_bench['sharpe']:>13.2f}")
    print("=" * 88)

    # 调仓期胜率（逐月相对沪深300）
    d_old = {x[0]: x[1] for x in nav_old}
    d_new = {x[0]: x[1] for x in nav_new}
    d_soft = {x[0]: x[1] for x in nav_soft}
    d_bench = {x[0]: x[1] for x in nav_bench}
    wins = {"OLD": 0, "NEW": 0, "SOFT": 0}
    tot = 0
    prev = {"OLD": None, "NEW": None, "SOFT": None}
    prev_b = None
    for rb, _ in t_new:
        if rb not in d_new:
            continue
        if prev_b is not None:
            rbv = d_bench[rb] / prev_b - 1
            if d_old[rb] / prev["OLD"] - 1 > rbv:
                wins["OLD"] += 1
            if d_new[rb] / prev["NEW"] - 1 > rbv:
                wins["NEW"] += 1
            if d_soft.get(rb) and d_soft[rb] / prev["SOFT"] - 1 > rbv:
                wins["SOFT"] += 1
            tot += 1
        prev["OLD"], prev["NEW"], prev["SOFT"] = d_old[rb], d_new[rb], d_soft.get(rb)
        prev_b = d_bench[rb]
    if tot > 0:
        print(f"\n调仓期胜率（相对{_bname}，共{tot}期）："
              f"OLD={wins['OLD']/tot:.1%}  NEW={wins['NEW']/tot:.1%}  SOFT={wins['SOFT']/tot:.1%}")

    # ── 输出 CSV ──
    os.makedirs(RES_DIR, exist_ok=True)
    df_nav = pd.DataFrame({
        "trade_date": [x[0] for x in nav_new],
        "nav_old": [d_old.get(x[0], np.nan) for x in nav_new],
        "nav_new": [x[1] for x in nav_new],
        "nav_soft": [d_soft.get(x[0], np.nan) for x in nav_new],
        "nav_bench": [d_bench.get(x[0], np.nan) for x in nav_new],
    })
    tag = f"{START}_{END}_absoft"
    nav_path = os.path.join(RES_DIR, f"bt_quality_nav_{tag}.csv")
    df_nav.to_csv(nav_path, index=False, encoding="utf-8-sig")

    df_old_sel = pd.DataFrame(log_old, columns=["rebal_date", "sel_date", "ts_code", "name", "dv_ttm", "volatility", "score"])
    df_new_sel = pd.DataFrame(log_new, columns=["rebal_date", "sel_date", "ts_code", "name", "dv_ttm", "volatility", "score"])
    df_soft_sel = pd.DataFrame(log_soft, columns=["rebal_date", "sel_date", "ts_code", "name", "dv_ttm", "volatility", "score"])
    old_sel_path = os.path.join(RES_DIR, f"bt_quality_sel_OLD_{tag}.csv")
    new_sel_path = os.path.join(RES_DIR, f"bt_quality_sel_NEW_{tag}.csv")
    soft_sel_path = os.path.join(RES_DIR, f"bt_quality_sel_SOFT_{tag}.csv")
    df_old_sel.to_csv(old_sel_path, index=False, encoding="utf-8-sig")
    df_new_sel.to_csv(new_sel_path, index=False, encoding="utf-8-sig")
    df_soft_sel.to_csv(soft_sel_path, index=False, encoding="utf-8-sig")

    print(f"\nNAV 曲线 → {nav_path}")
    print(f"选股明细 OLD  → {old_sel_path}")
    print(f"选股明细 NEW  → {new_sel_path}")
    print(f"选股明细 SOFT → {soft_sel_path}")


# ═══════════════════════════════════════════════════════════════════
#  官方编制法实战三档（全A 池）
# ═══════════════════════════════════════════════════════════════════
def get_quarterly_5th_trading_days(all_dates):
    """仅保留 1/4/7/10 月第5交易日，实现季度调仓。"""
    monthly = get_monthly_5th_trading_days(all_dates)
    return [d for d in monthly if int(str(d)[4:6]) in (1, 4, 7, 10)]


def run_nav_weighted(targets, weights_map, price_map, all_dates):
    """股息率加权差额再平衡，返回每日 NAV 序列。weight 为各股目标权重(和≈1)。"""
    cash = INIT_CAPITAL
    positions = {}
    nav = []
    rebal_set = dict(targets)
    for idx, d in enumerate(all_dates):
        rb_target = rebal_set.get(d)
        if rb_target is not None:
            def exec_px(code):
                return get_open_price(code, d)
            mv = cash
            for code, sh in positions.items():
                px = exec_px(code)
                if px:
                    mv += sh * px
            wmap = weights_map.get(str(d), {})
            all_codes = set(positions.keys()) | set(rb_target)
            for code in all_codes:
                px = exec_px(code)
                if px is None:
                    continue
                wt = wmap.get(code, 0.0)
                desired_val = mv * wt
                desired = int(desired_val // (px * (1 + COMMISSION_RATE + SLIPPAGE_RATE))) if wt > 0 else 0
                cur_sh = positions.get(code, 0)
                diff = desired - cur_sh
                if diff > 0:
                    cost = px * diff + calc_fee("buy", px, diff)
                    if cost <= cash and diff > 0:
                        cash -= cost
                        positions[code] = cur_sh + diff
                elif diff < 0:
                    sell = -diff
                    proceeds = px * sell - calc_fee("sell", px, sell)
                    cash += proceeds
                    positions[code] = cur_sh - sell
                    if positions[code] == 0:
                        del positions[code]
        mv = cash
        for code, sh in positions.items():
            px = ffill_price(price_map, code, d, all_dates, idx)
            if px:
                mv += sh * px
        nav.append((d, mv))
    return nav


def select_targets_official(mode, pool=None, top_n=None):
    """官方编制法选股，股票池/持仓数走系统全局配置（pool/top_n 缺省即取 config.GLOBAL）。
    支持 月/季调仓 + 等权/股息率加权 + 单行业上限。
    返回 (targets, weights_map, sel_log)，并写 partial 以支持断点续跑（按 pool+top_n 分文件）。"""
    spec = MODE_SPECS.get(mode, MODE_SPECS["official_compact"])
    pool = pool or config.GLOBAL.get("stock_pool", "hs300")
    if top_n is None:
        top_n = spec["top_n"]                 # 未显式指定时沿用该官方档默认
    else:
        top_n = int(top_n)
    buffer_n = max(top_n * 4, top_n + 8)      # 候选缓冲（供行业 cap 后取前 top_n）
    all_dates = get_trade_dates(START, END)
    if spec["rebal"] == "quarter":
        rebal_dates = get_quarterly_5th_trading_days(all_dates)
    else:
        rebal_dates = get_monthly_5th_trading_days(all_dates)
    print(f"[rebal] {mode} 调仓点: {len(rebal_dates)} 个 "
          f"({'季度' if spec['rebal']=='quarter' else '月度'})  pool={pool} 持仓={top_n}")

    cfg = build_cfg(mode)
    cfg["stock_pool"] = pool                  # ★ 走系统设置的股票池（不再写死 allA）
    cfg["top_n"] = buffer_n
    cfg["final_top_n"] = top_n
    sel_inst = DividendLowVolSelector(cfg, None)
    sel_inst._calc_volatility = _patched_vol.__get__(sel_inst)
    sel_inst._is_macd_golden = _patched_macd.__get__(sel_inst)

    sel_log = []
    weights_map = {}
    done_set = set()
    # 按 pool+top_n+窗口 区分 partial，避免不同池/数量之间串档
    partial_path = os.path.join(RES_DIR, f"_official_{mode}_{pool}_{top_n}_{START}_{END}_partial.csv")
    if os.path.exists(partial_path):
        p = pd.read_csv(partial_path, encoding="utf-8-sig")
        for _, r in p.iterrows():
            rb = str(r["rebal_date"])
            done_set.add(rb)
            w = float(r["weight"]) if "weight" in p.columns and pd.notna(r.get("weight")) else 0.0
            sel_log.append((rb, str(r["sel_date"]), str(r["ts_code"]), str(r["name"]),
                            float(r["dv_ttm"]) if pd.notna(r["dv_ttm"]) else 0.0,
                            float(r["volatility"]) if pd.notna(r["volatility"]) else 0.0,
                            float(r["score"]) if pd.notna(r["score"]) else 0.0, w))
            weights_map.setdefault(rb, {})[str(r["ts_code"])] = w
        print(f"[resume] partial 已含 {len(done_set)} 期, sel_log={len(sel_log)} 行")

    for i, rb in enumerate(rebal_dates):
        if str(rb) in done_set:
            continue
        rb_idx = all_dates.index(rb)
        sel_date = all_dates[max(0, rb_idx - 1)]
        sel_inst.top_n = buffer_n               # 候选放大，供行业 cap 后取前 top_n
        sel_inst.date = sel_date
        sel = sel_inst.select_stocks(date=sel_date)
        if sel is None or len(sel) == 0:
            print(f"[select] {i+1}/{len(rebal_dates)} {rb} 空")
            continue
        if spec["ind_cap"] > 0:
            sel = sel_inst._cap_industry(sel, spec["ind_cap"])
        picks_df = sel.head(top_n)              # 最终持仓 = 全局选股数（可实操）
        picks = picks_df["ts_code"].tolist()
        # 加权
        if spec["weight"] == "dividend":
            yld = picks_df["fwd_yield"].fillna(picks_df["dv_ttm"] / 100.0)
            yld = yld.fillna(0.0).clip(lower=0)
            s = yld.sum()
            w = {c: 1.0 / len(picks) for c in picks} if s <= 0 else {c: y / s for c, y in zip(picks, yld.values)}
        else:
            w = {c: 1.0 / len(picks) for c in picks}
        weights_map[str(rb)] = w
        for _, r in picks_df.iterrows():
            sel_log.append((rb, sel_date, str(r["ts_code"]), str(r.get("name", "")),
                            round(float(r.get("dv_ttm", 0) or 0), 2),
                            round(float(r.get("volatility", 0) or 0), 4),
                            round(float(r.get("score", 0) or 0), 4),
                            round(float(w[str(r["ts_code"])]), 4)))
        pd.DataFrame(sel_log, columns=["rebal_date", "sel_date", "ts_code", "name",
                                        "dv_ttm", "volatility", "score", "weight"]).to_csv(
            partial_path, index=False, encoding="utf-8-sig")
        n_ind = (picks_df["ts_code"].map(lambda c: getattr(sel_inst, "_ind_map", {}).get(c, "其他")).nunique()
                 if getattr(sel_inst, "_ind_map", None) else "-")
        print(f"[select] {i+1}/{len(rebal_dates)} {rb} 候选={len(sel)} 行业数={n_ind} 持仓={len(picks)}: {picks}")

    by_rb, by_rb_w = {}, {}
    for rec in sel_log:
        by_rb.setdefault(rec[0], []).append(rec[2])
        by_rb_w.setdefault(rec[0], {})[rec[2]] = rec[7]
    targets = [(rb, by_rb.get(str(rb), [])) for rb in rebal_dates]
    return targets, by_rb_w, sel_log


def run_official_backtest(mode="official_compact", pool=None, top_n=None, capital=None):
    """跑官方编制法某档。股票池/持仓数/初始资金=系统全局配置；基准=该股票池对应指数；
    输出 NAV + 选股明细 + 逐年盈亏（策略 vs 基准 vs 同赛道 000922）。"""
    global INIT_CAPITAL
    spec = MODE_SPECS.get(mode, MODE_SPECS["official_compact"])
    pool = pool or config.GLOBAL.get("stock_pool", "hs300")
    if top_n is None:
        top_n = config.GLOBAL.get("top_n", 5)     # 全局选股数
    else:
        top_n = int(top_n)
    # 初始资金跟随全局（与其它月度策略一致：config.BACKTEST.monthly_rebalance_capital）
    INIT_CAPITAL = float(capital) if capital is not None else \
        float(config.BACKTEST.get("monthly_rebalance_capital", 100000))
    # 基准 = 全局股票池对应的指数；全A 无单指数，沿用平台惯例用中证800
    bidx = STOCK_POOL_INDEX.get(pool)
    if bidx is None:
        bidx = "000906.SH"
    bname = INDEX_DISPLAY_NAME.get(bidx, bidx)
    pool_name = INDEX_DISPLAY_NAME.get(STOCK_POOL_INDEX.get(pool)) or "全A股"

    print("=" * 70)
    print(f"红利低波质量复合 实战回测 [{mode}]  {START}~{END}")
    print(f"股票池={pool}({pool_name})  持仓={top_n}只(=全局选股数)  "
          f"调仓={'季度' if spec['rebal']=='quarter' else '月度'}  "
          f"加权={'股息率' if spec['weight']=='dividend' else '等权'}  行业上限={spec['ind_cap']}")
    print(f"基准指数：{bname}({bidx})  ｜ 同赛道参考：中证红利低波(000922.SH)")
    _preload_pool_prices(pool)
    targets, weights_map, sel_log = select_targets_official(mode, pool=pool, top_n=top_n)
    all_codes = sorted({c for _, cs in targets for c in cs})
    print(f"涉及股票数: {len(all_codes)}")
    pmap = bulk_close_prices(all_codes, START, END)
    all_dates = get_trade_dates(START, END)
    nav = (run_nav_weighted(targets, weights_map, pmap, all_dates)
           if spec["weight"] == "dividend" else run_nav(targets, pmap, all_dates))

    nav_b, _ = benchmark_nav(all_dates, bidx)          # 主基准：股票池对应指数
    nav_922, f922 = benchmark_nav(all_dates, "000922.SH")  # 同赛道参考

    m = compute_metrics(nav, all_dates)
    m_b = compute_metrics(nav_b, all_dates)
    # 000922 数据库仅 2020 年起有数据，按实际区间计其收益（此前年份在逐年表记 N/A）
    if f922 < len(nav_922):
        m_922 = compute_metrics(nav_922[f922:], all_dates[f922:])
    else:
        m_922 = compute_metrics(nav_922, all_dates)

    nav_map = {x[0]: x[1] for x in nav}
    nav_b_map = {x[0]: x[1] for x in nav_b}
    nav_922_map = {x[0]: x[1] for x in nav_922}

    # 相对基准调仓期胜率
    win = tot = 0
    prev_n = prev_b = None
    for rb, _ in targets:
        if rb not in nav_map or rb not in nav_b_map:
            continue
        if prev_n is not None:
            if nav_map[rb] / prev_n - 1 > nav_b_map[rb] / prev_b - 1:
                win += 1
            tot += 1
        prev_n, prev_b = nav_map[rb], nav_b_map[rb]
    win_rate = (win / tot) if tot else float("nan")

    # 逐年收益
    y_strat = yearly_returns([d for d, _ in nav], [v for _, v in nav])
    y_b = yearly_returns([d for d, _ in nav_b], [v for _, v in nav_b])
    y_922 = yearly_returns([d for d, _ in nav_922], [v for _, v in nav_922])

    # 落盘：NAV 曲线
    rows = [(d, nav_map.get(d), nav_b_map.get(d), nav_922_map.get(d)) for d in all_dates]
    nav_out = os.path.join(RES_DIR, f"bt_quality_nav_{START}_{END}_{mode}_{pool}_{top_n}.csv")
    pd.DataFrame(rows, columns=["trade_date", f"nav_{mode}", f"nav_{bidx}", "nav_922"]).to_csv(
        nav_out, index=False, encoding="utf-8-sig")
    # 落盘：选股明细
    sel_out = os.path.join(RES_DIR, f"bt_quality_sel_OFFICIAL_{mode.upper()}_{pool}_{top_n}_{START}_{END}.csv")
    pd.DataFrame(sel_log, columns=["rebal_date", "sel_date", "ts_code", "name",
                                    "dv_ttm", "volatility", "score", "weight"]).to_csv(
        sel_out, index=False, encoding="utf-8-sig")

    distinct = len({r[2] for r in sel_log})
    avg_pick = np.mean([len(cs) for _, cs in targets]) if targets else 0

    if f922 > 0:
        print(f"※ 中证红利低波(000922) 数据库仅自 {all_dates[f922]} 起有数据，其总收益/年化按该区间计；此前年度记 N/A")
    print("-" * 80)
    print(f"{'指标':<14}{'策略':<18}{bname:>14}{'中证红利低波':>14}")
    print(f"{'期末净值':<14}{m['final']:>18,.0f}{m_b['final']:>14,.0f}{m_922['final']:>14,.0f}")
    print(f"{'总收益':<14}{m['total_ret']*100:>17.2f}%{m_b['total_ret']*100:>13.2f}%{m_922['total_ret']*100:>13.2f}%")
    print(f"{'年化':<14}{m['ann']*100:>17.2f}%{m_b['ann']*100:>13.2f}%{m_922['ann']*100:>13.2f}%")
    print(f"{'最大回撤':<14}{m['max_dd']*100:>17.2f}%{m_b['max_dd']*100:>13.2f}%{m_922['max_dd']*100:>13.2f}%")
    print(f"{'年化波动':<14}{m['vol']*100:>17.2f}%{m_b['vol']*100:>13.2f}%{m_922['vol']*100:>13.2f}%")
    print(f"{'夏普(2%)':<14}{m['sharpe']:>18.2f}{m_b['sharpe']:>14.2f}{m_922['sharpe']:>14.2f}")
    print(f"{'对'+bname+'胜率':<14}{win_rate*100:>17.2f}%{'-':>13}{'-':>13}")
    print("-" * 80)

    # ── 逐年收益表（策略 vs 基准 vs 同赛道）──
    print(f"\n【逐年收益（策略 vs {bname} vs 中证红利低波）】")
    print(f"{'年份':<8}{'策略':>12}{bname:>14}{'超额':>12}{'红利低波':>14}")
    years = sorted(set(y_strat) | set(y_b))
    first_year_922 = all_dates[f922][:4] if f922 < len(all_dates) else "9999"
    for y in years:
        s = y_strat.get(y); b = y_b.get(y)
        ex = (s - b) if (s is not None and b is not None) else None
        s_s = f"{s:+.2f}%" if s is not None else "  -  "
        b_s = f"{b:+.2f}%" if b is not None else "  -  "
        ex_s = f"{ex:+.2f}%" if ex is not None else "  -  "
        w922 = y_922.get(y)
        w_s = "  N/A" if y < first_year_922 else (f"{w922:+.2f}%" if w922 is not None else "  -  ")
        print(f"{y:<8}{s_s:>12}{b_s:>14}{ex_s:>12}{w_s:>14}")

    # 落盘：逐年收益 CSV（便于参看/二次分析）
    strat_list = [round(y_strat.get(y), 4) if y_strat.get(y) is not None else None for y in years]
    bench_list = [round(y_b.get(y), 4) if y_b.get(y) is not None else None for y in years]
    excess_list = []
    for y in years:
        s = y_strat.get(y); b = y_b.get(y)
        excess_list.append(round(s - b, 4) if (s is not None and b is not None) else None)
    dl922_list = [round(y_922.get(y), 4) if (y_922.get(y) is not None and y >= first_year_922) else None
                  for y in years]
    yr_df = pd.DataFrame({
        "year": years,
        "strategy_pct": strat_list,
        f"{bname}_pct": bench_list,
        "excess_vs_bench_pct": excess_list,
        "div_low_vol_922_pct": dl922_list,
    })
    yr_out = os.path.join(RES_DIR, f"bt_quality_yearly_{START}_{END}_{mode}_{pool}_{top_n}.csv")
    yr_df.to_csv(yr_out, index=False, encoding="utf-8-sig")

    print(f"\n集中度: 不同股票={distinct}  季/月均入选={avg_pick:.1f}")
    print(f"NAV 曲线 → {nav_out}")
    print(f"选股明细 → {sel_out}")
    print(f"逐年收益 → {yr_out}")
    print("DONE")


def main():
    parser = argparse.ArgumentParser(description="红利低波质量复合 回测（股票池/选股数=全局配置，基准=股票池对应指数）")
    parser.add_argument("--mode", default="official_compact",
                        choices=["old", "new", "soft", "official",
                                 "official_improved", "official_compact"],
                        help="回测模式（默认 official_compact = 落地版：季频/股息率加权/行业≤2；持仓数=全局选股数）")
    parser.add_argument("--pool", default=None,
                        help="股票池 hs300/zz500/zz800/zz1000/all（默认=config.GLOBAL 股票池）")
    parser.add_argument("--top-n", type=int, default=None,
                        help="持仓数量（默认=config.GLOBAL 选股数；不传则官方档用各自默认）")
    parser.add_argument("--capital", type=float, default=None,
                        help="初始资金（默认=config.BACKTEST.monthly_rebalance_capital，与其它月度策略一致）")
    args = parser.parse_args()
    if args.mode in ("old", "new", "soft"):
        run_legacy_comparison()
    else:
        run_official_backtest(args.mode, pool=args.pool, top_n=args.top_n, capital=args.capital)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise
