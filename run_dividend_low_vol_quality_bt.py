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
        "SELECT DISTINCT ts_code FROM index_constituent WHERE index_code='000300.SH' ",
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
        "SELECT DISTINCT ts_code FROM index_constituent WHERE index_code=? ",
        conn, params=(idx,))
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


# ──────────────────────────────────────────────────────────────
# 红利通道仓位 overlay（P0 实验）：用 000922 中证红利低波 的「通道位置」
# 决定权益仓位系数 k∈[k_min,k_max]。便宜(通道底)→满仓(k_max)；贵(通道顶)→减仓(k_min)。
# 通道位置严格因果（仅用 sel_date 之前的数据），无前视。
# ──────────────────────────────────────────────────────────────
_IDX922 = {}   # trade_date -> close (000922.SH)


def _preload_index_channel(ts_code="000922.SH"):
    """一次性把指数收盘价载入内存（与 _preload_pool_prices 同款内存加速）。"""
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT trade_date, close FROM index_daily WHERE ts_code=? ORDER BY trade_date",
        conn, params=(ts_code,))
    conn.close()
    for _, r in df.iterrows():
        _IDX922[str(r["trade_date"])] = float(r["close"])
    print(f"[channel] {ts_code} 预载 {len(_IDX922)} 行")
    return ts_code


def _latest_trade_date():
    """库里日线数据的最新交易日（真正的『今天』），用于 live-forward 前向选股。"""
    conn = get_conn()
    d = pd.read_sql_query("SELECT MAX(trade_date) AS d FROM daily", conn)["d"].iloc[0]
    conn.close()
    return str(d)


def _channel_pos(sel_date, mode="rolling", window=756, bottom=None, top=None):
    """000922 红利通道位置 pos∈[0,1]（0=通道底/便宜，1=通道顶/贵）。
    - rolling: sel_date 前 window 个交易日内 min/max 构成的通道（因果无前视）；
    - fixed:   固定 bottom/top 线。
    历史不足或 min==max（无展开）→ 返回 None（调用方视作不动作，k=k_max 满仓）。"""
    if mode == "fixed":
        if bottom is None or top is None or top <= bottom:
            return None
        c = _IDX922.get(sel_date)
        if c is None:
            return None
        return max(0.0, min(1.0, (c - bottom) / (top - bottom)))
    # rolling
    dates = sorted(d for d in _IDX922 if d <= sel_date)
    if len(dates) < min(252, window):
        return None
    vals = [_IDX922[d] for d in dates[-window:]]
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-6:
        return None
    c = _IDX922.get(sel_date)
    if c is None:
        return None
    return max(0.0, min(1.0, (c - lo) / (hi - lo)))


def _make_coef_fn(overlay, mode="rolling", window=756, bottom=None, top=None,
                  k_min=0.5, k_max=1.0):
    """返回 coef_fn(sel_date)->k。overlay 关闭时恒为 1.0（满仓，与基线完全等价）。"""
    if not overlay:
        return lambda sel_date: 1.0

    def coef(sel_date):
        pos = _channel_pos(sel_date, mode, window, bottom, top)
        if pos is None:
            return 1.0
        return k_max - (k_max - k_min) * pos   # 便宜(pos→0)→k_max；贵(pos→1)→k_min

    return coef


def _sel_date_of(all_dates, idx):
    """调仓日 d=all_dates[idx] 对应的选股日（前一日，与 select_targets 口径一致）。"""
    return all_dates[idx - 1] if idx > 0 else all_dates[idx]


def _print_live_signal(targets, weights_map, sel_log, coef_fn, all_dates, capital,
                       channel_mode, channel_window, channel_bottom, channel_top,
                       k_min, k_max, overlay):
    """--live：打印最近一期调仓的可执行买列表（通道系数缩放 + 现金比例）。
    仅做历史最近一期调仓的「可读化」，不重新选股、不含前视。"""
    last = next((t for t in reversed(targets) if t[1]), None)
    if last is None:
        print("[live] 无有效调仓记录，无法生成买列表")
        return
    rb_last, codes = last
    idx = all_dates.index(rb_last) if rb_last in all_dates else len(all_dates) - 1
    sel_date = _sel_date_of(all_dates, idx)
    k = coef_fn(sel_date)
    wmap = weights_map.get(str(rb_last), {})
    name_map = {str(r[2]): str(r[3]) for r in sel_log if str(r[0]) == str(rb_last)}
    print("\n" + "=" * 72)
    print(f"【LIVE 买列表 · 最近一期调仓 {rb_last}（选股日 {sel_date}）】")
    print(f"通道系数 k = {k:.3f}  →  权益 {k*100:.1f}% / 现金 {(1-k)*100:.1f}%")
    if capital:
        print(f"初始资金 {capital:,.0f}  →  投入权益 {capital*k:,.0f} / 留现金 {capital*(1-k):,.0f}")
    print("-" * 72)
    print(f"{'代码':<10}{'名称':<12}{'股息权重':>10}{'目标权益权重':>16}{'目标金额':>14}")
    for c in codes:
        w = wmap.get(c, 0.0)
        wk = w * k
        amt = capital * wk if capital else None
        nm = name_map.get(c, "")
        amt_s = f"{amt:>12,.0f}" if amt is not None else f"{'':>14}"
        print(f"{c:<10}{nm:<12}{w*100:>9.2f}%{wk*100:>15.2f}%{amt_s:>14}")
    if overlay and _IDX922:
        latest = max(_IDX922)
        pos_now = _channel_pos(latest, channel_mode, channel_window, channel_bottom, channel_top)
        if pos_now is not None:
            k_now = k_max - (k_max - k_min) * pos_now
            trend = "更贵" if k_now < k else ("更便宜" if k_now > k else "持平")
            print("-" * 72)
            print(f"※ 当前通道(最新 {latest})：pos={pos_now:.2f} → 若今日调仓 k={k_now:.3f}(权益{k_now*100:.1f}%)")
            print(f"  对比上次调仓 k={k:.3f}：通道较那时{trend}")
    print("=" * 72)
    print("[live] 上表为最近一期『历史』调仓，非今日前瞻选股；实操请以最新一期重选为准。")


def _forward_select(mode, pool, top_n, as_of_date):
    """前向选股：以 as_of_date 为选股日重跑 selector（不写历史 partial、不影响回测）。
    返回 (picks_df, sel_date)；无候选返回 (None, as_of_date)。"""
    spec = MODE_SPECS.get(mode, MODE_SPECS["official_compact"])
    pool = pool or config.GLOBAL.get("stock_pool", "hs300")
    if top_n is None:
        top_n = spec["top_n"]
    else:
        top_n = int(top_n)
    buffer_n = max(top_n * 4, top_n + 8)      # 候选缓冲（供行业 cap 后取前 top_n）
    cfg = build_cfg(mode)
    cfg["stock_pool"] = pool                  # ★ 走系统设置的股票池
    cfg["top_n"] = buffer_n
    cfg["final_top_n"] = top_n
    sel_inst = DividendLowVolSelector(cfg, None)
    sel_inst._calc_volatility = _patched_vol.__get__(sel_inst)
    sel_inst._is_macd_golden = _patched_macd.__get__(sel_inst)
    sel_inst.top_n = buffer_n
    sel_inst.date = as_of_date
    sel = sel_inst.select_stocks(date=as_of_date)
    if sel is None or len(sel) == 0:
        return None, as_of_date
    if spec["ind_cap"] > 0:
        sel = sel_inst._cap_industry(sel, spec["ind_cap"])
    picks_df = sel.head(top_n)                # 最终持仓 = 全局选股数（可实操）
    return picks_df, as_of_date


def _print_live_signal_forward(mode, pool, top_n, capital, coef_fn, as_of_date,
                               channel_mode, channel_window, channel_bottom, channel_top,
                               k_min, k_max, overlay):
    """--live-forward：以 as_of_date(今日) 为选股日重跑 selector，打印真正『今天该买什么』。"""
    picks_df, sel_date = _forward_select(mode, pool, top_n, as_of_date)
    if picks_df is None or len(picks_df) == 0:
        print(f"\n[live-forward] 以 {as_of_date} 为选股日重选失败（无候选），"
              f"请检查数据库是否含该日行情/财务。")
        return
    k = coef_fn(sel_date)
    spec = MODE_SPECS.get(mode, MODE_SPECS["official_compact"])
    if spec["weight"] == "dividend":
        yld = picks_df["fwd_yield"].fillna(picks_df["dv_ttm"] / 100.0)
        yld = yld.fillna(0.0).clip(lower=0)
        s = yld.sum()
        w = ({c: 1.0 / len(picks_df.index) for c in picks_df["ts_code"]}
             if s <= 0 else {c: y / s for c, y in zip(picks_df["ts_code"], yld.values)})
    else:
        w = {c: 1.0 / len(picks_df.index) for c in picks_df["ts_code"]}
    print("\n" + "=" * 72)
    print(f"【LIVE 前瞻买列表 · 选股日(as_of) {sel_date}】")
    print(f"通道系数 k = {k:.3f}  →  权益 {k*100:.1f}% / 现金 {(1-k)*100:.1f}%")
    if capital:
        print(f"初始资金 {capital:,.0f}  →  投入权益 {capital*k:,.0f} / 留现金 {capital*(1-k):,.0f}")
    print("-" * 72)
    print(f"{'代码':<10}{'名称':<12}{'股息权重':>10}{'目标权益权重':>16}{'目标金额':>14}")
    for _, r in picks_df.iterrows():
        c = str(r["ts_code"])
        ww = w[c]
        wk = ww * k
        amt = capital * wk if capital else None
        nm = str(r.get("name", ""))
        amt_s = f"{amt:>12,.0f}" if amt is not None else f"{'':>14}"
        print(f"{c:<10}{nm:<12}{ww*100:>9.2f}%{wk*100:>15.2f}%{amt_s:>14}")
    if overlay and _IDX922:
        pos = _channel_pos(sel_date, channel_mode, channel_window, channel_bottom, channel_top)
        if pos is not None:
            print("-" * 72)
            print(f"※ 选股日通道位置 pos={pos:.2f}（0=通道底/便宜，1=通道顶/贵）→ k={k:.3f}")
            print(f"  评估：{'红利指数偏贵，已减仓至权益 ' + format(k*100, '.1f') + '%' if k < 1.0 else '红利指数偏便宜，满仓'}")
        else:
            print("-" * 72)
            print("※ 通道历史不足，k 按满仓(1.0) 处理")
    print("=" * 72)
    print(f"[live-forward] 上表为以 {sel_date} 重新选股的前瞻清单，可直接用于建仓/调仓。")


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

# ── 计价口径开关（2026-09-01 新增）────────────────────────────────────────
# raw = daily.close 未复权（历史行为）：除息日价格下跌被当作真实亏损，
#       且分红从不计入 NAV → 对红利策略系统性低估收益，不可与全收益基准比较。
# hfq = close * adj_factor 后复权（含分红再投）：与 index_tr_official 全收益基准可比。
# adj_factor 取「当日」值，与平台 run_monthly_rebalance 同口径（无前视）。
# ✅ 2026-09-02 起默认 hfq（raw 变 opt-in）：hfq 才是与全收益基准可比的真实总回报口径。
#    本策略 raw→hfq 年化差 +3.67pp（实测）；用 raw 配全收益基准会误判成"跑输"（见报告 §8/§12.17）。
#    要复现 2026-09-02 之前的历史数字请用 --price-mode raw。
PRICE_MODE = "hfq"
EXEC_PMAP = {}          # hfq 模式下的成交价表 {ts_code: {trade_date: open*adj}}，与估值同价格空间
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
    """批量加载收盘价 {ts_code: {trade_date: close}}，含前向填充用最近值。

    PRICE_MODE='hfq' 时返回后复权价 = close * adj_factor（含分红再投），
    与 index_tr_official 全收益基准可比；adj_factor 取当日值，无前视。
    PRICE_MODE='raw'（默认）返回未复权收盘价，保持历史行为以便复现旧结果。
    """
    conn = get_conn()
    ph = ",".join("?" for _ in codes)
    if PRICE_MODE == "hfq":
        sql = (f"SELECT d.ts_code, d.trade_date, d.close AS px, a.adj_factor AS adj "
               f"FROM daily d "
               f"LEFT JOIN adj_factor a "
               f"       ON a.ts_code = d.ts_code AND a.trade_date = d.trade_date "
               f"WHERE d.ts_code IN ({ph}) AND d.trade_date BETWEEN ? AND ? "
               f"ORDER BY d.trade_date")
    else:
        sql = (f"SELECT ts_code, trade_date, close AS px FROM daily "
               f"WHERE ts_code IN ({ph}) AND trade_date BETWEEN ? AND ? "
               f"ORDER BY trade_date")
    df = pd.read_sql_query(sql, conn, params=(*codes, start, end))
    conn.close()
    out = {}
    for c in codes:
        out[c] = {}
    if PRICE_MODE != "hfq" or df.empty:
        for _, r in df.iterrows():
            out[str(r["ts_code"])][str(r["trade_date"])] = float(r["px"])
        return out
    df = df.sort_values(["ts_code", "trade_date"])
    df["adj"] = df.groupby("ts_code")["adj"].ffill().fillna(1.0)
    ref = {c: (float(g["adj"].iloc[0]) or 1.0) for c, g in df.groupby("ts_code")}
    for _, r in df.iterrows():
        c = str(r["ts_code"])
        out[c][str(r["trade_date"])] = float(r["px"]) * float(r["adj"]) / ref[c]
    return out


def bulk_open_prices(codes, start, end):
    """批量加载后复权开盘价 {ts_code: {trade_date: open*adj/adj_ref}}，用于 hfq 成交价。

    必须与 bulk_close_prices(hfq) 处在同一价格空间，否则会出现
    「按 raw 价买入、按 hfq 价估值」的虚增（实测可放大 30 倍）。
    """
    conn = get_conn()
    ph = ",".join("?" for _ in codes)
    sql = (f"SELECT d.ts_code, d.trade_date, d.open AS px, a.adj_factor AS adj "
           f"FROM daily d "
           f"LEFT JOIN adj_factor a "
           f"       ON a.ts_code = d.ts_code AND a.trade_date = d.trade_date "
           f"WHERE d.ts_code IN ({ph}) AND d.trade_date BETWEEN ? AND ? "
           f"ORDER BY d.trade_date")
    df = pd.read_sql_query(sql, conn, params=(*codes, start, end))
    conn.close()
    out = {c: {} for c in codes}
    if df.empty:
        return out
    # adj_factor 表存在「整交易日缺行」（2020-2026 共 132 天，全市场同缺），
    # 但 adj_factor 是阶跃函数（仅除权除息日变化），故前向填充在数学上是精确的，不丢信息。
    # 切忌 fillna(1.0)——那会让缺失日价格掉回 raw、次日跳回 hfq，制造巨额假跳空。
    df = df.sort_values(["ts_code", "trade_date"])
    df["adj"] = df.groupby("ts_code")["adj"].ffill().fillna(1.0)
    # 归一化基准：各股在窗口内首个交易日的因子。
    # 目的——hfq 绝对价位会被累计因子放大到几十~上百倍（如格力 raw 67.9 → hfq 10589），
    # 而回测按整股下单 int(per // px)，放大会导致买不进整股、资金全趴现金的伪现金拖累。
    # 归一化后价格量级≈raw，同时保留分红带来的相对增长。
    ref = {c: (float(g["adj"].iloc[0]) or 1.0) for c, g in df.groupby("ts_code")}
    for _, r in df.iterrows():
        c = str(r["ts_code"])
        out[c][str(r["trade_date"])] = float(r["px"]) * float(r["adj"]) / ref[c]
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


def run_nav(targets, price_map, all_dates, coef_fn=None):
    """按 targets 序列做等权差额再平衡，返回每日 NAV 序列 与 交易记录。
    coef_fn(sel_date)->仓位系数k∈(0,1]：红利通道 overlay（None=常满仓，与基线等价）。"""
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
                if PRICE_MODE == "hfq":
                    return ffill_price(EXEC_PMAP, code, d, all_dates, idx)
                return get_open_price(code, d)
            # 市值
            mv = cash
            for code, sh in positions.items():
                px = exec_px(code)
                if px:
                    mv += sh * px
            # 红利通道仓位系数：仅把 k 比例的市值铺进权益，余下留现金
            k = coef_fn(_sel_date_of(all_dates, idx)) if coef_fn else 1.0
            n_tgt = max(len(rb_target), 1)
            per = mv * k / n_tgt
            # 必须排序：买入受 `cost <= cash` 现金约束，集合迭代顺序（受 PYTHONHASHSEED
            # 随机化）会改变成交顺序进而改变组合 —— 这是回测非确定性的根因（同 livermore v2）。
            all_codes = sorted(set(positions.keys()) | set(rb_target))
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
    calmar = ann / abs(max_dd) if max_dd < 0 else 0.0   # 卡玛 = 年化 / |最大回撤|
    return dict(n=n, total_ret=total_ret, ann=ann, max_dd=max_dd,
                vol=vol, sharpe=sharpe, years=years, final=vals[-1], calmar=calmar)


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


_LAST_BENCH_META = {}   # index_code -> bench_index meta（供报告打印口径标签）


def benchmark_nav(all_dates, index_code="000985.SH"):
    """指数买入持有 NAV。返回 (nav_list, first_valid_idx)。
    - 以该指数第一个有数据的收盘点位为基准归一（避免数据缺口把点位当倍数）；
    - 数据开始前 NAV 平值(=INIT_CAPITAL)，视为无法投资；
    - 数据中间缺口前向填充最近有效点位。

    ⚠️ 2026-09-01 改：**基准口径必须跟随 PRICE_MODE**（两端同含或同不含分红）。
      旧实现直连 `index_daily`（价格指数），hfq 那跑会出现
      「NAV 含分红 vs 基准不含分红」→ 超额被**系统性高估约 2.5%/年**。
      现改走 `bench_index` 统一真相源：raw→价格指数，hfq→全收益（官方/自建）。
    """
    import bench_index as bi
    conn = get_conn()
    df, _bmeta = bi.load_benchmark(index_code, all_dates[0], all_dates[-1],
                                   conn=conn, nav_price_mode=PRICE_MODE)
    conn.close()
    # 口径错配告警（如中证全指暂无全收益而回退价格指数时，会在此显式提示）
    _warn = bi.check_consistency(PRICE_MODE, _bmeta)
    if _warn:
        print(f"  ⚠️ 基准 {index_code}: {_warn}")
    _LAST_BENCH_META[index_code] = _bmeta      # 供报告打印口径标签
    bmap = ({} if df is None or df.empty else
            dict(zip(df["trade_date"].astype(str), df["close"].astype(float))))
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
    if PRICE_MODE == "hfq":
        EXEC_PMAP.clear(); EXEC_PMAP.update(bulk_open_prices(codes_old, START, END))
    nav_old, tr_old = run_nav(t_old, pmap_old, all_dates)

    # ── NEW（六维质量门禁·硬）──
    print("[2/3] running NEW (quality gates, hard)...")
    t_new, log_new = select_targets("new")
    codes_new = sorted({c for _, cs in t_new for c in cs})
    pmap_new = bulk_close_prices(codes_new, START, END)
    if PRICE_MODE == "hfq":
        EXEC_PMAP.clear(); EXEC_PMAP.update(bulk_open_prices(codes_new, START, END))
    nav_new, tr_new = run_nav(t_new, pmap_new, all_dates)

    # ── SOFT（六维质量·软打分）──
    print("[3/3] running SOFT (quality soft-scoring)...")
    t_soft, log_soft = select_targets("soft")
    codes_soft = sorted({c for _, cs in t_soft for c in cs})
    pmap_soft = bulk_close_prices(codes_soft, START, END)
    if PRICE_MODE == "hfq":
        EXEC_PMAP.clear(); EXEC_PMAP.update(bulk_open_prices(codes_soft, START, END))
    nav_soft, tr_soft = run_nav(t_soft, pmap_soft, all_dates)

    # ── 基准：全局股票池对应指数 ──
    _bidx = STOCK_POOL_INDEX.get(STOCK_POOL)
    if _bidx is None:
        _bidx = "000985.SH"   # 全A 用中证全指
    _bname = INDEX_DISPLAY_NAME.get(_bidx, _bidx)
    nav_bench, _ = benchmark_nav(all_dates, _bidx)
    # 基准口径标签（透明标注：告诉读者这列数字含不含分红）
    _bmeta = _LAST_BENCH_META.get(_bidx) or {}
    _blabel = (_bmeta.get("note") or "?")
    print(f"  NAV 口径：{PRICE_MODE}"
          f"{'（含分红再投）' if PRICE_MODE == 'hfq' else '（不含分红）'}"
          f" | 基准 {_bname} [{_blabel}]")

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
def get_periodic_5th_trading_days(all_dates, months):
    """仅保留 months 集合内月份的第5交易日（返回有序 list，跨进程可复现）。"""
    monthly = get_monthly_5th_trading_days(all_dates)
    return [d for d in monthly if int(str(d)[4:6]) in months]


def get_quarterly_5th_trading_days(all_dates):
    """仅保留 1/4/7/10 月第5交易日，实现季度调仓。"""
    return get_periodic_5th_trading_days(all_dates, (1, 4, 7, 10))


def get_official_annual_rebal_days(all_dates):
    """中证红利低波(H30269)官方年度调仓日 = 每年 12 月第二个星期五的**下一交易日**。

    ⚠️ 与季度/半年度用的"第5交易日"口径不同：第5交易日约在 12/5~12/7，
    官方口径约在 12/11~12/17（晚 1~2 周）。照抄官方年度调仓时须用本函数，
    不要用 get_periodic_5th_trading_days(all_dates, (12,))，否则口径对不上官方。
    """
    s = pd.to_datetime(pd.Series([str(d) for d in all_dates]))
    df = pd.DataFrame({"trade_date": [str(d) for d in all_dates], "dt": s})
    out = []
    for ym, g in df.groupby(s.dt.strftime("%Y%m"), sort=True):
        if not str(ym).endswith("12"):
            continue
        fris = sorted(d for d in g["dt"] if d.weekday() == 4)
        if len(fris) < 2:
            continue                      # 该年 12 月不足两个周五（数据起点被截断）→ 跳过
        after = g[g["dt"] > fris[1]]
        if len(after):
            out.append(after.iloc[0]["trade_date"])
    return out


# rebal 取值 → (调仓日生成函数参数, 显示名)
REBAL_SPECS = {
    "month":   (None,            "月度"),
    "quarter": ((1, 4, 7, 10),   "季度"),
    "half":    ((6, 12),         "半年度"),
    # 年度走官方口径（12月第二个周五后一交易日），不是"12月第5交易日"
    "year":    (None,            "年度(官方12月二周五后)"),
}


def run_nav_weighted(targets, weights_map, price_map, all_dates, coef_fn=None):
    """股息率加权差额再平衡，返回每日 NAV 序列。weight 为各股目标权重(和≈1)。
    coef_fn(sel_date)->仓位系数k：红利通道 overlay（None=常满仓）。"""
    cash = INIT_CAPITAL
    positions = {}
    nav = []
    rebal_set = dict(targets)
    for idx, d in enumerate(all_dates):
        rb_target = rebal_set.get(d)
        if rb_target is not None:
            def exec_px(code):
                if PRICE_MODE == "hfq":
                    return ffill_price(EXEC_PMAP, code, d, all_dates, idx)
                return get_open_price(code, d)
            mv = cash
            for code, sh in positions.items():
                px = exec_px(code)
                if px:
                    mv += sh * px
            k = coef_fn(_sel_date_of(all_dates, idx)) if coef_fn else 1.0
            wmap = weights_map.get(str(d), {})
            # 必须排序：买入受 `cost <= cash` 现金约束，集合迭代顺序（受 PYTHONHASHSEED
            # 随机化）会改变成交顺序进而改变组合 —— 这是回测非确定性的根因（同 livermore v2）。
            all_codes = sorted(set(positions.keys()) | set(rb_target))
            for code in all_codes:
                px = exec_px(code)
                if px is None:
                    continue
                wt = wmap.get(code, 0.0)
                desired_val = mv * wt * k
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


def _ensure_ind_map(sel_inst):
    """确保 sel_inst._ind_map 已载入（B7 缓冲递补时的行业上限要用）。

    🔴 不能靠 _cap_industry(df, 0) 触发载入：_cap_industry 在 cap<=0 时直接早退，
       不会建立 _ind_map（曾导致 buffer_k>0 全部报错）。必须显式载入。"""
    if not getattr(sel_inst, "_ind_map", None):
        conn = sel_inst._get_conn()
        im = pd.read_sql_query("SELECT ts_code, industry FROM stock_basic", conn)
        conn.close()
        sel_inst._ind_map = {
            str(r["ts_code"]): (str(r["industry"]) if pd.notna(r["industry"]) else "其他")
            for _, r in im.iterrows()
        }
    return sel_inst._ind_map


def select_targets_official(mode, pool=None, top_n=None, buffer_k=0, turnover_cap=0.0,
                            final_key="fwd_yield"):
    """官方编制法选股，股票池/持仓数走系统全局配置（pool/top_n 缺省即取 config.GLOBAL）。
    支持 月/季/半年/年调仓 + 等权/股息率加权 + 单行业上限。

    final_key（2026-09-03 新增，opt-in，默认 "fwd_yield" = 旧行为）：
      最后一段筛子（行业 cap 之后取前 top_n）的排序键。
        "fwd_yield"  → 股息率降序（历史行为）
        "volatility" → 波动率升序（🔴 官方 930955 口径：股息率前 300 → 波动率升序取前 100）
      🔴 波动率是慢变量、股息率是快变量 → 换键预计大幅提高留存率、压低换手。

    buffer_k>0 时启用「指数编制法缓冲规则(B7)」：上期持仓若仍排进前 top_n+buffer_k 名
    （候选 rank 序，行业 cap 前口径）则保留不动，空出的位置按排名递补（递补时执行行业上限）。
    这是指数复制降换手的正统手段；buffer_k=0 保持原行为（纯重排名，无缓冲）。

    turnover_cap>0 时启用「B7' 官方同款调整比例硬上限」（与 buffer_k **互斥**，同时给则
    本开关优先、buffer_k 被忽略）。以**上期持仓**为起点，在「新进只数 ≤ max_change」的
    预算内把组合往本期最优改进，而不是"先重排再砍"：
      ① 老持仓（仍在本期候选池内）按 rank 保留，受行业上限约束；
      ② 老持仓坐不满 → **强制**递补（被动换仓，无法避免）；
      ③ 剩余预算内做改善型替换：最好的候补 ↔ 应被踢出者
         （目标行业已满 → 踢同行业最差，否则踢全局最差），换不出改善即停；
      ④ 恒真：持仓 = top_n、行业 ≤ ind_cap；每笔替换严格降低 rank 之和 → 必然收敛。
    依据：中证红利低波动指数(H30269)「每次调整的样本数量一般不超过样本总数的 20%」。
    🔴 官方的低换手**不是靠"一年调一次"实现的**，靠的就是这道硬上限。
    🔴 已自证判死的两种错法（勿回退）：
       (a) "全量重排→砍新进者→回补老持仓"：回补被 ind_cap 挡住 → 组合塌缩（12→5 只）；
       (b) 用 select_stocks 的 score 序当 rank：裸档最终按 **fwd_yield** 排序（见
           _cap_industry），口径对不上 → 首期就选出完全不同的 12 只。
    返回 (targets, weights_map, sel_log)，并写 partial 以支持断点续跑
    （按 pool+top_n+buffer_k+turnover_cap+rebal 分文件）。"""
    spec = MODE_SPECS.get(mode, MODE_SPECS["official_compact"])
    pool = pool or config.GLOBAL.get("stock_pool", "hs300")
    if top_n is None:
        top_n = spec["top_n"]                 # 未显式指定时沿用该官方档默认
    else:
        top_n = int(top_n)
    buffer_n = max(top_n * 4, top_n + 8)      # 候选缓冲（供行业 cap 后取前 top_n）
    all_dates = get_trade_dates(START, END)
    # rebal: month / quarter / half / year（half=6·12月第5交易日；year=官方12月二周五后一交易日）
    # CLI 可用 --rebal 覆盖 MODE_SPECS 的设定（opt-in，默认不变）
    _rb = spec["rebal"]
    if _rb not in REBAL_SPECS:
        raise ValueError(f"未知 rebal={_rb!r}，可选 {list(REBAL_SPECS)}")
    if _rb == "year":
        rebal_dates = get_official_annual_rebal_days(all_dates)
    elif _rb == "month":
        rebal_dates = get_monthly_5th_trading_days(all_dates)
    else:
        rebal_dates = get_periodic_5th_trading_days(all_dates, REBAL_SPECS[_rb][0])
    # 🔴 低频档（half/year）首个调仓日可能距回测起点 5~11 个月 → 前段**空仓**，
    # 年化被系统性低估，与季度档不可比。统一插入一次"期初建仓日"（第5个交易日，
    # 与其它档同口径）。季度/月度档 rebal_dates[0] 距起点仅数日，gap<=20 不触发（行为不变）。
    if rebal_dates:
        _gap = all_dates.index(rebal_dates[0]) if rebal_dates[0] in all_dates else 10 ** 6
        if _gap > 20:
            _bd = all_dates[4]                 # 第 5 个交易日（与月度/季度档同口径）
            rebal_dates = [_bd] + rebal_dates
            print(f"[rebal] 🔴 低频档首个调仓日距起点 {_gap} 个交易日（前段会空仓、年化被低估）"
                  f" → 插入期初建仓日 {_bd}")
    _cap_on = bool(turnover_cap and turnover_cap > 0)
    if _cap_on and buffer_k:
        print(f"[warn] 同时给了 buffer_k={buffer_k} 与 turnover_cap={turnover_cap} → "
              f"二者互斥，本次以 turnover_cap 为准（buffer_k 被忽略）")
    print(f"[rebal] {mode} 调仓点: {len(rebal_dates)} 个 "
          f"({REBAL_SPECS[_rb][1]})  pool={pool} 持仓={top_n}"
          f"  buffer_k={buffer_k}{'(缓冲规则开: 保留带=top_n+%d)' % buffer_k if buffer_k > 0 else '(关: 纯重排名)'}")
    # 🔴 样本量守卫：调仓点过少时收益/回撤/夏普无统计鉴别力（配对 t 必然不显著），
    #    只有换手是可信的（换手降幅是算术恒等式，不依赖样本量）。
    #    接入回测平台后用户可能随手传 2020 起点 → 年度档只剩 6 期，
    #    必须显式红牌，不能静默产出"看起来很好但不可信"的数字。
    if len(rebal_dates) < 10:
        print(f"[rebal] 🔴🔴 样本量不足：本窗口只有 {len(rebal_dates)} 个调仓点（<10）")
        print("        收益 / 回撤 / 夏普在此样本量下无统计鉴别力，**只有换手可信**。")
        print("        建议窗口：年度档 ≥13 年（20130101 起，此前股息覆盖率仅 46~62% 会饿死候选池）"
              "／半年档 ≥7 年／季度档 ≥3 年。")
    if _cap_on:
        print(f"[B7'] 调整比例硬上限 = {turnover_cap:.0%} → 每次最多新进 "
              f"max(1, int({top_n}×{turnover_cap})) = {max(1, int(top_n * turnover_cap))} 只")

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
    # 按 pool+top_n+buffer_k+窗口 区分 partial，避免不同参数之间串档
    # 🔴 频率也进文件名：--rebal half/year 与默认 quarter 会互相静默覆盖（第 7 次同类坑）
    _rbtag = f"_rb{spec['rebal']}" if spec["rebal"] != "quarter" else ""
    # 🔴 换手硬上限也进文件名（第 9 次同类坑）：cap 档与裸档会静默互相覆盖
    _tctag = f"_tc{int(round(turnover_cap * 100))}" if _cap_on else ""
    # 🔴 串档坑第 10 次：最后一段排序键也必须进文件名，否则 vol 档会覆盖 yield 档产物
    _fktag = "_kv" if final_key == "volatility" else ""
    partial_path = os.path.join(RES_DIR, f"_official_{mode}_{pool}_{top_n}"
                                         f"_bk{buffer_k}{_tctag}{_fktag}{_rbtag}_{START}_{END}_partial.csv")
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

    # 缓冲规则需要上一期持仓：含 partial 续跑（weights_map 里已有历史各期）
    picks_by_rb = {rb: set(w.keys()) for rb, w in weights_map.items()}
    ind_map = None
    if buffer_k > 0 or _cap_on:
        # 行业上限在递补时执行，需要行业映射（惰性载入一次）
        ind_map = _ensure_ind_map(sel_inst)
        ind_cap = spec["ind_cap"]

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
        _extra = ""
        if _cap_on:
            # ── B7' 官方同款「每次调整比例 ≤ turnover_cap」硬上限 ──
            # 🔴 第一版写法（全量重排→砍新进者→回补老持仓）已自证**判死**：
            #    砍完要回补的老持仓大量是银行股，被 ind_cap=2 挡住 → 组合从 12 只
            #    **塌缩到 5 只**（2023 冒烟实测 12→11→5→6）。故改为「以老持仓为起点」：
            #   ① 老持仓（仍在本期候选池内）按 rank 保留，受行业上限约束；
            #   ② 老持仓坐不满 → **强制**递补（被动换仓，无法避免，计入新进但不占预算判断）；
            #   ③ 剩余预算内做改善型替换：最好的候补 ↔ 应被踢出者
            #      （目标行业已满 → 踢同行业最差；否则踢全局最差），换不出改善即停。
            #   恒真：持仓 = top_n、行业 ≤ ind_cap；每笔替换都严格降低 rank 之和 → 必然收敛。
            prev_picks = picks_by_rb.get(str(rebal_dates[i - 1]), set()) if i > 0 else set()
            # 🔴 rank 序必须与裸档**完全一致**：裸档走 `_cap_industry(sel, ind_cap).head(top_n)`，
            #    而 **_cap_industry 内部按 fwd_yield 降序重排**（不是 select_stocks 的 score 序！）
            #    再按行业取前 cap。若这里拿 score 序当 rank，连首期（无老持仓）都会选出
            #    完全不同的 12 只（2023 冒烟实测：仅 4 只重合）→ cap 档无法与裸档对齐。
            # 🔴 候选池用**行业上限前**的全量（48 只）、上限只约束最终组合：
            #    _cap_industry 会把池子砍到 26 只，老持仓大量掉出池外 → 被动换仓激增，
            #    硬上限形同虚设。两者在首期等价（贪心按 fwd_yield 取前 top_n 且行业≤cap）。
            _sr = sel.copy()
            _sr["ts_code"] = _sr["ts_code"].astype(str)
            _sr = _sr.sort_values(final_key,
                                  ascending=(final_key != "fwd_yield")).reset_index(drop=True)
            cand = [str(c) for c in _sr["ts_code"]]
            rank = {c: j for j, c in enumerate(cand)}        # rank = fwd_yield 序（裸档口径）
            prev_avail = {c for c in cand if c in prev_picks}   # 老持仓中仍合格的
            max_change = max(1, int(top_n * turnover_cap))

            def _ic(c):
                return ind_map.get(str(c), "其他")

            cur, ind_cnt = [], {}
            # ① 老持仓按 rank 保留（上期组合已满足 ind_cap → 正常情况全部保住）
            for c in cand:
                if len(cur) >= top_n:
                    break
                if c not in prev_avail:
                    continue
                _k = _ic(c)
                if ind_cap > 0 and ind_cnt.get(_k, 0) >= ind_cap:
                    continue
                ind_cnt[_k] = ind_cnt.get(_k, 0) + 1
                cur.append(c)
            # ② 强制递补到坐满（老持仓不足 / 被行业上限挤掉）
            cur_set = set(cur)
            for c in cand:
                if len(cur) >= top_n:
                    break
                if c in cur_set:
                    continue
                _k = _ic(c)
                if ind_cap > 0 and ind_cnt.get(_k, 0) >= ind_cap:
                    continue
                ind_cnt[_k] = ind_cnt.get(_k, 0) + 1
                cur.append(c)
                cur_set.add(c)
            # ③ 预算内改善型替换（每轮取全局最优候补，rank 和严格下降 → 收敛）
            while True:
                swapped = False
                for best in cand:
                    if best in cur_set:
                        continue
                    _kb = _ic(best)
                    pool = [c for c in cur if _ic(c) == _kb] \
                        if (ind_cap > 0 and ind_cnt.get(_kb, 0) >= ind_cap) else list(cur)
                    if not pool:
                        continue
                    v = max(pool, key=lambda c: rank[c])       # rank 大 = 差
                    if rank[best] >= rank[v]:
                        continue                               # 换不出改善
                    trial = [c for c in cur if c != v] + [best]
                    if i > 0 and len([c for c in trial if c not in prev_picks]) > max_change:
                        continue                               # 预算用完（但可能还有老持仓可换回）
                    cur.remove(v)
                    cur_set.discard(v)
                    ind_cnt[_ic(v)] -= 1
                    cur.append(best)
                    cur_set.add(best)
                    ind_cnt[_kb] = ind_cnt.get(_kb, 0) + 1
                    swapped = True
                if not swapped:
                    break
            final_codes = sorted(cur, key=lambda c: rank[c])   # 按 rank 复原（下游依赖 rank 序）
            sel_r = _sr
            picks_df = sel_r[sel_r["ts_code"].isin(final_codes)].drop_duplicates("ts_code", keep="first")
            picks_df = picks_df.set_index("ts_code").loc[final_codes].reset_index()
            picks = picks_df["ts_code"].tolist()
            n_kept = None
            n_new = len([c for c in final_codes if c not in prev_picks]) if i > 0 else len(final_codes)
            _over = " ⚠超预算(强制)" if (i > 0 and n_new > max_change) else ""
            _extra = f" 新进={n_new}/{max_change}{_over}"
        elif buffer_k > 0:
            # ── B7 指数编制法缓冲规则 ──
            prev_picks = picks_by_rb.get(str(rebal_dates[i - 1]), set()) if i > 0 else set()
            prev_picks = {c for c in prev_picks if c in set(sel["ts_code"])}
            sel_r = sel.reset_index(drop=True)   # rank 序 = select_stocks 输出序（score 降序）
            keep_band = top_n + buffer_k
            kept = sel_r[sel_r["ts_code"].isin(prev_picks)].head(max(0, keep_band)) \
                if keep_band > 0 else sel_r.iloc[0:0]
            kept = kept[kept.index < keep_band]  # 上期持仓仍排进 keep_band 内才保留
            kept_codes = kept["ts_code"].tolist()
            kept_set = set(kept_codes)
            # 递补：按 rank 序遍历非保留候选，行业计数 = 已保留 + 已递补
            ind_cnt = {}
            if ind_cap > 0:
                for c in kept_codes:
                    _ic = ind_map.get(str(c), "其他")
                    ind_cnt[_ic] = ind_cnt.get(_ic, 0) + 1
            fill = []
            for _, r in sel_r.iterrows():
                if len(kept_codes) + len(fill) >= top_n:
                    break
                c = str(r["ts_code"])
                if c in kept_set:
                    continue
                if ind_cap > 0:
                    _ic = ind_map.get(c, "其他")
                    if ind_cnt.get(_ic, 0) >= ind_cap:
                        continue
                    ind_cnt[_ic] = ind_cnt.get(_ic, 0) + 1
                fill.append(c)
            final_codes = kept_codes + fill
            picks_df = sel_r[sel_r["ts_code"].isin(final_codes)].drop_duplicates("ts_code", keep="first")
            picks_df = picks_df.set_index("ts_code").loc[final_codes].reset_index()
            picks = picks_df["ts_code"].tolist()
            n_kept = len(kept_codes)
        else:
            if spec["ind_cap"] > 0:
                sel = sel_inst._cap_industry(sel, spec["ind_cap"], sort_key=final_key)
            picks_df = sel.head(top_n)              # 最终持仓 = 全局选股数（可实操）
            picks = picks_df["ts_code"].tolist()
            n_kept = None
        # 加权
        if spec["weight"] == "dividend":
            yld = picks_df["fwd_yield"].fillna(picks_df["dv_ttm"] / 100.0)
            yld = yld.fillna(0.0).clip(lower=0)
            s = yld.sum()
            w = {c: 1.0 / len(picks) for c in picks} if s <= 0 else {c: y / s for c, y in zip(picks, yld.values)}
        else:
            w = {c: 1.0 / len(picks) for c in picks}
        weights_map[str(rb)] = w
        picks_by_rb[str(rb)] = set(picks)       # 缓冲规则：下一期沿用本期持仓
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
        _kb = f" 保留={n_kept}" if n_kept is not None else ""
        print(f"[select] {i+1}/{len(rebal_dates)} {rb} 候选={len(sel)} 行业数={n_ind} "
              f"持仓={len(picks)}{_kb}{_extra}: {picks}")

    by_rb, by_rb_w = {}, {}
    for rec in sel_log:
        by_rb.setdefault(rec[0], []).append(rec[2])
        by_rb_w.setdefault(rec[0], {})[rec[2]] = rec[7]
    targets = [(rb, by_rb.get(str(rb), [])) for rb in rebal_dates]
    return targets, by_rb_w, sel_log


def run_official_backtest(mode="official_compact", pool=None, top_n=None, capital=None,
                          overlay=True, channel_mode="rolling", channel_window=756,
                          channel_bottom=None, channel_top=None, k_min=0.5, k_max=1.0,
                          live=False, live_forward=False, buffer_k=0, turnover_cap=0.0,
                          rebal=None, final_key="fwd_yield"):
    """跑官方编制法某档。股票池/持仓数/初始资金=系统全局配置；基准=该股票池对应指数；
    输出 NAV + 选股明细 + 逐年盈亏（策略 vs 基准 vs 同赛道 000922）。

    红利通道仓位 overlay 默认开启（rolling/w756/k0.5）：在官方 DL 选股之上叠加一层
    按 000922 通道位置的权益仓位调节（贵→减仓留现金、便宜→满仓），属风控/仓位层、
    不改选股。需复现普通满仓红利低波基线请用 --no-div-channel-overlay。

    rebal=None 表示沿用 MODE_SPECS 里该 mode 的默认频率（quarter）。
    传 "year"/"half" 可显式降频（降换手第一杠杆，详见 divlow_b7_demystify.md §3.3）。
    🔴 实现上必须改 MODE_SPECS 本体（全局），不能只改局部副本 ——
       select_targets_official 内部也会读 MODE_SPECS[mode]["rebal"]（同 PRICE_MODE 那类坑）。
    """
    global INIT_CAPITAL
    spec = MODE_SPECS.get(mode, MODE_SPECS["official_compact"])
    # 调仓频率覆盖（显式参数 > MODE_SPECS 默认）。平台层调用请走这个形参，
    # 不要自己改 dlq.MODE_SPECS —— 那是可变全局状态，同一进程跑多档会互相污染。
    if rebal:
        if rebal not in REBAL_SPECS:
            raise ValueError(f"未知 rebal={rebal!r}，可选 {list(REBAL_SPECS)}")
        MODE_SPECS[mode]["rebal"] = rebal
        print(f"[rebal] 覆盖 {mode} 的调仓频率 → {rebal}（{REBAL_SPECS[rebal][1]}）")
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
          f"调仓={REBAL_SPECS.get(spec['rebal'], (None, spec['rebal']))[1]}  "
          f"加权={'股息率' if spec['weight']=='dividend' else '等权'}  行业上限={spec['ind_cap']}")
    print(f"基准指数：{bname}({bidx})  ｜ 同赛道参考：中证红利低波(000922.SH)")
    if overlay:
        _cm = (f"rolling(窗口{channel_window}交易日)"
               if channel_mode == "rolling"
               else f"fixed[{channel_bottom},{channel_top}]")
        print(f"红利通道 overlay：开  模式={_cm}  仓位系数 k∈[{k_min},{k_max}]（便宜→{k_max}/贵→{k_min}）")
    else:
        print(f"红利通道 overlay：关（满仓基线＝普通红利低波，无任何仓位调节）")
    if overlay or live or live_forward:
        _preload_index_channel("000922.SH")
    coef_fn = _make_coef_fn(overlay, channel_mode, channel_window,
                            channel_bottom, channel_top, k_min, k_max)
    _preload_pool_prices(pool)
    targets, weights_map, sel_log = select_targets_official(
        mode, pool=pool, top_n=top_n, buffer_k=buffer_k, turnover_cap=turnover_cap,
        final_key=final_key)
    all_codes = sorted({c for _, cs in targets for c in cs})
    print(f"涉及股票数: {len(all_codes)}")
    pmap = bulk_close_prices(all_codes, START, END)
    if PRICE_MODE == "hfq":
        EXEC_PMAP.clear()
        EXEC_PMAP.update(bulk_open_prices(all_codes, START, END))
    all_dates = get_trade_dates(START, END)
    nav = (run_nav_weighted(targets, weights_map, pmap, all_dates, coef_fn)
           if spec["weight"] == "dividend" else run_nav(targets, pmap, all_dates, coef_fn))

    nav_b, _ = benchmark_nav(all_dates, bidx)          # 主基准：股票池对应指数
    nav_922, f922 = benchmark_nav(all_dates, "000922.SH")  # 同赛道参考
    # 口径透明标注：告诉读者这两列数字含不含分红（与 NAV 同口径才可比）
    import bench_index as _bi
    _mb = _LAST_BENCH_META.get(bidx) or {}
    _m922 = _LAST_BENCH_META.get("000922.SH") or {}
    print(f"  NAV 口径：{PRICE_MODE}"
          f"{'（含分红再投）' if PRICE_MODE == 'hfq' else '（不含分红）'}"
          f" | 基准 {INDEX_DISPLAY_NAME.get(bidx, bidx)} [{_mb.get('note', '?')}]"
          f" | 参考 中证红利低波 [{_m922.get('note', '?')}]")
    for _ic, _m_ in ((bidx, _mb), ("000922.SH", _m922)):
        _w = _bi.check_consistency(PRICE_MODE, _m_)
        if _w:
            print(f"    ⚠️ 基准 {INDEX_DISPLAY_NAME.get(_ic, _ic)}: {_w}")

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
    # hfq 结果加 _hfq 后缀，raw 保持原文件名（compare_tr_benchmark.py 依赖该路径，勿改）
    # 🔴 缓冲档必须进文件名：buffer_k>0 时若沿用原名，各档会静默互相覆盖（第 6 次同类坑）。
    _pmtag = "_hfq" if PRICE_MODE == "hfq" else ""
    _bktag = f"_bk{buffer_k}" if buffer_k else ""
    _rbtag = f"_rb{spec['rebal']}" if spec["rebal"] != "quarter" else ""
    # 🔴 换手硬上限也进文件名（第 9 次同类坑）：cap 档与裸档会静默互相覆盖
    _tctag = f"_tc{int(round(turnover_cap * 100))}" if (turnover_cap and turnover_cap > 0) else ""
    # 🔴 串档坑第 10 次：最后一段排序键（yield/vol）也必须进落盘名
    _fktag = "_kv" if final_key == "volatility" else ""
    nav_out = os.path.join(RES_DIR, f"bt_quality_nav_{START}_{END}_{mode}_{pool}_{top_n}"
                                    f"{_bktag}{_tctag}{_fktag}{_rbtag}{_pmtag}.csv")
    pd.DataFrame(rows, columns=["trade_date", f"nav_{mode}", f"nav_{bidx}", "nav_922"]).to_csv(
        nav_out, index=False, encoding="utf-8-sig")
    # 落盘：选股明细
    sel_out = os.path.join(RES_DIR, f"bt_quality_sel_OFFICIAL_{mode.upper()}_{pool}_{top_n}"
                                    f"{_bktag}{_tctag}{_fktag}{_rbtag}_{START}_{END}.csv")
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
    print(f"{'卡玛(Calmar)':<14}{m['calmar']:>18.2f}{m_b['calmar']:>14.2f}{m_922['calmar']:>14.2f}")
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
    # 🔴 缓冲档与调仓频率都必须进文件名（第 8 次同类坑）：bk>0 或 --rebal half/year
    #    若沿用裸名，各档逐年收益会静默互相覆盖，对照表直接作废。
    yr_out = os.path.join(RES_DIR, f"bt_quality_yearly_{START}_{END}_{mode}_{pool}_{top_n}"
                                   f"{_bktag}{_tctag}{_fktag}{_rbtag}{_pmtag}.csv")
    yr_df.to_csv(yr_out, index=False, encoding="utf-8-sig")

    print(f"\n集中度: 不同股票={distinct}  季/月均入选={avg_pick:.1f}")
    print(f"NAV 曲线 → {nav_out}")
    print(f"选股明细 → {sel_out}")
    print(f"逐年收益 → {yr_out}")
    if live:
        _print_live_signal(targets, weights_map, sel_log, coef_fn, all_dates, capital,
                           channel_mode, channel_window, channel_bottom, channel_top,
                           k_min, k_max, overlay)
    if live_forward:
        _print_live_signal_forward(mode, pool, top_n, capital, coef_fn, _latest_trade_date(),
                                   channel_mode, channel_window, channel_bottom, channel_top,
                                   k_min, k_max, overlay)
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
    # ── 红利通道仓位 overlay（P0 实验）──
    parser.add_argument("--div-channel-overlay", action="store_true",
                        help="开启红利通道仓位 overlay（现已默认开启，此开关可省略）；按000922通道位置缩放权益仓位")
    parser.add_argument("--no-div-channel-overlay", action="store_true",
                        help="关闭红利通道仓位 overlay，回退为满仓基线（＝普通红利低波策略，用于复现/对照）")
    parser.add_argument("--div-channel-mode", default="rolling", choices=["rolling", "fixed"],
                        help="通道算法：rolling=sel_date前N日min/max通道(无前视)；fixed=固定上下轨线")
    parser.add_argument("--div-channel-window", type=int, default=756,
                        help="rolling 模式回看窗口(交易日)，默认756≈3年")
    parser.add_argument("--div-channel-bottom", type=float, default=None,
                        help="fixed 模式通道下轨(便宜线)，如 5100")
    parser.add_argument("--div-channel-top", type=float, default=None,
                        help="fixed 模式通道上轨(贵线)，如 6100")
    parser.add_argument("--k-min", type=float, default=0.5,
                        help="通道顶(最贵)时的仓位系数，默认0.5（即最多减至半仓）")
    parser.add_argument("--k-max", type=float, default=1.0,
                        help="通道底(最便宜)时的仓位系数，默认1.0（满仓）")
    # ── 计价口径（2026-09-01 新增，用于与全收益基准做同口径比较）──
    parser.add_argument("--price-mode", default="hfq", choices=["raw", "hfq"],
                        help="NAV 计价口径：hfq=后复权(默认,close*adj_factor,含分红再投,与全收益基准可比)；"
                             "raw=未复权(旧口径,不含分红,仅供复现 2026-09-02 前的历史结果)")
    # ── B7 指数编制法缓冲规则（2026-09-02 新增，opt-in 降换手）──
    parser.add_argument("--buffer-k", type=int, default=0,
                        help="B7 缓冲规则保留带宽 k：上期持仓若仍排进前 top_n+k 名则保留，空位按排名递补"
                             "（递补时执行行业上限）。0=关闭(默认,纯重排名=旧行为)。经验值 6~12")
    # ── B7' 官方同款「每次调整比例硬上限」（2026-09-03 新增，opt-in 降换手）──
    parser.add_argument("--final-key", choices=["fwd_yield", "volatility"], default="fwd_yield",
                        help="最后一段筛子（行业 cap 之后取前 top_n）的排序键："
                             "fwd_yield=股息率降序(默认,旧行为)；"
                             "volatility=波动率升序(官方 930955 口径：股息率前300→波动率升序取前100)。"
                             "🔴 波动率是慢变量 → 换键预计大幅提高留存率、压低换手")
    parser.add_argument("--turnover-cap", type=float, default=0.0,
                        help="每次调仓最多新进的比例上限（官方 H30269 = 0.20）。"
                             "做法：先与无约束档完全一致地全量重排名，再限制新进只数 ≤ int(top_n*cap)，"
                             "超限则砍掉排名最差的新进者、回补排名最好的落选老持仓。"
                             "0=关闭(默认,纯重排名=旧行为)。与 --buffer-k 互斥，同时给则以本开关为准")
    # ── 调仓频率（2026-09-03 新增，opt-in；不传则沿用 MODE_SPECS 的设定）
    parser.add_argument("--rebal", default=None, choices=list(REBAL_SPECS),
                        help="调仓频率，覆盖该 mode 在 MODE_SPECS 里的默认值："
                             "month=月度 / quarter=季度 / half=半年度(6·12月第5交易日) / "
                             "year=年度(官方口径:12月第二个周五后一交易日)。"
                             "🔴 降换手的第一杠杆是频率，不是缓冲带（详见 divlow_b7_demystify.md §3.3）")
    parser.add_argument("--live", action="store_true",
                        help="回测结束后打印最近一期调仓的可执行买列表（k缩放权重+现金%%），供实盘部署参考")
    parser.add_argument("--live-forward", action="store_true",
                        help="以库里最新交易日为选股日重跑 selector，打印真正『今天该买什么』的前瞻买列表（独立于历史回测）")
    args = parser.parse_args()
    global PRICE_MODE
    PRICE_MODE = args.price_mode
    if PRICE_MODE == "hfq":
        print("[计价口径] hfq = close * adj_factor（含分红再投），可与全收益基准比较")
    # 调仓频率覆盖（opt-in）。🔴 必须改 MODE_SPECS 本体：select_targets_official 内部
    # 读的是 MODE_SPECS[mode]["rebal"]，只改局部副本不会生效（同 PRICE_MODE 那类坑）。
    if args.rebal and args.mode in MODE_SPECS:
        MODE_SPECS[args.mode]["rebal"] = args.rebal
        print(f"[rebal] 覆盖 {args.mode} 的调仓频率 → {args.rebal}（{REBAL_SPECS[args.rebal][1]}）")
    if args.mode in ("old", "new", "soft"):
        run_legacy_comparison()
    else:
        run_official_backtest(args.mode, pool=args.pool, top_n=args.top_n, capital=args.capital,
                              overlay=not args.no_div_channel_overlay, channel_mode=args.div_channel_mode,
                              channel_window=args.div_channel_window, channel_bottom=args.div_channel_bottom,
                              channel_top=args.div_channel_top, k_min=args.k_min, k_max=args.k_max,
                              live=args.live, live_forward=args.live_forward,
                              buffer_k=args.buffer_k, turnover_cap=args.turnover_cap,
                              final_key=args.final_key)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise
