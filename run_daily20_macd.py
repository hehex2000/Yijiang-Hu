# -*- coding: utf-8 -*-
"""
v3-M1：红利低波 top20 + 再平衡 + 沪深300 MACD 开关
================================================
设计（docs/dev_plan_factor20_macd.md v3 定稿，选股改红利低波）：
  建仓/重入 : 金叉日（空仓时）用红利低波双排序从 zz800 选 top20，
             前一交易日数据选股（ex-ante）、当日收盘价等权买入（含费）
  再平衡    : --rebal daily    -> 持仓期每日收盘拉回等权（锁篮，不重选）
             --rebal monthly   -> 每月首个交易日【重新选股】+等权再平衡
             + --no-reselect   -> 每月首个交易日仅按当前持仓等权维护（锁篮，隔离频控）
  死叉     : 收盘全清仓（含费），现金冻结（不转货基）
  现金     : 空仓期冻结，无收益
口径：
  - 收盘价成交（与 apply_overlay 语义一致：转空日吃当日收益、转多日不吃当日收益）
  - 不复权 close（与平台 get_price 一致；红利类收益系统性低估为已知口径，M3 复核）
  - 停牌日无法交易（按最后价估值挂账，恢复后处理）
选股（红利低波，宽松版，不含逐只 MACD/杠杆/红利质量过滤）：
  daily_basic 估值过滤(0<PE_TTM<50, 0<PB<10, dv_ttm>0, total_mv>0)
  ∩ zz800 时点成分 ∩ 非ST → 逐只年化波动率 → 股息率pct降序 + 波动率pct升序 双排序取 topN
输出：对照表（干净 A/B：日频锁篮 / 月度锁篮 / 月度重选 × 满仓 vs +MACD）+ 净值CSV + 逐笔CSV
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import macd_plugin_validate as M
from regime_cash_overlay import load_index_close
from run_monthly_rebalance import (get_conn, calc_fee, get_trade_dates,
                                     get_index_constituents)

SIG = '000300.SH'          # MACD 信号基准
POOL_INDEX = '000906.SH'   # 股票池（中证800 时点成分）
CLOSES_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_closes_cache.pkl')
CLOSES_CACHE_HFQ = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '_closes_cache_hfq.pkl')
CLOSES_MAX_DATE = 20251231   # load_closes 的查询上限；超出则价格为 NaN（main 中有硬校验）

# ════════════════════════════════════════════════════════════════
#  NAV 计价口径（2026-09-01 新增；2026-09-02 起默认 hfq，raw 变 opt-in）
# ════════════════════════════════════════════════════════════════
# "raw" = 不复权（旧行为）：NAV 不含现金分红，且**不处理送转股**
#         → shares 从不随送转增加，除权日市值凭空蒸发 (1 - 1/送转比例)，单次 17%~45%。
# "hfq" = 后复权（含分红再投 + 免疫送转）：NAV 是**总回报**口径，须配全收益基准。
# ⚠️ 本模块的 closes 矩阵同时供给【选股信号】(波动率/MACD) 与【NAV 估值+成交】。
#    切换口径时**信号侧必须锁定 raw**、只切 NAV 侧，否则双跑差异无法归因
#    （分不清是口径修正还是选股改变）。见 main() 中的 raw_closes_full / closes_full。
PRICE_MODE = "hfq"     # 2026-09-02 起默认总回报口径；命令行 --price-mode 可覆盖


# ───────────────────────── 波动率预计算 ─────────────────────────
def build_vol_lookup(closes_full, window=120):
    """向量化年化波动率矩阵（index=trade_date, columns=ts_code）。
    用 ffill 后的收盘算日收益，rolling std × sqrt(252)；min_periods=window//2
    容忍停牌缺口。一次性计算，重入时直接查表，避免逐股 SQL。"""
    rets = closes_full.ffill().pct_change()
    vol = rets.rolling(window, min_periods=window // 2).std() * np.sqrt(252)
    return vol


# ───────────────────────── 选股：红利低波 ─────────────────────────
def select_div_low_vol(prev_date, top_n, vol_lookup, verbose=False):
    """红利低波双排序（宽松版）。返回 list[ts_code]（ex-ante：用 prev_date 时点数据）。
    vol_lookup: build_vol_lookup 产物，查表得到各股年化波动率。"""
    conn = get_conn()
    # 取 <= prev_date 最近的有 daily_basic 的交易日（防未来函数）
    actual_date = prev_date
    while True:
        cnt = conn.execute(
            "SELECT COUNT(*) AS n FROM daily_basic WHERE trade_date = ?",
            (actual_date,)).fetchone()[0]
        if cnt > 0:
            break
        prev = conn.execute(
            "SELECT MAX(trade_date) AS mx FROM daily_basic WHERE trade_date < ?",
            (actual_date,)).fetchone()[0]
        if prev is None:
            conn.close()
            return []
        actual_date = prev

    df = pd.read_sql_query("""
        SELECT ts_code, dv_ttm, total_mv
        FROM daily_basic
        WHERE trade_date = ?
          AND pe_ttm > 0 AND pe_ttm < 50
          AND pb > 0 AND pb < 10
          AND dv_ttm > 0
          AND total_mv > 0
    """, conn, params=(actual_date,))
    if df.empty:
        conn.close()
        return []

    # 股票池过滤（zz800 时点成分）
    zz = get_index_constituents(POOL_INDEX, trade_date=actual_date)
    if zz is not None:
        df = df[df['ts_code'].isin(zz)]
        if df.empty:
            conn.close()
            return []

    # 排除 ST
    st = pd.read_sql_query(
        "SELECT ts_code FROM stock_basic WHERE name LIKE '%ST%' OR name LIKE '%*%'",
        conn)
    conn.close()
    if len(st):
        df = df[~df['ts_code'].isin(set(st['ts_code'].tolist()))]
    if df.empty:
        return []

    # 波动率查表（ex-ante：<=actual_date 最近一日）
    vrow = vol_lookup.loc[vol_lookup.index <= int(actual_date)].iloc[-1]
    vol = vrow.to_dict()
    df = df[df['ts_code'].map(lambda c: vol.get(c) is not None and np.isfinite(vol.get(c)))]
    if df.empty:
        return []
    df['volatility'] = df['ts_code'].map(lambda c: vol.get(c))

    # 双排序：股息率越高越好(降序pct) + 波动率越低越好(升序pct)
    df['dv_rank'] = df['dv_ttm'].rank(pct=True)
    df['vol_rank'] = df['volatility'].rank(pct=True, ascending=False)
    df['score'] = (df['dv_rank'] + df['vol_rank']) / 2
    res = df.sort_values('score', ascending=False).head(top_n)
    if verbose:
        print(f"  [选股 {prev_date}|数据日{actual_date}] 候选 {len(df)} 只 → 取 {len(res)} 只")
    return res['ts_code'].tolist()


# ───────────────────────── 数据 ─────────────────────────
def load_closes(hfq=False):
    """zz800 全部历史成分日收盘，整段缓存后按需切片。

    hfq=False（默认）：不复权（旧行为，历史可复现）。
    hfq=True        ：后复权 = close × adj_factor / 该股首个已知因子（归一化）。

    ⚠️ 调用方必须区分用途分别取：
       · 选股信号（波动率 build_vol_lookup / 逐只 MACD 金叉）→ **永远用 hfq=False**，
         锁定 raw 以隔离变量，让 raw/hfq 双跑的差异纯粹来自 NAV 口径、可归因。
       · NAV 估值与成交 → 按 PRICE_MODE 切换。
       两处共用同一矩阵会导致"改口径"与"改选股"两个变量耦合，双跑结论不可解释。

    hfq 口径两个必须守住的坑：
      ① adj_factor 是阶跃函数且 2020-2026 有 132 个**整交易日全市场同缺** → 缺行必须按股
         ffill+bfill，绝不能 fillna(1.0)（会让缺行日掉回不复权、次日跳回复权，假跳空）。
      ② 必须除以该股首个已知因子做**归一化**，否则绝对价位被累计因子放大数十倍
         （000651.SZ raw 67.9 → hfq 10589），按整股下单 int(cash//px) 会买入 0 股 → 伪现金拖累。
    """
    cache = CLOSES_CACHE_HFQ if hfq else CLOSES_CACHE
    if os.path.exists(cache):
        codes, df = pd.read_pickle(cache)
        return codes, df
    conn = get_conn()
    codes = [r[0] for r in conn.execute(
        "SELECT DISTINCT ts_code FROM index_constituent WHERE index_code='000906.SH'")]
    print(f"[load] zz800 历史成分 {len(codes)} 只，bulk 日线"
          f"{'(后复权)' if hfq else '(不复权)'}...", flush=True)
    frames = []
    for i in range(0, len(codes), 400):
        b = codes[i:i + 400]
        ph = ",".join("?" * len(b))
        if hfq:
            d = pd.read_sql_query(
                f"SELECT t.ts_code, t.trade_date, t.close, a.adj_factor "
                f"FROM daily t "
                f"LEFT JOIN adj_factor a ON t.ts_code=a.ts_code AND t.trade_date=a.trade_date "
                f"WHERE t.ts_code IN ({ph}) "
                f"AND t.trade_date>=? AND t.trade_date<=? ORDER BY t.ts_code,t.trade_date",
                conn, params=b + [20100101, 20251231])
        else:
            d = pd.read_sql_query(
                f"SELECT ts_code,trade_date,close FROM daily WHERE ts_code IN ({ph}) "
                f"AND trade_date>=? AND trade_date<=? ORDER BY ts_code,trade_date",
                conn, params=b + [20100101, 20251231])
        if len(d):
            frames.append(d)
    conn.close()
    alld = pd.concat(frames, ignore_index=True)
    alld['trade_date'] = alld['trade_date'].astype(int)
    if hfq:
        alld = alld.sort_values(["ts_code", "trade_date"])
        # ① ffill 补中间缺行（阶跃函数，数学精确）；bfill 补上市初期未覆盖段；
        #    整列全缺才兜底 1.0（此时整股用 raw，无跳变风险）
        alld["adj_factor"] = (alld.groupby("ts_code")["adj_factor"]
                                  .ffill().bfill().fillna(1.0))
        # ② 归一化：除以该股首个已知因子，使首日 hfq 严格 == raw
        ref = alld.groupby("ts_code")["adj_factor"].transform("first")
        alld["close"] = alld["close"] * alld["adj_factor"] / ref
    df = alld.pivot(index='trade_date', columns='ts_code', values='close').sort_index()
    pd.to_pickle((codes, df), cache)
    print(f"[load] 收盘矩阵 {df.shape} 已缓存 -> {os.path.basename(cache)}", flush=True)
    return codes, df


# ───────────────────────── 模拟 ─────────────────────────
def _rebalance_to(td, raw, val, codes, cash, positions, tot_fee, turn_over,
                  trades, reason, weights=None):
    """把仓位调整到目标篮子 codes 的目标权重。
    weights: dict(code->float, 归一化 sum≈1)；为 None 时等权。
    codes 为空则只做现有持仓的等权维护。
    返回 (cash, positions, tot_fee, turn_over)。"""
    codes = list(codes)
    if weights is None:
        weights = {c: 1.0 / max(len(codes), 1) for c in codes}
    # 1) 卖出不在目标篮子的持仓
    for c in list(positions):
        if c not in codes:
            px = raw[c] if (raw is not None and c in raw.index) else np.nan
            if not np.isfinite(px) or px <= 0:
                continue
            fee = calc_fee('sell', px, positions[c], trade_date=td)
            tot_fee += fee
            cash += positions[c] * px - fee
            turn_over += positions[c] * px
            trades.append({"date": td, "action": "SELL-rebal", "code": c,
                           "shares": positions[c], "price": px, "fee": fee,
                           "reason": reason})
            del positions[c]
    # 2) 目标市值（按 weights 逐只）
    eq = cash + sum(sh * (val[c] if val is not None and c in val.index else 0.0)
                    for c, sh in positions.items())
    def _tgt(c):
        return eq * weights.get(c, 1.0 / max(len(codes), 1))
    # 3) 先卖超额（现有持仓内）
    for c in list(positions):
        if c in codes:
            px = raw[c] if (raw is not None and c in raw.index) else np.nan
            if not np.isfinite(px):
                continue
            cur = positions[c] * px
            delta = _tgt(c) - cur
            if delta < -0.5:
                sell_sh = min(positions[c], -delta / px)
                fee = calc_fee('sell', px, sell_sh, trade_date=td)
                tot_fee += fee
                cash += sell_sh * px - fee
                positions[c] -= sell_sh
                turn_over += sell_sh * px
                if positions[c] <= 1e-9:
                    del positions[c]
    # 4) 买齐目标（含新进 + 现有补到目标）
    for c in codes:
        px = raw[c] if (raw is not None and c in raw.index) else np.nan
        if not np.isfinite(px) or px <= 0:
            continue
        cur = positions.get(c, 0) * px
        delta = _tgt(c) - cur
        if delta > 0.5:
            buy_sh = delta / px
            fee = calc_fee('buy', px, buy_sh, trade_date=td)
            tot_fee += fee
            cost = buy_sh * px + fee
            if cost <= cash:
                cash -= cost
                positions[c] = positions.get(c, 0) + buy_sh
                turn_over += buy_sh * px
    return cash, positions, tot_fee, turn_over


def run_sim(trade_dates, dates_i, golden, closes, closes_ff, top_n, capital,
            select_fn, vol_lookup, rebal_freq='monthly', month_starts=None,
            no_reselect=False, verbose=False, weight_mode='equal'):
    """golden: list[bool] 与 trade_dates 对齐。
    rebal_freq: 'daily' 每日等权维护 | 'monthly' 每月首个交易日再平衡。
    no_reselect: 仅对 monthly 生效——每月仅按当前持仓等权维护（锁篮），不重新选股。
    weight_mode: 'equal' 等权 | 'div' 股息率加权（要求 select_fn 返回 dict{code:dv_ttm}）。
    返回 (nav, trades, stats)。"""
    cash = float(capital)
    positions = {}
    nav = np.zeros(len(trade_dates))
    trades = []
    rebal_days = 0
    turn_over = 0.0
    n_empty_reentry = 0
    n_reselect = 0
    tot_fee = 0.0

    for i, td in enumerate(trade_dates):
        di = dates_i[i]
        g = bool(golden[i])
        val = closes_ff.loc[di] if di in closes_ff.index else None
        raw = closes.loc[di] if di in closes.index else None
        is_ms = month_starts is not None and td in month_starts

        if g:
            def _resolve(res):
                """select_fn 可能返回 list 或 dict{code:dv_ttm(可为None)}。
                返回 (codes, weights)；codes 空时 weights=None。
                div 模式下：dv<=0 或缺失( None) 的 code 用可用 dv 的中位数填补，
                保证篮子数量恒定（不被缺失 dv 偷偷缩小），仅改变权重分布。"""
                if isinstance(res, dict):
                    codes = [c for c in res if c is not None]
                    if not codes:
                        return [], None
                    if weight_mode == 'div':
                        vals = []
                        for c in codes:
                            v = res.get(c)
                            vals.append(v if (isinstance(v, (int, float)) and v > 0) else None)
                        good = [v for v in vals if v is not None]
                        if good:
                            med = float(np.median(good))
                            filled = [v if v is not None else med for v in vals]
                            s = float(sum(filled))
                            weights = {c: f / s for c, f in zip(codes, filled)}
                        else:
                            weights = {c: 1.0 / len(codes) for c in codes}
                    else:
                        weights = {c: 1.0 / len(codes) for c in codes}
                else:
                    codes = list(res) if res else []
                    if not codes:
                        return [], None
                    weights = {c: 1.0 / len(codes) for c in codes}
                return codes, weights

            if not positions:
                # 重入：空仓且金叉 → 选股建仓（必要的一次选股）
                prev_td = trade_dates[i - 1] if i > 0 else td
                codes, weights = _resolve(select_fn(prev_td, top_n, vol_lookup))
                if not codes:
                    n_empty_reentry += 1
                else:
                    for c in codes:
                        px = raw[c] if (raw is not None and c in raw.index) else np.nan
                        if not np.isfinite(px) or px <= 0:
                            continue
                        per = cash * weights.get(c, 1.0 / len(codes))
                        sh = per / px
                        fee = calc_fee('buy', px, sh, trade_date=td)
                        tot_fee += fee
                        cost = sh * px + fee
                        if cost <= cash:
                            cash -= cost
                            positions[c] = sh
                            turn_over += sh * px
                            trades.append({"date": td, "action": "BUY-reentry",
                                           "code": c, "shares": sh, "price": px,
                                           "fee": fee, "reason": "golden_reentry"})
                    if verbose:
                        print(f"  [金叉 {td}] 重入 {len(positions)} 只")
            else:
                do_rebal = (rebal_freq == 'daily') or (rebal_freq == 'monthly' and is_ms)
                if do_rebal:
                    prev_td = trade_dates[i - 1] if i > 0 else td
                    if rebal_freq == 'monthly' and is_ms and not no_reselect:
                        # 月度【重选】篮子
                        codes, weights = _resolve(select_fn(prev_td, top_n, vol_lookup))
                        if codes:
                            n_reselect += 1
                            cash, positions, tot_fee, turn_over = _rebalance_to(
                                td, raw, val, codes, cash, positions, tot_fee,
                                turn_over, trades, 'monthly_reselect', weights=weights)
                            rebal_days += 1
                        else:
                            cash, positions, tot_fee, turn_over = _rebalance_to(
                                td, raw, val, list(positions.keys()), cash, positions,
                                tot_fee, turn_over, trades, 'monthly_maint', weights=None)
                            rebal_days += 1
                    else:
                        # 日频维护 / 月度锁篮维护：仅按当前持仓等权
                        cash, positions, tot_fee, turn_over = _rebalance_to(
                            td, raw, val, list(positions.keys()), cash, positions,
                            tot_fee, turn_over, trades, 'maint', weights=None)
                        rebal_days += 1
            eq = cash + sum(sh * (val[c] if val is not None and c in val.index else 0.0)
                            for c, sh in positions.items())
            nav[i] = eq
        else:
            if positions and raw is not None:
                for c in list(positions):
                    px = raw[c] if c in raw.index else np.nan
                    if not np.isfinite(px):
                        continue
                    fee = calc_fee('sell', px, positions[c], trade_date=td)
                    tot_fee += fee
                    cash += positions[c] * px - fee
                    turn_over += positions[c] * px
                    trades.append({"date": td, "action": "SELL-liquidate",
                                   "code": c, "shares": positions[c], "price": px,
                                   "fee": fee, "reason": "death_cross"})
                    del positions[c]
                if verbose and not positions:
                    print(f"  [死叉 {td}] 全清仓")
            eq = cash + sum(sh * (val[c] if val is not None and c in val.index else 0.0)
                            for c, sh in positions.items())
            nav[i] = eq

    stats = {"rebal_days": rebal_days, "turnover": turn_over,
             "n_reentry": sum(1 for t in trades if t["action"] == "BUY-reentry"),
             "n_liquidate": sum(1 for t in trades if t["action"] == "SELL-liquidate"),
             "n_reselect": n_reselect,
             "total_fee": tot_fee,
             "n_empty_reentry": n_empty_reentry}
    return nav, trades, stats


# ───────────────────────── 主流程 ─────────────────────────
def main():
    ap = argparse.ArgumentParser(description='v3-M1: 红利低波20只+再平衡+MACD')
    ap.add_argument('--start', default='20150101')
    ap.add_argument('--end', default='20251231')
    ap.add_argument('--top-n', type=int, default=20)
    ap.add_argument('--capital', type=float, default=1_000_000)
    ap.add_argument('--rebal', choices=['daily', 'monthly'], default='monthly')
    ap.add_argument('--no-reselect', action='store_true',
                    help='月度模式下不重选篮子，仅按当前持仓等权维护（隔离频控效应）')
    ap.add_argument('--all-modes', action='store_true',
                    help='跑全部三档(日频锁篮/月度锁篮/月度重选)做干净A/B对比')
    ap.add_argument('--out', default='data/results/daily20_divlow')
    ap.add_argument('--price-mode', choices=['raw', 'hfq'], default='hfq',
                    help='NAV 计价口径: hfq=后复权(默认,总回报,含分红再投+免疫送转,自动配全收益基准) / '
                         'raw=不复权(旧口径,漏分红且不处理送转,仅供复现历史结论;须配价格指数基准)。'
                         '仅影响 NAV 侧，选股信号(波动率/MACD)恒定用 raw 以隔离变量')
    args = ap.parse_args()

    global PRICE_MODE          # main() 是函数，此处必须 global
    PRICE_MODE = args.price_mode
    if PRICE_MODE == 'hfq' and not args.out.endswith('_hfq'):
        args.out = args.out + '_hfq'   # 避免 hfq 结果静默覆盖 raw 留证

    trade_dates = get_trade_dates(args.start, args.end)
    dates_i = [int(d) for d in trade_dates]
    print(f"[info] 交易日 {len(trade_dates)} | {args.start}~{args.end} | top{args.top_n} 红利低波 | 池={POOL_INDEX}")

    # 月度再平衡的调仓日 = 每月首个交易日
    month_starts = set()
    prev_ym = None
    for d in trade_dates:
        ym = d[:6]
        if ym != prev_ym:
            month_starts.add(d)
            prev_ym = ym

    # 信号侧（波动率因子 + 逐只 MACD 金叉）**永远锁 raw**：隔离变量，
    # 让 raw/hfq 双跑的差异纯粹来自 NAV 口径，否则无法归因。
    _, raw_closes_full = load_closes(hfq=False)
    vol_lookup = build_vol_lookup(raw_closes_full)
    # NAV 侧按 PRICE_MODE 切换
    if PRICE_MODE == "hfq":
        _, closes_full = load_closes(hfq=True)
    else:
        closes_full = raw_closes_full
    if int(args.end) > CLOSES_MAX_DATE:
        print(f"\n[错误] 回测终点 {args.end} 超出收盘矩阵缓存上限 {CLOSES_MAX_DATE}。"
              f"\n       超区间价格全为 NaN → 持仓估值静默归零（曾跑出 -92.90% 的假结果）。"
              f"\n       如需更长区间，先扩大 load_closes() 的查询范围并删除 "
              f"{os.path.basename(CLOSES_CACHE)} / {os.path.basename(CLOSES_CACHE_HFQ)} 重建缓存。")
        sys.exit(2)
    closes = closes_full.loc[(closes_full.index >= int(args.start)) & (closes_full.index <= int(args.end))]
    closes_ff = closes.ffill()
    # 二次防御：末日若整行无有效价，说明区间与数据错位，宁可报错也不要输出假净值
    if closes.shape[0] and closes.iloc[-1].notna().sum() == 0:
        print(f"\n[错误] 回测末日 {closes.index[-1]} 无任何有效收盘价，区间与数据错位。")
        sys.exit(2)

    # 沪深300 MACD 信号（对齐交易日）
    hs = load_index_close(SIG, args.start, args.end)
    hs = hs.reindex(closes.index).ffill()
    golden_s = M.macd_golden(hs.values.astype(float))
    golden_map = dict(zip(closes.index, golden_s.values))
    golden_arr = [bool(golden_map.get(di, False)) for di in dates_i]
    print(f"[info] 沪深300 MACD 金叉持仓日占比 {np.mean(golden_arr)*100:.1f}%")

    sel_fn = lambda pd_, tn, vl: select_div_low_vol(pd_, tn, vl, verbose=False)

    # 基准指数（口径自动跟随 NAV：raw→价格指数 / hfq→全收益）
    # 两端必须同含或同不含分红，否则超额系统性失真（中证800 约 2.2%/年、沪深300 约 3.0%/年）。
    bench = {}
    _bmeta = None
    for code, nm in [('000906.SH', '中证800'), ('000300.SH', '沪深300')]:
        if PRICE_MODE == 'hfq':
            import bench_index as bi
            df_b, meta = bi.load_benchmark(code, args.start, args.end,
                                           nav_price_mode='hfq')
            if df_b is not None and len(df_b) >= 2:
                bench[nm] = pd.Series(
                    (df_b['close'] / float(df_b['close'].iloc[0])).values,
                    index=df_b['trade_date'])
                _bmeta = meta
                continue
        b = M.load_base_index(code, args.start, args.end)
        if b is not None:
            bench[nm] = b

    def metrics(nav):
        rb, ab, mdb, sb = M.metrics(pd.Series(nav))
        return rb, ab, mdb, sb

    def pct(x): return f"{x*100:+.2f}%"
    years = len(trade_dates) / 252

    # ── 三档定义：(rebal_freq, no_reselect, tier名, 文件key) ──
    if args.all_modes:
        modes = [('daily', False, '日频锁篮', 'daily'),
                 ('monthly', True, '月度锁篮', 'monthly_static'),
                 ('monthly', False, '月度重选', 'monthly')]
    else:
        modes = [(args.rebal, args.no_reselect,
                  ('月度锁篮' if (args.rebal == 'monthly' and args.no_reselect)
                   else ('月度重选' if args.rebal == 'monthly' else '日频锁篮')),
                  ('monthly_static' if (args.rebal == 'monthly' and args.no_reselect)
                   else args.rebal))]

    results = {}   # (tier, kind) -> (nav, st)
    for rebal_freq, no_reselect, tier, key in modes:
        nav_bh, tr_bh, st_bh = run_sim(trade_dates, dates_i, [True] * len(trade_dates),
                                       closes, closes_ff, args.top_n, args.capital,
                                       sel_fn, vol_lookup, rebal_freq=rebal_freq,
                                       month_starts=month_starts, no_reselect=no_reselect)
        nav_macd, tr_macd, st_macd = run_sim(trade_dates, dates_i, golden_arr,
                                             closes, closes_ff, args.top_n, args.capital,
                                             sel_fn, vol_lookup, rebal_freq=rebal_freq,
                                             month_starts=month_starts,
                                             no_reselect=no_reselect, verbose=False)
        results[(tier, '满仓')] = (nav_bh, st_bh)
        results[(tier, 'MACD')] = (nav_macd, st_macd)
        # 保存
        os.makedirs(args.out, exist_ok=True)
        pd.DataFrame({'date': trade_dates, 'nav_buyhold': nav_bh,
                      'nav_macd': nav_macd}).to_csv(
            f"{args.out}/nav_{key}_{args.start}_{args.end}.csv", index=False)
        pd.DataFrame(tr_macd).to_csv(
            f"{args.out}/trades_{key}_{args.start}_{args.end}.csv", index=False)

    # ── 打印对照表 ──
    run_tiers = [t for (t, k) in results if k == '满仓']   # 实际跑出的档
    print(f"\n{'='*96}")
    print(f"  v3-M1 红利低波top{args.top_n} zz800 | 干净A/B：{' / '.join(run_tiers)} | {args.start}~{args.end}")
    print(f"  费用: 佣金万2.5(最低5元)+印花税分段+滑点0.1%/边 | 收盘价成交")
    print(f"{'='*96}")
    print(f"  {'方案':<26}{'总收益':>10}{'年化':>9}{'最大回撤':>10}{'Sharpe':>9}{'持币%':>8}")
    for tier in run_tiers:
        rb, ab, mdb, sb = metrics(results[(tier, '满仓')][0])
        rm, am_, mdm, sm = metrics(results[(tier, 'MACD')][0])
        cr = 100 * (1 - np.mean(golden_arr))
        print(f"  {'满仓·'+tier:<22}{pct(rb):>10}{pct(ab):>9}{pct(mdb):>10}{sb:>9.2f}{0.0:>7.1f}%")
        print(f"  {'+MACD·'+tier:<22}{pct(rm):>10}{pct(am_):>9}{pct(mdm):>10}{sm:>9.2f}{cr:>7.1f}%")
    for nm, b in bench.items():
        rb_, ab_, mdb_, _ = metrics(b)
        print(f"  {'基准' + nm + '满仓':<22}{pct(rb_):>10}{pct(ab_):>9}{pct(mdb_):>10}")

    print(f"\n  ── 成本/换手 ──")
    for tier in run_tiers:
        for kind, nav_, st in [('满仓', results[(tier, '满仓')][0], results[(tier, '满仓')][1]),
                               ('+MACD', results[(tier, 'MACD')][0], results[(tier, 'MACD')][1])]:
            ann_to = st['turnover'] / max(years, 1e-9) / args.capital
            print(f"  {kind}·{tier:<10} 总费{st['total_fee']:>10,.0f}元 | 年化换手{ann_to*100:>5.0f}% | "
                  f"重入{st['n_reentry']:>4} 清仓{st['n_liquidate']:>4} | 再平衡{st['rebal_days']:>4}天 重选{st['n_reselect']:>4}次")

    # 分年表（仅对 +MACD vs 满仓 在各档内的 Δpp）
    def yearly(nav):
        df = pd.DataFrame({'d': [int(d) for d in trade_dates], 'v': nav})
        df['y'] = df['d'] // 10000
        return {y: g['v'].iloc[-1] / g['v'].iloc[0] - 1 for y, g in df.groupby('y')}
    print(f"\n  ── 分年表（+MACD − 满仓，单位pp；正=MACD占优）──")
    print(f"  {'年份':<8}" + "".join(f"{t:>14}" for t in run_tiers))
    years_list = sorted({y for (tier, _), (nav, _) in results.items()
                         for y in yearly(nav)})
    for y in years_list:
        line = f"  {y:<8}"
        for tier in run_tiers:
            yb = yearly(results[(tier, '满仓')][0])
            ym = yearly(results[(tier, 'MACD')][0])
            dd = (ym.get(y, 0) - yb.get(y, 0)) * 100
            line += f"{dd:>+14.1f}"
        print(line)

    print(f"\n  [输出] {args.out}/nav_{{daily,monthly_static,monthly}}_{args.start}_{args.end}.csv 等")


if __name__ == '__main__':
    main()
