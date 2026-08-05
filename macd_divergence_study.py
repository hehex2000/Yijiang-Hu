# -*- coding: utf-8 -*-
"""
MACD 背离诊断器（对照实验验证指标 / MACD 科普视频"元方法论"落地）
========================================================================
纯研究诊断脚本，不修改任何现有策略。设计严格对应视频核心方法论：
  "把全部符合条件的背离找出来 → 按趋势/震荡分组 → 看背离后 5/10/20 天
   → 加交易摩擦 → 用相近背景的对照组比较"

本工具回答：MACD 顶/底背离在 A 股到底有没有 edge、在哪类市场状态下有。

两套对照（关键，避免"极点均值回归"假象）：
  ① 背景对照（plan 原设计）：同(状态桶+同月)的"任意非背离日"前向收益均值相减。
     缺陷：背离事件坐在价格极点，而背景日含随机日 → 差值会混入"买在低点/卖在
     高点"的机械反弹。
  ② 极点对照（本工具新增，更严谨）：同(状态桶+同月)的"普通极点"（普通 pivot 低/
     高点，但 MACD 未确认）前向收益均值相减。这才是 MACD 确认带来的"纯增量"。
  结论以 ② 为主、① 为辅。

状态分组（视频要点③：趋势和震荡分开看）：
  - 中枢期（consolidation）：20 日布林带宽滚动分位 < 阈值（复用中枢计划口径）
  - 趋势：个股 T-1 收盘 vs 自身 MA200 → 上升 / 下降
  桶优先级：中枢期 > 趋势（盘整中背离最易失效）
复用：run_magic_v2.py 的 hfq/adj_factor 后向填充/MACD/MA 口径模式。
"""
import sys, os, sqlite3, bisect, argparse
import numpy as np
import pandas as pd
import talib as ta
import run_magic_formula as mf

DB_PATH = "D:/tu-shareData/astock_daily.db"
_POOL_INDEX = {"hs300": "000300.SH", "zz500": "000905.SH",
               "zz800": "000906.SH", "zz1000": "000852.SH"}

_PX = {}; _FAC = {}; _IDX = {}

def _conn():
    return sqlite3.connect(DB_PATH)

def _load_code(code):
    if code in _PX:
        return
    c = _conn()
    rows = c.execute(
        "SELECT CAST(trade_date AS TEXT), open, high, low, close "
        "FROM daily WHERE ts_code=? ORDER BY trade_date", (code,)).fetchall()
    fr = c.execute(
        "SELECT CAST(trade_date AS TEXT), adj_factor FROM adj_factor "
        "WHERE ts_code=? ORDER BY trade_date", (code,)).fetchall()
    c.close()
    _PX[code] = ([r[0] for r in rows], [r[1] for r in rows],
                 [r[2] for r in rows], [r[3] for r in rows], [r[4] for r in rows])
    _FAC[code] = ([r[0] for r in fr], [r[1] for r in fr])

def _raw(code, td, kind="close"):
    _load_code(code)
    dates, opens, highs, lows, closes = _PX[code]
    i = bisect.bisect_right(dates, td) - 1
    if i < 0:
        return None
    v = {"open": opens, "high": highs, "low": lows, "close": closes}[kind][i]
    return float(v) if v is not None else None

def _factor(code, td):
    _load_code(code)
    dates, facs = _FAC[code]
    i = bisect.bisect_right(dates, td) - 1
    if i >= 0 and facs[i] is not None:
        return float(facs[i])
    for f in facs:
        if f is not None:
            return float(f)
    return None

def hfq_price(code, td, kind="close"):
    p = _raw(code, td, kind)
    if p is None:
        return None
    f = _factor(code, td)
    return p * f if f else p

def _load_index(idx):
    if idx in _IDX:
        return
    c = _conn()
    rows = c.execute(
        "SELECT CAST(trade_date AS TEXT), close FROM index_daily "
        "WHERE ts_code=? ORDER BY trade_date", (idx,)).fetchall()
    c.close()
    dates = [r[0] for r in rows]
    closes = [float(r[1]) for r in rows]
    _IDX[idx] = (dates, closes, np.concatenate([[0.0], np.cumsum(closes)]))

def index_close(idx, td):
    _load_index(idx)
    dates, closes, _ = _IDX[idx]
    i = bisect.bisect_right(dates, td) - 1
    return closes[i] if i >= 0 else None

def index_above_ma(idx, td, win=200):
    _load_index(idx)
    dates, closes, csum = _IDX[idx]
    i = bisect.bisect_right(dates, td) - 1
    if i < win - 1:
        return True
    ma = (csum[i + 1] - csum[i + 1 - win]) / win
    return closes[i] >= ma

def get_universe(pool, end_date):
    idx_code = _POOL_INDEX.get(pool, "000300.SH")
    c = _conn()
    rows = c.execute(
        "SELECT DISTINCT ts_code FROM index_constituent WHERE index_code=?",
        (idx_code,)).fetchall()
    c.close()
    codes = [r[0] for r in rows]
    if not codes:
        codes = list(mf._get_pool_constituents(pool, end_date))
    return sorted(set(codes))


def _bb_width_pct(width_arr, i, lookback):
    lo = max(0, i - lookback + 1)
    win = width_arr[lo:i + 1]
    win = win[~np.isnan(win)]
    if len(win) < max(5, int(lookback * 0.5)):
        return None
    cur = width_arr[i]
    if np.isnan(cur):
        return None
    return float((win < cur).mean())


def process_stock(code, start, end, P, sanity=False):
    """返回 (events, bg_ctrl, pl_ctrl, ph_ctrl)
       events: list of dict{...fwd.., date,type,bucket}
       bg_ctrl: 背景对照组  list of (bucket,ym,f5n,f10n,f20n)  每月一个非背离日
       pl_ctrl: 普通 pivot 低点对照  list of (bucket,ym,f5n,f10n,f20n)
       ph_ctrl: 普通 pivot 高点对照  list of (bucket,ym,f5n,f10n,f20n)
    """
    _load_code(code)
    dates, opens, highs, lows, closes = _PX[code]
    si = bisect.bisect_left(dates, start)
    ei = bisect.bisect_right(dates, end)
    if ei - si < 200:
        return [], [], [], []
    dates = dates[si:ei]
    highs = np.asarray(highs[si:ei], dtype=float)
    lows = np.asarray(lows[si:ei], dtype=float)
    closes = np.asarray(closes[si:ei], dtype=float)
    n = len(closes)
    if n < 250:
        return [], [], [], []

    hfq = np.array([(hfq_price(code, d, "close") or np.nan) for d in dates], dtype=float)
    if np.isnan(hfq).mean() > 0.1:
        return [], [], [], []

    dif, _, _ = ta.MACD(closes, fastperiod=P["fast"], slowperiod=P["slow"],
                        signalperiod=P["signal"])
    if sanity:
        dif = np.random.permutation(dif)
    valid = ~np.isnan(dif)

    s = pd.Series(closes)
    roll_mean = s.rolling(P["bb_win"]).mean()
    roll_std = s.rolling(P["bb_win"]).std()
    width = (roll_std / roll_mean).values
    ma200 = s.rolling(200).mean().values

    W = P["pivot_window"]
    sh = pd.Series(highs)
    rmax = sh.rolling(2 * W + 1, center=True, min_periods=2 * W + 1).max()
    ph_mask = (sh == rmax) & (sh > sh.shift(1)) & (sh >= sh.shift(-1))
    ph_idx = np.where(ph_mask.values)[0]
    sl = pd.Series(lows)
    rmin = sl.rolling(2 * W + 1, center=True, min_periods=2 * W + 1).min()
    pl_mask = (sl == rmin) & (sl < sl.shift(1)) & (sl <= sl.shift(-1))
    pl_idx = np.where(pl_mask.values)[0]

    def bucket_at(i):
        pct = _bb_width_pct(width, i, P["bb_lookback"])
        if pct is not None and pct < P["bb_th"]:
            return "consolidation"
        m = ma200[i]
        if m is None or np.isnan(m) or np.isnan(closes[i]):
            return "unknown"
        return "uptrend" if closes[i] >= m else "downtrend"

    def fwd_ret(idx, K):
        j = idx + K
        if j >= n:
            return None
        a, b = hfq[idx], hfq[j]
        if a is None or b is None or not np.isfinite(a) or a == 0:
            return None
        return b / a - 1.0

    events = []
    seen_types = []
    bull_event_idx = set()
    bear_event_idx = set()

    def add_event(i2, etype):
        for (d2, t2) in seen_types:
            if t2 == etype and abs(bisect.bisect_left(dates, d2)
                                    - bisect.bisect_left(dates, dates[i2])) < P["min_gap"]:
                return False
        seen_types.append((dates[i2], etype))
        bk = bucket_at(i2)
        if bk == "unknown":
            return False
        f5, f10, f20 = fwd_ret(i2, 5), fwd_ret(i2, 10), fwd_ret(i2, 20)
        if f5 is None or f10 is None or f20 is None:
            return False
        cost = P["cost"]
        events.append({
            "date": dates[i2], "type": etype, "bucket": bk,
            "fwd5g": f5, "fwd10g": f10, "fwd20g": f20,
            "fwd5n": f5 - cost, "fwd10n": f10 - cost, "fwd20n": f20 - cost,
        })
        return True

    for a in range(1, len(ph_idx)):
        i1, i2 = ph_idx[a - 1], ph_idx[a]
        if i2 - i1 < P["min_gap"] or not valid[i1] or not valid[i2]:
            continue
        if highs[i2] > highs[i1] and dif[i2] < dif[i1]:
            if P["lookback"] and highs[i2] >= np.nanmax(highs[max(0, i2 - P["lookback"]):i2 + 1]):
                if add_event(i2, "bearish"):
                    bear_event_idx.add(i2)
            elif not P["lookback"]:
                if add_event(i2, "bearish"):
                    bear_event_idx.add(i2)
    for a in range(1, len(pl_idx)):
        i1, i2 = pl_idx[a - 1], pl_idx[a]
        if i2 - i1 < P["min_gap"] or not valid[i1] or not valid[i2]:
            continue
        if lows[i2] < lows[i1] and dif[i2] > dif[i1]:
            if P["lookback"] and lows[i2] <= np.nanmin(lows[max(0, i2 - P["lookback"]):i2 + 1]):
                if add_event(i2, "bullish"):
                    bull_event_idx.add(i2)
            elif not P["lookback"]:
                if add_event(i2, "bullish"):
                    bull_event_idx.add(i2)

    # 背景对照（每月一个非背离日）
    event_dates = set(e["date"] for e in events)
    bg_ctrl = []
    cur_month = None
    for i, d in enumerate(dates):
        ym = d[:6]
        if ym != cur_month:
            cur_month = ym
            if d in event_dates:
                continue
            bk = bucket_at(i)
            if bk == "unknown":
                continue
            f5, f10, f20 = fwd_ret(i, 5), fwd_ret(i, 10), fwd_ret(i, 20)
            if f5 is None or f10 is None or f20 is None:
                continue
            bg_ctrl.append((bk, ym, f5 - P["cost"], f10 - P["cost"], f20 - P["cost"]))

    # 极点对照（普通 pivot 低/高点，排除背离事件本身）
    pl_ctrl = []
    for i2 in pl_idx:
        if i2 in bull_event_idx:
            continue
        bk = bucket_at(i2)
        if bk == "unknown":
            continue
        f5, f10, f20 = fwd_ret(i2, 5), fwd_ret(i2, 10), fwd_ret(i2, 20)
        if f5 is None or f10 is None or f20 is None:
            continue
        pl_ctrl.append((bk, dates[i2][:6], f5 - P["cost"], f10 - P["cost"], f20 - P["cost"]))
    ph_ctrl = []
    for i2 in ph_idx:
        if i2 in bear_event_idx:
            continue
        bk = bucket_at(i2)
        if bk == "unknown":
            continue
        f5, f10, f20 = fwd_ret(i2, 5), fwd_ret(i2, 10), fwd_ret(i2, 20)
        if f5 is None or f10 is None or f20 is None:
            continue
        ph_ctrl.append((bk, dates[i2][:6], f5 - P["cost"], f10 - P["cost"], f20 - P["cost"]))

    return events, bg_ctrl, pl_ctrl, ph_ctrl


def run_study(pool, start, end, P, sanity=False, progress=True):
    codes = get_universe(pool, end)
    if progress:
        print(f"[study] 池={pool} 区间={start}~{end} 候选成分 {len(codes)} 只")
    all_events = []
    bg = {}      # (bucket, ym) -> [[f5,f10,f20]]
    pl = {}      # (bucket, ym) -> [[f5,f10,f20]]  普通 pivot 低点
    ph = {}      # (bucket, ym) -> [[f5,f10,f20]]  普通 pivot 高点
    total = len(codes)
    for ci, code in enumerate(codes):
        try:
            ev, bgc, plc, phc = process_stock(code, start, end, P, sanity=sanity)
        except Exception:
            continue
        all_events.extend(ev)
        for (bk, ym, f5, f10, f20) in bgc:
            bg.setdefault((bk, ym), []).append([f5, f10, f20])
        for (bk, ym, f5, f10, f20) in plc:
            pl.setdefault((bk, ym), []).append([f5, f10, f20])
        for (bk, ym, f5, f10, f20) in phc:
            ph.setdefault((bk, ym), []).append([f5, f10, f20])
        if progress and (ci + 1) % 100 == 0:
            print(f"  ... {ci+1}/{total}  事件 {len(all_events)}")

    bg_mean = {k: np.mean(np.array(v), axis=0) for k, v in bg.items()}
    pl_mean = {k: np.mean(np.array(v), axis=0) for k, v in pl.items()}
    ph_mean = {k: np.mean(np.array(v), axis=0) for k, v in ph.items()}

    rows = []
    for e in all_events:
        key = (e["bucket"], e["date"][:6])
        b = bg_mean.get(key, [0, 0, 0])
        ext = pl_mean.get(key) if e["type"] == "bullish" else ph_mean.get(key)
        ext = ext if ext is not None else [0, 0, 0]
        rows.append({
            "date": e["date"], "type": e["type"], "bucket": e["bucket"],
            "fwd5g": e["fwd5g"], "fwd10g": e["fwd10g"], "fwd20g": e["fwd20g"],
            "fwd5n": e["fwd5n"], "fwd10n": e["fwd10n"], "fwd20n": e["fwd20n"],
            "ctrl5": b[0], "ctrl10": b[1], "ctrl20": b[2],
            "diff5": e["fwd5n"] - b[0], "diff10": e["fwd10n"] - b[1], "diff20": e["fwd20n"] - b[2],
            "ext5": ext[0], "ext10": ext[1], "ext20": ext[2],
            "dext5": e["fwd5n"] - ext[0], "dext10": e["fwd10n"] - ext[1], "dext20": e["fwd20n"] - ext[2],
        })

    df = pd.DataFrame(rows)
    # 单行汇总（类型 × 桶）
    summary = []
    for (tp, bk), g in df.groupby(["type", "bucket"]):
        d = g["diff20"].dropna().values
        de = g["dext20"].dropna().values
        fg = g["fwd20g"].values
        n = len(d)
        mean = d.mean() if n else 0.0
        sd = d.std() if n > 1 else 0.0
        t = mean / (sd / np.sqrt(n)) if (n > 1 and sd > 0) else 0.0
        emean = de.mean() if len(de) else 0.0
        esd = de.std() if len(de) > 1 else 0.0
        et = emean / (esd / np.sqrt(len(de))) if (len(de) > 1 and esd > 0) else 0.0
        win = (g["fwd20n"].values > 0).mean() if n else 0.0
        summary.append({"type": tp, "bucket": bk, "n": n,
                        "avg_fwd20n": g["fwd20n"].mean() if n else 0.0,
                        "win20": win,
                        "diff20_bg": mean, "t20_bg": t,
                        "diff20_ext": emean, "t20_ext": et})
    sdf = pd.DataFrame(summary)
    return df, sdf


def print_summary(sdf, label=""):
    print(f"\n{'='*86}\n  MACD 背离诊断汇总 {label}\n{'='*86}")
    print(f"  {'类型':<9}{'状态桶':<14}{'n':>6}{'净20':>8}{'胜率':>7}"
          f"{'差(背景)':>11}{'t_bg':>7}{'差(极点)':>11}{'t_ext':>7}")
    print(f"  {'─'*82}")
    for _, r in sdf.iterrows():
        print(f"  {r['type']:<9}{r['bucket']:<14}{int(r['n']):>6}"
              f"{r['avg_fwd20n']:>+7.2%}{r['win20']:>6.1%}"
              f"{r['diff20_bg']:>+10.2%}{r['t20_bg']:>7.2f}"
              f"{r['diff20_ext']:>+10.2%}{r['t20_ext']:>7.2f}")
    print(f"\n  判定（n≥50 且 |t|≥2 且 背景/极点同号 → 该状态有 edge）：")
    for _, r in sdf.iterrows():
        if int(r['n']) >= 50 and abs(r['t20_bg']) >= 2 and abs(r['t20_ext']) >= 2 \
           and r['diff20_bg'] * r['diff20_ext'] > 0:
            sign = "正" if r['diff20_ext'] > 0 else "负"
            print(f"    ★ {r['type']}/{r['bucket']}: n={int(r['n'])} "
                  f"背景差={r['diff20_bg']:+.2%} 极点差={r['diff20_ext']:+.2%} "
                  f"t_ext={r['t20_ext']:.2f} → 有{sign}向 edge（MACD 确认贡献）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="MACD 背离诊断器（对照实验验证指标）")
    ap.add_argument("--pool", default="hs300")
    ap.add_argument("--start", default="20140301")
    ap.add_argument("--end", default="20260731")
    ap.add_argument("--macd-fast", type=int, default=12)
    ap.add_argument("--macd-slow", type=int, default=26)
    ap.add_argument("--macd-signal", type=int, default=9)
    ap.add_argument("--pivot-window", type=int, default=10)
    ap.add_argument("--lookback", type=int, default=60)
    ap.add_argument("--min-gap", type=int, default=10)
    ap.add_argument("--cost", type=float, default=0.002)
    ap.add_argument("--bb-win", type=int, default=20)
    ap.add_argument("--bb-lookback", type=int, default=120)
    ap.add_argument("--bb-th", type=float, default=0.25)
    ap.add_argument("--sanity", action="store_true", help="打乱 MACD 重跑（edge 应≈0）")
    ap.add_argument("--grid", action="store_true", help="跑 pivot×lookback 网格")
    ap.add_argument("--no-save", action="store_true", help="不落 CSV")
    a = ap.parse_args()

    base_P = dict(fast=a.macd_fast, slow=a.macd_slow, signal=a.macd_signal,
                  pivot_window=a.pivot_window, lookback=a.lookback, min_gap=a.min_gap,
                  cost=a.cost, bb_win=a.bb_win, bb_lookback=a.bb_lookback, bb_th=a.bb_th)

    if a.grid:
        for pw in (8, 10, 15):
            for lb in (40, 60):
                P = dict(base_P); P["pivot_window"] = pw; P["lookback"] = lb
                df, sdf = run_study(a.pool, a.start, a.end, P, progress=False)
                print_summary(sdf, label=f"[grid pw={pw} lb={lb}]")
        sys.exit(0)

    df, sdf = run_study(a.pool, a.start, a.end, base_P, sanity=a.sanity)
    tag = "_sanity" if a.sanity else ""
    print_summary(sdf, label=f"[{a.pool} {a.start}~{a.end}]{tag}")
    if not a.no_save:
        out = "data/results/macd_study"
        os.makedirs(out, exist_ok=True)
        ev_csv = f"{out}/macd_div_events_{a.pool}_{a.start}_{a.end}{tag}.csv"
        su_csv = f"{out}/macd_div_summary_{a.pool}_{a.start}_{a.end}{tag}.csv"
        df.to_csv(ev_csv, index=False)
        sdf.to_csv(su_csv, index=False)
        print(f"\n  事件明细 → {ev_csv}\n  汇总 → {su_csv}")
