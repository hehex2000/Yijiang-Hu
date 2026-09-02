#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
逐笔 raw-NAV 暴露测量器 (只读, 不改任何策略代码)
==================================================
用途: 在给某个回测脚本接入 hfq 之前, 先用它**已导出的 trades CSV** 量化
      "raw 口径到底漏记了多少收益", 从而按 cheapest/most-impact 排序决定
      哪些脚本值得花力气改造。

原理:
  每笔 round-trip(买入→卖出) 期间, 后复权因子 f 会因分红/送转而上升。
  正确记账下, 卖出所得应为  shares × sell_px × f(sell)/f(buy)。
  raw 口径只用 shares × sell_px, 于是漏掉:
      phantom_i = shares × sell_px × (f_sell/f_buy − 1)
  —— 现金分红部分 = 真漏记收益; 送转股部分 = 假亏损(raw 价除权暴跌但股数没变)。

核心指标(跨策略可比):
      drag = Σphantom_i / Σ(capital_i × years_i)     [每元每年漏记多少收益]
  其中 capital_i = shares × buy_px, years_i = 持有交易日/252。

用法:
  ./venv_ml/Scripts/python.exe _measure_raw_exposure.py <trades.csv> [<trades2.csv> ...]
  ./venv_ml/Scripts/python.exe _measure_raw_exposure.py --dir data/results/darvas
"""
import os
import sys
import sqlite3
from collections import defaultdict, deque

import pandas as pd
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
DB = r"D:\tu-shareData\astock_daily.db"


# ──────────────────────────────────────────────────────────────
#  因子加载: 按 code 批量取 [min,max] 区间, 再做 as-of(ffill) 查找
# ──────────────────────────────────────────────────────────────
def load_factors(codes, d0, d1):
    """返回 {code: (dates_ndarray, factors_ndarray)}, 已按日期升序。"""
    conn = sqlite3.connect(DB)
    out = {}
    q = ("SELECT ts_code, trade_date, adj_factor FROM adj_factor "
         "WHERE ts_code=? AND trade_date BETWEEN ? AND ? ORDER BY trade_date")
    for c in codes:
        df = pd.read_sql_query(q, conn, params=(c, str(d0), str(d1)))
        if len(df) == 0:
            out[c] = None
            continue
        out[c] = (df["trade_date"].astype(str).to_numpy(),
                  df["adj_factor"].astype(float).to_numpy())
    conn.close()
    return out


def asof(fmap, code, date):
    """as-of 因子: 取 <= date 的最近一条。天然 ffill(继承缺行日)。
    买入日无因子 → 返回 None(调用方须把该持仓锁 1.0, 绝不能当 1.0 用绝对比值)。"""
    e = fmap.get(code)
    if e is None:
        return None
    ds, fs = e
    i = np.searchsorted(ds, str(date), side="right") - 1
    if i < 0:
        return None
    return float(fs[i])


# ──────────────────────────────────────────────────────────────
#  FIFO 配对
# ──────────────────────────────────────────────────────────────
def is_buy(a):
    s = str(a)
    return ("买" in s) or ("BUY" in s.upper())


def is_sell(a):
    s = str(a)
    return ("卖" in s) or ("SELL" in s.upper())


def pair_roundtrips(df):
    """FIFO 配对, 返回 list of dict(buy_date,sell_date,code,shares,buy_px,sell_px,ndays)"""
    lots = defaultdict(deque)
    rt = []
    for _, r in df.iterrows():
        code = str(r["code"])
        d = str(r["date"])
        sh = float(r["shares"])
        px = float(r["price"])
        act = str(r["action"])
        if is_buy(act):
            lots[code].append([d, px, sh])
        elif is_sell(act):
            need = sh
            while need > 1e-9 and lots[code]:
                lot = lots[code][0]
                take = min(lot[2], need)
                rt.append(dict(code=code, buy_date=lot[0], sell_date=d,
                               shares=take, buy_px=lot[1], sell_px=px))
                lot[2] -= take
                need -= take
                if lot[2] <= 1e-9:
                    lots[code].popleft()
    open_lots = sum(len(v) for v in lots.values())
    open_sh = sum(float(l[2]) for v in lots.values() for l in v)
    return rt, open_lots, open_sh


# ──────────────────────────────────────────────────────────────
#  主测量
# ──────────────────────────────────────────────────────────────
def measure(path, verbose=True):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.dropna(subset=["date", "action", "code", "price", "shares"])
    df["date"] = df["date"].astype(str).str.replace(r"\.0$", "", regex=True)
    df = df.sort_values("date").reset_index(drop=True)

    rt, open_lots, open_sh = pair_roundtrips(df)
    if not rt:
        print(f"[跳过] {path}: 无法配对任何 round-trip")
        return None

    codes = sorted({r["code"] for r in rt})
    d0 = min(r["buy_date"] for r in rt)
    d1 = max(r["sell_date"] for r in rt)
    fmap = load_factors(codes, d0, d1)

    rows = []
    for r in rt:
        fb = asof(fmap, r["code"], r["buy_date"])
        fs = asof(fmap, r["code"], r["sell_date"])
        if fb is None or fs is None or fb <= 0:
            ratio = 1.0        # 因子缺失 → 无法测量, 保守记为无漏记
            missing = True
        else:
            ratio = fs / fb
            missing = False
        cap = r["shares"] * r["buy_px"]
        raw_pnl = r["shares"] * (r["sell_px"] - r["buy_px"])
        hfq_pnl = r["shares"] * (r["sell_px"] * ratio - r["buy_px"])
        phantom = hfq_pnl - raw_pnl
        # 持有年数: 用交易日序号差近似(无日历表时按 252 折算)
        rows.append(dict(code=r["code"], buy_date=r["buy_date"],
                         sell_date=r["sell_date"], shares=r["shares"],
                         buy_px=r["buy_px"], sell_px=r["sell_px"],
                         ratio=ratio, cap=cap, raw_pnl=raw_pnl,
                         hfq_pnl=hfq_pnl, phantom=phantom,
                         missing=missing))
    t = pd.DataFrame(rows)

    # 持有天数: 用全局交易日表算准确值
    span = _trading_days(d0, d1)
    if span is not None:
        pos = {d: i for i, d in enumerate(span)}
        def nd(r):
            a, b = pos.get(r["buy_date"]), pos.get(r["sell_date"])
            return (b - a) if (a is not None and b is not None) else np.nan
        t["ndays"] = t.apply(nd, axis=1)
    else:
        t["ndays"] = np.nan
    t["years"] = t["ndays"] / 252.0

    ok = t[~t["missing"]]
    capyr = float((t["cap"] * t["years"]).sum())
    phantom_tot = float(t["phantom"].sum())
    raw_tot = float(t["raw_pnl"].sum())
    hfq_tot = float(t["hfq_pnl"].sum())
    drag = phantom_tot / capyr if capyr > 0 else float("nan")

    # ── 两种年化口径, 必须分开看 (短持仓策略会被 drag 严重放大) ──
    #  drag_deployed = 漏记/部署资本·年  → 衡量"持仓期内的漏记强度", 会被短持仓放大
    #  drag_nav      = 漏记/(初始资本×回测年数) → 一阶近似"头条年化收益被低估多少"
    #                  (未计复利, 实为下界: 真实 hfq 后期仓位更大, 漏记复利更多)
    span_days = _trading_days(d0, d1)
    yrs_span = (len(span_days) / 252.0) if span_days else float("nan")
    cap0 = _guess_capital(path)
    drag_nav = (phantom_tot / (cap0 * yrs_span)
                if (cap0 and yrs_span and yrs_span > 0) else float("nan"))

    # 事件分级
    split = ok[ok["ratio"] >= 1.5]          # 送转股(除权暴跌) → raw 假亏损
    big_div = ok[(ok["ratio"] > 1.0) & (ok["ratio"] < 1.5)]
    flat = ok[np.isclose(ok["ratio"], 1.0)]

    res = dict(
        file=os.path.basename(path), n_rt=len(t), n_missing=int(t["missing"].sum()),
        span=f"{d0[:4]}~{d1[:4]}", years=yrs_span, capital=cap0,
        raw_pnl=raw_tot, hfq_pnl=hfq_tot, phantom=phantom_tot,
        capyr=capyr, drag=drag, drag_nav=drag_nav,
        n_split=len(split), n_div=len(big_div), n_flat=len(flat),
        pct_split=len(split) / len(ok) * 100 if len(ok) else 0,
        pct_div=len(big_div) / len(ok) * 100 if len(ok) else 0,
        med_days=float(t["ndays"].median()) if t["ndays"].notna().any() else float("nan"),
        open_lots=open_lots, open_shares=open_sh,
    )

    if verbose:
        print("=" * 92)
        print(f"{res['file']}")
        print("=" * 92)
        print(f"  区间 {d0} ~ {d1}   配对 round-trip {res['n_rt']} 笔"
              f"   期末未平仓 {open_lots} 批 / {open_sh:,.0f} 股")
        print(f"  中位持有 {res['med_days']:.0f} 交易日")
        print(f"  已实现盈亏   raw {raw_tot:+,.0f}   hfq {hfq_tot:+,.0f}"
              f"   ★漏记 {phantom_tot:+,.0f}")
        print(f"  因子缺失(无法测量) {res['n_missing']} 笔")
        print(f"  ── 事件分级 (按 f卖/f买) ──")
        print(f"  送转股(≥1.5)      : {res['n_split']:>4} 笔  ({res['pct_split']:.1f}%)"
              f"   ← raw 记成假亏损, 最危险")
        print(f"  分红(1.0<r<1.5)   : {res['n_div']:>4} 笔  ({res['pct_div']:.1f}%)"
              f"   ← raw 漏记真收益")
        print(f"  无变化            : {res['n_flat']:>4} 笔")
        print(f"  ── 年化拖累 (两个口径都要看) ──")
        print(f"  ① 漏记强度 drag_deployed = {drag*100:+.2f}%/年"
              f"   (部署资本·年 {capyr:,.0f})  ← 会被短持仓放大")
        print(f"  ② 头条影响 drag_nav      = {drag_nav*100:+.2f}pp/年"
              f"   (初始资本 {cap0:,.0f} × {yrs_span:.1f}年)  ← 一阶下界, 决定要不要改")
        verdict = ("★ 值得改造" if drag_nav >= 0.01 else
                   "· 边缘, 可只标注" if drag_nav >= 0.003 else
                   "· 可忽略, 不值得改")
        print(f"  判定: {verdict}")
        if len(split):
            print(f"  ── 送转股地雷明细 (前10) ──")
            s = split.reindex(split["phantom"].abs().sort_values(ascending=False).index).head(10)
            for _, r in s.iterrows():
                print(f"   {r['code']}  {r['buy_date']}→{r['sell_date']}"
                      f"  f比 {r['ratio']:.3f}  假亏损 {r['phantom']:+,.0f}")
        print()
    return res


def _guess_capital(path):
    """猜初始资本: 文件名里的 _c<digits>_ 优先(peg 等), 否则用平台默认 INIT_CAPITAL。
    可用 --capital 覆盖。"""
    if _CLI_CAPITAL[0]:
        return float(_CLI_CAPITAL[0])
    import re
    m = re.search(r"_c(\d+)_", os.path.basename(path))
    if m:
        v = int(m.group(1))
        if v >= 1000:
            return float(v)
    return 200000.0      # run_monthly_rebalance.INIT_CAPITAL


_CLI_CAPITAL = [None]


_TD_CACHE = {}


def _trading_days(d0, d1):
    key = (d0, d1)
    if key in _TD_CACHE:
        return _TD_CACHE[key]
    conn = sqlite3.connect(DB)
    try:
        df = pd.read_sql_query(
            "SELECT trade_date FROM index_daily WHERE ts_code='000300.SH' "
            "AND trade_date BETWEEN ? AND ? ORDER BY trade_date",
            conn, params=(str(d0), str(d1)))
        v = df["trade_date"].astype(str).tolist() if len(df) else None
    except Exception:
        v = None
    conn.close()
    _TD_CACHE[key] = v
    return v


def main():
    args = [a for a in sys.argv[1:]]
    if "--capital" in args:
        i = args.index("--capital")
        if i + 1 < len(args):
            _CLI_CAPITAL[0] = float(args[i + 1])
            args = args[:i] + args[i + 2:]
    if not args:
        print(__doc__)
        return
    files = []
    if args[0] == "--dir":
        d = os.path.join(BASE, args[1]) if len(args) > 1 else BASE
        pat = args[2] if len(args) > 2 else "*trades*.csv"
        import glob
        files = sorted(glob.glob(os.path.join(d, pat)))
    else:
        files = [a if os.path.isabs(a) else os.path.join(BASE, a) for a in args]

    res = []
    for f in files:
        if not os.path.exists(f):
            print(f"[跳过] 不存在 {f}")
            continue
        r = measure(f)
        if r:
            res.append(r)

    if len(res) > 1:
        print("=" * 92)
        print("汇总 (按 drag_nav 降序 = 头条年化被低估最多者优先改造)")
        print("=" * 92)
        t = pd.DataFrame(res)
        t = t.sort_values("drag_nav", ascending=False)
        show = t[["file", "span", "n_rt", "med_days", "n_split", "n_div",
                  "phantom", "drag", "drag_nav"]].copy()
        show["med_days"] = show["med_days"].map(lambda x: f"{x:.0f}")
        show["phantom"] = show["phantom"].map(lambda x: f"{x:+,.0f}")
        show["drag"] = show["drag"].map(lambda x: f"{x*100:+.2f}%")
        show["drag_nav"] = show["drag_nav"].map(lambda x: f"{x*100:+.2f}pp")
        print(show.to_string(index=False))
        print("\n  drag_nav ≥1pp/年 → 值得改造; 0.3~1pp → 边缘; <0.3pp → 不值得")
        out = os.path.join(BASE, "data", "results", "tr_index",
                           "raw_exposure_measure.csv")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        t.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"\n已保存: {out}")


if __name__ == "__main__":
    main()
