# -*- coding: utf-8 -*-
"""
P6 · 定期调仓检验：多久换一次仓最划算？
=====================================================================

用户拍板的行动优先级：**扩大池子（5→15+，P5 已完成）> 定期调仓 > 调因子权重**。
本轮做第二条。

§3.8 已给出铁证——持有期是主效应：同样 2020-01-02 选出的 5 只白酒，
**持有 1 年 BH=+60.23%，持有 6.6 年 BH=−20.44%**。所以"多久换一次仓"
可能是三条里收益最大的一条。

★ 实现上的关键：不做「每段强平再买」
------------------------------------------------------------------
朴素做法是每段末全部平仓、下段重新买入。但那会给**仍留在池里的票**
凭空多算一次卖+买的双边成本，月度调仓 80 段 ⇒ 80 次虚假换手，
严重高估调仓成本、低估调仓收益。

本脚本改为**全局时间轴单次 simulate**：
  · 每段选股后抽该段信号（按 [seg_start, seg_end] 截断）
  · 段末只对「下段不在池」的票插入强制 SELL（换仓）
  · 仍在池的票：持仓自然结转到下段（simulate 的 shares_vec 是全局状态）
  · 买入受 `if shares_vec[j] > 0: continue` 保护 ⇒ 不会重复建仓
  ⇒ 换手成本只发生在真正换掉的票上，与真实调仓一致

频率：static（只选一次）/ annual / semi / quarterly / monthly

用法:  venv_ml/Scripts/python.exe run_rebalance_freq_check.py [freq ...]
       例: ... monthly quarterly      # 只跑指定频率
       不传参数则跑全部
"""
import sqlite3
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import config  # noqa: E402
import run_backtest as rb  # noqa: E402
from backtest.mean_reversion_plugin import MeanReversionStrategyPlugin  # noqa: E402
from backtest.portfolio_engine import extract_signals, simulate  # noqa: E402

START, END = "20200103", "20260825"
TOTAL = float(config.BACKTEST["total_capital"])
N_POOL = 20                 # P5 推荐值（f = 2/20 = 0.10）
F_POS = 0.10
CAL_CODE = "600519.SH"      # 仅用于取交易日历
OUT = Path("data/results/position_sizing/rebalance_freq.csv")

FREQS = ["static", "annual", "semi", "quarterly", "monthly"]
FREQ_PD = {"static": None, "annual": "YS", "semi": "6MS",
           "quarterly": "QS", "monthly": "MS"}


def get_calendar(conn) -> list:
    df = rb.load_stock_prices(CAL_CODE, START, END, conn, lookback_days=0)
    return sorted(d for d in df["trade_date"].tolist() if d >= START)


def make_segments(freq: str, cal: list) -> list:
    """返回 [(seg_start, seg_end), ...]，seg_end = 下一段开始的前一交易日。"""
    if freq == "static" or not cal:
        return [(START, END)]
    dr = pd.date_range(start=START, end=END, freq=FREQ_PD[freq])
    bounds = []
    for d in dr:
        ymd = d.strftime("%Y%m%d")
        nxt = next((c for c in cal if c >= ymd), None)
        if nxt and nxt not in bounds:
            bounds.append(nxt)
    bounds = [cal[0]] + [b for b in bounds if b > cal[0]]
    segs = []
    for i, b in enumerate(bounds):
        e = bounds[i + 1] if i + 1 < len(bounds) else None
        if e is None:
            segs.append((b, cal[-1]))
        else:
            idx = cal.index(e) - 1
            segs.append((b, cal[idx] if idx >= 0 else b))
    return [s for s in segs if s[0] <= s[1]]


def main():
    freqs = [f for f in sys.argv[1:] if f in FREQS] or FREQS
    print("=" * 92)
    print("P6 · 定期调仓检验（全局时间轴单次 simulate，换仓成本只算真换掉的票）")
    print("=" * 92)
    print(f"  区间 {START}~{END}｜池 N={N_POOL}（f={F_POS}）｜总资金 {TOTAL:,.0f}"
          f"｜频率 {freqs}")

    conn = sqlite3.connect(config.DATA["local_db_path"])
    cal = get_calendar(conn)
    print(f"  交易日 {len(cal)} 天（{cal[0]} ~ {cal[-1]}）")

    orig_top_n = config.SELECTION["top_n"]
    df_cache = {}
    rows = []

    def get_df(code):
        if code not in df_cache:
            try:
                df_cache[code] = rb.load_stock_prices(
                    code, START, END, conn, lookback_days=250)
            except Exception:  # noqa: BLE001
                df_cache[code] = None
        return df_cache[code]

    try:
        for freq in freqs:
            t0 = time.time()
            segs = make_segments(freq, cal)
            print("\n" + "=" * 92)
            print(f"  【{freq}】{len(segs)} 段")
            print("=" * 92)

            seg_codes = []
            for si, (s, e) in enumerate(segs):
                config.BACKTEST["start_date"] = s
                config.BACKTEST["end_date"] = e
                config.SELECTION["top_n"] = N_POOL
                try:
                    sel = rb.run_selection()
                except Exception:  # noqa: BLE001
                    print(f"    [FAIL] {s} 选股异常")
                    traceback.print_exc()
                    seg_codes.append([])
                    continue
                if sel is None or sel.empty:
                    seg_codes.append([])
                    continue
                codes = [str(c).zfill(6) for c in sel["code"].tolist()[:N_POOL]]
                seg_codes.append(codes)
                if (si + 1) % 10 == 0 or si == len(segs) - 1:
                    print(f"    段 {si+1}/{len(segs)}  {s}~{e}  选出 {len(codes)} 只")

            # ── 抽信号：每段独立抽，按段截断 ──
            cfg = dict(config.STRATEGIES["mean_reversion"])
            cfg["use_kelly"] = False
            cfg["real_cost"] = False
            all_events = []
            for si, (s, e) in enumerate(segs):
                codes = seg_codes[si] if si < len(seg_codes) else []
                if not codes:
                    continue
                sd = {}
                for c in codes:
                    df = get_df(c)
                    if df is not None and len(df) > 30:
                        sd[c] = ("", df, 0)
                if not sd:
                    continue
                try:
                    ev, _ = extract_signals(sd, MeanReversionStrategyPlugin, cfg, s)
                except Exception:  # noqa: BLE001
                    continue
                all_events.extend(x for x in ev if s <= x["date"] <= e)

            # ── 全局 px（用于盯市 + 换仓卖价）──
            px_series = {}
            for c, df in df_cache.items():
                if df is None:
                    continue
                d = df[["trade_date", "adj_close"]].copy()
                d = d[d["trade_date"] >= START]
                px_series[c] = d.set_index("trade_date")["adj_close"].astype(float)
            if not px_series:
                print("    [SKIP] 无价格数据")
                continue
            px = pd.DataFrame(px_series).sort_index().ffill().fillna(0.0)

            # ── 段末换仓：只对「下段不在池」的票插 SELL ──
            n_switch = 0
            for si, (s, e) in enumerate(segs):
                codes = seg_codes[si] if si < len(seg_codes) else []
                if not codes:
                    continue
                nxt = set(seg_codes[si + 1]) if si + 1 < len(seg_codes) else set()
                if si + 1 >= len(segs):
                    continue          # 最后一段交给引擎期末强平
                if e not in px.index:
                    continue
                for c in codes:
                    if c in nxt:
                        continue      # 仍在池 ⇒ 持仓自然结转，不卖
                    if c not in px.columns:
                        continue
                    p = float(px.loc[e, c])
                    if p <= 0:
                        continue
                    all_events.append(
                        {"date": e, "code": c, "action": "SELL", "price": p,
                         "_switch": True})
                    n_switch += 1

            all_events.sort(key=lambda x: (x["date"], x["code"]))
            if not all_events:
                print("    [SKIP] 无信号")
                continue

            m, nav = simulate(all_events, px, TOTAL, f=F_POS, cap=0,
                              label=f"REBAL_{freq}")
            years = (pd.Timestamp(END) - pd.Timestamp(START)).days / 365.25
            term = m["terminal"]
            cagr = ((term / TOTAL) ** (1 / years) - 1) * 100 if term > 0 else -100.0

            rows.append({
                "freq": freq, "n_seg": len(segs), "n_switch_sell": n_switch,
                "terminal": term, "total_ret_pct": m["total_ret_pct"],
                "cagr_pct": cagr, "sharpe": m["sharpe"], "mdd_pct": m["mdd_pct"],
                "exposure": m["exposure"] * 100, "n_taken": m["n_taken"],
                "n_closed": m["n_closed"], "win_rate_pct": m["win_rate_pct"],
            })
            print(f"    → 终值 {term:>12,.0f}  总收益 {m['total_ret_pct']:>7.2f}%  "
                  f"CAGR {cagr:>6.2f}%  Sharpe {m['sharpe']:>5.2f}  "
                  f"MDD {m['mdd_pct']:>7.2f}%  暴露 {m['exposure']*100:>5.1f}%  "
                  f"成交 {m['n_taken']}  换仓卖出 {n_switch}  "
                  f"[{time.time()-t0:.0f}s]")
    finally:
        config.SELECTION["top_n"] = orig_top_n
        conn.close()

    if not rows:
        print("\n[FAIL] 无结果")
        return 1

    R = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    R.to_csv(OUT, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 92)
    print("  汇总 · 调仓频率对比")
    print("=" * 92)
    print(f"  {'频率':<11}{'段数':>5}{'总收益%':>10}{'CAGR%':>8}{'Sharpe':>8}"
          f"{'MDD%':>9}{'暴露%':>8}{'成交':>7}{'换仓卖':>8}")
    print("  " + "-" * 76)
    for _, r in R.iterrows():
        print(f"  {r['freq']:<11}{int(r['n_seg']):>5}{r['total_ret_pct']:>10.2f}"
              f"{r['cagr_pct']:>8.2f}{r['sharpe']:>8.2f}{r['mdd_pct']:>9.2f}"
              f"{r['exposure']:>8.1f}{int(r['n_taken']):>7}{int(r['n_switch_sell']):>8}")

    b = R[R["freq"] == "static"]
    if not b.empty:
        bc = float(b.iloc[0]["cagr_pct"])
        print("  " + "-" * 76)
        print(f"  相对 static（CAGR {bc:.2f}%）的改善：")
        for _, r in R.iterrows():
            if r["freq"] == "static":
                continue
            print(f"    {r['freq']:<10} {r['cagr_pct'] - bc:>+7.2f}pp"
                  f"   Sharpe {r['sharpe'] - float(b.iloc[0]['sharpe']):>+6.2f}"
                  f"   MDD {r['mdd_pct'] - float(b.iloc[0]['mdd_pct']):>+7.2f}pp")

    print("=" * 92)
    print(f"  明细 → {OUT}")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    sys.exit(main())
