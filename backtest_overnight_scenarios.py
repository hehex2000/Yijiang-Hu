# -*- coding: utf-8 -*-
"""
隔夜持股 · 封板/延长持有 情景建模
复用 backtest_overnight 的 load_data / build_signals / select。

情景：
  base_open        = 次日开盘卖出（基准，= backtest_overnight 的 gross_ret）
  hold_close       = 次日收盘卖出（延长半天）
  forced_ld        = 次日开盘跌停→强制持有到首个非跌停开盘日卖出
  limitup_sellnow  = 次日开盘涨停→开盘立刻卖（+9.5%）
  limitup_holdcls  = 次日开盘涨停→持有到次日收盘
"""
import sys, os, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backtest_overnight as B

LIMIT_DOWN = 0.905   # 开盘 <= 前收*0.905 视为跌停开盘
LIMIT_UP   = 1.095   # 开盘 >= 前收*1.095 视为涨停开盘
FORCE_CAP  = 15      # 强制持有最多 15 个交易日


def add_next(df):
    g = df.groupby("ts_code", sort=False)
    df["next_close"] = g["close"].shift(-1)
    df["next_high"]  = g["high"].shift(-1)
    df["next_low"]   = g["low"].shift(-1)
    return df


def build_price_map(df):
    """per-ts_code 有序序列，用于强制持有walk"""
    pm = {}
    for code, sub in df.groupby("ts_code", sort=False):
        s = sub.sort_values("trade_date")
        pm[code] = (
            s["trade_date"].to_numpy(),
            s["open"].to_numpy(dtype="float64"),
            s["pre_close"].to_numpy(dtype="float64"),
            s["close"].to_numpy(dtype="float64"),
        )
    return pm


def walk_forced_hold(pm, code, next_td, buy_close):
    """从 next_td 起逐日找首个非跌停开盘日卖出；返回(ret, steps, hit_cap)"""
    tds, opens, pres, _ = pm.get(code, (None, None, None, None))
    if tds is None:
        return np.nan, 0, False
    # 定位 next_td
    idx = np.searchsorted(tds, next_td)
    if idx >= len(tds):
        return np.nan, 0, False
    steps = 0
    for k in range(idx, min(idx + FORCE_CAP, len(tds))):
        o, pre = opens[k], pres[k]
        if pre is None or np.isnan(o) or np.isnan(pre) or pre <= 0:
            break
        steps += 1
        if o > pre * LIMIT_DOWN:   # 非跌停开盘 → 可卖
            return o / buy_close - 1.0, steps, False
        # 仍为跌停开盘，继续持有
    # 触及上限仍未解封：以最后一日开盘价卖出（最差情形）
    last = opens[min(idx + FORCE_CAP - 1, len(tds) - 1)]
    return (last / buy_close - 1.0) if (last and buy_close) else np.nan, steps, True


def scenario_stats(rets):
    r = np.asarray(rets, dtype="float64")
    r = r[~np.isnan(r)]
    if len(r) == 0:
        return dict(n=0, mean=np.nan, median=np.nan, win=np.nan, max_loss=np.nan)
    return dict(n=len(r), mean=r.mean() * 100, median=np.median(r) * 100,
                win=(r > 0).mean() * 100, max_loss=r.min() * 100)


def run_mode(df, pm, mode):
    sub = B.select(df, mode).copy()
    if len(sub) == 0:
        return {}
    c = sub["close"].to_numpy(dtype="float64")
    no = sub["next_open"].to_numpy(dtype="float64")
    nc = sub["next_close"].to_numpy(dtype="float64")
    ntd = sub["next_td"].to_numpy()
    codes = sub["ts_code"].to_numpy()
    ld = sub["next_limit_down_open"].to_numpy()
    lu = sub["next_limit_up_open"].to_numpy()

    base = no / c - 1.0
    holdcls = nc / c - 1.0

    # 强制持有（仅跌停开盘的做 walk；其余用 base）
    forced = base.copy()
    n_forced = 0
    worst = 0.0
    for i in np.where(ld)[0]:
        ret, steps, cap = walk_forced_hold(pm, codes[i], int(ntd[i]), c[i])
        if not np.isnan(ret):
            forced[i] = ret
            n_forced += 1
            worst = min(worst, ret)

    # 涨停开盘：开盘卖 vs 持有到收盘
    lu_idx = np.where(lu)[0]
    lu_sellnow = base[lu_idx] if len(lu_idx) else np.array([])
    lu_holdcls = holdcls[lu_idx] if len(lu_idx) else np.array([])

    return {
        "base_open":      scenario_stats(base),
        "hold_close":     scenario_stats(holdcls),
        "forced_ld":      scenario_stats(forced),
        "forced_ld_n":    n_forced,
        "forced_ld_worst": worst * 100,
        "limitup_n":      len(lu_idx),
        "limitup_sellnow": scenario_stats(lu_sellnow),
        "limitup_holdcls": scenario_stats(lu_holdcls),
    }


def main():
    start, end = "20150101", "20260630"
    t0 = time.time()
    print(f"[load] {start}~{end} ...", flush=True)
    df = B.load_data(start, end)
    df = B.build_signals(df)
    df = add_next(df)
    pm = build_price_map(df)
    print(f"[load done] {len(df):,} rows, {time.time()-t0:.1f}s", flush=True)

    out = {}
    for mode in ["all", "momentum", "sector"]:
        out[mode] = run_mode(df, pm, mode)
        s = out[mode]
        print(f"\n=== {mode} (N={s['base_open']['n']:,}) ===", flush=True)
        print(f"  base_open     : {s['base_open']['mean']:+.3f}%  胜率{s['base_open']['win']:.1f}%  中位{s['base_open']['median']:+.3f}%")
        print(f"  hold_close    : {s['hold_close']['mean']:+.3f}%  胜率{s['hold_close']['win']:.1f}%  中位{s['hold_close']['median']:+.3f}%")
        print(f"  forced_ld     : {s['forced_ld']['mean']:+.3f}%  胜率{s['forced_ld']['win']:.1f}%  强制持有{s['forced_ld_n']}笔")
        print(f"     ↳ 强制持有笔数={s['forced_ld_n']}  最差单笔={s['forced_ld_worst']:+.1f}%")
        if s['limitup_n']:
            print(f"  涨停开盘 {s['limitup_n']} 笔: 开盘卖 {s['limitup_sellnow']['mean']:+.2f}% vs 持有到收盘 {s['limitup_holdcls']['mean']:+.2f}%")

    # 存 CSV
    rows = []
    for mode, s in out.items():
        for sc in ["base_open", "hold_close", "forced_ld", "limitup_sellnow", "limitup_holdcls"]:
            st = s[sc]
            rows.append(dict(mode=mode, scenario=sc, n=st["n"], mean_ret_pct=round(st["mean"],4),
                             median_pct=round(st["median"],4), win_pct=round(st["win"],2),
                             max_loss_pct=round(st["max_loss"],2)))
    od = "data/results/overnight"
    os.makedirs(od, exist_ok=True)
    pd.DataFrame(rows).to_csv(f"{od}/scenarios.csv", index=False, encoding="utf-8-sig")
    print(f"\n[done] 已保存 {od}/scenarios.csv  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
