"""
跨策略 PK 驾驶舱
================
把四套「已验证/改良」策略放在同一区间、同一初始资金、同一基准下对比：
  - PEG 最优版 (质量+动量+VaR2.5%)        —— run_peg.py
  - 神奇公式 (Magic Formula, 年度等权)    —— run_magic_formula.py
  - 神奇公式 V2.1 (EBIT3年均值+MA200+⑧暴涨护栏1.5+分散15只) —— run_magic_v2.py
  - 红利低波 (官方compact: 季频/股息率加权) —— run_dividend_low_vol_quality_bt.py

统一口径：
  - 区间 20140101~20260715（红利低波模块默认 20200101~20260723，已显式覆盖为 2014 对齐）
  - 初始资金 1,000,000
  - 基准：中证全指(000985.SH) 买入持有 + 沪深300(000300.SH) 买入持有，均自实现同一 NAV 函数
  - 指标：总收益/年化/最大回撤/夏普，三家使用同一套公式（与 run_peg/run_magic 原生口径一致）
  - 红利低波 NAV 由其原生引擎(run_nav_weighted)产出后，用统一函数复算指标，确保夏普口径一致
"""
import os, sys, time, glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_peg
import run_magic_formula as mf
import run_dividend_low_vol_quality_bt as dlv
import run_magic_v2 as mv2
from run_monthly_rebalance import get_trade_dates
import sqlite3, config

START, END = "20140101", "20260715"
CAP = 1_000_000
DB = os.path.abspath(config.DATA["local_db_path"])

# 让红利低波模块用与另两家相同的窗口（其模块全局默认 20200101~20260723）
dlv.START = START
dlv.END = END
dlv.INIT_CAPITAL = float(CAP)


def index_nav(code):
    """指数买入持有 NAV（与策略同 all_dates 对齐）。"""
    conn = sqlite3.connect(DB)
    df = pd.read_sql(
        f"SELECT trade_date,close FROM index_daily WHERE ts_code='{code}' "
        f"AND trade_date BETWEEN '{START}' AND '{END}' ORDER BY trade_date", conn)
    conn.close()
    bmap = dict(zip(df["trade_date"].astype(str), df["close"].astype(float)))
    alld = get_trade_dates(START, END)
    lv = [bmap.get(d) for d in alld]
    fv = next((i for i, v in enumerate(lv) if v is not None), None)
    if fv is None:
        return [CAP] * len(alld)
    base = lv[fv]
    out = []
    for i, v in enumerate(lv):
        if v is None:
            j = i - 1
            while j > fv and lv[j] is None:
                j -= 1
            out.append((lv[j] / base * CAP) if j >= fv else CAP)
        else:
            out.append(v / base * CAP)
    return out


def metrics(dates, vals):
    """统一指标：总收益/年化/最大回撤/夏普（与 run_peg / run_magic 原生口径一致）。"""
    vals = np.asarray(vals, float)
    total = vals[-1] / vals[0] - 1
    years = (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days / 365.25
    ann = (vals[-1] / vals[0]) ** (1 / years) - 1 if years > 0 else 0
    peak = np.maximum.accumulate(vals)
    mdd = (vals / peak - 1).min()
    rets = np.diff(vals) / vals[:-1]
    sharpe = (np.mean(rets) * 252 - 0.025) / (np.std(rets) * np.sqrt(252)) \
        if len(rets) > 1 and np.std(rets) > 0 else 0.0
    return total * 100, ann * 100, mdd * 100, sharpe


def yearly(dates, vals):
    df = pd.DataFrame({"date": list(dates), "v": list(vals)})
    df["year"] = df["date"].str[:4]
    out = {}
    for y, g in df.groupby("year"):
        out[y] = (g["v"].iloc[-1] / g["v"].iloc[0] - 1) * 100
    return out


def _safe_to_csv(df, path, **kw):
    """pandas to_csv 带重试——绕开 Windows 瞬时文件锁(如 IDE 预览句柄)"""
    import time
    last = None
    for _ in range(6):
        try:
            df.to_csv(path, **kw)
            return
        except PermissionError as e:
            last = e
            time.sleep(2)
    raise last


def main():
    t0 = time.time()
    all_dates = get_trade_dates(START, END)
    bench_csi = index_nav("000985.SH")
    bench_hs = index_nav("000300.SH")
    bcsi = (bench_csi[-1] / bench_csi[0] - 1) * 100
    bhs = (bench_hs[-1] / bench_hs[0] - 1) * 100
    print(f"基准(买入持有) 中证全指 +{bcsi:.2f}%  沪深300 +{bhs:.2f}%\n")

    rows = []
    yr_data = {}

    # —— 1) PEG 最优版 ——
    print(">>> [1/4] PEG 最优版 (质量+动量+VaR2.5%) ...")
    rp = run_peg.run_backtest(start_date=START, end_date=END, top_n=30, verbose=False,
                              stock_pool="all", freq="annual", stab_years=3,
                              min_roe=8, max_debt=70, momentum_months=12,
                              var_guard=True, var_cap=0.025)
    peg_csv = sorted(glob.glob("data/results/peg/backtest_*_v25_20140101_20260715.csv"))[-1]
    pdf = pd.read_csv(peg_csv)
    pnav = pdf["value"].astype(float).tolist()
    pdt, pa, pm, ps = metrics(pdf["date"].astype(str).tolist(), pnav)
    rows.append(("PEG 最优版", "年度(5月)等权", pdt, pa, pm, ps,
                 pdt - bcsi, pdt - bhs))
    yr_data["PEG 最优版"] = yearly(pdf["date"].astype(str).tolist(), pnav)
    print(f"    总收益 {pdt:+.2f}%  年化 {pa:+.2f}%  最大回撤 {pm:.2f}%  夏普 {ps:.2f}\n")

    # —— 2) 神奇公式 ——
    print(">>> [2/4] 神奇公式 (Magic Formula) ...")
    mg = mf.run_backtest(start_date=START, end_date=END, top_n=30, verbose=False)
    mdt, ma, mm, ms = mg["total_return"], mg["annual_return"], mg["max_drawdown"], mg["sharpe"]
    rows.append(("神奇公式", "年度(5月)等权", mdt, ma, mm, ms,
                 mdt - bcsi, mdt - bhs))
    yr_data["神奇公式"] = {y: r for y, r in mg["yearly"].items()}
    print(f"    总收益 {mdt:+.2f}%  年化 {ma:+.2f}%  最大回撤 {mm:.2f}%  夏普 {ms:.2f}\n")

    # —— 3) 红利低波 (官方 compact) ——
    print(">>> [3/4] 红利低波 (官方compact: 季频/股息率加权) ...")
    targets, weights_map, _ = dlv.select_targets_official("official_compact", pool="all", top_n=12)
    all_codes = sorted({c for _, cs in targets for c in cs})
    pmap = dlv.bulk_close_prices(all_codes, START, END)
    nav = dlv.run_nav_weighted(targets, weights_map, pmap, all_dates)
    ddv = [x[0] for x in nav]
    vv = [x[1] for x in nav]
    ddt, da, dm, ds = metrics(ddv, vv)
    rows.append(("红利低波", "季频·股息率加权", ddt, da, dm, ds,
                 ddt - bcsi, ddt - bhs))
    yr_data["红利低波"] = yearly(ddv, vv)
    print(f"    总收益 {ddt:+.2f}%  年化 {da:+.2f}%  最大回撤 {dm:.2f}%  夏普 {ds:.2f}\n")

    # —— 4) 神奇公式 V2.1（⑧暴涨护栏1.5 + 分散15只）——
    print(">>> [4/4] 神奇公式 V2.1 (EBIT3年均值+MA200+⑧护栏1.5+15只) ...")
    v2 = mv2.run_backtest_v2(start_date=START, end_date=END, top_n=15, spike_guard=1.5,
                             stock_pool="zz800", capital=CAP)
    v2_csv = f"data/results/magic_v2/backtest_v2_sg15_{START}_{END}.csv"
    vdf = pd.read_csv(v2_csv, dtype={"date": str})
    vnav = vdf["value_real"].astype(float).tolist()
    vdt, va, vm, vs = metrics(vdf["date"].astype(str).tolist(), vnav)
    rows.append(("神奇公式V2.1", "年度·15只·⑧1.5", vdt, va, vm, vs,
                 vdt - bcsi, vdt - bhs))
    yr_data["神奇公式V2.1"] = yearly(vdf["date"].astype(str).tolist(), vnav)
    print(f"    总收益 {vdt:+.2f}%  年化 {va:+.2f}%  最大回撤 {vm:.2f}%  夏普 {vs:.2f}\n")

    # —— 汇总表 ——
    df = pd.DataFrame(rows, columns=[
        "策略", "调仓/加权", "总收益%", "年化%", "最大回撤%", "夏普",
        "超额_中证全指", "超额_沪深300"])
    print("=" * 110)
    print(f"跨策略 PK（{START}~{END}，初始 {CAP:,.0f}，基准买入持有）")
    print("=" * 110)
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(df.to_string(index=False))
    os.makedirs("data/results", exist_ok=True)
    out = "data/results/cross_strategy_pk_v21.csv"
    _safe_to_csv(df, out, index=False, encoding="utf-8-sig")
    print(f"\n汇总已保存：{out}")

    # —— 逐年表 ——
    years = sorted({y for s in yr_data.values() for y in s.keys()})
    yrows = []
    csi_yr = yearly(get_trade_dates(START, END), bench_csi)
    for y in years:
        yrows.append({
            "年份": y,
            "PEG最优版%": round(yr_data["PEG 最优版"].get(y, float("nan")), 2),
            "神奇公式%": round(yr_data["神奇公式"].get(y, float("nan")), 2),
            "神奇公式V2.1%": round(yr_data["神奇公式V2.1"].get(y, float("nan")), 2),
            "红利低波%": round(yr_data["红利低波"].get(y, float("nan")), 2),
            "中证全指%": round(csi_yr.get(y, float("nan")), 2),
        })
    ydf = pd.DataFrame(yrows)
    print("\n逐年收益（%）:")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(ydf.to_string(index=False))
    yout = "data/results/cross_strategy_yearly_v21.csv"
    _safe_to_csv(ydf, yout, index=False, encoding="utf-8-sig")
    print(f"\n逐年表已保存：{yout}")
    print(f"\n总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
