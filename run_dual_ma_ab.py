"""双均线 A/B 专用回测台：基础版(dual_ma) vs Jim状态机版(dual_ma_jim) vs 买入持有。
把两个插件跑在整只股票池(默认zz800)上逐票聚合为等权组合，对照买入持有与基准指数。
与 run_backtest 的区别：不经过多因子选股(TOP5)，直接在全宇宙上评估择时框架的 edge。
用法:
  ./venv_ml/Scripts/python.exe run_dual_ma_ab.py --start 20100101 --end 20251231 --pool zz800
"""
import sys, os, logging, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.WARNING)
import numpy as np, pandas as pd
import config as C
import run_backtest as RB
from backtest.dual_ma_plugin import DualMAStrategyPlugin
from backtest.dual_ma_jim_plugin import DualMAJimPlugin

POOL_TO_BENCH = {"hs300": "000300.SH", "zz500": "000905.SH", "zz800": "000906.SH",
                 "zz1000": "000852.SH", "all": "000300.SH"}

def get_universe(pool, as_of):
    if pool == "hs300":   return RB._get_hs300_from_db(as_of)
    if pool == "zz500":   return RB._get_zz500_from_db(as_of)
    if pool == "zz800":   return RB._get_zz800_from_db(as_of)
    if pool == "zz1000":  return RB._get_zz1000_from_db(as_of)
    return RB._get_all_stocks_from_db()

def max_drawdown(nav):
    nav = np.asarray(nav, dtype=float)
    peak = np.maximum.accumulate(nav)
    dd = nav / peak - 1.0
    return float(dd.min() * 100)

def ann_sharpe(nav):
    nav = np.asarray(nav, dtype=float)
    rets = nav[1:] / nav[:-1] - 1.0
    rets = rets[np.isfinite(rets)]
    if len(rets) < 2 or rets.std() == 0:
        return 0.0
    return float(rets.mean() / rets.std() * np.sqrt(252) * 100)

def build_nav(results, all_dates, cap):
    """results: dict code -> list[(date, portfolio_value)]; 逐票等权(等额资金)聚合成组合NAV"""
    cols = {}
    for code, dvs in results.items():
        s = pd.Series({d: v for d, v in dvs})
        s = s.reindex(all_dates).ffill().fillna(cap)
        cols[code] = s
    return pd.DataFrame(cols).sum(axis=1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="20100101")
    ap.add_argument("--end", default="20251231")
    ap.add_argument("--pool", default="zz800")
    ap.add_argument("--per-stock", type=int, default=100000)
    args = ap.parse_args()

    start, end, pool = args.start, args.end, args.pool
    bench = POOL_TO_BENCH[pool]
    cap = args.per_stock

    print(f"双均线 A/B 台 | 池={pool} | {start}→{end} | 基准={bench} | 每股资金={cap}")
    universe = get_universe(pool, start)
    print(f"  宇宙成分数: {len(universe)} (as_of={start}, 平台缺失快照时回退最早)")
    codes = [str(c) for c in universe["code"].tolist()]

    conn = __import__("sqlite3").connect(RB.DB_PATH)
    base_res, jim_res, bh_res = {}, {}, {}
    base_trades = jim_trades = 0
    stock_meta = {}
    all_dates = set()
    skipped = 0
    for i, code in enumerate(codes, 1):
        df = RB.load_stock_prices(code, start, end, conn, lookback_days=250)
        if df is None or len(df) < 30:
            skipped += 1
            continue
        sd = df[df["trade_date"] >= start].index.min()
        if pd.isna(sd):
            skipped += 1
            continue
        sd = int(sd)
        pc = "adj_close" if "adj_close" in df.columns else "close"
        bh_start = df[pc].iloc[sd]
        # BH 等权NAV(逐票): cap * price/price_start
        bh_series = cap * (df[pc] / bh_start)
        bh_res[code] = list(zip(df["trade_date"].tolist(), bh_series.tolist()))
        all_dates.update(df["trade_date"].tolist())

        base_cfg = dict(C.STRATEGIES["dual_ma"]); base_cfg["total_capital"] = cap * 5
        jim_cfg = dict(C.STRATEGIES["dual_ma_jim"]); jim_cfg["total_capital"] = cap * 5

        r_base = DualMAStrategyPlugin(cap, base_cfg).run(df, sd)
        r_jim = DualMAJimPlugin(cap, jim_cfg).run(df, sd)
        base_res[code] = [(v["date"], v["portfolio_value"]) for v in r_base.get("daily_values", [])]
        jim_res[code] = [(v["date"], v["portfolio_value"]) for v in r_jim.get("daily_values", [])]
        base_trades += len(r_base.get("trades", []))
        jim_trades += len(r_jim.get("trades", []))
        if i % 100 == 0:
            print(f"  ...已处理 {i}/{len(codes)} 只")
    conn.close()
    print(f"  有效股票: {len(base_res)} 只 | 跳过(数据不足): {skipped} 只")

    if not base_res:
        print("  [ERR] 无有效股票，无法回测"); return

    all_dates = sorted(all_dates)
    nav_base = build_nav(base_res, all_dates, cap)
    nav_jim = build_nav(jim_res, all_dates, cap)
    nav_bh = build_nav(bh_res, all_dates, cap)

    def total_ret(nav): return (nav.iloc[-1] / nav.iloc[0] - 1) * 100

    rb_ = total_ret(nav_base); rj_ = total_ret(nav_jim); rbh = total_ret(nav_bh)
    # 基准指数
    idx = RB.load_benchmark(bench, start, end, __import__("sqlite3").connect(RB.DB_PATH))
    idx_ret = (idx["close"].iloc[-1] / idx["close"].iloc[0] - 1) * 100 if idx is not None else float("nan")

    # 逐票均值收益 & 跑赢BH比例
    def per_stock_means(res_a, res_b):
        ra, rb_ = [], []
        for code in res_a:
            a = res_a[code][-1][1] / cap - 1 if res_a[code] else 0
            b = res_b[code][-1][1] / cap - 1 if res_b[code] else 0
            # 用BH末值
            bh_end = bh_res[code][-1][1] / cap - 1 if bh_res[code] else 0
            ra.append(a); rb_.append(b)
        return ra, rb_
    base_ps = [v[-1][1]/cap - 1 for v in base_res.values() if v]
    jim_ps = [v[-1][1]/cap - 1 for v in jim_res.values() if v]
    bh_ps = [v[-1][1]/cap - 1 for v in bh_res.values() if v]
    n = len(base_ps)
    beat_bh_base = sum(1 for a, b in zip(base_ps, bh_ps) if a > b)
    beat_bh_jim = sum(1 for a, b in zip(jim_ps, bh_ps) if a > b)
    pos_base = sum(1 for a in base_ps if a > 0)
    pos_jim = sum(1 for a in jim_ps if a > 0)

    from datetime import datetime as _dt
    _yrs = max((_dt.strptime(end, "%Y%m%d") - _dt.strptime(start, "%Y%m%d")).days / 365.25, 1e-9)
    def _ann(total_pct): return ((1 + total_pct / 100) ** (1 / _yrs) - 1) * 100

    print("\n" + "=" * 82)
    print(f"  双均线 A/B 结果 (等权组合, {n} 只有效, {start}→{end}, 区间{_yrs:.1f}年)")
    print("=" * 82)
    hdr = f"  {'策略':<22}{'组合收益':>10}{'年化收益':>10}{'最大回撤':>10}{'年化Sharpe':>12}{'总交易':>10}"
    print(hdr)
    print(f"  {'双均线基础(金叉死叉)':<20}{rb_:>+9.2f}%{_ann(rb_):>+9.2f}%{max_drawdown(nav_base):>9.2f}%{ann_sharpe(nav_base):>12.2f}{base_trades:>10}")
    print(f"  {'双均线Jim状态机':<20}{rj_:>+9.2f}%{_ann(rj_):>+9.2f}%{max_drawdown(nav_jim):>9.2f}%{ann_sharpe(nav_jim):>12.2f}{jim_trades:>10}")
    print(f"  {'买入持有(等权)':<20}{rbh:>+9.2f}%{_ann(rbh):>+9.2f}%{max_drawdown(nav_bh):>9.2f}%{ann_sharpe(nav_bh):>12.2f}{0:>10}")
    print(f"  {bench+'指数':<22}{idx_ret:>+9.2f}%")
    print("-" * 78)
    print(f"  逐票均值收益: 基础 {np.mean(base_ps)*100:+.2f}% | Jim {np.mean(jim_ps)*100:+.2f}% | BH {np.mean(bh_ps)*100:+.2f}%")
    print(f"  正收益占比:   基础 {pos_base}/{n} ({pos_base/n*100:.0f}%) | Jim {pos_jim}/{n} ({pos_jim/n*100:.0f}%)")
    print(f"  跑赢BH占比:   基础 {beat_bh_base}/{n} ({beat_bh_base/n*100:.0f}%) | Jim {beat_bh_jim}/{n} ({beat_bh_jim/n*100:.0f}%)")
    print(f"  交易数(全宇宙): 基础 {base_trades} | Jim {jim_trades}  (漏斗: Jim<=基础 ? {'OK' if jim_trades<=base_trades else 'FAIL'})")
    print("=" * 78)

if __name__ == "__main__":
    main()
