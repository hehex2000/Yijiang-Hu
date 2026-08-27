# -*- coding: utf-8 -*-
"""
净值发生器：把指定择时插件在股票池(默认zz800)上逐票回测、等权聚合为组合净值，
输出 date,nav 两列的 csv，供 macd_plugin_validate.py --base-nav 做「真头对头」对照。

复用 run_dual_ma_ab.py 已验证的逐票回测 + 等权聚合框架（不经过多因子选股，
直接在整只股票池上评估该择时框架的 edge）。

用法:
  ./venv_ml/Scripts/python.exe gen_plugin_nav.py --plugin macd_jim --pool zz800 --out macd_jim_nav.csv
  ./venv_ml/Scripts/python.exe gen_plugin_nav.py --plugin macd_jim --pool zz800 --start 20100101 --end 20251231

csv 格式: 第1列=date(YYYYMMDD 整数), 第2列=nav(组合净值, 货币单位, validate 脚本会归一化)。
"""
import sys, os, argparse, logging
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.WARNING)
import numpy as np, pandas as pd
import config as C
import run_backtest as RB
from backtest.macd_jim_plugin import MacdJimPlugin

PLUGIN_REGISTRY = {
    "macd_jim": ("macd_jim", MacdJimPlugin),
    # 红利低波等非逐股择时插件(选股+持有型)暂不支持此路径；见文末说明。
}

POOL_TO_BENCH = {"hs300": "000300.SH", "zz500": "000905.SH", "zz800": "000906.SH",
                 "zz1000": "000852.SH", "all": "000300.SH"}


def get_universe(pool, as_of):
    if pool == "hs300":
        return RB._get_hs300_from_db(as_of)
    if pool == "zz500":
        return RB._get_zz500_from_db(as_of)
    if pool == "zz800":
        return RB._get_zz800_from_db(as_of)
    if pool == "zz1000":
        return RB._get_zz1000_from_db(as_of)
    return RB._get_all_stocks_from_db()


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
    ap.add_argument("--plugin", default="macd_jim", help="插件名(见 PLUGIN_REGISTRY)")
    ap.add_argument("--pool", default="zz800")
    ap.add_argument("--start", default="20100101")
    ap.add_argument("--end", default="20251231")
    ap.add_argument("--out", default=None, help="输出 csv 路径(默认 <plugin>_nav.csv)")
    ap.add_argument("--per-stock", type=int, default=100000)
    args = ap.parse_args()

    if args.plugin not in PLUGIN_REGISTRY:
        print(f"[ERR] 未知插件 {args.plugin}; 当前支持: {list(PLUGIN_REGISTRY)}")
        return
    cfg_key, PluginCls = PLUGIN_REGISTRY[args.plugin]
    out = args.out or f"{args.plugin}_nav.csv"

    start, end, pool, cap = args.start, args.end, args.pool, args.per_stock
    universe = get_universe(pool, start)
    print(f"净值发生器 | 插件={args.plugin}(cfg:{cfg_key}) | 池={pool} | {start}→{end} | 每股资金={cap}")
    print(f"  宇宙成分数: {len(universe)} (as_of={start})")
    codes = [str(c) for c in universe["code"].tolist()]

    conn = __import__("sqlite3").connect(RB.DB_PATH)
    results = {}
    all_dates = set()
    skipped = 0
    for i, code in enumerate(codes, 1):
        df = RB.load_stock_prices(code, start, end, conn, lookback_days=250)
        if df is None or len(df) < 60:
            skipped += 1
            continue
        sd = df[df["trade_date"] >= start].index.min()
        if pd.isna(sd):
            skipped += 1
            continue
        sd = int(sd)
        # 与 run_dual_ma_ab 同口径: 凯利 max_position=0.20 → total_capital=cap*5 使满仓暴露=cap
        cfg = dict(C.STRATEGIES[cfg_key])
        cfg["total_capital"] = cap * 5
        r = PluginCls(cap, cfg).run(df, sd)
        dvs = [(v["date"], v["portfolio_value"]) for v in r.get("daily_values", [])]
        if dvs:
            results[code] = dvs
            all_dates.update(df["trade_date"].tolist())
        if i % 100 == 0:
            print(f"  ...已处理 {i}/{len(codes)} 只")
    conn.close()
    print(f"  有效股票: {len(results)} 只 | 跳过(数据不足): {skipped} 只")

    if not results:
        print("  [ERR] 无有效股票，无法生成净值"); return

    all_dates = sorted(all_dates)
    nav = build_nav(results, all_dates, cap)
    out_df = pd.DataFrame({"date": all_dates, "nav": nav.values})
    out_df.to_csv(out, index=False)
    print(f"  已写出 {out} | 行数={len(out_df)} | 首={out_df['date'].iloc[0]} 末={out_df['date'].iloc[-1]} "
          f"| 末净值/初始={nav.iloc[-1]/nav.iloc[0]:.4f}")


if __name__ == "__main__":
    main()
