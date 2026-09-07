"""B9 NAV 同窗口对照：任意多个 bt_quality_nav_*.csv，切到同一窗口后并排比指标。

为什么需要它：
  B9 实验室（`divlow_b9_pool_lab.py`）只算**换手**（候选池已落盘，离线秒算），不算收益。
  而"取消行业计数上限"真正的代价在**集中度 vs 收益/回撤**，必须落到 NAV 上判断。
  又：各档回测窗口起点常不一致（2013 起 vs 2020 起），直接比总收益会白捡/白亏窗口差异
  （B5' 报告踩过：年度档优势 5.55pp → 统一窗口后只剩 3.30pp）。
  → 本脚本把所有 NAV 统一切到 [start, end]，用引擎**同一个 compute_metrics** 复算，
    并额外做配对 t 检验，杜绝"窗口不同导致的假差异"。

用法：
  venv_ml/Scripts/python.exe divlow_b9_nav_cmp.py --start 20200101 --end 20260723 \
      data/results/dividend_low_vol/bt_quality_nav_20130101_20260903_official_compact_all_12_hfq.csv \
      data/results/dividend_low_vol/bt_quality_nav_20200101_20260723_official_compact_all_12_ic0_hfq.csv

  第一个文件视为基准（差值/配对检验都以它为对照）。
  用 --labels 给每档起短名（逗号分隔，个数须与文件数一致）。
"""
import argparse
import os

import numpy as np
import pandas as pd

import run_dividend_low_vol_quality_bt as E


def load_slice(path, start, end):
    """读 NAV → 切窗口 → 返回 (dates, {col: vals})。trade_date 是 TEXT，按字符串比较即可。"""
    df = pd.read_csv(path, dtype={"trade_date": str}, encoding="utf-8-sig")
    df["trade_date"] = df["trade_date"].astype(str)
    m = (df["trade_date"] >= start) & (df["trade_date"] <= end)
    d = df[m].reset_index(drop=True)
    if len(d) < 2:
        raise SystemExit(f"[err] {os.path.basename(path)} 在 {start}~{end} 内只有 {len(d)} 行，切不出窗口")
    cols = [c for c in d.columns if c != "trade_date"]
    return list(d["trade_date"]), {c: d[c].astype(float).tolist() for c in cols}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="bt_quality_nav_*.csv（第一个=基准）")
    ap.add_argument("--start", default="20200101")
    ap.add_argument("--end", default="20260723")
    ap.add_argument("--labels", default=None, help="短名，逗号分隔，个数须与文件数一致")
    args = ap.parse_args()

    labels = args.labels.split(",") if args.labels else [os.path.basename(f).replace("bt_quality_nav_", "")
                                                          .replace(".csv", "") for f in args.files]
    if len(labels) != len(args.files):
        raise SystemExit("[err] --labels 个数与文件数不一致")

    series = []
    for f, lb in zip(args.files, labels):
        if not os.path.exists(f):
            raise SystemExit(f"[err] 文件不存在 {f}")
        dates, cols = load_slice(f, args.start, args.end)
        # 策略列 = 第一个非 trade_date 列（nav_<mode>）；其余视为基准
        keys = list(cols.keys())
        strat = keys[0]
        series.append(dict(label=lb, dates=dates, cols=cols, strat=strat, path=f))

    # 交集对齐（各文件交易日应一致；不一致时取交集，避免配对检验错位）
    common = set(series[0]["dates"])
    for s in series[1:]:
        common &= set(s["dates"])
    common = sorted(common)
    if len(common) != len(series[0]["dates"]):
        print(f"⚠️ 各文件交易日不完全一致，已取交集 {len(common)} 天（基准 {len(series[0]['dates'])} 天）")

    print(f"窗口 {args.start}~{args.end}   {len(common)} 交易日")
    print(f"{'档位':<34}{'总收益':>10}{'年化':>9}{'最大回撤':>10}{'波动':>8}{'夏普':>7}{'卡玛':>7}")
    print("-" * 85)
    base_rets = None
    rows = []
    for i, s in enumerate(series):
        idx = [s["dates"].index(d) for d in common]
        col = s["cols"][s["strat"]]
        vals = [col[j] for j in idx]
        m = E.compute_metrics(list(zip(common, vals)), common)
        rows.append((s["label"], m, vals))
        print(f"{s['label']:<34}{m['total_ret']*100:>9.2f}%{m['ann']*100:>8.2f}%"
              f"{m['max_dd']*100:>9.2f}%{m['vol']*100:>7.2f}%{m['sharpe']:>7.2f}{m['calmar']:>7.2f}")
        r = np.diff(np.array(vals, dtype=float))
        r = r / np.array(vals[:-1], dtype=float)
        if i == 0:
            base_rets = r
    print("-" * 85)

    # 基准列（各文件应一致，取第一档的）与配对检验
    bkeys = [k for k in series[0]["cols"].keys() if k != series[0]["strat"]]
    for bk in bkeys:
        col = series[0]["cols"][bk]
        idx = [series[0]["dates"].index(d) for d in common]
        vals = [col[j] for j in idx]
        m = E.compute_metrics(list(zip(common, vals)), common)
        print(f"{'[基准] '+bk:<34}{m['total_ret']*100:>9.2f}%{m['ann']*100:>8.2f}%"
              f"{m['max_dd']*100:>9.2f}%{m['vol']*100:>7.2f}%{m['sharpe']:>7.2f}{m['calmar']:>7.2f}")

    if len(rows) > 1:
        print(f"\n【配对检验 vs 基准（{rows[0][0]}）｜日收益，n={len(common)-1}】")
        for lb, m, vals in rows[1:]:
            r = np.diff(np.array(vals, dtype=float)) / np.array(vals[:-1], dtype=float)
            d = r - base_rets
            t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if d.std(ddof=1) > 0 else 0.0
            d_ann = m["ann"] - rows[0][1]["ann"]
            d_dd = m["max_dd"] - rows[0][1]["max_dd"]
            flag = "显著" if abs(t) >= 1.96 else ("噪声" if abs(t) < 0.9 else "弱")
            print(f"  {lb:<32} 年化差 {d_ann*100:+7.2f}pp   回撤差 {d_dd*100:+7.2f}pp   t={t:+.2f}  → {flag}")
        print("\n🔴 |t|<0.9 一律判噪声，不得当卖点；回撤为负，改善幅度须 abs(基准)-abs(新档)。")


if __name__ == "__main__":
    main()
