# -*- coding: utf-8 -*-
"""
个股级移动止盈 · 快速重放（持仓源 = 引擎真实选股明细 CSV）
================================================================
红利低波质量复合（official_compact · 全A · 股息率加权）的真实持仓序列，
直接复用引擎 run_backtest 菜单[6]→[4] 落盘的选股明细 CSV，避免重跑慢速全A 选股。

- 基线 NAV：直接调用引擎自身的 run_nav_weighted() 重放真实持仓（零改造真值）；
  并额外与引擎已落盘的 NAV CSV（nav_official_compact 列）交叉校验，证明重放≡引擎实跑。
- 移动止盈：复用 individual_trail.run_nav_weighted_trail（逐股非锁存 trailing）。
- 自证：thr=∞ 须逐位≡基线重放。
- 窗口：选股 CSV 覆盖 2013-2026（策略支持的最早区间，2010-2012 候选池饥饿不可用），
  重放统一裁到 --end（默认 20251231）。

产物（data/results/negative_cost/）：nav_<tag>_*.csv / metrics_<tag>.csv / sel_replay.csv / summary.json
信任但验证：落盘 CSV 反算 总收益/最大回撤，与内存指标逐位比对。
"""
import os, sys, json, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import run_dividend_low_vol_quality_bt as dlq
from run_monthly_rebalance import (
    get_trade_dates, calc_fee, COMMISSION_RATE, SLIPPAGE_RATE,
)
import individual_trail as it

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "results", "negative_cost")


def load_targets_from_sel(sel_csv, end=None):
    """从引擎选股明细 CSV 还原 (targets, weights_map, sel_log)。
    targets: [(rebal_date, [ts_code...]), ...]，按 rebal_date 升序、ts_code 排序（确定性）。"""
    df = pd.read_csv(sel_csv, encoding="utf-8-sig")
    # 🔴 CSV 读入的 rebal_date 为 int64，str() 会变成 "20130110.0" 导致与 all_dates 字符串键错配、
    #   run_nav_weighted 的 rebal_set.get(d) 永远 miss（再平衡失效、NAV 变平）。统一 int→str。
    df["rebal_date"] = df["rebal_date"].astype(int).astype(str)
    if end:
        df = df[df["rebal_date"] <= str(end)]
    targets = []
    weights_map = {}
    for rb, g in df.groupby("rebal_date"):
        codes = sorted(g["ts_code"].astype(str).tolist())
        targets.append((rb, codes))
        weights_map[rb] = {str(r.ts_code): float(r.weight) for _, r in g.iterrows()}
    # 权重和校验（股息率加权应≈1）
    sums = [sum(weights_map[rb].values()) for rb, _ in targets]
    print(f"  持仓期数={len(targets)}  涉及股票={len({c for _,cs in targets for c in cs})}"
          f"  权重和[min,max]=[{min(sums):.4f},{max(sums):.4f}]")
    return targets, weights_map, df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sel-csv",
                    default="data/results/dividend_low_vol/bt_quality_sel_OFFICIAL_OFFICIAL_COMPACT_all_12_20130101_20260903.csv")
    ap.add_argument("--nav-csv",
                    default="data/results/dividend_low_vol/bt_quality_nav_20130101_20260903_official_compact_all_12_hfq.csv")
    ap.add_argument("--start", default="20130101")
    ap.add_argument("--end", default="20251231")
    ap.add_argument("--thr", default="10,15,20,25")
    ap.add_argument("--overlay", choices=["on", "off", "both"], default="both")
    ap.add_argument("--price-mode", default="hfq", choices=["raw", "hfq"])
    args = ap.parse_args()

    thr_list = [int(x) / 100.0 for x in args.thr.split(",")]
    dlq.PRICE_MODE = args.price_mode
    dlq.START = args.start
    dlq.END = args.end

    print(f"[重放] 持仓源={os.path.basename(args.sel_csv)}  引擎NAV基准={os.path.basename(args.nav_csv)}")
    print(f"      窗口 {args.start}~{args.end}  price={args.price_mode}  overlay={args.overlay}  thr={args.thr}%")
    t0 = time.time()
    targets, weights_map, sel_df = load_targets_from_sel(args.sel_csv, args.end)
    all_codes = sorted({c for _, cs in targets for c in cs})
    # 行情仅加载持仓股（秒级，不碰全A 慢预载）
    pmap = dlq.bulk_close_prices(all_codes, args.start, args.end)
    if args.price_mode == "hfq":
        dlq.EXEC_PMAP.clear()
        dlq.EXEC_PMAP.update(dlq.bulk_open_prices(all_codes, args.start, args.end))
    all_dates = get_trade_dates(args.start, args.end)
    print(f"  行情加载 {len(all_codes)} 只 / {len(all_dates)} 交易日（{time.time()-t0:.1f}s）")

    # 引擎真实 NAV 基准（落盘 CSV）
    nav_df = pd.read_csv(args.nav_csv, encoding="utf-8-sig")
    nav_col = [c for c in nav_df.columns if c.startswith("nav_") and "000" not in c and "922" not in c][0]
    eng_nav = {str(int(r.trade_date)): float(r[nav_col]) for _, r in nav_df.iterrows()}

    scenarios = []
    if args.overlay in ("on", "both"):
        scenarios.append(True)
    if args.overlay in ("off", "both"):
        scenarios.append(False)

    pkg = {}
    verify_log = []
    for overlay in scenarios:
        tag = f"ov{1 if overlay else 0}"
        print(f"\n── 场景 overlay={'ON' if overlay else 'OFF'} ──")
        if overlay:
            dlq._preload_index_channel("000922.SH")
        coef_fn = dlq._make_coef_fn(overlay, "rolling", 756, None, None, 0.5, 1.0)

        # 基线 = 引擎自身 run_nav_weighted 重放（真值）
        nav_base = dlq.run_nav_weighted(targets, weights_map, pmap, all_dates, coef_fn)
        m_base = dlq.compute_metrics(nav_base, all_dates)

        # 交叉校验：我的重放 vs 引擎落盘 NAV
        myv = np.array([v for _, v in nav_base])
        engv = np.array([eng_nav.get(d, np.nan) for d, _ in nav_base])
        m_eng = dlq.compute_metrics([(d, eng_nav[d]) for d in all_dates if d in eng_nav], all_dates)
        # 对齐比较（取两者都有值的时点）
        both = [(d, v, eng_nav.get(d)) for d, v in nav_base if eng_nav.get(d) is not None]
        diffs = [v - e for d, v, e in both]
        max_abs = float(np.max(np.abs(diffs)))
        print(f"  基线重放: 总收益 {m_base['total_ret']*100:+.2f}%  最大回撤 {m_base['max_dd']*100:.2f}%")
        print(f"  引擎落盘NAV: 总收益 {m_eng['total_ret']*100:+.2f}%  最大回撤 {m_eng['max_dd']*100:.2f}%")
        print(f"  [交叉校验] 我的重放 vs 引擎NAV 最大绝对差 = {max_abs:.3f}  → "
              f"{'PASS(重放≡引擎实跑)' if max_abs < 1.0 else 'WARN 偏差较大'}")

        results = {"base": (nav_base, m_base)}
        for thr in thr_list:
            nav_t = it.run_nav_weighted_trail(targets, weights_map, pmap, all_dates, coef_fn, thr)
            m_t = dlq.compute_metrics(nav_t, all_dates)
            results[f"trail{int(thr*100)}"] = (nav_t, m_t)
            print(f"  trail {int(thr*100)}%: 总收益 {m_t['total_ret']*100:+.2f}%  最大回撤 {m_t['max_dd']*100:.2f}%")

        # 自证 thr=∞ ≡ 基线
        nav_inf = it.run_nav_weighted_trail(targets, weights_map, pmap, all_dates, coef_fn, 1e12)
        vb = np.array([v for _, v in nav_base]); vi = np.array([v for _, v in nav_inf])
        max_abs_diff = float(np.max(np.abs(vb - vi)))
        identity_ok = max_abs_diff < 1e-6
        print(f"  [自证] thr=∞ vs 基线 最大绝对差 = {max_abs_diff:.2e}  → {'PASS' if identity_ok else 'FAIL ❌'}")

        # 落盘
        for key, (nav, m) in results.items():
            navname = "base" if key == "base" else f"trail{key[5:]}"
            pd.DataFrame(nav, columns=["trade_date", f"nav_{navname}"]).to_csv(
                os.path.join(OUTDIR, f"nav_{tag}_{navname}.csv"), index=False, encoding="utf-8-sig")
            tr, mdd, ok = _verify_csv(os.path.join(OUTDIR, f"nav_{tag}_{navname}.csv"), m)
            verify_log.append((f"nav_{tag}_{navname}.csv", ok))
        sel_df.to_csv(os.path.join(OUTDIR, f"sel_replay_{tag}.csv"), index=False, encoding="utf-8-sig")
        rows = [it.metrics_row("基线(无个股止盈)", m_base, m_base)]
        for thr in thr_list:
            rows.append(it.metrics_row(f"移动止盈 {int(thr*100)}%", results[f"trail{int(thr*100)}"][1], m_base))
        pd.DataFrame(rows).to_csv(os.path.join(OUTDIR, f"metrics_{tag}.csv"), index=False, encoding="utf-8-sig")
        pkg[tag] = dict(overlay=overlay, m_base=m_base, m_eng=m_eng, max_abs=max_abs,
                        identity_ok=identity_ok, rows=rows)

    summary = dict(
        args=vars(args), thr_list=thr_list,
        cross_check_max_abs={k: v["max_abs"] for k, v in pkg.items()},
        cross_check_pass={k: v["max_abs"] < 1.0 for k, v in pkg.items()},
        identity_ok={k: v["identity_ok"] for k, v in pkg.items()},
        csv_verify_all_ok=all(ok for _, ok in verify_log),
    )
    with open(os.path.join(OUTDIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    print("\n" + "=" * 100)
    print("  个股级移动止盈 A/B（红利低波质量复合 · 真实持仓 · 引擎NAV交叉校验）")
    print("=" * 100)
    for tag, sc in pkg.items():
        print(f"\n── 场景 overlay={'ON' if sc['overlay'] else 'OFF'} ──")
        print(f"  引擎NAV交叉校验最大差={sc['max_abs']:.3f}  自证thr=∞={'PASS' if sc['identity_ok'] else 'FAIL'}")
        print(f"  {'方案':<16}{'总收益':>9}{'年化':>8}{'最大回撤':>10}{'Sharpe':>8}{'DD_cut':>8}{'Ret_cost':>10}{'Eff':>8}")
        for r in sc["rows"]:
            eff_s = f"{r['eff']:>8.2f}" if not np.isnan(r['eff']) else f"{'N/A':>8}"
            print(f"  {r['name']:<16}{r['total_ret']:>+8.2f}%{r['ann']:>+7.2f}%{r['max_dd']:>9.2f}%"
                  f"{r['sharpe']:>8.2f}{r['dd_cut']:>+7.2f}pp{r['ret_cost']:>+9.2f}pp{eff_s}")
    print(f"\n[信任但验证] 落盘CSV反算全部吻合={summary['csv_verify_all_ok']}；"
          f"交叉校验PASS={all(summary['cross_check_pass'].values())}；"
          f"thr=∞自证PASS={all(sc['identity_ok'] for sc in pkg.values())}")
    print(f"产物目录: {OUTDIR}")


def _verify_csv(csv_path, m_inmem):
    df = pd.read_csv(csv_path)
    col = [c for c in df.columns if c.lower().startswith("nav")][0]
    vals = df[col].dropna().values.astype(float)
    total_ret = vals[-1] / vals[0] - 1
    peak = np.maximum.accumulate(vals)
    max_dd = (vals / peak - 1).min()
    ok = (abs(total_ret - m_inmem["total_ret"]) < 1e-9 and abs(max_dd - m_inmem["max_dd"]) < 1e-9)
    return total_ret, max_dd, ok


if __name__ == "__main__":
    main()
