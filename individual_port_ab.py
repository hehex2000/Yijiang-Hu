# -*- coding: utf-8 -*-
"""
个股层保护 vs 组合层 A15：叠加 / 互斥（增量 or 重复表达）对照
=================================================================
承接 individual_latch.py：「锁存20% + 沪深300 MACD 解锁」在个股层是首个提升 Sharpe 的方案。
但组合层 decompose_trail_ab 早已验证 A15（组合锁存15% + MACD 解锁）有正贡献（+400% vs +83%）。
→ 关键未决问题：**个股层保护是 A15 之外的【增量】，还是同一 alpha 的【重复表达】？**

实验设计（四档，同一真实持仓 / 同一基线 / 同一窗口）：
  S0 基线          ：无个股层、无组合层
  S1 仅个股层      ：锁存 thr_s + 指数 MACD 解锁
  S2 仅组合层 A15  ：组合峰回撤 thr_p + 沪深300 MACD 解锁（复用 decompose_trail_ab.locked_stop）
  S3 叠加          ：S1 + S2 同时启用

判定法则（增量 vs 重复）：
  Interaction = (S3 − S0) − (S1 − S0) − (S2 − S0) = S3 − S1 − S2 + S0
    Interaction ≈ 0        → 两层效应可加 = 【增量】
    Interaction 与单层反号 / 使 S3 不优于 max(S1,S2) → 【重复表达】（同一 alpha 被表达了两次）

🔑 方法论关键：组合层信号用【影子基线净值】触发，而非本方案自身净值。
   - 实盘可实现：并行维护一条"不加个股层保护"的影子账面净值即可。
   - 分析价值：S2 与 S3 的组合层信号【逐位相同】∎ 交互项可干净分解，
     不会被"个股层改变了净值 → 组合层触发时点漂移"这条路径依赖污染。
   - 代价：S3 里组合层的实际保护力度会偏保守（真实净值比影子净值高 → 更晚触发），
     故 S3 是叠加效应的【下界】。已在报告中标注。

自证（两重）：
  1. 个股 mask 恒 True + port_mask 恒 True ⇒ ≡ 基线（最大差须 = 0）
  2. port_mask 恒 True ⇒ ≡ 仅个股层（组合层代码路径未引入偏差）

产物（data/results/negative_cost/）：
  nav_portab_*_<key>.csv / metrics_portab.csv / summary_portab.json
信任但验证：落盘后从 CSV 反算 总收益/最大回撤，与内存指标逐位比对。
"""
import os, sys, json, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import run_dividend_low_vol_quality_bt as dlq
from run_monthly_rebalance import get_trade_dates, calc_fee
import individual_trail as it
import individual_latch as il
import macd_plugin_validate as M
from decompose_trail_ab import locked_stop          # 组合层语义逐位复用
from regime_cash_overlay import load_index_close, BENCH

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "results", "negative_cost")
DEFAULT_SEL = "data/results/dividend_low_vol/bt_quality_sel_OFFICIAL_OFFICIAL_COMPACT_all_12_20130101_20260903.csv"
DEFAULT_NAV = "data/results/dividend_low_vol/bt_quality_nav_20130101_20260903_official_compact_all_12_hfq.csv"


def _verify_csv(csv_path, m_inmem):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    col = [c for c in df.columns if c.lower().startswith("nav")][0]
    vals = df[col].dropna().values.astype(float)
    total_ret = vals[-1] / vals[0] - 1
    peak = np.maximum.accumulate(vals)
    max_dd = (vals / peak - 1).min()
    return (abs(total_ret - m_inmem["total_ret"]) < 1e-9 and
            abs(max_dd - m_inmem["max_dd"]) < 1e-9)


def run_variant(targets, weights_map, pmap, all_dates, coef_fn,
                mask_cache, port_mask, enabled):
    nav, posc = il.run_nav_weighted_masked(
        targets, weights_map, pmap, all_dates, coef_fn, mask_cache,
        enabled=enabled, port_mask=port_mask)
    return nav, float(np.mean(posc)), dlq.compute_metrics(nav, all_dates)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sel-csv", default=DEFAULT_SEL)
    ap.add_argument("--nav-csv", default=DEFAULT_NAV)
    ap.add_argument("--start", default="20130101")
    ap.add_argument("--end", default="20251231")
    ap.add_argument("--stock-thr", default="20", help="个股层锁存阈值(%)，可逗号分隔")
    ap.add_argument("--port-thr", default="15,20,25", help="组合层 A15 阈值(%)，可逗号分隔")
    ap.add_argument("--overlay", choices=["on", "off"], default="off")
    ap.add_argument("--tag", default="portab")
    args = ap.parse_args()

    s_thrs = [int(x) / 100.0 for x in args.stock_thr.split(",")]
    p_thrs = [int(x) / 100.0 for x in args.port_thr.split(",")]
    dlq.START, dlq.END = args.start, args.end

    os.makedirs(OUTDIR, exist_ok=True)
    t0 = time.time()
    print(f"[个股层 vs 组合层 A15] 窗口 {args.start}~{args.end}  "
          f"个股thr={args.stock_thr}%  组合thr={args.port_thr}%  overlay={args.overlay}")

    import individual_trail_replay as ir
    targets, weights_map, sel_df = ir.load_targets_from_sel(args.sel_csv, args.end)
    all_codes = sorted({c for _, cs in targets for c in cs})
    pmap = dlq.bulk_close_prices(all_codes, args.start, args.end)
    dlq.EXEC_PMAP.clear()
    dlq.EXEC_PMAP.update(dlq.bulk_open_prices(all_codes, args.start, args.end))
    all_dates = get_trade_dates(args.start, args.end)
    print(f"  行情 {len(all_codes)} 只 / {len(all_dates)} 交易日（{time.time()-t0:.1f}s）")

    # 指数 MACD 金叉（个股层与组合层共用同一信号源，便于判定"是否同一 alpha"）
    ic = load_index_close(BENCH, args.start, args.end)
    ic = ic.reindex(pd.Index([int(d) for d in all_dates])).ffill()
    golden = M.macd_golden(ic.values.astype(float)).values
    print(f"  解锁信号源={BENCH}  金叉天数={int(golden.sum())}/{len(golden)}")

    coef_fn = dlq._make_coef_fn(args.overlay == "on", "rolling", 756, None, None, 0.5, 1.0)
    if args.overlay == "on":
        dlq._preload_index_channel("000922.SH")

    # ── 影子基线 NAV（组合层信号源，固定不变）──
    nav_base = dlq.run_nav_weighted(targets, weights_map, pmap, all_dates, coef_fn)
    m_base = dlq.compute_metrics(nav_base, all_dates)
    base_v = np.array([v for _, v in nav_base], dtype=float)
    peak = np.maximum.accumulate(base_v)

    ones_cache = {c: np.ones(len(all_dates), bool) for c in all_codes}
    port_ones = np.ones(len(all_dates), bool)

    variants = {}          # key -> (nav, avg_pos, metrics)

    # S0 基线
    nav0, pos0, m0 = run_variant(targets, weights_map, pmap, all_dates, coef_fn,
                                 ones_cache, None, enabled=False)
    variants["S0_基线"] = (nav0, pos0, m0)
    print(f"  S0 基线          总收益 {m0['total_ret']*100:>+8.2f}%  回撤 {m0['max_dd']*100:>7.2f}%"
          f"  Sharpe {m0['sharpe']:>5.2f}  持仓 {pos0:>5.2f}")

    # S2 仅组合层 A15（先跑，信号与 S3 共用）
    for pthr in p_thrs:
        pmask = locked_stop(base_v, peak, pthr, golden)
        nav2, pos2, m2 = run_variant(targets, weights_map, pmap, all_dates, coef_fn,
                                     ones_cache, pmask, enabled=False)
        key = f"S2_组合{int(pthr*100)}"
        variants[key] = (nav2, pos2, m2)
        print(f"  {key:<14} 总收益 {m2['total_ret']*100:>+8.2f}%  回撤 {m2['max_dd']*100:>7.2f}%"
              f"  Sharpe {m2['sharpe']:>5.2f}  持仓 {pos2:>5.2f}  持币 {100*(1-pmask.mean()):>5.1f}%")

    # S1 仅个股层 + S3 叠加
    for sthr in s_thrs:
        mc = il.build_mask_cache(all_codes, pmap, all_dates, sthr, "index", golden)
        nav1, pos1, m1 = run_variant(targets, weights_map, pmap, all_dates, coef_fn,
                                     mc, None, enabled=True)
        key1 = f"S1_个股{int(sthr*100)}"
        variants[key1] = (nav1, pos1, m1)
        print(f"  {key1:<14} 总收益 {m1['total_ret']*100:>+8.2f}%  回撤 {m1['max_dd']*100:>7.2f}%"
              f"  Sharpe {m1['sharpe']:>5.2f}  持仓 {pos1:>5.2f}")
        for pthr in p_thrs:
            pmask = locked_stop(base_v, peak, pthr, golden)
            nav3, pos3, m3 = run_variant(targets, weights_map, pmap, all_dates, coef_fn,
                                         mc, pmask, enabled=True)
            key3 = f"S3_叠加{int(sthr*100)}+{int(pthr*100)}"
            variants[key3] = (nav3, pos3, m3)
            print(f"  {key3:<14} 总收益 {m3['total_ret']*100:>+8.2f}%  回撤 {m3['max_dd']*100:>7.2f}%"
                  f"  Sharpe {m3['sharpe']:>5.2f}  持仓 {pos3:>5.2f}")

    # ── 自证（两重）──
    nav_ii, _, m_ii = run_variant(targets, weights_map, pmap, all_dates, coef_fn,
                                  ones_cache, port_ones, enabled=True)
    d1 = float(np.max(np.abs(base_v - np.array([v for _, v in nav_ii], float))))
    ok1 = d1 < 1e-6
    print(f"\n  [自证1] 双 mask 恒 True vs 基线               最大差 = {d1:.2e} → {'PASS' if ok1 else 'FAIL ❌'}")
    mc20 = il.build_mask_cache(all_codes, pmap, all_dates, s_thrs[0], "index", golden)
    nav_i2, _, _ = run_variant(targets, weights_map, pmap, all_dates, coef_fn,
                               mc20, port_ones, enabled=True)
    nav_s1 = np.array([v for _, v in variants[f"S1_个股{int(s_thrs[0]*100)}"][0]], float)
    d2 = float(np.max(np.abs(nav_s1 - np.array([v for _, v in nav_i2], float))))
    ok2 = d2 < 1e-6
    print(f"  [自证2] 组合层恒 True + 个股{int(s_thrs[0]*100)}% ≡ 仅个股层   最大差 = {d2:.2e} → {'PASS' if ok2 else 'FAIL ❌'}")

    # ── 落盘 + 反算校验 ──
    verify_log = []
    for key, (nav, pos, m) in variants.items():
        fn = os.path.join(OUTDIR, f"nav_{args.tag}_{key}.csv")
        pd.DataFrame(nav, columns=["trade_date", f"nav_{key}"]).to_csv(
            fn, index=False, encoding="utf-8-sig")
        verify_log.append((os.path.basename(fn), _verify_csv(fn, m)))

    rows = [dict(it.metrics_row("S0_基线", m0, m0), 平均持仓=pos0)]
    for key in variants:
        if key == "S0_基线":
            continue
        nav, pos, m = variants[key]
        r = it.metrics_row(key, m, m0)
        r["平均持仓"] = pos
        rows.append(r)
    dfm = pd.DataFrame(rows)
    dfm.to_csv(os.path.join(OUTDIR, f"metrics_{args.tag}.csv"), index=False, encoding="utf-8-sig")

    # ── 交互项分解 ──
    print("\n" + "=" * 112)
    print("  增量 vs 重复表达：Interaction = S3 − S1 − S2 + S0")
    print("=" * 112)
    inter_rows = []
    for sthr in s_thrs:
        s1k = f"S1_个股{int(sthr*100)}"
        if s1k not in variants:
            continue
        s1 = variants[s1k][2]
        for pthr in p_thrs:
            s2 = variants[f"S2_组合{int(pthr*100)}"][2]
            s3 = variants[f"S3_叠加{int(sthr*100)}+{int(pthr*100)}"][2]
            i_ret = (s3["total_ret"] - s1["total_ret"] - s2["total_ret"] + m0["total_ret"]) * 100
            i_dd = (s3["max_dd"] - s1["max_dd"] - s2["max_dd"] + m0["max_dd"]) * 100
            i_sh = s3["sharpe"] - s1["sharpe"] - s2["sharpe"] + m0["sharpe"]
            best_single = max(s1["sharpe"], s2["sharpe"])
            verdict = ("增量(可加)" if abs(i_sh) < 0.02 else
                       ("重复表达(叠加≤单层)" if s3["sharpe"] <= best_single + 0.01 else "超加性"))
            inter_rows.append(dict(组合=f"个股{int(sthr*100)}+A15-{int(pthr*100)}",
                                   交互_收益=i_ret, 交互_回撤=i_dd, 交互_Sharpe=i_sh,
                                   S3Sharpe=s3["sharpe"], 单层最优Sharpe=best_single,
                                   判定=verdict))
            print(f"  个股{int(sthr*100)}% + A15-{int(pthr*100)}%:  "
                  f"交互收益 {i_ret:>+8.2f}pp  交互回撤 {i_dd:>+7.2f}pp  交互Sharpe {i_sh:>+6.3f}  "
                  f"| S3 Sharpe {s3['sharpe']:.2f} vs 单层最优 {best_single:.2f} → {verdict}")
    if inter_rows:
        pd.DataFrame(inter_rows).to_csv(os.path.join(OUTDIR, f"interaction_{args.tag}.csv"),
                                        index=False, encoding="utf-8-sig")

    print(f"\n[信任但验证] 落盘CSV反算全部吻合={all(ok for _, ok in verify_log)}；"
          f"自证1={ok1} 自证2={ok2}")
    print("  判读：DD_cut/Eff 为【负】=回撤变浅(改善)；|Eff|=每牺牲1pp收益换回的回撤削减(pp)，≪1=不划算")
    print(f"产物目录: {OUTDIR}")

    with open(os.path.join(OUTDIR, f"summary_{args.tag}.json"), "w", encoding="utf-8") as f:
        json.dump(dict(args=vars(args), self_proof=dict(identity=ok1, port_ones=ok2),
                       csv_verify_all_ok=all(ok for _, ok in verify_log),
                       interaction=inter_rows), f, ensure_ascii=False, indent=2, default=str)


if __name__ == "__main__":
    main()
