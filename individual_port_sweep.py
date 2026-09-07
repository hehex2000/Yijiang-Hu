# -*- coding: utf-8 -*-
"""
组合层 A15 阈值【稳健性检验】：细网格扫描 + 分时段稳定性
==========================================================
动机：individual_port_ab.py 全样本出现非单调结果 —— 组合层 thr=15% 反而劣于基线
（+148% / Sharpe 0.38），thr=20% 却全面碾压（+288% / -20.50% / Sharpe 0.64），
thr=25% 居中（+246% / Sharpe 0.55）。三点网格上"中间点最好且远超两端"是典型的
**参数过拟合指纹**，直接采信 thr=20% 会踩坑（六闸门之「参数面过拟合」）。

检验三条：
  1. 细网格扫描（8%~40%，14 点）：看 thr 与结果的关系是【平滑平台】还是【尖峰】。
     平滑/平台 → 参数稳健；孤立尖峰 → 过拟合，不可采信。
  2. 分时段稳定性（全样本 / 2013-2018 / 2019-2025）：最优阈值在不同子样本是否一致。
  3. 触发事件诊断：列出每次组合层触发/解锁的日期与当时的基线峰回撤，
     判断 15% vs 20% 的巨大差异是否由【少数几次关键事件】驱动（若是 → 更不可信）。

判定标准（事前声明，避免事后挑参）：
  ✅ 稳健   ：相邻 thr 结果接近，且最优档在 ≥2 个子样本中同样占优
  ❌ 过拟合 ：thr=20% 是孤立尖峰（相邻 18%/22% 明显塌陷），或最优档随子样本漂移

产物：data/results/negative_cost/sweep_*.csv / sweep_run_*.txt
"""
import os, sys, json, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import run_dividend_low_vol_quality_bt as dlq
from run_monthly_rebalance import get_trade_dates
import individual_trail as it
import individual_latch as il
import individual_trail_replay as ir
import macd_plugin_validate as M
from decompose_trail_ab import locked_stop
from regime_cash_overlay import load_index_close, BENCH

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "results", "negative_cost")
DEFAULT_SEL = "data/results/dividend_low_vol/bt_quality_sel_OFFICIAL_OFFICIAL_COMPACT_all_12_20130101_20260903.csv"

SWEEP = [8, 10, 12, 14, 15, 16, 18, 20, 22, 25, 28, 30, 35, 40]
WINDOWS = [("全样本", "20130101", "20251231"),
           ("前半 2013-2018", "20130101", "20181231"),
           ("后半 2019-2025", "20190101", "20251231")]


def slice_window(targets, all_dates, start, end):
    tg = [(d, cs) for d, cs in targets if start <= str(d) <= end]
    ad = [d for d in all_dates if start <= str(d) <= end]
    return tg, ad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sel-csv", default=DEFAULT_SEL)
    ap.add_argument("--start", default="20130101")
    ap.add_argument("--end", default="20251231")
    ap.add_argument("--sweep", default=",".join(str(x) for x in SWEEP))
    ap.add_argument("--layer", choices=["port", "stock"], default="port",
                    help="port=组合层A15全局掩码 / stock=个股层锁存+指数MACD解锁")
    ap.add_argument("--tag", default="sw")
    args = ap.parse_args()

    thr_list = [int(x) / 100.0 for x in args.sweep.split(",")]
    dlq.START, dlq.END = args.start, args.end
    os.makedirs(OUTDIR, exist_ok=True)

    t0 = time.time()
    layer_cn = "组合层 A15(全局掩码)" if args.layer == "port" else "个股层(锁存+指数MACD解锁)"
    print(f"[{layer_cn} 阈值稳健性检验] 扫描 {args.sweep}%  窗口 {args.start}~{args.end}")
    targets, weights_map, _ = ir.load_targets_from_sel(args.sel_csv, args.end)
    all_codes = sorted({c for _, cs in targets for c in cs})
    pmap = dlq.bulk_close_prices(all_codes, args.start, args.end)
    dlq.EXEC_PMAP.clear()
    dlq.EXEC_PMAP.update(dlq.bulk_open_prices(all_codes, args.start, args.end))
    all_dates = get_trade_dates(args.start, args.end)
    print(f"  行情 {len(all_codes)} 只 / {len(all_dates)} 交易日（{time.time()-t0:.1f}s）\n")

    ic = load_index_close(BENCH, args.start, args.end)
    ic = ic.reindex(pd.Index([int(d) for d in all_dates])).ffill()
    golden = M.macd_golden(ic.values.astype(float)).values
    coef_fn = dlq._make_coef_fn(False, "rolling", 756, None, None, 0.5, 1.0)
    ones_cache = {c: np.ones(len(all_dates), bool) for c in all_codes}

    all_rows, events = [], []
    for wname, ws, we in WINDOWS:
        tg, ad = slice_window(targets, all_dates, ws, we)
        if len(tg) < 3:
            continue
        nav_b = dlq.run_nav_weighted(tg, weights_map, pmap, ad, coef_fn)
        m_b = dlq.compute_metrics(nav_b, ad)
        bv = np.array([v for _, v in nav_b], float)
        pk = np.maximum.accumulate(bv)
        print(f"── {wname}（{ws[:4]}~{we[:4]}，{len(ad)} 交易日，{len(tg)} 期）──")
        print(f"  基线  总收益 {m_b['total_ret']*100:>+8.2f}%  回撤 {m_b['max_dd']*100:>7.2f}%  "
              f"Sharpe {m_b['sharpe']:>5.2f}")
        print(f"  {'thr':>5}{'总收益':>10}{'最大回撤':>10}{'Sharpe':>8}{'Calmar':>8}"
              f"{'持币%':>8}{'触发次数':>9}{'vs基线(收益pp)':>15}{'vs基线(Sharpe)':>15}")
        rows = []
        for thr in thr_list:
            if args.layer == "port":
                pmask = locked_stop(bv, pk, thr, golden)
                nav, posc = il.run_nav_weighted_masked(
                    tg, weights_map, pmap, ad, coef_fn, ones_cache,
                    enabled=False, port_mask=pmask)
                # 触发次数：mask 由 True→False 的跳变数
                trig = int(np.sum((~pmask[1:]) & pmask[:-1]))
                cash_pct = 100 * (1 - pmask.mean())
            else:
                mc = il.build_mask_cache(all_codes, pmap, ad, thr, "index", golden)
                nav, posc = il.run_nav_weighted_masked(
                    tg, weights_map, pmap, ad, coef_fn, mc,
                    enabled=True, port_mask=None)
                # 个股层：统计所有个股的累计离场切换次数 + 实际现金占比代理（持仓数下降）
                trig = int(sum(int(np.sum((~mc[c][1:]) & mc[c][:-1])) for c in all_codes))
                cash_pct = 100 * (1 - np.mean(posc) / max(
                    np.mean([len(cs) for _, cs in tg]), 1e-9))
            m = dlq.compute_metrics(nav, ad)
            rows.append(dict(窗口=wname, thr=int(thr * 100), total_ret=m["total_ret"] * 100,
                             max_dd=m["max_dd"] * 100, sharpe=m["sharpe"], calmar=m["calmar"],
                             cash_pct=cash_pct, 触发次数=trig,
                             d_ret=(m["total_ret"] - m_b["total_ret"]) * 100,
                             d_sharpe=m["sharpe"] - m_b["sharpe"],
                             d_dd=(m["max_dd"] - m_b["max_dd"]) * 100))
            print(f"  {int(thr*100):>4}%{m['total_ret']*100:>+9.2f}%{m['max_dd']*100:>9.2f}%"
                  f"{m['sharpe']:>8.2f}{m['calmar']:>8.3f}{cash_pct:>7.1f}%"
                  f"{trig:>9}{(m['total_ret']-m_b['total_ret'])*100:>+14.2f}pp"
                  f"{m['sharpe']-m_b['sharpe']:>+14.2f}")
        all_rows += rows
        # 最优档（按 Sharpe）是否随窗口漂移
        best = max(rows, key=lambda r: r["sharpe"])
        print(f"  → 本窗口 Sharpe 最优档 = thr {best['thr']}%  (Sharpe {best['sharpe']:.2f}，"
              f"vs 基线 {best['d_sharpe']:+.2f})\n")

        # 触发事件诊断（仅全样本、仅 15%/20% 两档）
        if wname == "全样本" and args.layer == "port":
            for thr in (0.15, 0.20):
                pmask = locked_stop(bv, pk, thr, golden)
                for i in range(1, len(pmask)):
                    if pmask[i - 1] and not pmask[i]:
                        d0 = ad[i]
                        j = i
                        while j < len(pmask) and not pmask[j]:
                            j += 1
                        d1 = ad[j] if j < len(pmask) else None
                        events.append(dict(thr=int(thr * 100), 触发日=d0,
                                           解锁日=d1, 持币天数=(j - i),
                                           触发时峰回撤=(bv[i] / pk[i] - 1) * 100))
    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(OUTDIR, f"sweep_{args.tag}.csv"), index=False, encoding="utf-8-sig")
    if events:
        edf = pd.DataFrame(events)
        edf.to_csv(os.path.join(OUTDIR, f"sweep_events_{args.tag}.csv"), index=False, encoding="utf-8-sig")
        print("── 触发事件诊断（组合层离场/回场）──")
        for thr in (15, 20):
            sub = edf[edf["thr"] == thr]
            print(f"  thr={thr}%：触发 {len(sub)} 次，累计持币 {sub['持币天数'].sum()} 天")
            for _, r in sub.iterrows():
                print(f"     触发 {r['触发日']}（峰回撤 {r['触发时峰回撤']:.1f}%）→ "
                      f"解锁 {r['解锁日']}  持币 {int(r['持币天数'])} 天")

    # 稳健性判定
    full = df[df["窗口"] == "全样本"]
    best_thr = int(full.loc[full["sharpe"].idxmax(), "thr"])
    neigh = [int(t * 100) for t in thr_list if abs(int(t * 100) - best_thr) <= 2 and int(t * 100) != best_thr]
    nb = full[full["thr"].isin(neigh)]["sharpe"]
    spike = (nb.max() < full.loc[full["thr"] == best_thr, "sharpe"].iloc[0] - 0.10) if len(nb) else False
    bests = {w: int(g.loc[g["sharpe"].idxmax(), "thr"]) for w, g in df.groupby("窗口")}
    stable = len(set(bests.values())) <= 2
    print(f"\n── 稳健性判定 ──")
    print(f"  全样本最优档 thr={best_thr}%  相邻档(±2pp) Sharpe = "
          f"{', '.join(f'{v:.2f}' for v in nb.values) if len(nb) else '无'}")
    print(f"  各窗口最优档：{bests}")
    print(f"  → {'❌ 孤立尖峰（相邻档塌陷）→ 参数过拟合嫌疑，不可采信' if spike else '✅ 非尖峰：相邻档与最优档接近（平台/平滑）'}")
    print(f"  → {'✅ 跨子样本最优档一致/邻近 → 稳健' if stable else '❌ 最优档随子样本漂移 → 不稳'}")
    with open(os.path.join(OUTDIR, f"sweep_verdict_{args.tag}.json"), "w", encoding="utf-8") as f:
        json.dump(dict(best_thr_full=best_thr, spike=bool(spike), stable=bool(stable),
                       bests_by_window=bests), f, ensure_ascii=False, indent=2)
    print(f"\n产物: {OUTDIR}/sweep_{args.tag}.csv, sweep_events_{args.tag}.csv")


if __name__ == "__main__":
    main()
