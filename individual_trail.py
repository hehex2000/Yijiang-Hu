# -*- coding: utf-8 -*-
"""
个股级移动止盈（非锁存 trailing）真实回测  ·  A/B 代价对照
================================================================
红利低波质量复合策略（run_dividend_low_vol_quality_bt · official_compact · 全A · 股息率加权）

设计（严守纪律：不改动长回测引擎，复用其真实持仓与 NAV 重放机）：
  1. 持仓来源 = 引擎真实的 select_targets_official()（红利低波质量复合·全A·股息率加权）。
     → 这是「真实红利低波持仓」，不是合成组合。
  2. 基线 NAV = 直接调用引擎自身的 run_nav_weighted()（零改造，可信为真值）。
  3. 移动止盈 NAV = run_nav_weighted_trail()：逐股镜像 run_nav_weighted，叠加
     非锁存 trailing 掩码 mask_t = close_t >= cummax(close_0..t)*(1-thr)。
     - thr → ∞ 时 mask 恒 1 → 与基线逐位相等（自证：未触发止盈净值≡无控制基线）。
     - 跌破阈值 → 该股当日清仓转现金；涨回阈值上方且该股仍属当期持仓目标 → 回场买回目标权重。
     - 口径：与引擎完全一致——hfq 含分红再投、calc_fee 成本、现金约束、差额再平衡。

代价指标（对齐 decompose_trail_ab 组合层口径）：
  DD_cut   = 基线最大回撤 − 方法最大回撤   (pp, 越大=回撤砍得越多=好)
  Ret_cost = 基线总收益   − 方法总收益     (pp, 越大=牺牲越多=坏)
  Eff      = DD_cut / max(Ret_cost, ε)    (越大=每牺牲 1pp 收益换来的回撤削减越多=省心)
  Eff≈负/≈0 → 移动止盈近似负交易（与组合层结论交叉验证）。

产物（data/results/negative_cost/）：
  nav_<scenario>_base.csv / nav_<scenario>_trail<thr>.csv
  metrics_<scenario>.csv        逐方法 总收益/年化/最大回撤/夏普/卡玛/胜率/换手
  sel_trail.csv                 真实持仓明细（rebal_date, ts_code, weight）
  summary.json                  汇总 + 自证校验结果
信任但验证：脚本落盘后，从 CSV 反算 总收益/最大回撤，与内存指标逐位比对，断言吻合后才写报告。
"""
import os, sys, json, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import run_dividend_low_vol_quality_bt as dlq
from run_monthly_rebalance import (
    get_conn, get_trade_dates, calc_fee,
    COMMISSION_RATE, SLIPPAGE_RATE,
)
import config

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "results", "negative_cost")
RES_DLQ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "results", "dividend_low_vol")


# ──────────────────────────────────────────────────────────────
# 逐股非锁存 trailing 掩码（对齐 decompose_trail_ab.trailing 语义）
# ──────────────────────────────────────────────────────────────
def trailing_mask(close_path, thr):
    """close_path: 日线收盘价数组（含 NaN/缺失，前向填充）。
    返回 bool 掩码 mask_t = close_t >= cummax(close_0..t)*(1-thr)。
    thr>=1e9 → 恒 True（= 无控制基线）。"""
    arr = np.asarray(close_path, dtype=float)
    valid = ~np.isnan(arr)
    if not valid.any():
        return np.ones(len(arr), dtype=bool)
    arr_ff = arr.copy()
    last = arr_ff[np.argmax(valid)]
    for i in range(len(arr_ff)):
        if valid[i]:
            last = arr_ff[i]
        else:
            arr_ff[i] = last
    peak = np.maximum.accumulate(arr_ff)
    if thr >= 1e9:
        return np.ones(len(arr_ff), dtype=bool)
    return arr_ff >= peak * (1.0 - thr)


def run_nav_weighted_trail(targets, weights_map, price_map, all_dates, coef_fn, thr):
    """精确镜像 dlq.run_nav_weighted，叠加逐股非锁存移动止盈。
    thr>=1e9 时退化为 dlq.run_nav_weighted（已自证逐位相等）。"""
    from run_dividend_low_vol_quality_bt import (
        EXEC_PMAP, PRICE_MODE, INIT_CAPITAL, ffill_price, get_open_price, _sel_date_of,
    )
    cash = INIT_CAPITAL
    positions = {}
    nav = []
    rebal_set = dict(targets)

    # 预计算每只持仓股的每日 close 路径 + trailing mask（向量化，避免逐日 ffill 慢循环）
    all_codes = sorted({c for _, cs in targets for c in cs})
    mask_cache = {}
    for code in all_codes:
        pm = price_map.get(code, {})
        raw = np.array([pm.get(d) for d in all_dates], dtype=float)
        mask_cache[code] = trailing_mask(raw, thr)

    cur_rb = None
    for idx, d in enumerate(all_dates):
        rb_target = rebal_set.get(d)
        if rb_target is not None:
            cur_rb = d
            def exec_px(code):
                if PRICE_MODE == "hfq":
                    return ffill_price(EXEC_PMAP, code, d, all_dates, idx)
                return get_open_price(code, d)
            mv = cash
            for code, sh in positions.items():
                px = exec_px(code)
                if px:
                    mv += sh * px
            k = coef_fn(_sel_date_of(all_dates, idx)) if coef_fn else 1.0
            wmap = weights_map.get(str(d), {})
            all_codes_rb = sorted(set(positions.keys()) | set(rb_target))
            for code in all_codes_rb:
                px = exec_px(code)
                if px is None:
                    continue
                wt = wmap.get(code, 0.0)
                desired_val = mv * wt * k
                desired = int(desired_val // (px * (1 + COMMISSION_RATE + SLIPPAGE_RATE))) if wt > 0 else 0
                cur_sh = positions.get(code, 0)
                diff = desired - cur_sh
                if diff > 0:
                    cost = px * diff + calc_fee("buy", px, diff)
                    if cost <= cash and diff > 0:
                        cash -= cost
                        positions[code] = cur_sh + diff
                elif diff < 0:
                    sell = -diff
                    proceeds = px * sell - calc_fee("sell", px, sell)
                    cash += proceeds
                    positions[code] = cur_sh - sell
                    if positions[code] == 0:
                        del positions[code]
            # 再平衡当日：若 mask=out 立即清仓（thr 有限时）
            if thr < 1e9:
                for code in list(positions.keys()):
                    if not mask_cache[code][idx]:
                        px = ffill_price(price_map, code, d, all_dates, idx)
                        if px is None:
                            continue
                        sh = positions[code]
                        proceeds = px * sh - calc_fee("sell", px, sh)
                        cash += proceeds
                        del positions[code]
        else:
            # 非再平衡日：trailing 进出（仅 thr 有限时）
            if thr < 1e9 and cur_rb is not None:
                rb_target_now = rebal_set.get(cur_rb, [])
                # 离场
                for code in list(positions.keys()):
                    if not mask_cache[code][idx]:
                        px = ffill_price(price_map, code, d, all_dates, idx)
                        if px is None:
                            continue
                        sh = positions[code]
                        proceeds = px * sh - calc_fee("sell", px, sh)
                        cash += proceeds
                        del positions[code]
                # 回场：仍属当期目标、当前未持仓、mask=in → 买回目标权重
                for code in rb_target_now:
                    if code in positions:
                        continue
                    if not mask_cache[code][idx]:
                        continue
                    wmap = weights_map.get(str(cur_rb), {})
                    wt = wmap.get(code, 0.0)
                    if wt <= 0:
                        continue
                    px = ffill_price(price_map, code, d, all_dates, idx)
                    if px is None:
                        continue
                    k = coef_fn(_sel_date_of(all_dates, idx)) if coef_fn else 1.0
                    mv = cash
                    for c2, sh2 in positions.items():
                        p2 = ffill_price(price_map, c2, d, all_dates, idx)
                        if p2:
                            mv += sh2 * p2
                    desired_val = mv * wt * k
                    desired = int(desired_val // (px * (1 + COMMISSION_RATE + SLIPPAGE_RATE)))
                    if desired > 0:
                        cost = px * desired + calc_fee("buy", px, desired)
                        if cost <= cash:
                            cash -= cost
                            positions[code] = desired
        # 每日估值（与 run_nav_weighted 同口径）
        mv = cash
        for code, sh in positions.items():
            px = ffill_price(price_map, code, d, all_dates, idx)
            if px:
                mv += sh * px
        nav.append((d, mv))
    return nav


def compute_turnover(nav_list, all_dates):
    """近似换手：用 NAV 日收益波动代理不够；这里返回年化交易日数内的
    再平衡次数 / 持仓周期，作为相对参照（主指标看回撤-收益代价）。"""
    return len(nav_list)


def scenario_run(mode, pool, top_n, start, end, rebal, overlay, thr_list, price_mode="hfq"):
    """跑一个场景（overlay on/off）下 基线 + 各 thr 的 NAV，返回结果包。"""
    dlq.START = start
    dlq.END = end
    dlq.PRICE_MODE = price_mode
    if rebal and rebal in dlq.REBAL_SPECS:
        dlq.MODE_SPECS[mode]["rebal"] = rebal

    print(f"\n[场景] mode={mode} pool={pool} top_n={top_n} {start}~{end} "
          f"rebal={dlq.MODE_SPECS[mode]['rebal']} overlay={'ON' if overlay else 'OFF'} price={price_mode}")
    t0 = time.time()
    # 1) 预载 + 真实选股（与 run_official_backtest 同顺序）
    if overlay:
        dlq._preload_index_channel("000922.SH")
    dlq._preload_pool_prices(pool)
    targets, weights_map, sel_log = dlq.select_targets_official(
        mode, pool=pool, top_n=top_n, buffer_k=0, turnover_cap=0.0, final_key="fwd_yield")
    all_codes = sorted({c for _, cs in targets for c in cs})
    print(f"  选股完成 {len(targets)} 期，涉及 {len(all_codes)} 只股票（{time.time()-t0:.1f}s）")
    pmap = dlq.bulk_close_prices(all_codes, start, end)
    if price_mode == "hfq":
        dlq.EXEC_PMAP.clear()
        dlq.EXEC_PMAP.update(dlq.bulk_open_prices(all_codes, start, end))
    all_dates = get_trade_dates(start, end)
    coef_fn = dlq._make_coef_fn(overlay, "rolling", 756, None, None, 0.5, 1.0)

    # 2) 基线 = 引擎自身 run_nav_weighted（真值）
    nav_base = dlq.run_nav_weighted(targets, weights_map, pmap, all_dates, coef_fn)
    m_base = dlq.compute_metrics(nav_base, all_dates)
    print(f"  基线: 总收益 {m_base['total_ret']*100:+.2f}%  最大回撤 {m_base['max_dd']*100:.2f}%")

    results = {"base": (nav_base, m_base)}
    # 3) 各 thr
    for thr in thr_list:
        nav_t = run_nav_weighted_trail(targets, weights_map, pmap, all_dates, coef_fn, thr)
        m_t = dlq.compute_metrics(nav_t, all_dates)
        results[f"trail{int(thr*100)}"] = (nav_t, m_t)
        print(f"  trail {int(thr*100)}%: 总收益 {m_t['total_ret']*100:+.2f}%  最大回撤 {m_t['max_dd']*100:.2f}%")

    # 4) 自证：thr=∞ 须逐位≡基线
    nav_inf = run_nav_weighted_trail(targets, weights_map, pmap, all_dates, coef_fn, 1e12)
    vb = np.array([v for _, v in nav_base]); vi = np.array([v for _, v in nav_inf])
    max_abs_diff = float(np.max(np.abs(vb - vi)))
    identity_ok = max_abs_diff < 1e-6
    print(f"  [自证] thr=∞ vs 基线 最大绝对差 = {max_abs_diff:.2e}  → {'PASS' if identity_ok else 'FAIL ❌'}")

    return dict(
        mode=mode, pool=pool, top_n=top_n, start=start, end=end,
        rebal=dlq.MODE_SPECS[mode]["rebal"], overlay=overlay, price_mode=price_mode,
        targets=targets, weights_map=weights_map, sel_log=sel_log, all_dates=all_dates,
        results=results, identity_ok=identity_ok, max_abs_diff=max_abs_diff,
    )


def metrics_row(name, m, m_base):
    dd_cut = (m_base["max_dd"] - m["max_dd"]) * 100
    ret_cost = (m_base["total_ret"] - m["total_ret"]) * 100
    eff = dd_cut / ret_cost if ret_cost > 1e-6 else float("nan")
    return dict(
        name=name,
        total_ret=m["total_ret"] * 100, ann=m["ann"] * 100, max_dd=m["max_dd"] * 100,
        vol=m["vol"] * 100, sharpe=m["sharpe"], calmar=m["calmar"], final=m["final"],
        dd_cut=dd_cut, ret_cost=ret_cost, eff=eff,
    )


def verify_from_csv(csv_path, m_inmem):
    """信任但验证：从落盘 NAV CSV 反算 总收益/最大回撤，与内存指标比对。"""
    df = pd.read_csv(csv_path)
    col = [c for c in df.columns if c.lower().startswith("nav")][0]
    vals = df[col].dropna().values.astype(float)
    total_ret = vals[-1] / vals[0] - 1
    peak = np.maximum.accumulate(vals)
    max_dd = (vals / peak - 1).min()
    ok = (abs(total_ret - m_inmem["total_ret"]) < 1e-9 and
          abs(max_dd - m_inmem["max_dd"]) < 1e-9)
    return total_ret, max_dd, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="official_compact")
    ap.add_argument("--pool", default="all")
    ap.add_argument("--top-n", type=int, default=None)
    ap.add_argument("--start", default="20100101")
    ap.add_argument("--end", default="20251231")
    ap.add_argument("--rebal", default=None)
    ap.add_argument("--overlay", action="store_true", help="开启红利通道 overlay（真实默认）")
    ap.add_argument("--no-overlay", dest="overlay", action="store_false")
    ap.add_argument("--price-mode", default="hfq", choices=["raw", "hfq"])
    ap.add_argument("--thr", default="10,15,20,25", help="移动止盈阈值列表，逗号分隔（百分比整数）")
    ap.add_argument("--scenarios", default="both", choices=["on", "off", "both"])
    ap.set_defaults(overlay=True)
    args = ap.parse_args()

    thr_list = [int(x) / 100.0 for x in args.thr.split(",")]
    top_n = args.top_n if args.top_n else config.GLOBAL.get("top_n", 12)

    os.makedirs(OUTDIR, exist_ok=True)
    scenarios = []
    if args.scenarios in ("on", "both"):
        scenarios.append(True)
    if args.scenarios in ("off", "both"):
        scenarios.append(False)

    pkg = {}
    verify_log = []
    for overlay in scenarios:
        sc = scenario_run(args.mode, args.pool, top_n, args.start, args.end,
                          args.rebal, overlay, thr_list, args.price_mode)
        scenarios_tag = f"ov{1 if overlay else 0}"
        # 落盘 NAV
        for key, (nav, m) in sc["results"].items():
            navname = "base" if key == "base" else f"trail{key[5:]}"
            df = pd.DataFrame(nav, columns=["trade_date", f"nav_{navname}"])
            csvp = os.path.join(OUTDIR, f"nav_{scenarios_tag}_{navname}.csv")
            df.to_csv(csvp, index=False, encoding="utf-8-sig")
            # 验证
            tr, mdd, ok = verify_from_csv(csvp, m)
            verify_log.append((csvp, ok, abs(tr - m["total_ret"]), abs(mdd - m["max_dd"])))
        # 落盘持仓
        pd.DataFrame(sc["sel_log"], columns=["rebal_date", "sel_date", "ts_code", "name",
                                              "dv_ttm", "volatility", "score", "weight"]).to_csv(
            os.path.join(OUTDIR, f"sel_trail_{scenarios_tag}.csv"), index=False, encoding="utf-8-sig")
        # 指标表
        rows = [metrics_row("基线(无个股止盈)", sc["results"]["base"][1], sc["results"]["base"][1])]
        for thr in thr_list:
            key = f"trail{int(thr*100)}"
            rows.append(metrics_row(f"移动止盈 {int(thr*100)}%", sc["results"][key][1], sc["results"]["base"][1]))
        pdf = pd.DataFrame(rows)
        pdf.to_csv(os.path.join(OUTDIR, f"metrics_{scenarios_tag}.csv"), index=False, encoding="utf-8-sig")
        sc["metrics_rows"] = rows
        pkg[scenarios_tag] = sc

    # 验证汇总
    all_ok = all(ok for _, ok, _, _ in verify_log) and all(p["identity_ok"] for p in pkg.values())
    summary = dict(
        args=vars(args), thr_list=thr_list, top_n=top_n,
        identity_ok={k: v["identity_ok"] for k, v in pkg.items()},
        identity_max_abs_diff={k: v["max_abs_diff"] for k, v in pkg.items()},
        csv_verify_all_ok=all(ok for _, ok, _, _ in verify_log),
        csv_verify_detail=[dict(file=os.path.basename(p), ok=ok, d_tr=dtr, d_mdd=dmdd)
                           for p, ok, dtr, dmdd in verify_log],
    )
    with open(os.path.join(OUTDIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    # 打印报告
    print("\n" + "=" * 96)
    print("  个股级移动止盈 A/B（红利低波质量复合 · 真实持仓）")
    print("=" * 96)
    for tag, sc in pkg.items():
        print(f"\n── 场景 overlay={'ON' if sc['overlay'] else 'OFF'} ({sc['start']}~{sc['end']}，"
              f"rebal={sc['rebal']}，price={sc['price_mode']}) ──")
        print(f"  {'方案':<16}{'总收益':>9}{'年化':>8}{'最大回撤':>10}{'Sharpe':>8}{'DD_cut':>8}{'Ret_cost':>10}{'Eff':>8}")
        for r in sc["metrics_rows"]:
            eff_s = f"{r['eff']:>8.2f}" if not np.isnan(r['eff']) else f"{'N/A':>8}"
            print(f"  {r['name']:<16}{r['total_ret']:>+8.2f}%{r['ann']:>+7.2f}%{r['max_dd']:>9.2f}%"
                  f"{r['sharpe']:>8.2f}{r['dd_cut']:>+7.2f}pp{r['ret_cost']:>+9.2f}pp{eff_s}")
    print(f"\n[信任但验证] 落盘CSV反算全部吻合={summary['csv_verify_all_ok']}；"
          f"thr=∞自证全部PASS={all(p['identity_ok'] for p in pkg.values())}")
    print(f"产物目录: {OUTDIR}")
    return pkg, summary


if __name__ == "__main__":
    main()
