# -*- coding: utf-8 -*-
"""
个股级【锁存硬止损 + MACD 金叉解锁】真实回测 · A/B 代价对照
================================================================
承接 individual_trail.py（非锁存移动止盈 ≈ 负交易）的下一步：
报告 §七 建议「若确需个股层保护，应转向锁存硬止损 + MACD 金叉解锁（组合层 A15 方案）」，
本脚本在【同一真实持仓、同一基线】上把 A15 语义落到个股层，与移动止盈做同口径对照。

设计（严守纪律：不改动长回测引擎，复用其真实持仓与 NAV 重放机制）
  1. 持仓来源 = 引擎落盘选股明细 CSV（official_compact · 全A · 股息率加权 · TOP12 · 52 期），
     与 individual_trail_replay.py 完全同源 → 与移动止盈结论可直接对比。
  2. 基线 NAV = 引擎自身 run_nav_weighted() 重放（零改造真值），并与引擎落盘 NAV 交叉校验。
  3. 锁存掩码 = 逐股复制 decompose_trail_ab.locked_stop 的状态机（含判定顺序）：
         hit[i] = close[i] < cummax(close[0..i]) * (1-thr)
         逐日：if hit[i]: s=True ; if golden[i]: s=False ; stopped[i]=s ; mask = ~stopped
     ⚠️ 顺序与组合层逐位一致：同一日既触锁又出现金叉 → 金叉优先（解锁）。
  4. 解锁信号三档（--unlock）：
       none  → 永不解锁（触锁后永久离场，对应组合层 B15）
       index → 沪深300 MACD 金叉解锁（对应组合层 A15 的信号源）
       stock → 个股自身 MACD 金叉解锁（纯个股层语义）
  5. 自证：thr→∞ 时 hit 恒 False ⇒ mask 恒 True ⇒ 与基线逐位相等（最大差须 = 0）。

代价指标（与组合层 decompose_trail_ab 同口径；⚠️ 按 |Eff| 判读，不按符号）：
  DD_cut   = 基线最大回撤 − 方法最大回撤   (pp；最大回撤以负数存储 ⇒ 负=回撤变浅=改善)
  Ret_cost = 基线总收益   − 方法总收益     (pp，>0=牺牲了收益)
  Eff      = DD_cut / max(Ret_cost, ε)
  |Eff|    = 每牺牲 1pp 收益换回的回撤削减(pp) ≫1 划算 / ≪1 不划算

产物（data/results/negative_cost/）：
  nav_latch_<tag>_*.csv / metrics_latch_<tag>.csv / summary_latch.json
信任但验证：落盘后从 CSV 反算 总收益/最大回撤，与内存指标逐位比对。
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
import macd_plugin_validate as M
from regime_cash_overlay import load_index_close, BENCH

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "results", "negative_cost")
DEFAULT_SEL = "data/results/dividend_low_vol/bt_quality_sel_OFFICIAL_OFFICIAL_COMPACT_all_12_20130101_20260903.csv"
DEFAULT_NAV = "data/results/dividend_low_vol/bt_quality_nav_20130101_20260903_official_compact_all_12_hfq.csv"


# ──────────────────────────────────────────────────────────────
# 逐股【锁存】掩码（语义逐位复制 decompose_trail_ab.locked_stop）
# ──────────────────────────────────────────────────────────────
def _ffill(arr):
    a = np.asarray(arr, dtype=float).copy()
    valid = ~np.isnan(a)
    if not valid.any():
        return np.zeros(len(a))
    last = a[np.argmax(valid)]
    for i in range(len(a)):
        if valid[i]:
            last = a[i]
        else:
            a[i] = last
    return a


def latch_mask(close_path, thr, golden=None):
    """锁存硬止损：close < cummax*(1-thr) 触锁→离场，直到 golden[i] 解锁。
    golden=None → 永不解锁（永久离场）。thr>=1e9 → 恒 True（= 无控制基线）。"""
    arr = _ffill(close_path)
    if thr >= 1e9:
        return np.ones(len(arr), dtype=bool)
    peak = np.maximum.accumulate(arr)
    hit = arr < peak * (1.0 - thr)
    stopped = np.zeros(len(arr), dtype=bool)
    s = False
    for i in range(len(arr)):
        if hit[i]:
            s = True
        if golden is not None and golden[i]:
            s = False     # 同日既触锁又金叉 → 金叉优先（与组合层逐位一致）
        stopped[i] = s
    return ~stopped       # True=可持仓


def run_nav_weighted_masked(targets, weights_map, price_map, all_dates,
                            coef_fn, mask_cache, enabled=True, port_mask=None):
    """镜像 dlq.run_nav_weighted，叠加【逐股锁存掩码】+【可选·组合层 A15 全局掩码】。

    与 individual_trail.run_nav_weighted_trail 的两处关键差别（为锁存语义所必需）：
      1. 再平衡日：掩码为 out 的目标股【不建仓】（记为 stopped），而不是"买了再立刻卖"。
         否则每期都要为锁存股白付一次买卖双边费用，把成本算到止盈头上（口径失真）。
      2. 回场：只回补【确实被止盈踢出去】的个股（stopped 集合），
         不回补"再平衡时因整手取整/现金约束没买满"的票——否则 thr→∞ 时也会额外建仓，
         导致自证（掩码恒 True 应 ≡ 基线）失败。

    ⚠️ 锁存跨再平衡持续：只在 MACD 金叉解锁（mask 转 in）后才重新建仓。

    port_mask（组合层 A15，可选，None=不启用）：
      长度 = len(all_dates) 的 bool 数组，True=允许持仓。False 日 → 全部清仓转现金并锁存，
      转 True 后按本期目标权重回补。语义与组合层 decompose_trail_ab.locked_stop 一致
      （全局清仓 + MACD 金叉解锁），差别只在作用层级（这里作用于个股持仓集合）。
      ⚠️ 信号源固定为【影子基线净值】而非本方案自身净值（见 individual_port_ab.py），
      以保证 S2/S3 两档的组合层信号逐位相同 → 交互项可干净分解（port_mask=None 时行为不变）。
    """
    from run_dividend_low_vol_quality_bt import (
        EXEC_PMAP, PRICE_MODE, INIT_CAPITAL, ffill_price, get_open_price, _sel_date_of,
    )
    cash = INIT_CAPITAL
    positions = {}
    stopped = set()          # 当前处于「锁存·未解锁」的个股
    port_reentry = set()     # 组合层解锁后待回补的个股
    port_stopped = False     # 组合层是否已触发全局清仓
    nav = []
    pos_cnt = []
    rebal_set = dict(targets)
    cur_rb = None

    for idx, d in enumerate(all_dates):
        # ── 组合层 A15：先决定「今天能不能持仓」，再走个股层逻辑 ──
        p_ok = True if port_mask is None else bool(port_mask[idx])
        if not p_ok:
            if positions:
                for code in sorted(positions.keys()):   # sorted 保确定性
                    px = ffill_price(price_map, code, d, all_dates, idx)
                    if px is None:
                        continue
                    sh = positions[code]
                    cash += px * sh - calc_fee("sell", px, sh)
                    del positions[code]
            port_stopped = True
        elif port_stopped:
            port_stopped = False
            # 再平衡日由正常再平衡流程自然建仓；否则标记待回补（下方回场循环 sorted 买入）
            if rebal_set.get(d) is None and cur_rb is not None:
                port_reentry |= set(rebal_set.get(cur_rb, []))

        rb_target = rebal_set.get(d)
        if rb_target is not None:
            cur_rb = d
            stopped = set()   # 新一期：清空后按当前 mask 重新判定，锁存者会立刻被重新加入

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
                if not p_ok:
                    wt = 0.0            # 组合层 A15 触发 → 全局不建仓
                elif enabled and wt > 0 and not mask_cache[code][idx]:
                    wt = 0.0            # 锁存未解 → 本期跳过建仓（不付无谓的双边费用）
                    stopped.add(code)
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
            # 仍持仓但 mask=out（上期遗留且不在本期目标）→ 清仓
            if enabled:
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
            if enabled and cur_rb is not None:
                # 离场：持仓股跌破阈值 → 清仓并记入 stopped
                for code in list(positions.keys()):
                    if not mask_cache[code][idx]:
                        px = ffill_price(price_map, code, d, all_dates, idx)
                        if px is None:
                            continue
                        sh = positions[code]
                        proceeds = px * sh - calc_fee("sell", px, sh)
                        cash += proceeds
                        del positions[code]
                        stopped.add(code)
                # 回场：仅限 stopped 集合（真被踢出去的）+ 组合层解锁待回补，
                #       mask 转 in 且仍属本期目标
                # 🔴 必须 sorted()：set 迭代顺序受 PYTHONHASHSEED 跨进程随机化影响，
                #    现金不足以买回全部待回场个股时，"谁先买到"会随进程变化 → 同参不同结果。
                for code in sorted(set(stopped) | port_reentry):
                    if mask_cache[code][idx] and code in rebal_set.get(cur_rb, []):
                        wmap = weights_map.get(str(cur_rb), {})
                        wt = wmap.get(code, 0.0)
                        px = ffill_price(price_map, code, d, all_dates, idx)
                        if wt > 0 and px is not None:
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
                                    stopped.discard(code)
                                    port_reentry.discard(code)
                                    continue
                    # 已不在本期目标 → 移出跟踪
                    if code not in rebal_set.get(cur_rb, []):
                        stopped.discard(code)
                        port_reentry.discard(code)

        mv = cash
        for code, sh in positions.items():
            px = ffill_price(price_map, code, d, all_dates, idx)
            if px:
                mv += sh * px
        nav.append((d, mv))
        pos_cnt.append(len(positions))
    return nav, pos_cnt


def build_mask_cache(all_codes, price_map, all_dates, thr, unlock, golden_idx=None):
    """为每只持仓股构建锁存掩码；unlock ∈ {none, index, stock}。"""
    cache = {}
    for code in all_codes:
        pm = price_map.get(code, {})
        raw = np.array([pm.get(d) for d in all_dates], dtype=float)
        if unlock == "stock":
            g = M.macd_golden(_ffill(raw)).values
        elif unlock == "index":
            g = golden_idx
        else:
            g = None
        cache[code] = latch_mask(raw, thr, g)
    return cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sel-csv", default=DEFAULT_SEL)
    ap.add_argument("--nav-csv", default=DEFAULT_NAV)
    ap.add_argument("--start", default="20130101")
    ap.add_argument("--end", default="20251231")
    ap.add_argument("--thr", default="15,20,25")
    ap.add_argument("--unlock", default="none,index,stock")
    ap.add_argument("--overlay", choices=["on", "off", "both"], default="off")
    ap.add_argument("--price-mode", default="hfq", choices=["raw", "hfq"])
    ap.add_argument("--tag", default="latch")
    args = ap.parse_args()

    thr_list = [int(x) / 100.0 for x in args.thr.split(",")]
    unlock_list = [u.strip() for u in args.unlock.split(",")]
    dlq.PRICE_MODE = args.price_mode
    dlq.START = args.start
    dlq.END = args.end

    os.makedirs(OUTDIR, exist_ok=True)
    t0 = time.time()
    print(f"[锁存回测] 窗口 {args.start}~{args.end}  thr={args.thr}%  unlock={unlock_list}  price={args.price_mode}")
    targets, weights_map, sel_df = it.load_targets_from_sel(args.sel_csv, args.end) \
        if hasattr(it, "load_targets_from_sel") else (None, None, None)
    if targets is None:
        import individual_trail_replay as ir
        targets, weights_map, sel_df = ir.load_targets_from_sel(args.sel_csv, args.end)

    all_codes = sorted({c for _, cs in targets for c in cs})
    pmap = dlq.bulk_close_prices(all_codes, args.start, args.end)
    if args.price_mode == "hfq":
        dlq.EXEC_PMAP.clear()
        dlq.EXEC_PMAP.update(dlq.bulk_open_prices(all_codes, args.start, args.end))
    all_dates = get_trade_dates(args.start, args.end)
    print(f"  行情 {len(all_codes)} 只 / {len(all_dates)} 交易日（{time.time()-t0:.1f}s）")

    # 指数 MACD 金叉（unlock=index 的信号源）
    golden_idx = None
    if "index" in unlock_list:
        ic = load_index_close(BENCH, args.start, args.end)
        ic = ic.reindex(pd.Index([int(d) for d in all_dates])).ffill()
        golden_idx = M.macd_golden(ic.values.astype(float)).values
        print(f"  指数解锁信号源={BENCH}  金叉天数={int(golden_idx.sum())}/{len(golden_idx)}")

    scenarios = []
    if args.overlay in ("on", "both"):
        scenarios.append(True)
    if args.overlay in ("off", "both"):
        scenarios.append(False)

    pkg, verify_log = {}, []
    for overlay in scenarios:
        stag = f"ov{1 if overlay else 0}"
        print(f"\n── 场景 overlay={'ON' if overlay else 'OFF'} ──")
        if overlay:
            dlq._preload_index_channel("000922.SH")
        coef_fn = dlq._make_coef_fn(overlay, "rolling", 756, None, None, 0.5, 1.0)

        nav_base = dlq.run_nav_weighted(targets, weights_map, pmap, all_dates, coef_fn)
        m_base = dlq.compute_metrics(nav_base, all_dates)
        _, posc_base = run_nav_weighted_masked(targets, weights_map, pmap, all_dates,
                                               coef_fn, {c: np.ones(len(all_dates), bool) for c in all_codes},
                                               enabled=False)
        # 交叉校验：重放 vs 引擎落盘 NAV
        try:
            nav_df = pd.read_csv(args.nav_csv, encoding="utf-8-sig")
            nav_col = [c for c in nav_df.columns if c.startswith("nav_")
                       and "000" not in c and "922" not in c][0]
            eng_nav = {str(int(r.trade_date)): float(r[nav_col]) for _, r in nav_df.iterrows()}
            both = [(v, eng_nav[d]) for d, v in nav_base if d in eng_nav]
            max_abs = float(np.max(np.abs([a - b for a, b in both]))) if both else float("nan")
        except Exception as e:
            max_abs = float("nan")
            print(f"  [交叉校验] 跳过（{e}）")
        print(f"  基线: 总收益 {m_base['total_ret']*100:+.2f}%  最大回撤 {m_base['max_dd']*100:.2f}%"
              f"  平均持仓 {np.mean(posc_base):.2f} 只")
        if np.isfinite(max_abs):
            print(f"  [交叉校验] 重放 vs 引擎NAV 最大绝对差 = {max_abs:.3f} → "
                  f"{'PASS' if max_abs < 1.0 else 'WARN'}")

        results = {"base": (nav_base, m_base, np.mean(posc_base))}
        for unlock in unlock_list:
            for thr in thr_list:
                key = f"{unlock}{int(thr*100)}"
                mc = build_mask_cache(all_codes, pmap, all_dates, thr, unlock, golden_idx)
                nav_l, posc = run_nav_weighted_masked(targets, weights_map, pmap, all_dates,
                                                      coef_fn, mc, enabled=True)
                m_l = dlq.compute_metrics(nav_l, all_dates)
                results[key] = (nav_l, m_l, float(np.mean(posc)))
                print(f"  {key:<10} 总收益 {m_l['total_ret']*100:>+8.2f}%  "
                      f"最大回撤 {m_l['max_dd']*100:>7.2f}%  平均持仓 {np.mean(posc):>5.2f} 只")

        # 自证：thr→∞ ⇒ 掩码恒 True ⇒ ≡ 基线
        mc_inf = build_mask_cache(all_codes, pmap, all_dates, 1e12, "none", None)
        nav_inf, _ = run_nav_weighted_masked(targets, weights_map, pmap, all_dates, coef_fn, mc_inf, enabled=True)
        vb = np.array([v for _, v in nav_base]); vi = np.array([v for _, v in nav_inf])
        max_diff = float(np.max(np.abs(vb - vi)))
        identity_ok = max_diff < 1e-6
        print(f"  [自证] thr=∞ vs 基线 最大绝对差 = {max_diff:.2e} → {'PASS' if identity_ok else 'FAIL ❌'}")

        # 落盘 + 反算校验
        for key, (nav, m, ap_) in results.items():
            fn = os.path.join(OUTDIR, f"nav_{args.tag}_{stag}_{key}.csv")
            pd.DataFrame(nav, columns=["trade_date", f"nav_{key}"]).to_csv(fn, index=False, encoding="utf-8-sig")
            ok = _verify_csv(fn, m)
            verify_log.append((os.path.basename(fn), ok))
        rows = [dict(it.metrics_row("基线(无个股止盈)", m_base, m_base),
                     平均持仓=results["base"][2])]
        for unlock in unlock_list:
            for thr in thr_list:
                key = f"{unlock}{int(thr*100)}"
                r = it.metrics_row(f"锁存{int(thr*100)}%-{unlock}", results[key][1], m_base)
                r["平均持仓"] = results[key][2]
                rows.append(r)
        pd.DataFrame(rows).to_csv(os.path.join(OUTDIR, f"metrics_{args.tag}_{stag}.csv"),
                                  index=False, encoding="utf-8-sig")
        pkg[stag] = dict(overlay=overlay, m_base=m_base, rows=rows, max_abs=max_abs,
                         identity_ok=identity_ok)

    summary = dict(args=vars(args), thr_list=thr_list, unlock_list=unlock_list,
                   cross_check_max_abs={k: v["max_abs"] for k, v in pkg.items()},
                   identity_ok={k: v["identity_ok"] for k, v in pkg.items()},
                   csv_verify_all_ok=all(ok for _, ok in verify_log))
    with open(os.path.join(OUTDIR, f"summary_{args.tag}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    print("\n" + "=" * 108)
    print("  个股级【锁存硬止损+MACD解锁】A/B（红利低波质量复合 · 真实持仓）")
    print("=" * 108)
    for stag, sc in pkg.items():
        print(f"\n── overlay={'ON' if sc['overlay'] else 'OFF'} ──")
        print(f"  {'方案':<22}{'总收益':>9}{'年化':>8}{'最大回撤':>10}{'Sharpe':>8}"
              f"{'DD_cut':>9}{'Ret_cost':>10}{'Eff':>8}{'|Eff|':>8}{'持仓':>7}")
        for r in sc["rows"]:
            eff = r["eff"]
            eff_s = f"{eff:>8.2f}" if np.isfinite(eff) else f"{'N/A':>8}"
            abs_s = f"{abs(eff):>8.3f}" if np.isfinite(eff) else f"{'N/A':>8}"
            print(f"  {r['name']:<22}{r['total_ret']:>+8.2f}%{r['ann']:>+7.2f}%{r['max_dd']:>9.2f}%"
                  f"{r['sharpe']:>8.2f}{r['dd_cut']:>+8.2f}pp{r['ret_cost']:>+9.2f}pp{eff_s}{abs_s}"
                  f"{r.get('平均持仓', float('nan')):>7.2f}")
    print(f"\n[信任但验证] 落盘CSV反算全部吻合={summary['csv_verify_all_ok']}；"
          f"thr=∞自证PASS={all(v['identity_ok'] for v in pkg.values())}")
    print(f"  判读：DD_cut/Eff 为【负】= 回撤变浅(改善)；|Eff| = 每牺牲1pp收益换回的回撤削减(pp)，≪1=不划算")
    print(f"产物目录: {OUTDIR}")


def _verify_csv(csv_path, m_inmem):
    df = pd.read_csv(csv_path)
    col = [c for c in df.columns if c.lower().startswith("nav")][0]
    vals = df[col].dropna().values.astype(float)
    total_ret = vals[-1] / vals[0] - 1
    peak = np.maximum.accumulate(vals)
    max_dd = (vals / peak - 1).min()
    return (abs(total_ret - m_inmem["total_ret"]) < 1e-9 and
            abs(max_dd - m_inmem["max_dd"]) < 1e-9)


if __name__ == "__main__":
    main()
