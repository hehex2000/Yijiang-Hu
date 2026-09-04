# -*- coding: utf-8 -*-
"""工具②：活跃税 / 处置效应 体检。

对标 Barber & Odean (2000)《Trading Is Hazardous to Your Wealth》核心发现：
最活跃交易者年化收益 11.4% vs 市场 17.9%，"活跃税"约 6.5%/年——频繁交易本身
就在吞噬收益。

本工具扫描一组策略的 trades CSV，用平台真实成本函数（市价/taker 假设）重建 NAV，
计算：
  - 年化换手成本率（活跃税）= 总交易成本 / (平均组合市值 × 年数)
  - 总交易笔数、样本年数、平均组合市值
并对照 6.5%/年 基准，排序标红超阈值策略，提示"该策略的alpha可能被交易摩擦吃光"。

纯分析，不改任何回测引擎。

用法：
  # 默认扫描一组代表性策略（价值/PEG/神奇公式/多因子/EP中性/高股息/日20红利低波）
  python run_activity_tax_check.py

  # 自定义目录 + glob
  python run_activity_tax_check.py --scan data/results --glob "trades_*.csv" --max 40

  # 市场模型用平方根冲击（更真实，小票影响更大）
  MFS_SQRT_IMPACT=1 python run_activity_tax_check.py
"""
import os
import sys
import glob
import argparse

import numpy as np
import pandas as pd

_p = argparse.ArgumentParser(add_help=False)
_p.add_argument("--sqrt", action="store_true")
_args0, _ = _p.parse_known_args()
if _args0.sqrt:
    os.environ["MFS_SQRT_IMPACT"] = "1"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nav_recon_util as U

# 代表性策略样本（与用户现有 trades 产物对应；非穷举，按研究价值挑选）
DEFAULT_SAMPLE = [
    ("价值选股",        "data/results/monthly_rebalance/trades_blend50_value_20100101_20251231.csv"),
    ("红利低波质量",     "data/results/monthly_rebalance/trades_div_low_vol_20200103_20260815.csv"),
    ("PEG(年度)",       "data/results/peg/trades_n30_c1000000_annual_s3_20140101_20260715.csv"),
    ("神奇公式",        "data/results/magic_formula/trades_n30_c1000000_20230101_20260715.csv"),
    ("多因子Q(长样本)",  "data/results/multifactor/trades_n30_open_Q_m_c1000000_20100101_20260715.csv"),
    ("EP中性",          "data/results/ep_neutral/trades_nG5_open_c5000000_20200101_20260715.csv"),
    ("周度高股息量价",   "data/results/weekly_highdiv_vol/trades_n10_d25_t85_db55_s50_cost_20200103_20260715.csv"),
    # 注意：`data/results/daily20_divlow/`（raw 旧产物）的 trades 因
    # run_daily20_macd.py 导出缺 trades.append 而**不自洽**（负持仓占 60%），
    # 不能用于重建 NAV。指向 *_bugfixed_20260902：该目录由修复后的代码重跑，
    # 同时修好了两处资金分配 bug（见 run_daily20_macd._fit_budget 文档）：
    #   ① 旧代码 `capital/top_n` 等额 → 第 20 只被 `cost<=cash` 拒单（少持 1 只、5% 闲置）；
    #   ② 中间版本 `cash * w` 循环内衰减 → 每次重入只部署 64%（36% 闲置），
    #      曾把 +MACD 最大回撤从真实的 -25.03% 美化成 -19.15%。
    # 修复后部署 ≈99.9%，流水自洽，活跃税 2.60% → 3.38%/年（部署更满，成本更高）。
    ("日20红利低波(月调)", "data/results/daily20_divlow_bugfixed_20260902/trades_monthly_20150101_20251231.csv"),
]

BENCH_TAX = 0.065  # Barber&Odean 6.5%/年


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", help="扫描目录（覆盖 DEFAULT_SAMPLE）")
    ap.add_argument("--glob", default="trades_*.csv")
    ap.add_argument("--max", type=int, default=40)
    ap.add_argument("--sqrt", action="store_true", help="市价模型用平方根冲击")
    ap.add_argument("--cache", action="store_true",
                    help="缓存收盘价到 data/results/.cache/（重跑可跳过约 70s 取数；"
                         "数据库补数后需手动删该目录）")
    ap.add_argument("--out", default="data/results/activity_tax_check.csv")
    args = ap.parse_args()

    if args.scan:
        files = sorted(glob.glob(os.path.join(args.scan, "**", args.glob),
                                 recursive=True))
        # 🔴 批量扫描不得静默截断（--max 按路径排序砍尾部，可能整族漏扫）。
        #    默认 40 远小于全平台 211 个流水，单独跑极易漏掉问题最严重的一族。
        if args.max and len(files) > args.max:
            print(f"⚠️ 共 {len(files)} 个文件，--max {args.max} 只处理前 {args.max} 个，"
                  f"漏掉 {len(files) - args.max} 个（按路径排序，可能整族漏扫）"
                  f"→ 用 --max 0 表示不限")
        sel = files[: args.max] if args.max else files
        # 用相对路径作显示名：递归扫描会命中子目录里的同名文件
        # （如 daily20_divlow/n5/trades_monthly_*.csv 与主目录同名），
        # 只显示 basename 会产生无法区分的两行。
        items = [(os.path.relpath(f, args.scan), f) for f in sel]
    else:
        items = DEFAULT_SAMPLE

    # ── 跨策略共享一次批量取数 ──────────────────────────────────────
    # 各策略标的池高度重叠，且长样本策略（2010 起）基本覆盖全日期区间，
    # 故「一次取并集」远快于「每策略各查一遍」（实测 43.5s vs 200s+）。
    # 见 nav_recon_util.fetch_closes_bulk 中 +ts_code / 勿分块 两条铁律。
    loaded = []
    for name, path in items:
        if not os.path.exists(path):
            print(f"  ! 缺失 {path}")
            continue
        try:
            loaded.append((name, path, U.load_trades(path)))
        except Exception as e:
            print(f"  ! {name} 读取失败: {e}")
    if not loaded:
        print("无可用结果")
        return

    all_codes, d0, d1 = [], None, None
    for _, _, tr in loaded:
        all_codes.extend(tr["code"].astype(str).tolist())
        lo, hi = int(tr["date"].min()), int(tr["date"].max())
        d0 = lo if d0 is None else min(d0, lo)
        d1 = hi if d1 is None else max(d1, hi)
    uniq_codes = list(dict.fromkeys(all_codes))
    print(f"预取收盘价：{len(uniq_codes)} 标的 / {d0}~{d1}"
          f"（跨策略共享一次查询）")
    # 三套口径一次取齐，避免重复全表扫描：
    #   · raw：不复权收盘（detect 基线）
    #   · hfq：后复权，按首因子归一化（锚点 HFQ_ANCHOR）
    #   · qfq_engine：引擎同源口径 raw×fac_t/fac_last（weekly_highdiv_vol 用）
    closes = U.fetch_closes_bulk(uniq_codes, d0, d1, verbose=True,
                                 cache=args.cache)
    closes_hfq = U.fetch_closes_bulk(uniq_codes, U.HFQ_ANCHOR, d1,
                                     verbose=True, cache=args.cache, hfq=True)
    closes_qfq = U.fetch_closes_bulk(uniq_codes, d0, d1,
                                     verbose=True, cache=args.cache,
                                     qfq_engine=True)

    # ── 三路价格口径识别：trades 与 closes 必须同一把尺子 ──────────
    # 活跃税 = 总成本(trades 的 price) / (平均净值(closes 重建) × 年数)。
    # 旧二选一检测会把 qfq_engine 流水误判成 hfq（两者都不是 raw），
    # 导致 trades 与 closes 尺度错配、NAV 失真（实测 n10 重建 mdd −68% vs 引擎 −10%）。
    # 现在三选一：谁与 trade price 的中位比最接近 1，就选谁。
    modes = {}
    for name, _path, tr in loaded:
        modes[name] = U.pick_price_mode(tr, closes, closes_hfq, closes_qfq)
    n_raw = sum(1 for m, _, _ in modes.values() if m == "raw")
    n_hfq = sum(1 for m, _, _ in modes.values() if m == "hfq")
    n_qfq = sum(1 for m, _, _ in modes.values() if m == "qfq")
    print(f"价格口径识别：raw={n_raw}  hfq={n_hfq}  qfq_engine={n_qfq}")

    _px_map = {"raw": closes, "hfq": closes_hfq, "qfq": closes_qfq}
    rows = []
    for name, path, trades in loaded:
        init_cap = U.compute_init_cap(trades)
        mode, ratio, n_cmp = modes[name]
        px = _px_map.get(mode, closes)
        res = U.reconstruct(trades, U.slippage_frac_market,
                            init_cap=init_cap, closes=px)
        if res is None:
            print(f"  ! {name} 重建失败（无行情数据）")
            continue
        rows.append(dict(
            name=name, file=os.path.basename(path), price_mode=mode,
            px_ratio=ratio, n_cmp=n_cmp,
            n_trades=res["n_trades"], years=round(res["years"], 2),
            init_cap=res["init_cap"], total_cost=res["total_cost"],
            total_traded=res["total_traded"],
            active_tax_yr=res["active_tax_yr"],
            active_tax_nav=res["active_tax_nav"],
            round_trip_cost=res["round_trip_cost"],
            total_return=res["total_return"], annualized=res["annualized"],
            max_dd=res["max_dd"], mean_nav=res["mean_nav"],
            n_unpriced=res["n_unpriced"], n_offcal=res["n_offcal"],
            neg_hold_codes=res["neg_hold_codes"], nav_min=res["nav_min"],
        ))

    if not rows:
        print("无可用结果")
        return

    df = pd.DataFrame(rows)
    df = df.sort_values("active_tax_nav", ascending=False).reset_index(drop=True)
    # 分母退化守卫：init_cap 为 0 时 active_tax_yr 会被静默算成 0.00%，
    # 看起来像"零成本"实则毫无意义（曾因 action 标签未被识别而触发）。
    df["bad"] = (df["init_cap"] <= 0) | (df["neg_hold_codes"] > 0)
    df["flag"] = np.where(df["init_cap"] <= 0, "⚠️分母异常",
                          np.where(df["neg_hold_codes"] > 0, "⚠️流水不自洽",
                                   np.where(df["active_tax_nav"] > BENCH_TAX,
                                            "⚠️超阈值", "")))

    print(f"\n=== 活跃税 / 处置效应 体检（市价/taker 成本，B&O 基准 {BENCH_TAX*100:.1f}%/年）===")
    print(f"{'策略':<14}{'笔数':>6}{'年数':>7}{'建仓资金(万)':>13}{'平均净值(万)':>13}"
          f"{'活跃税/年(净值)':>16}{'活跃税/年(本金)':>16}{'单边摩擦':>10}{'标记':>10}")
    for _, r in df.iterrows():
        init_w = r["init_cap"] / 1e4
        mean_w = r["mean_nav"] / 1e4
        if r["bad"]:   # 分母塌缩或流水不自洽 → 数字不可信，不展示
            tax_nav = tax_cap = "n/a"
        else:
            tax_nav = f"{r['active_tax_nav']*100:.2f}%"
            tax_cap = f"{r['active_tax_yr']*100:.2f}%"
        print(f"{r['name']:<14}{int(r['n_trades']):>6}{r['years']:>7.1f}{init_w:>13.0f}"
              f"{mean_w:>13.0f}{tax_nav:>16}{tax_cap:>16}"
              f"{r['round_trip_cost']*100:>9.2f}%{r['flag']:>10}")

    ok = df[~df["bad"]]
    n_over = int((ok["active_tax_nav"] > BENCH_TAX).sum())
    print(f"\n→ {n_over}/{len(ok)} 个有效策略活跃税超过 B&O 6.5%/年基准"
          f"（频繁交易本身在吞噬alpha）；{len(df)-len(ok)} 个因流水不可信已剔除。")
    print("  注1：单边摩擦=总成本/总成交额，跨策略可比（佣金+滑点+卖出印花税）。")
    print("  注2：(净值)口径=年成本/平均净值——与 B&O「占财富比例」可比，为主判据；")
    print("       (本金)口径=年成本/初始本金——不受估值影响的保守上界。")
    print("       两者差距 = 组合增值对税负的稀释；差距越大说明策略增长越快。")

    # ── 数据质量自检 ──────────────────────────────────────────────
    # 三类问题都会让 NAV/活跃税失真，且都不报错，只安静地给出错误数字。
    probs = []
    for _, r in df.iterrows():
        msgs = []
        if r["init_cap"] <= 0:
            msgs.append("建仓资金=0（成交方向未被识别或流水缺买入）→ 活跃税分母塌缩")
        if r["neg_hold_codes"] > 0:
            msgs.append(f"{int(r['neg_hold_codes'])} 个标的持仓曾为负"
                        f"（卖出了日志中不存在的买入）→ CSV 非完整逐笔流水")
        if r["n_unpriced"] > 0:
            msgs.append(f"{int(r['n_unpriced'])} 笔成交无收盘价（按 0 估值）")
        if r["n_offcal"] > 0:
            msgs.append(f"{int(r['n_offcal'])} 笔成交日不在行情日历（已顺延）")
        if msgs:
            probs.append((r["name"], msgs))
    if probs:
        print(f"\n⚠️ 数据质量告警（{len(probs)} 个策略，其数字不可信）：")
        for name, msgs in probs:
            print(f"   {name}：")
            for m in msgs:
                print(f"      - {m}")
    else:
        print("\n✓ 数据质量自检通过：所有成交方向可识别、均有收盘价、"
              "成交日全部落在行情日历内、无负持仓。")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"明细已写出：{args.out}")


if __name__ == "__main__":
    main()
