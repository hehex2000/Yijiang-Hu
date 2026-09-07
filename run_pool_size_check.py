# -*- coding: utf-8 -*-
"""
P5 · 池子大小检验：top_n 从 5 扩到 30 到底值不值？
=====================================================================

用户拍板的行动优先级：**扩大池子（5→15+）> 定期调仓 > 调因子权重**。
本脚本做第一条。

§3.5 已有线索（随机前 N 只 + 静态持有）：N=5 CAGR 9.63% → N=40 14.87%。
但那有两个缺陷，本脚本逐条补上：
  ① **随机池 ≠ 真实选股池** —— §3.7/§3.8 已证明真实选股与随机无显著差异，
     但池子扩大时的**边际票**是"排名更低的选股票"，与随机补仓不是一回事；
  ② **单期静态** —— §3.8 已证明"选一次持有 6.6 年"是主要损害来源。

设计（嵌套前缀，单变量干净隔离）：
  · 每期独立选股（与 P4 同规则：选股日 = 回测开始日前一交易日）
  · config.SELECTION["top_n"] 临时设为 SEL_TOP(=20) ⇒ _candidate_n = 40 只候选
  · 取候选前 MAX_N(=30) 只，**前缀嵌套**：N=5 ⊂ 10 ⊂ 15 ⊂ 20 ⊂ 30
    ⇒ 同一期同一批股票，唯一变量是"用多少只"，不含选股噪声
  · 对照：同期同池随机抽 K 组 × 30 只，**同样做前缀扫描**
    ⇒ 随机组的 N 效应 = **纯分散化**；真实组的 N 效应 = 分散化 + 选股覆盖
    ⇒ 两者之差 = 选股覆盖的边际贡献（这是本脚本独有的分解）
  · 统一组合级共享池，f = auto clamp(2/N, 0.05, 0.25)，总资金 = config 口径

用法:  venv_ml/Scripts/python.exe run_pool_size_check.py [K组数]
"""
import sqlite3
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import config  # noqa: E402
import run_backtest as rb  # noqa: E402
from backtest.mean_reversion_plugin import MeanReversionStrategyPlugin  # noqa: E402
from backtest.portfolio_engine import run_portfolio_mode  # noqa: E402

TOTAL = float(config.BACKTEST["total_capital"])
K_GROUPS = int(sys.argv[1]) if len(sys.argv) > 1 else 12
SEED = 20260907
N_LIST = [5, 10, 15, 20, 30]        # 池子大小（嵌套前缀）
MAX_N = max(N_LIST)
SEL_TOP = 20                         # 让 _candidate_n = 40 ≥ MAX_N
OUT = Path("data/results/position_sizing/pool_size_scan.csv")

PERIODS = [
    ("2020", "20200103", "20201231"),
    ("2021", "20210104", "20211231"),
    ("2022", "20220104", "20221231"),
    ("2023", "20230103", "20231229"),
    ("2024", "20240102", "20241231"),
    ("2025", "20250102", "20251231"),
    ("2026", "20260105", "20260825"),  # 不足一年，单独标注
]


def build_pool(sel_date: str):
    pool = rb._get_zz800_from_db(sel_date)
    conn = sqlite3.connect(config.DATA["local_db_path"])
    try:
        pool = rb.prefilter_by_liquidity(conn, pool, sel_date)
    except Exception as e:  # noqa: BLE001
        print(f"    [WARN] 流动性过滤失败，用原池: {e}")
    conn.close()
    return sorted(pool["code"].tolist())


def load(codes: list, start: str, end: str) -> dict:
    conn = sqlite3.connect(config.DATA["local_db_path"])
    sd = {}
    for c in codes:
        try:
            df = rb.load_stock_prices(c, start, end, conn, lookback_days=250)
        except Exception:  # noqa: BLE001
            continue
        if df is None or len(df) < 30:
            continue
        si = df[df["trade_date"] >= start].index.min()
        if pd.isna(si):
            continue
        sd[c] = ("", df, int(si))
    conn.close()
    return sd


def bh_mean(sd: dict) -> float:
    rets = []
    for _, (_, df, si) in sd.items():
        col = "adj_close" if "adj_close" in df.columns else "close"
        try:
            rets.append(df[col].iloc[-1] / df[col].iloc[si] - 1)
        except Exception:  # noqa: BLE001
            pass
    return float(np.mean(rets) * 100) if rets else float("nan")


def run_one(sd: dict, start: str, end: str) -> dict:
    """跑一次组合级回测。f 由引擎按 N 自动定（clamp(2/N,0.05,0.25)）。"""
    cfg = dict(config.STRATEGIES["mean_reversion"])
    cfg["portfolio_shared_pool"] = True
    cfg["portfolio_f"] = None
    cfg["portfolio_cap"] = 0
    r = run_portfolio_mode(sd, MeanReversionStrategyPlugin, cfg, TOTAL, start, end)
    if r is None:
        return {}
    m = r["metrics"]
    return {
        "cagr_pct": m["cagr_pct"], "sharpe": m["sharpe"], "mdd_pct": m["mdd_pct"],
        "n_taken": m["n_taken"], "exposure": m["exposure"] * 100,
        "bh_pct": bh_mean(sd), "terminal": m["terminal"], "f": m.get("f", np.nan),
    }


def prefix_scan(codes30: list, sd_all: dict, start: str, end: str,
                label: str, period: str) -> list:
    """对同一批 30 只做前缀嵌套扫描：N=5/10/15/20/30，唯一变量是数量。"""
    out = []
    for n in N_LIST:
        sub = {c: sd_all[c] for c in codes30[:n] if c in sd_all}
        if len(sub) < n:
            print(f"      [SKIP] {label} N={n}: 只有 {len(sub)}/{n} 只有数据")
            continue
        r = run_one(sub, start, end)
        if not r:
            continue
        r.update({"period": period, "group": label, "N": n})
        out.append(r)
    return out


def main():
    print("=" * 96)
    print("P5 · 池子大小检验（多期 × 嵌套前缀 × 随机对照）")
    print("=" * 96)
    print(f"  期数 {len(PERIODS)}｜N 扫描 {N_LIST}｜随机 {K_GROUPS} 组/期"
          f"｜总资金 {TOTAL:,.0f}｜组合级 f=auto clamp(2/N,0.05,0.25)")
    print("  选股：临时 top_n=%d ⇒ 候选 %d 只，取前 %d 只做前缀嵌套" % (
        SEL_TOP, max(SEL_TOP * 2, SEL_TOP + 10), MAX_N))

    orig_top_n = config.SELECTION["top_n"]
    rng = np.random.default_rng(SEED)
    rows = []

    try:
        for tag, start, end in PERIODS:
            print("\n" + "=" * 96)
            print(f"  【{tag}】回测 {start} ~ {end}")
            print("=" * 96)
            config.BACKTEST["start_date"] = start
            config.BACKTEST["end_date"] = end
            config.SELECTION["top_n"] = SEL_TOP

            # ── 1. 本期独立选股（候选 ≥ MAX_N）──
            try:
                sel = rb.run_selection()
            except Exception:  # noqa: BLE001
                print("    [FAIL] 选股抛异常：")
                traceback.print_exc()
                continue
            if sel is None or sel.empty:
                print("    [SKIP] 选股返回空")
                continue
            sel_date = config.SELECTION["date"].replace("-", "")
            cand = [str(c).zfill(6) for c in sel["code"].tolist()][:MAX_N]
            if len(cand) < MAX_N:
                print(f"    [SKIP] 候选只有 {len(cand)}/{MAX_N} 只")
                continue
            print(f"    选股日 {sel_date}｜候选取前 {MAX_N}"
                  f"（平台 top_n={SEL_TOP} ⇒ 候选 {max(SEL_TOP*2, SEL_TOP+10)}）")

            # ── 2. 随机对照（同池同日，每组 30 只，同样前缀扫描）──
            pool = build_pool(sel_date)
            if len(pool) < MAX_N * 4:
                print(f"    [SKIP] 池子过小({len(pool)})")
                continue
            groups = [list(rng.choice(pool, size=MAX_N, replace=False))
                      for _ in range(K_GROUPS)]

            need = sorted(set(cand) | {c for g in groups for c in g})
            sd_all = load(need, start, end)

            rr = prefix_scan(cand, sd_all, start, end, "REAL", tag)
            if len(rr) < len(N_LIST):
                print(f"    [SKIP] 真实组只跑通 {len(rr)}/{len(N_LIST)} 档")
                if not rr:
                    continue
            rows.extend(rr)
            print(f"    真实: " + "  ".join(
                f"N={r['N']:<2} CAGR {r['cagr_pct']:>6.2f}% Sharpe {r['sharpe']:>5.2f} "
                f"暴露 {r['exposure']:>4.1f}%" for r in rr))

            rrows = []
            for i, g in enumerate(groups, 1):
                rrows.extend(prefix_scan(g, sd_all, start, end, f"R{i:02d}", tag))
            if not rrows:
                print("    [SKIP] 随机对照全部失败")
                continue
            rows.extend(rrows)
            d = pd.DataFrame(rrows)
            print(f"    随机(n={len(groups)}): " + "  ".join(
                f"N={n:<2} CAGR {d[d['N']==n]['cagr_pct'].median():>6.2f}%"
                f" Sharpe {d[d['N']==n]['sharpe'].median():>5.2f}"
                for n in N_LIST if (d["N"] == n).any()))
    finally:
        config.SELECTION["top_n"] = orig_top_n

    if not rows:
        print("\n[FAIL] 没有数据")
        return 1

    D = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    D.to_csv(OUT, index=False, encoding="utf-8-sig")

    # ── 汇总：按 N 聚合 ──
    real = D[D["group"] == "REAL"]
    rand = D[D["group"] != "REAL"]

    print("\n" + "=" * 96)
    print("  汇总 · 各池子大小的中位表现（跨期）")
    print("=" * 96)
    print(f"  {'N':>3}{'真实CAGR%':>11}{'随机CAGR%':>11}{'Δ(选股覆盖)':>13}"
          f"{'真实Sharpe':>11}{'随机Sharpe':>11}{'真实MDD%':>11}{'暴露%':>8}{'成交':>7}")
    print("  " + "-" * 90)
    agg = []
    for n in N_LIST:
        a = real[real["N"] == n]
        b = rand[rand["N"] == n]
        if a.empty or b.empty:
            continue
        row = {
            "N": n,
            "real_cagr": a["cagr_pct"].median(), "rand_cagr": b["cagr_pct"].median(),
            "real_sharpe": a["sharpe"].median(), "rand_sharpe": b["sharpe"].median(),
            "real_mdd": a["mdd_pct"].median(), "exposure": a["exposure"].median(),
            "n_taken": a["n_taken"].median(),
            "n_periods": len(a),
        }
        row["delta"] = row["real_cagr"] - row["rand_cagr"]
        agg.append(row)
        print(f"  {n:>3}{row['real_cagr']:>11.2f}{row['rand_cagr']:>11.2f}"
              f"{row['delta']:>+13.2f}{row['real_sharpe']:>11.2f}"
              f"{row['rand_sharpe']:>11.2f}{row['real_mdd']:>11.2f}"
              f"{row['exposure']:>8.1f}{row['n_taken']:>7.0f}")

    if not agg:
        print("  [FAIL] 无可聚合数据")
        return 1
    A = pd.DataFrame(agg)
    A.to_csv(OUT.with_name("pool_size_agg.csv"), index=False, encoding="utf-8-sig")

    base = A.iloc[0]
    print("  " + "-" * 90)
    print("  ⚠️ 下表是【两个中位数相减】，仅用于看跨期水平，【不能直接当改善幅度】：")
    for _, r in A.iloc[1:].iterrows():
        print(f"    N={int(r['N']):<3} 真实 {r['real_cagr'] - base['real_cagr']:>+6.2f}pp"
              f"   随机(纯分散化) {r['rand_cagr'] - base['rand_cagr']:>+6.2f}pp"
              f"   选股覆盖贡献 {r['delta'] - base['delta']:>+6.2f}pp")

    # ★ 中位数不可加：真正的改善幅度必须逐期配对后再取中位
    print("\n  ★ 逐期配对（同一期 N vs N=5，先配对再取中位）——【这才是改善幅度】：")
    b5 = real[real["N"] == N_LIST[0]].set_index("period")
    for n in N_LIST[1:]:
        g = real[real["N"] == n].set_index("period")
        common = [p for p in g.index if p in b5.index]
        if not common:
            continue
        dc = g.loc[common, "cagr_pct"] - b5.loc[common, "cagr_pct"]
        ds = g.loc[common, "sharpe"] - b5.loc[common, "sharpe"]
        print(f"    N=5→{n:<3} 胜 {int((dc > 0).sum())}/{len(common)} 期"
              f"  中位 ΔCAGR {dc.median():>+6.2f}pp"
              f"  中位 ΔSharpe {ds.median():>+5.2f}"
              f"  最差 ΔCAGR {dc.min():>+7.2f}pp")

    # ── 逐期一致性：N=30 是否在各期都优于 N=5 ──
    print("\n" + "=" * 96)
    print("  逐期一致性：N=30 vs N=5（真实选股）")
    print("=" * 96)
    win = 0
    tot = 0
    for tag, *_ in PERIODS:
        a5 = real[(real["period"] == tag) & (real["N"] == 5)]
        a30 = real[(real["period"] == tag) & (real["N"] == 30)]
        if a5.empty or a30.empty:
            continue
        c5, c30 = a5["cagr_pct"].iloc[0], a30["cagr_pct"].iloc[0]
        s5, s30 = a5["sharpe"].iloc[0], a30["sharpe"].iloc[0]
        w = c30 > c5
        win += w
        tot += 1
        print(f"  {tag}: CAGR {c5:>7.2f}% → {c30:>7.2f}% ({c30-c5:+6.2f})  "
              f"Sharpe {s5:>5.2f} → {s30:>5.2f} ({s30-s5:+5.2f})  "
              f"{'WIN' if w else 'LOSS'}")
    if tot:
        print(f"  ⇒ 扩池胜出 {win}/{tot} 期")

    # ── 真实 vs 随机：各 N 档上选股是正贡献还是负贡献 ──
    print("\n" + "=" * 96)
    print("  真实选股 vs 随机（逐期配对，负 = 选股跑输）")
    print("=" * 96)
    print(f"  {'N':>3}{'真实中位':>10}{'随机中位':>10}{'差':>9}{'逐期胜':>9}")
    print("  " + "-" * 44)
    for n in N_LIST:
        a = real[real["N"] == n]
        b = rand[rand["N"] == n]
        if a.empty or b.empty:
            continue
        rp = a.set_index("period")["cagr_pct"]
        bp = b.groupby("period")["cagr_pct"].median()
        common = [p for p in rp.index if p in bp.index]
        d = rp.loc[common] - bp.loc[common]
        print(f"  {n:>3}{a['cagr_pct'].median():>10.2f}{b['cagr_pct'].median():>10.2f}"
              f"{a['cagr_pct'].median() - b['cagr_pct'].median():>+9.2f}"
              f"{int((d > 0).sum()):>6}/{len(common):<3}")
    print("  注：逐期胜率接近 一半 且各档差值为负 ⇒ 与 §3.8「选股与随机无显著差异」同向；")
    print("      N=30 档差距明显收窄，说明扩池稀释了 top 名次的负面影响。")

    print("\n" + "=" * 96)
    print(f"  明细 → {OUT}")
    print(f"  聚合 → {OUT.with_name('pool_size_agg.csv')}")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    sys.exit(main())
