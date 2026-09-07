"""B9 候选池实验室：池子已存盘 → 所有"换口径"实验离线秒算，不再重跑选股。

为什么单开这个脚本：
  选股（全A 波动率 + 分红档案）占 95% 耗时，一趟 2020-2026 季度档要 40~50min。
  而**候选池（行业 cap 前的 buffer_n 只）与最后的排序键、行业约束、持仓数完全无关** ——
  把这些池子存盘（divlow_b8_key_smoke.py 的 pools/ 目录）之后，
  换任何下游口径都只是对同一批池子重新截取，秒级出结果。

本脚本回答的问题（2026-09-03 B8 被证伪后提出）：
  B8 假设「最后一段排序键从股息率换成交波动率（官方口径）能压换手」——**已被 2023 短窗口证伪**
  （两键留存率完全相同 58.3%）。真正的嫌疑转向**行业约束的形式**：
    官方 930955 = **行业权重 ≤20%**（软约束：可以调权重满足，不必换股）
    我们        = **每行业最多 2 只**（硬约束：48 只池里银行扎堆，cap 后只剩 27~31 只，
                   砍掉谁每期都在变 → 强制换手）
  → 用同一批池子做对照，看看到底是"排序键"还是"行业约束"在制造换手。

用法：
  venv_ml/Scripts/python.exe divlow_b9_pool_lab.py
  venv_ml/Scripts/python.exe divlow_b9_pool_lab.py 20230101 20231231   # 指定窗口（目录后缀）
"""
import os
import sys

import numpy as np
import pandas as pd

import run_dividend_low_vol_quality_bt as E

POOL_DIR = os.path.join(E.RES_DIR, "_b8_keydiag", "pools")
IND_CAP_BASE = 2          # 现状：每行业最多 2 只
W_CAP = 0.20              # 官方 930955：中证二级行业权重上限


# ─────────────────────────────────────────────────────────────────────
#  行业映射
# ─────────────────────────────────────────────────────────────────────
def load_ind_map():
    conn = E.get_conn()
    im = pd.read_sql_query("SELECT ts_code, industry FROM stock_basic", conn)
    conn.close()
    return {str(r["ts_code"]): (str(r["industry"]) if pd.notna(r["industry"]) else "其他")
            for _, r in im.iterrows()}


# ─────────────────────────────────────────────────────────────────────
#  各种下游口径
# ─────────────────────────────────────────────────────────────────────
def pick_by_rank(df, top_n, ind_cap, ind_map):
    """按 df 当前顺序（调用方排好）贪心取 top_n，每行业最多 ind_cap 只。ind_cap<=0 = 不限。"""
    cnt, out = {}, []
    for c in df["ts_code"].astype(str):
        if len(out) >= top_n:
            break
        if ind_cap and ind_cap > 0:
            k = ind_map.get(c, "其他")
            if cnt.get(k, 0) >= ind_cap:
                continue
            cnt[k] = cnt.get(k, 0) + 1
        out.append(c)
    return out


def pick_by_wcap(df, top_n, w_cap, ind_map, wcol="_w"):
    """官方口径：先按序取 top_n，再用**权重缩放**把行业权重压到 ≤ w_cap（不换股）。

    与"每行业最多 N 只"的本质区别：超限时**改权重**而不是**踢股票** → 不制造强制换手。
    实现：迭代缩放超限行业的权重因子并重新归一化（对应官方"权重因子介于 0 和 1 之间"）。
    """
    picks = [str(c) for c in df["ts_code"].astype(str).head(top_n)]
    if w_cap <= 0:
        return picks
    inds = [ind_map.get(c, "其他") for c in picks]
    raw = df.set_index(df["ts_code"].astype(str)).loc[picks, wcol].astype(float).values
    raw = np.where(np.isfinite(raw) & (raw > 0), raw, 1e-6)
    fac = np.ones(len(picks))
    for _ in range(50):
        w = raw * fac
        w = w / w.sum()
        by_ind = {}
        for i, k in enumerate(inds):
            by_ind[k] = by_ind.get(k, 0.0) + w[i]
        over = {k: v for k, v in by_ind.items() if v > w_cap + 1e-9}
        if not over:
            break
        for k, v in over.items():
            fac[[i for i, kk in enumerate(inds) if kk == k]] *= (w_cap / v)
    return picks


def weights_of(df, picks, mode="dividend"):
    """与引擎同口径的权重：dividend = 按 fwd_yield 归一化（现有基准）。"""
    n = len(picks)
    if n == 0:
        return {}
    if mode not in ("dividend", "dy_vol"):
        return {c: 1.0 / n for c in picks}
    sub = df.drop_duplicates("ts_code").set_index("ts_code").loc[picks]
    y = sub["fwd_yield"].fillna(sub.get("dv_ttm", 0) / 100.0).fillna(0.0).clip(lower=0).astype(float)
    if mode == "dy_vol":                      # 官方 930955 加权：股息率 ÷ 波动率
        v = sub["volatility"].astype(float).replace(0, np.nan).fillna(sub["volatility"].median())
        y = (y / v).clip(lower=0)
    s = float(y.sum())
    if s <= 0:
        return {c: 1.0 / n for c in picks}
    return {c: float(v) / s for c, v in zip(picks, y.values)}


def turnover(seq_w, per_year=4):
    """per_one_way = 0.5·Σ|w_t − w_{t−1}| → 年化单边（×期数/年）。"""
    per = []
    for i in range(1, len(seq_w)):
        a, b = seq_w[i - 1], seq_w[i]
        keys = set(a) | set(b)
        per.append(0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys))
    if not per:
        return 0.0, 0.0, []
    m = float(np.mean(per))
    return m, m * per_year, per


def verify_repro(pools, ind_map, sel_path, top_n=12, ind_cap=2):
    """🔴 复现率自检：池子选出的持仓 vs **引擎真实持仓**（sel CSV）逐期对拍。

    2026-09-04 事故：本脚本以 **4.41/12** 的复现率输出了「取消行业上限 −34.3pp」的结论，
    全盘作废（引擎真实口径是 volatility 键，而脚本默认按 fwd_yield 建模"现状"）。
    → **任何离线实验室都必须先自证"我选的股票就是引擎选的股票"，才准引用它的数字。**

    返回 {排序键: (平均交集, 完全一致期数, 总期数)}；调用方应只信任复现率最高的键。
    """
    if not sel_path or not os.path.exists(sel_path):
        print("⚠️ 未提供引擎 sel 文件 → **跳过复现率自检，以下数字可信度未知**。"
              "建议加 --verify-sel <bt_quality_sel_*.csv>")
        return {}
    sel = pd.read_csv(sel_path, dtype={"rebal_date": str, "ts_code": str}, encoding="utf-8-sig")
    eng = {str(rb): [str(c) for c in g["ts_code"]]
           for rb, g in sel.groupby(sel["rebal_date"].astype(str))}
    out = {}
    for key in ("fwd_yield", "volatility"):
        tot = []
        for p in pools:
            rb = str(p["rebal_date"].iloc[0])
            if rb not in eng:
                continue
            d = p.sort_values(key, ascending=(key != "fwd_yield")).reset_index(drop=True)
            pk = pick_by_rank(d, top_n, ind_cap, ind_map)
            tot.append(len(set(pk) & set(eng[rb])))
        if tot:
            out[key] = (float(np.mean(tot)), sum(1 for x in tot if x == top_n), len(tot))
    return out


# ─────────────────────────────────────────────────────────────────────
def main():
    import argparse
    ap = argparse.ArgumentParser(description="B9 候选池实验室（离线秒算换手对比）")
    ap.add_argument("window", nargs="*", help="起始/结束日期，用于选 pool 目录后缀")
    ap.add_argument("--verify-sel", default=None,
                    help="引擎 bt_quality_sel_*.csv，用于**复现率自检**（强烈建议提供）")
    ap.add_argument("--min-repro", type=float, default=11.0,
                    help="复现率阈值（平均交集/持仓数），低于则**拒绝输出结论**（默认 11/12）")
    args = ap.parse_args()

    global POOL_DIR
    if args.window:
        POOL_DIR = POOL_DIR + "_" + "_".join(args.window)
    if not os.path.isdir(POOL_DIR):
        print(f"[err] 没有候选池目录 {POOL_DIR}，先跑 divlow_b8_key_smoke.py")
        return
    ind_map = load_ind_map()
    files = sorted(f for f in os.listdir(POOL_DIR) if f.startswith("pool_") and f.endswith(".csv"))
    if not files:
        print("[err] 池子目录为空")
        return
    pools, dates = [], []
    for f in files:
        d = pd.read_csv(os.path.join(POOL_DIR, f), dtype={"ts_code": str, "rebal_date": str})
        d["ts_code"] = d["ts_code"].astype(str)
        d["_w"] = d["fwd_yield"].fillna(0).clip(lower=0)
        pools.append(d)
        dates.append(str(d["rebal_date"].iloc[0]))
    n_pool = len(pools[0])
    print(f"候选池 {len(pools)} 期 × {n_pool} 只   {dates[0]} ~ {dates[-1]}")

    # 池子自身留存
    ov = [len(set(pools[i]["ts_code"]) & set(pools[i - 1]["ts_code"])) for i in range(1, len(pools))]
    print(f"候选池自身逐期留存：{ov}  平均 {np.mean(ov)/n_pool:.1%}")

    def srt(df, key):
        return df.sort_values(key, ascending=(key != "fwd_yield")).reset_index(drop=True)

    # ── 🔴 复现率自检：先证明"我选的股票就是引擎选的股票"，否则拒绝出结论 ──
    repro = verify_repro(pools, ind_map, args.verify_sel, top_n=12, ind_cap=2)
    best_key = None
    if repro:
        print("\n【复现率自检】池子选出的持仓 vs 引擎真实持仓（同池同法，只换排序键）")
        for k, (m, exact, n) in repro.items():
            print(f"  {k:<12} 平均交集 {m:>5.2f}/12   完全一致 {exact}/{n} 期")
        best_key = max(repro, key=lambda k: repro[k][0])
        print(f"  → 引擎实际生效的排序键 = **{best_key}**（复现率最高）")
        if repro[best_key][0] < args.min_repro:
            print(f"\n🔴 复现率 {repro[best_key][0]:.2f}/12 < 阈值 {args.min_repro} → "
                  f"**拒绝输出换手结论**：池子与引擎真实输入存在未定位偏差。\n"
                  f"   （2026-09-04 事故：本脚本曾以 4.41/12 的复现率输出「−34.3pp」错误结论）")
            return
        print(f"  ✅ 复现率达标（≥{args.min_repro}），继续。")

    cases = [
        ("A 现状：yield键 + 每行业≤2", dict(key="fwd_yield", mode="cnt", ind_cap=2, top_n=12)),
        ("B B8：vol键 + 每行业≤2", dict(key="volatility", mode="cnt", ind_cap=2, top_n=12)),
        ("C yield键 + 不限行业", dict(key="fwd_yield", mode="cnt", ind_cap=0, top_n=12)),
        ("D vol键 + 不限行业", dict(key="volatility", mode="cnt", ind_cap=0, top_n=12)),
        ("E vol键 + 行业权重≤20%", dict(key="volatility", mode="wcap", ind_cap=0, top_n=12)),
        ("F yield键 + 行业权重≤20%", dict(key="fwd_yield", mode="wcap", ind_cap=0, top_n=12)),
        ("G vol键 + 每行业≤5 + 30只", dict(key="volatility", mode="cnt", ind_cap=5, top_n=30)),
        ("H vol键 + 权重≤20% + 30只", dict(key="volatility", mode="wcap", ind_cap=0, top_n=30)),
        ("I vol键 + 权重≤20% + dy/vol加权", dict(key="volatility", mode="wcap", ind_cap=0,
                                              top_n=12, wmode="dy_vol")),
    ]

    # 🔴 现状口径必须按"引擎实际生效的键"建模，否则"vs 现状"的差值全是假的
    if best_key and cases[0][1]["key"] != best_key:
        print(f"⚠️ case A「现状」的排序键 {cases[0][1]['key']} → 自动改为 {best_key}"
              f"（否则'现状'不是引擎的现状，差值无意义）")
        cases[0][1]["key"] = best_key

    print(f"\n{'口径':<26}{'持仓':>4}{'每期新进均值':>12}{'留存率':>9}"
          f"{'每期单边':>10}{'年化单边':>10}{'每期双边':>10}")
    print("-" * 82)
    rows = []
    for name, cfg in cases:
        seq_w, seq_p = [], []
        for p in pools:
            d = srt(p, cfg["key"])
            if cfg["mode"] == "cnt":
                pk = pick_by_rank(d, cfg["top_n"], cfg["ind_cap"], ind_map)
            else:
                pk = pick_by_wcap(d, cfg["top_n"], W_CAP, ind_map)
            seq_p.append(pk)
            seq_w.append(weights_of(p, pk, cfg.get("wmode", "dividend")))
        newn = [len(set(seq_p[i]) - set(seq_p[i - 1])) for i in range(1, len(seq_p))]
        per, ann, _ = turnover(seq_w)
        rows.append((name, cfg["top_n"], float(np.mean(newn)), 1 - np.mean(newn) / cfg["top_n"],
                     per, ann, per * 2))
        print(f"{name:<26}{cfg['top_n']:>4}{np.mean(newn):>12.2f}"
              f"{1 - np.mean(newn)/cfg['top_n']:>9.1%}{per:>10.1%}{ann:>10.1%}{per*2:>10.1%}")
    print("-" * 82)
    base = rows[0]
    print(f"\n判读（vs A 现状 年化单边 {base[5]:.1%}）：")
    for r in rows[1:]:
        d = r[5] - base[5]
        flag = "✅降" if d < -0.02 else ("⚠️升" if d > 0.02 else "＝持平")
        print(f"  {r[0]:<26} 年化单边 {r[5]:>7.1%}  {d:+7.1%}pp  {flag}")
    print("\n🔴 换手只看权重口径（年化单边/每期双边）；收益需另跑 NAV 重放，本脚本不算收益。")


if __name__ == "__main__":
    main()
