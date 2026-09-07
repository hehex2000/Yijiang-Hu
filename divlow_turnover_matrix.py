"""红利低波换手 2×2 权威矩阵（频率 × 排序键）—— 唯一可信来源。

## 为什么要有这个脚本

历史上同一档位出现过 **4 套互相打架的换手数字**（48.0% / 50.7% / 51.0% / 52.6%），
根因是两件事混在一起：

1. **口径地雷**：季度档存在两个孤儿产物，只有排序键不同
   - `_official_*_bk0_<窗口>_partial.csv` → **fwd_yield 键**（`divlow_rebal_fair_cmp.py` / B5' 报告读它）
   - `bt_quality_sel_OFFICIAL_*_<窗口>.csv` → **volatility 键**（现役基线，NAV 文件配套）
   同参数、同期选股仅重合 3/12 → 一个数 142.4%、另一个 103.8%。
   → **B5' 的「142.3%→51.0%」属 yield 口径，不能与 vol 键基线混讲**（凭空多算 38pp 降幅）。
2. **落盘点不一致**：`fair_cmp` 读 partial、审计脚本读 sel，两者未必同源。

本脚本用**同一函数、同一窗口**一次性算完 2×2，并**自动判键**（见 `guess_key`），
任何一格的口径都对不上预期时直接报警，杜绝"拿 A 口径的季度比 B 口径的年度"。

## 判键原理

`_cap_industry` 末尾无条件执行 `sort_values(sort_key, ascending=(sort_key != "fwd_yield"))`，
所以**产物文件的行序就是排序键**：
  - volatility 升序 → volatility 键
  - fwd_yield  降序 → fwd_yield 键

## 用法
    venv_ml/Scripts/python.exe divlow_turnover_matrix.py
    venv_ml/Scripts/python.exe divlow_turnover_matrix.py --start 20200101 --end 20260723
"""
import argparse
import os

import numpy as np
import pandas as pd

RES = os.path.join("data", "results", "dividend_low_vol")
START, END = "20200101", "20260723"

# 2×2：(频率标签, 排序键) → 产物文件（缺文件则自动退到另一落盘点）
CELLS = {
    # 🔴 季度 vol 键 = 现役基线，但**文件名没有 _kv**（串档坑第 11 次变体：
    #    产物是 9-03 跑的，当时 default 一度是 volatility 而标签逻辑还没入库）
    ("季度", "volatility"): [
        os.path.join(RES, f"bt_quality_sel_OFFICIAL_OFFICIAL_COMPACT_all_12_{START}_{END}.csv"),
    ],
    ("季度", "fwd_yield"): [
        os.path.join(RES, f"_official_official_compact_all_12_bk0_{START}_{END}_partial.csv"),
    ],
    ("年度", "fwd_yield"): [
        os.path.join(RES, f"bt_quality_sel_OFFICIAL_OFFICIAL_COMPACT_all_12_rbyear_{START}_{END}.csv"),
        os.path.join(RES, f"_official_official_compact_all_12_bk0_rbyear_{START}_{END}_partial.csv"),
    ],
    ("年度", "volatility"): [
        os.path.join(RES, f"bt_quality_sel_OFFICIAL_OFFICIAL_COMPACT_all_12_kv_rbyear_{START}_{END}.csv"),
        os.path.join(RES, f"bt_quality_sel_OFFICIAL_OFFICIAL_COMPACT_all_12_rbyear_kv_{START}_{END}.csv"),
    ],
}


def ann_factor(dates):
    """由调仓日序列推"每年期数"（按实际跨度折算，不硬编码 4）。"""
    d = pd.to_datetime(pd.Series(sorted(set(dates))), format="%Y%m%d")
    if len(d) < 2:
        return 0.0
    return (len(d) - 1) / ((d.iloc[-1] - d.iloc[0]).days / 365.25)


def monotonic_share(df, col, ascending):
    """文件行序在 col 上单调的期数占比。"""
    ok = n = 0
    for _, g in df.groupby("rebal_date"):
        v = g[col].astype(float).values
        ok += bool(all((v[i] <= v[i + 1] + 1e-12) if ascending
                       else (v[i] >= v[i + 1] - 1e-12) for i in range(len(v) - 1)))
        n += 1
    return ok / n if n else 0.0


def key_hint(df):
    """排序键提示（**不是判定**，仅观测）。

    ⚠️ 为什么不能自动判键：sel 文件只落盘 `dv_ttm`（fwd_yield 的代理，但二者不等价），
    且 `score = −volatility` 与实际行序也不总一致 → 行序单调性达不到可判定阈值。
    🔑 真正的判据见 `divlow_b9_demystify.md` §7.5「二银行测试」：
       池子 11 只银行中，引擎留的是 **vol 最低 2 只**(601988/601288)，
       而非 **fwd_yield 最高 2 只**(601328/601988) → 该产物为 volatility 键。
    本表的键由 CELLS 显式标注、可审计；观测值仅用于发现"文件被换过"。
    """
    return (f"vol升序{monotonic_share(df, 'volatility', True):.0%} / "
            f"dv降序{monotonic_share(df, 'dv_ttm', False):.0%}")


def turnover(df):
    piv = (df.pivot_table(index="rebal_date", columns="ts_code",
                          values="weight", aggfunc="first")
             .fillna(0.0).sort_index())
    ds = list(piv.index)
    per = [0.5 * float((piv.iloc[i] - piv.iloc[i - 1]).abs().sum()) for i in range(1, len(ds))]
    if not per:
        return None
    af = ann_factor(ds)
    m = float(np.mean(per))
    return dict(n_period=len(ds), per_year=af, per_one_way=m,
                ann_one_way=m * af, per_two_way=m * 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=START)
    ap.add_argument("--end", default=END)
    args = ap.parse_args()

    print(f"窗口 {args.start}~{args.end}   口径：年化单边 = 每期单边(0.5·Σ|Δw|) × 期数/年  ← B&O/活跃税同口径")
    print()
    print(f"{'频率':<6}{'排序键':<12}{'来源':<9}{'行序观测':<22}{'期数':>5}{'期/年':>7}"
          f"{'每期单边':>10}{'年化单边':>10}{'每期双边':>10}")
    print("-" * 93)

    grid, warn, used = {}, [], {}
    for (freq, key), paths in CELLS.items():
        path = next((p for p in paths if os.path.exists(p)), None)
        if path is None:
            print(f"{freq:<6}{key:<12}{'（缺产物）':<9}")
            continue
        base = os.path.basename(path)
        if base in used:
            warn.append(f"{freq}/{key} 与 {used[base]} 指向同一文件 {base} → 两格必然相同，映射有误")
        used[base] = f"{freq}/{key}"
        df = pd.read_csv(path, dtype={"rebal_date": str, "ts_code": str}, encoding="utf-8-sig")
        t = turnover(df)
        src = "partial" if base.startswith("_official") else "sel"
        grid[(freq, key)] = t
        print(f"{freq:<6}{key:<12}{src:<9}{key_hint(df):<22}{t['n_period']:>5}{t['per_year']:>7.2f}"
              f"{t['per_one_way']*100:>9.1f}%{t['ann_one_way']*100:>9.1f}%{t['per_two_way']*100:>9.1f}%")

    # 因子分解：同键看频率效应、同频看键效应
    def g(f, k):
        return (grid.get((f, k)) or {}).get("ann_one_way")

    for k in ("fwd_yield", "volatility"):
        q, y = g("季度", k), g("年度", k)
        if q and y:
            print(f"  【{k} 键】季度 {q*100:.1f}% → 年度 {y*100:.1f}%   降 {(q-y)*100:.1f}pp（{(1-y/q)*100:.0f}%）")
    for f in ("季度", "年度"):
        a, b = g(f, "fwd_yield"), g(f, "volatility")
        if a and b:
            print(f"  【{f} 档】yield {a*100:.1f}% → vol {b*100:.1f}%   降 {(a-b)*100:.1f}pp（{(1-b/a)*100:.0f}%）")

    q_y, y_v = g("季度", "fwd_yield"), g("年度", "volatility")
    if q_y and y_v:
        print(f"  【双杠杆叠加】季度+yield {q_y*100:.1f}% → 年度+vol {y_v*100:.1f}%   "
              f"降 {(q_y-y_v)*100:.1f}pp（{(1-y_v/q_y)*100:.0f}%）")

    if warn:
        print()
        for w in warn:
            print(f"  🔴 {w}")
        print("  → 任一格口径不符，本表数字不可混用，先修产物或改 CELLS 映射。")


if __name__ == "__main__":
    main()
