# -*- coding: utf-8 -*-
"""
全市场换手率择时信号检验（时序型，Gate 1 / Gate 4 / Gate 5）

要回答的问题
------------
「全市场换手率处于高位时，未来市场（中证800）收益是更高还是更低？」

方向约定
--------
  换手率高 = 市场过热 = 未来该跌 → **IC < 0**（与破净率相反：破净率高=便宜=该涨）
  换手率低 = 地量见地价 = 未来该涨 → 分位 Q0 应显著优于 Q4

为什么用市场（中证800）而不是某个策略作标的
------------------------------------------
与破净率同款设计：overlay 的第一性依据是市场层面冷热。策略自身的
真实 A/B 留到 net_break_screen.py / 引擎双账验证（那才是最终判据）。

统计口径（样本重叠三口径并列）
----------------------------
  t_naive  明知虚高，仅对照
  t_NW(H)  Newey-West，lag=H  ← 主口径
  t_NWauto 自动带宽
  t_nonovl 每 H 日抽一个  ← 最保守

⚠️ 复用 broker_pe_test 的工具，勿重写：NW 方差 V=(X'X)⁻¹S(X'X)⁻¹
   **不能再乘 n**（S 已对全部 t 求和）。

因子清单
--------
  turn_flow   流通市值加权（主口径）
  turn_mean   简单平均（视频 5-6% 所在口径）
  turn_free   自由流通加权
  各自 750 日滚动分位：pct_flow / pct_mean / pct_free

Gate 5 冗余对照
--------------
  1. 已实现波动率 rv20（20 日滚动 std × √250）—— 换手率与波动率天然高度相关，
     这是最可能的"马甲"来源
  2. 破净率分位 pct_nobj（上一轮 T1-① 的信号，同出自 daily_basic）

用法
----
  python turnover_test.py
"""
import os
import sys
import argparse
import warnings

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import bench_index  # noqa: E402
import broker_pe_factor as bpf  # noqa: E402
from broker_pe_test import nw_tstat, nw_tstat_auto, naive_tstat, spearman_ic  # noqa: E402

TURNOVER_CSV = os.path.join("data", "results", "turnover", "market_turnover.csv")
NB_CSV = os.path.join("data", "results", "net_break", "market_net_break.csv")
OUT = os.path.join("data", "results", "turnover", "turnover_test.csv")
OUT_YEARLY = os.path.join("data", "results", "turnover", "turnover_yearly.csv")

BENCH = "000906.SH"   # 中证800（🔴 000905 才是中证500）
HORIZONS = (60, 120, 250)
START = "20100101"
END = "20260831"
RV_WIN = 20

FACTORS = ["turn_flow", "turn_mean", "turn_free"]
PCTS = ["pct_flow", "pct_mean", "pct_free"]


def load_bench():
    con = bpf.get_conn()
    bdf, meta = bench_index.load_benchmark(BENCH, START, END, conn=con,
                                           nav_price_mode="hfq")
    con.close()
    print(f"[基准] {BENCH} mode={meta.get('mode')} "
          f"source={meta.get('source_table')} resolved={meta.get('resolved_code')}")
    b = bdf[["trade_date", "close"]].rename(columns={"close": "bench_close"})
    b["trade_date"] = b["trade_date"].astype(str)
    return b.sort_values("trade_date").reset_index(drop=True)


def build():
    tv = pd.read_csv(TURNOVER_CSV, dtype={"trade_date": str})
    b = load_bench()
    df = tv.merge(b, on="trade_date", how="inner").sort_values("trade_date").reset_index(drop=True)
    print(f"[合并] 换手率 {len(tv)} × 基准 {len(b)} → {len(df)} 行")

    # 未来 H 日市场收益（T+1 进场，避免当日收盘可知的偷价）
    nav = df["bench_close"].values
    for H in HORIZONS:
        f = np.full(len(nav), np.nan)
        f[: len(nav) - H - 1] = nav[H + 1:] / nav[1: len(nav) - H] - 1
        df[f"fwd{H}"] = f

    # 已实现波动率（Gate 5 对照信号 1）
    r = pd.Series(nav).pct_change()
    df["rv20"] = r.rolling(RV_WIN).std() * np.sqrt(250) * 100.0   # 年化，%
    df["pct_rv20"] = (df["rv20"].rolling(750, min_periods=250)
                      .apply(lambda w: (w[-1] >= w[:-1]).mean() * 100.0, raw=True))

    # 破净率分位（Gate 5 对照信号 2）
    if os.path.exists(NB_CSV):
        nb = pd.read_csv(NB_CSV, dtype={"trade_date": str})
        df = df.merge(nb[["trade_date", "pct_nobj", "rate_nobj"]],
                      on="trade_date", how="left")
        print(f"[对照] 已并入破净率分位 pct_nobj（{nb['pct_nobj'].notna().sum()} 天有效）")
    return df


def run_horizon(df, factor, H):
    d = df[["trade_date", factor, f"fwd{H}"]].dropna()
    if len(d) < 60:
        return None
    x = d[factor].values.astype(float)
    y = d[f"fwd{H}"].values.astype(float)

    ic = spearman_ic(x, y)                                   # 只回相关系数，不回 p 值
    p_ic = stats.spearmanr(x, y).pvalue
    _, t_naive = naive_tstat(y, x)
    b_nw, t_nw = nw_tstat(y, x, lag=H)
    _, t_nw_auto = nw_tstat_auto(y, x)

    idx = np.arange(0, len(d), H)
    dsub = d.iloc[idx]
    ic_sub = spearman_ic(dsub[factor].values, dsub[f"fwd{H}"].values, min_n=8)
    _, t_sub = naive_tstat(dsub[f"fwd{H}"].values, dsub[factor].values)

    return dict(factor=factor, H=H, n_obs=len(d), n_eff=len(dsub),
                IC=ic, p_ic=p_ic, t_naive=t_naive, beta_nw=b_nw,
                t_NW=t_nw, t_NWauto=t_nw_auto, IC_sub=ic_sub, t_非重叠=t_sub)


def run_quantile(df, factor, H, q=5):
    """分层：Q0 = 换手率最低（地量）… Q4 = 最高（过热）。"""
    d = df[[factor, f"fwd{H}"]].dropna()
    if len(d) < 200:
        return None
    d = d.copy()
    d["q"] = pd.qcut(d[factor], q, labels=False, duplicates="drop")
    g = d.groupby("q")[f"fwd{H}"].agg(["mean", "median", "count", "std"])
    g["胜率%"] = d.groupby("q")[f"fwd{H}"].apply(lambda s: (s > 0).mean() * 100)
    g["档位含义"] = ["Q0 地量", "Q1", "Q2", "Q3", "Q4 过热"][: len(g)]
    return g


def yearly(df, factor, H):
    """逐年利差：Q4(过热) − Q0(地量)。

    🔴 阈值必须用**全样本**分位，不能用逐年 qcut：
       逐年 qcut 会把 fwd250 在年末必然为 NaN 的样本也算进档位，
       导致 15 年里只有 3 年算得出利差（net_break 踩过的坑）。
    """
    d = df[["trade_date", factor, f"fwd{H}"]].dropna()
    if len(d) < 200:
        return None
    qs = np.nanpercentile(d[factor], [20, 80])
    lo_m = d[factor] <= qs[0]
    hi_m = d[factor] >= qs[1]
    d = d.copy()
    d["year"] = d["trade_date"].astype(str).str[:4]
    rows = []
    for y, grp in d.groupby("year"):
        lo = grp[lo_m.reindex(grp.index, fill_value=False)][f"fwd{H}"]
        hi = grp[hi_m.reindex(grp.index, fill_value=False)][f"fwd{H}"]
        rows.append(dict(year=y, n=len(grp),
                         Q0地量=lo.mean() * 100 if len(lo) else np.nan,
                         Q4过热=hi.mean() * 100 if len(hi) else np.nan,
                         利差=(hi.mean() - lo.mean()) * 100 if len(lo) and len(hi) else np.nan))
    out = pd.DataFrame(rows)
    out["利差为负(有效)"] = np.where(out["利差"] < 0, "✅", "❌")
    return out


def exclude_test(df, factor, H):
    """剔除检验：信号是否只由某一段行情撑起来。"""
    out = []
    cases = [
        ("全样本", None),
        ("剔2015牛市", ("20150101", "20151231")),
        ("剔2018", ("20180101", "20181231")),
        ("剔2024", ("20240901", "20241231")),
        ("剔2025-2026", ("20250101", "20261231")),
    ]
    for label, excl in cases:
        d = df[["trade_date", factor, f"fwd{H}"]].dropna()
        if excl:
            d = d[~((d["trade_date"] >= excl[0]) & (d["trade_date"] <= excl[1]))]
        if len(d) < 60:
            continue
        x, y = d[factor].values.astype(float), d[f"fwd{H}"].values.astype(float)
        ic = spearman_ic(x, y)
        _, t_nw = nw_tstat(y, x, lag=H)
        out.append(dict(剔除=label, n=len(d), IC=ic, t_NW=t_nw))
    return pd.DataFrame(out)


def redundancy(df, factor, H):
    """Gate 5：换手率信号 vs 两个对照信号的相关性 + 残差检验。"""
    rows = []
    for ctrl, cname in [("pct_rv20", "已实现波动率分位"), ("pct_nobj", "破净率分位")]:
        if ctrl not in df.columns:
            continue
        d = df[[factor, ctrl, f"fwd{H}"]].dropna()
        if len(d) < 200:
            continue
        rho = stats.spearmanr(d[factor], d[ctrl]).correlation
        # 残差化（无前视滚动）：y_t = beta0 + beta1 * ctrl_t，beta 只用 [t-750, t)
        resid = np.full(len(d), np.nan)
        fv, cv = d[factor].values, d[ctrl].values
        for i in range(750, len(d)):
            m = np.isfinite(fv[i - 750:i]) & np.isfinite(cv[i - 750:i])
            if m.sum() < 100:
                continue
            b1, b0 = np.polyfit(cv[i - 750:i][m], fv[i - 750:i][m], 1)
            resid[i] = fv[i] - (b0 + b1 * cv[i])
        d = d.assign(resid=resid).dropna(subset=["resid"])
        ic_raw = spearman_ic(d[factor], d[f"fwd{H}"])
        ic_res = spearman_ic(d["resid"], d[f"fwd{H}"])
        _, t_res = nw_tstat(d[f"fwd{H}"].values, d["resid"].values, lag=H)
        rows.append(dict(对照信号=cname, Spearman=rho, n=len(d),
                         IC_原始=ic_raw, IC_残差=ic_res, t_残差=t_res,
                         残差保留比=(ic_res / ic_raw if ic_raw else np.nan)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="换手率择时信号检验")
    ap.add_argument("--factor", default="pct_flow", help="主因子列（默认 pct_flow）")
    ap.add_argument("--horizon", type=int, default=250)
    a = ap.parse_args()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df = build()

    print("\n" + "=" * 78)
    print("Gate 1  时序 IC —— 三口径并列（IC<0 才支持「高换手=顶部」）")
    print("=" * 78)
    rows = []
    for f in PCTS + FACTORS:
        for H in HORIZONS:
            r = run_horizon(df, f, H)
            if r:
                rows.append(r)
    res = pd.DataFrame(rows)
    print(res.round(4).to_string(index=False))
    res.to_csv(OUT, index=False, encoding="utf-8-sig")

    H = a.horizon
    fac = a.factor
    print("\n" + "=" * 78)
    print(f"分层检验  factor={fac}  H={H}")
    print("=" * 78)
    g = run_quantile(df, fac, H)
    if g is not None:
        print((g.assign(**{c: (g[c] * 100).round(2) for c in ["mean", "median", "std"]}))
              .round(2).to_string())

    print("\n" + "=" * 78)
    print(f"逐年利差（Q4过热 − Q0地量，负值=有效）  factor={fac}  H={H}")
    print("=" * 78)
    y = yearly(df, fac, H)
    if y is not None:
        print(y.round(2).to_string(index=False))
        y.to_csv(OUT_YEARLY, index=False, encoding="utf-8-sig")
        neg = (y["利差"] < 0).sum()
        print(f"\n-> 利差为负（支持假说）的年份: {neg}/{len(y)}")

    print("\n" + "=" * 78)
    print(f"剔除检验  factor={fac}  H={H}")
    print("=" * 78)
    print(exclude_test(df, fac, H).round(4).to_string(index=False))

    print("\n" + "=" * 78)
    print(f"Gate 5 冗余  factor={fac}  H={H}")
    print("=" * 78)
    print(redundancy(df, fac, H).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
