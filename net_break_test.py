# -*- coding: utf-8 -*-
"""
破净率择时信号检验（时序型，六道闸门 Gate 1 / Gate 4 前半段）

要回答的问题
------------
「全市场破净率处于高位时，未来市场（中证800）收益是否更高？」
  - IC > 0：破净率高 → 未来收益高 → **信号有效**（支持做 overlay）
  - IC ≈ 0：无效
  - IC < 0：反向（破净率高反而危险）

为什么用市场（中证800）而不是价值策略本身作标的
---------------------------------------------
overlay 的形式是「破净率高 → 提高价值策略仓位」。这个决策的**第一性**依据应是
市场层面的便宜程度。价值策略自身的表现留到 `net_break_overlay.py` 用真实
A/B 回测验证（那才是最终判据）。两步分开，避免用同一份数据既造因子又评因子。

统计口径（同券商因子，避免样本重叠虚高）
----------------------------------------
日频采样 + H 日 forward return → 相邻样本重叠 (H−1)/H。三口径并列：
  t_naive  明知虚高，仅作对照
  t_NW(H)  Newey-West，lag=H 覆盖重叠窗口  ← 主口径
  t_NWauto 自动带宽 floor(4*(n/100)^(2/9))，交叉验证
  t_nonovl 每 H 日抽一个非重叠子样本 ← 最保守

⚠️ 已踩过的坑：NW 方差 `V = (X'X)⁻¹ S (X'X)⁻¹` 后面**不能再乘 n**
   （S 已对全部 t 求和）。乘了 se 会放大 √n 倍，把强信号误判成不显著。

用法
----
  python net_break_test.py
  python net_break_test.py --factor pct_nobj     # 换口径
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

CACHE = os.path.join("data", "results", "net_break", "market_net_break.csv")
OUT = os.path.join("data", "results", "net_break", "nb_test.csv")

BENCH = "000906.SH"   # 中证800（🔴 000905 才是中证500，勿混）
HORIZONS = (60, 120, 250)
START = "20100101"
END = "20260831"


def load_bench():
    """中证800 全收益日序列（hfq 口径，与后续 value 策略 NAV 同口径）。"""
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
    nb = pd.read_csv(CACHE, dtype={"trade_date": str})
    b = load_bench()
    df = nb.merge(b, on="trade_date", how="inner").sort_values("trade_date").reset_index(drop=True)
    print(f"[合并] 破净率 {len(nb)} 行 × 基准 {len(b)} 行 → {len(df)} 行")

    # 未来 H 日市场收益（T+1 进场，避免当日收盘价可知的偷价）
    nav = df["bench_close"].values
    for H in HORIZONS:
        f = np.full(len(nav), np.nan)
        f[: len(nav) - H - 1] = nav[H + 1:] / nav[1: len(nav) - H] - 1
        df[f"fwd{H}"] = f
    return df


def run_horizon(df, factor, H):
    """一个 (因子, horizon) 组合的完整检验。"""
    d = df[["trade_date", factor, f"fwd{H}"]].dropna()
    x = d[factor].values.astype(float)
    y = d[f"fwd{H}"].values.astype(float)
    if len(d) < 60:
        return None
    ic = spearman_ic(x, y)
    # 注意：broker_pe_test.spearman_ic 只回传相关系数（不回 p 值），p 值单独算
    icp = stats.spearmanr(x, y).pvalue
    b_naive, t_naive = naive_tstat(y, x)
    _, t_nw = nw_tstat(y, x, lag=H)
    _, t_auto = nw_tstat_auto(y, x)

    # 非重叠子样本
    idx = np.arange(0, len(d), H)
    xs, ys = x[idx], y[idx]
    ic_n = spearman_ic(xs, ys)
    _, t_n = nw_tstat(ys, xs, lag=1) if len(idx) >= 8 else (np.nan, np.nan)

    return {
        "factor": factor, "H": H, "n": len(d), "n_nonovl": len(idx),
        "IC": round(ic, 4), "IC_p": icp,
        "beta": round(b_naive, 6),
        "t_naive": round(t_naive, 3),
        "t_NW": round(t_nw, 3),
        "t_NWauto": round(t_auto, 3),
        "IC_nonovl": round(ic_n, 4) if ic_n == ic_n else np.nan,
        "t_nonovl": round(t_n, 3) if t_n == t_n else np.nan,
    }


def run_quantile(df, factor, H, q=5):
    """分层：按破净率分位分 5 档，看各档未来 H 日市场收益。"""
    d = df[["trade_date", factor, f"fwd{H}"]].dropna().copy()
    d["grp"] = pd.qcut(d[factor], q, labels=[f"Q{i}" for i in range(q)])
    g = d.groupby("grp", observed=True)[f"fwd{H}"].agg(["count", "mean", "median"])
    g["mean"] = (g["mean"] * 100).round(2)
    g["median"] = (g["median"] * 100).round(2)
    g.columns = ["次数", f"未来{H}日均值%", f"未来{H}日中位%"]
    return g


def yearly(df, factor, H):
    """逐年 IC + 强弱分组利差（Q4 − Q0）。

    🔴 分档阈值必须用**全样本**分位，不能用逐年 qcut：
       逐年 qcut 会把 fwd250 在年末必然为 NaN 的样本也算进档位，
       导致 Q0/Q4 均值大量 NaN（实测 15 年里只有 3 年算得出利差）。
       用全样本阈值后每年都有可比的高/低破净率样本。
    """
    d = df[["trade_date", factor, f"fwd{H}"]].dropna().copy()
    d["year"] = d["trade_date"].str[:4]
    try:
        qs = np.nanpercentile(d[factor], [20, 80])
    except Exception:
        return pd.DataFrame()
    lo_m = d[factor] <= qs[0]
    hi_m = d[factor] >= qs[1]
    rows = []
    for y, s in d.groupby("year"):
        if len(s) < 30:
            continue
        ic = spearman_ic(s[factor].values, s[f"fwd{H}"].values)
        sl, sh = s[lo_m.reindex(s.index)], s[hi_m.reindex(s.index)]
        lo = sl[f"fwd{H}"].mean() if len(sl) else np.nan
        hi = sh[f"fwd{H}"].mean() if len(sh) else np.nan
        rows.append({"year": y, "n": len(s), "n_lo": len(sl), "n_hi": len(sh),
                     "IC": round(ic, 3),
                     "Q0%": round(lo * 100, 2) if lo == lo else np.nan,
                     "Q4%": round(hi * 100, 2) if hi == hi else np.nan,
                     "Q4-Q0%": round((hi - lo) * 100, 2) if (hi == hi and lo == lo) else np.nan})
    return pd.DataFrame(rows)


def exclude_test(df, factor, H, lo_d, hi_d):
    """剔除某段时间后重算 IC（检验信号是否被单次极端行情撑起）。"""
    d = df[["trade_date", factor, f"fwd{H}"]].dropna()
    m = ~((d["trade_date"] >= lo_d) & (d["trade_date"] <= hi_d))
    s = d[m]
    if len(s) < 60:
        return np.nan, np.nan
    ic = spearman_ic(s[factor].values, s[f"fwd{H}"].values)
    _, t = nw_tstat(s[f"fwd{H}"].values, s[factor].values, lag=H)
    return round(ic, 4), round(t, 3)


def redundancy(df, factor, H=250):
    """Gate 5 冗余：破净率分位 vs 全市场 PB 分位。

    券商因子已证明「全市场 PB 分位」是有效择时信号。
    若破净率与它高度相关，则破净率只是同一信息的另一种算法
    → 不是致命问题（破净率的定位本就是市场择时），但必须如实标注，
      否则会误以为发现了新信号。
    """
    p = os.path.join("data", "results", "broker_pe", "market_daily.csv")
    if not os.path.exists(p):
        print("\n[冗余] 缺 market_daily.csv，跳过（先跑 industry_pe_panel.py）")
        return
    m = pd.read_csv(p, dtype={"trade_date": str})
    d = df.merge(m[["trade_date", "pb_pct750"]], on="trade_date", how="inner")
    d = d[["trade_date", factor, "pb_pct750", f"fwd{H}"]].dropna()
    if len(d) < 60:
        print("\n[冗余] 样本不足，跳过")
        return
    rho = stats.spearmanr(d[factor], d["pb_pct750"]).correlation
    print("\n" + "=" * 96)
    print("Gate 5  冗余检验：破净率分位 vs 全市场 PB 分位")
    print("=" * 96)
    print(f"  样本 {len(d)} 日，Spearman 相关 = **{rho:.3f}**")
    if abs(rho) >= 0.7:
        print("  → 高度相关：破净率是「全市场 PB 分位」的另一种算法（同一信息）。")
        print("     不是致命问题（定位本就是市场择时），但不能声称是新发现。")
    elif abs(rho) >= 0.4:
        print("  → 中度相关：有相当部分信息重叠，剩余部分需残差化后再判。")
    else:
        print("  → 低相关：破净率提供的是独立信息。")

    # 残差化（滚动、无前视）后看 IC 是否还显著
    from broker_pe_redundancy import rolling_residual
    resid = rolling_residual(d[factor].values.astype(float),
                             d["pb_pct750"].values.astype(float), window=750)
    dd = d.iloc[-len(resid):].copy()
    dd["resid"] = resid
    ic_r = spearman_ic(dd["resid"].values, dd[f"fwd{H}"].values)
    _, t_r = nw_tstat(dd[f"fwd{H}"].values, dd["resid"].values, lag=H)
    print(f"  滚动残差化后（window=750，无前视）：IC = {ic_r:.4f}  t_NW = {t_r:.3f}"
          f"  n={len(dd)}")
    if t_r is not None and abs(t_r) >= 2:
        print("  → 残差仍显著：破净率在剔除市场 PB 后**仍有独立增量**。")
    else:
        print("  → 残差不显著：增量信息归零，破净率 ≈ 市场 PB 分位的马甲。")


def main():
    ap = argparse.ArgumentParser(description="破净率择时信号检验")
    ap.add_argument("--factor", default="pct_all",
                    choices=["pct_all", "pct_nobj", "pct_clean"],
                    help="破净率口径：pct_all 全A / pct_nobj 剔北交所 / pct_clean 再剔ST")
    ap.add_argument("--raw-factor", action="store_true",
                    help="用原始破净率（rate_*）而非滚动分位")
    a = ap.parse_args()

    factor = a.factor.replace("pct_", "rate_") if a.raw_factor else a.factor
    df = build()
    print(f"\n因子 = {factor}（{'原始破净率' if a.raw_factor else '750日滚动分位'}）")
    print("方向约定：IC > 0 = 破净率越高未来市场收益越高 = 信号有效")

    # ---- Gate 1：IC 与显著性
    print("\n" + "=" * 96)
    print("Gate 1  时序 IC 与显著性")
    print("=" * 96)
    rows = [run_horizon(df, factor, H) for H in HORIZONS]
    rep = pd.DataFrame([r for r in rows if r])
    print(rep.to_string(index=False))
    rep.to_csv(OUT, index=False, encoding="utf-8-sig")

    # ---- 分层
    for H in (120, 250):
        print(f"\n--- 分层（{factor} → 未来{H}日中证800收益，Q0=破净率最低 … Q4=最高）---")
        print(run_quantile(df, factor, H).to_string())

    # ---- 逐年
    H = 250
    y = yearly(df, factor, H)
    print(f"\n--- 逐年（{factor} → 未来{H}日）---")
    print(y.to_string(index=False))
    pos = (y["IC"] > 0).sum()
    print(f"\nIC>0 年份 {pos}/{len(y)}；Q4-Q0 利差>0 年份 {(y['Q4-Q0%'] > 0).sum()}/{len(y)}")
    tot = y["Q4-Q0%"].abs().sum()
    top2 = y.reindex(y["Q4-Q0%"].abs().sort_values(ascending=False).index)[:2]
    print(f"利差绝对值 top2 年份：{list(top2['year'])} 贡献 "
          f"{top2['Q4-Q0%'].abs().sum()/tot*100:.1f}%")

    # ---- 剔除检验：信号是否被单次极端行情撑起
    print("\n--- 剔除检验（IC / t_NW，对照全样本）---")
    for H2 in HORIZONS:
        full_ic = rep.loc[rep["H"] == H2, "IC"].iloc[0]
        full_t = rep.loc[rep["H"] == H2, "t_NW"].iloc[0]
        print(f"  H={H2}:  全样本 IC={full_ic:+.4f} t={full_t:+.3f}")
        for tag, lo_d, hi_d in [
            ("剔 2014.06-2015.12 牛市", "20140601", "20151231"),
            ("剔 2018 全年熊市", "20180101", "20181231"),
            ("剔 2024 全年(924行情)", "20240101", "20241231"),
        ]:
            ic2, t2 = exclude_test(df, factor, H2, lo_d, hi_d)
            print(f"     {tag:<24} IC={ic2:+.4f} t={t2:+.3f}")

    redundancy(df, factor, H)

    # ---- 🔴 2021.02 反例独立考察
    print("\n" + "=" * 96)
    print("🔴 2021.02 反例：破净率 >10% 却迎来核心资产见顶")
    print("=" * 96)
    ev = df[(df["trade_date"] >= "20210101") & (df["trade_date"] <= "20211231")]
    if len(ev):
        i = ev["rate_all"].idxmax()
        r = df.loc[i]
        print(f"  2021 年破净率峰值日 {r['trade_date']}，全A {r['rate_all']*100:.2f}%")
        for H2 in HORIZONS:
            v = r.get(f"fwd{H2}")
            print(f"    该日之后 {H2} 日中证800收益："
                  f"{v*100:+.2f}%" if v == v else f"    之后{H2}日：无数据")
    print("  注：这是全样本里唯一一个「破净率高但随后大跌」的案例，")
    print("      它决定了 overlay 在 2021 年会系统性挨一记（后面 A/B 回测验证）。")


if __name__ == "__main__":
    main()
