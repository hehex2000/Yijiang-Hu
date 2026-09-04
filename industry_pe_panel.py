# -*- coding: utf-8 -*-
"""
多行业「估值分位 → 未来超额」对照 + Gate 5 冗余检验（剥出身）

要回答的核心问题：券商 PB 分位的预测力，到底是
  (A) 券商特有的估值信息，还是
  (B) 「市场整体估值择时」换了个马甲（即券商只是市场 beta 放大器）？

方法
----
1) 对 10 个代表性行业各自构建板块序列（与券商完全同口径），
   计算 PB 分位 → 未来 250 日行业超额 的 IC 与非重叠调仓绩效。
   → 若所有行业都显著，说明是市场级现象（B）；若只有券商显著，说明是行业特有（A）。

2) 冗余检验：把「行业 PB 分位」对「全市场 PB 分位」做 OLS 残差化，
   看残差 IC 是否还显著。残差 IC ≈ 0 → 冗余（Gate 5 判据）。

用法
----
  python industry_pe_panel.py            # 全量构建 + 检验（首次较慢）
  python industry_pe_panel.py --reuse    # 复用已缓存的行业序列
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

from config import DATA  # noqa: E402
import broker_pe_factor as bpf  # noqa: E402

OUT_DIR = os.path.join("data", "results", "broker_pe")
PANEL_CACHE = os.path.join(OUT_DIR, "industry_panel.csv")
MKT_CACHE = os.path.join(OUT_DIR, "market_daily.csv")

# 对照组：周期 / 金融 / 稳定 / 科技
INDUSTRIES = [
    ("IND_0029", "证券"),
    ("IND_0036", "银行"),
    ("IND_0053", "煤炭开采"),
    ("IND_0023", "小金属"),
    ("IND_0006", "化工原料"),
    ("IND_0063", "白酒"),
    ("IND_0015", "食品"),
    ("IND_0016", "家用电器"),
    ("IND_0008", "医疗保健"),
    ("IND_0007", "半导体"),
]

H = 250
WINDOW = 750


# ------------------------------------------------------------ 全市场估值
def build_market_series():
    """全市场市值加权 PB/PE（整体法），用于残差化控制。"""
    con = bpf.get_conn()
    print("[全市场] 聚合 daily_basic 全表（约需 1-3 分钟）...")
    q = """
    SELECT trade_date,
           SUM(total_mv) / NULLIF(SUM(CASE WHEN pb>0 AND total_mv>0
                                           THEN total_mv/pb END), 0) AS pb,
           SUM(total_mv) / NULLIF(SUM(CASE WHEN pe_ttm>0 AND total_mv>0
                                           THEN total_mv/pe_ttm END), 0) AS pe,
           COUNT(*) AS n_stock
    FROM daily_basic
    WHERE trade_date >= ? AND total_mv > 0
    GROUP BY trade_date ORDER BY trade_date
    """
    df = pd.read_sql(q, con, params=[bpf.START])
    con.close()
    df["trade_date"] = df["trade_date"].astype(str)
    for c in ("pb", "pe"):
        r = df[c].rolling(WINDOW, min_periods=int(WINDOW * 0.8))
        df[f"{c}_pct{WINDOW}"] = r.apply(lambda x: (x < x.iloc[-1]).mean() * 100,
                                         raw=False)
    df.to_csv(MKT_CACHE, index=False, encoding="utf-8-sig")
    print(f"[全市场] {len(df)} 行 → {MKT_CACHE}")
    return df


def load_market():
    if os.path.exists(MKT_CACHE):
        return pd.read_csv(MKT_CACHE, dtype={"trade_date": str})
    return build_market_series()


# ------------------------------------------------------------ 行业面板
def build_industry_panel(force=False):
    if os.path.exists(PANEL_CACHE) and not force:
        return pd.read_csv(PANEL_CACHE, dtype={"trade_date": str})
    frames = []
    for code, name in INDUSTRIES:
        print(f"[构建] {code} {name} ...", flush=True)
        try:
            sec, _ = bpf.build(code, verbose=False)
            sec["ind_name"] = name
            frames.append(sec)
        except Exception as e:
            print(f"   ⚠️ 跳过 {name}: {e}")
    panel = pd.concat(frames, ignore_index=True)
    panel.to_csv(PANEL_CACHE, index=False, encoding="utf-8-sig")
    print(f"[输出] {PANEL_CACHE}  ({len(panel)} 行)")
    return panel


# ------------------------------------------------------------ 绩效
def nonoverlap_perf(d, factor_col, ret_col, H=H):
    """非重叠调仓：Q0 多 / Q4 空，各半仓。返回 dict。"""
    dd = d.iloc[::H].copy()
    if len(dd) < 8:
        return None
    dd["q"] = pd.qcut(dd[factor_col], 5, labels=False, duplicates="drop")
    long_ = (dd["q"] == 0).astype(float)
    short_ = (dd["q"] == 4).astype(float)
    dd["ls"] = 0.5 * long_ * dd[ret_col] - 0.5 * short_ * dd[ret_col]
    n = len(dd)
    nav = (1 + dd["ls"]).cumprod()
    tot = nav.iloc[-1] - 1
    ann = (1 + tot) ** (250.0 / (H * n)) - 1 if tot > -1 else np.nan
    dd_ = (nav / nav.cummax() - 1).min()
    win = (dd["ls"] > 0).mean() * 100
    t = dd["ls"].mean() / (dd["ls"].std() / np.sqrt(n)) if dd["ls"].std() > 0 else np.nan
    return dict(n_rebal=n, total=tot, ann=ann, mdd=dd_, win=win, t=t)


def spearman(x, y, min_n=30):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < min_n:
        return np.nan
    return stats.spearmanr(np.asarray(x)[m], np.asarray(y)[m]).correlation


# ------------------------------------------------------------ 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="强制重建行业面板")
    ap.add_argument("--market", action="store_true", help="重建全市场估值序列")
    args = ap.parse_args()

    panel = build_industry_panel(force=args.rebuild)
    if args.market or not os.path.exists(MKT_CACHE):
        mkt = build_market_series()
    else:
        mkt = load_market()

    factor = f"pb_pct{WINDOW}"
    ret = f"fwd_exc_{H}"

    print("\n" + "=" * 100)
    print(f"【对照】{len(INDUSTRIES)} 个行业：PB 分位({WINDOW}日) → 未来 {H} 日行业超额")
    print("=" * 100)
    print(f"{'行业':<10}{'成分':>6}{'n_obs':>7}{'IC':>9}{'调仓':>6}"
          f"{'总收益%':>10}{'年化%':>9}{'回撤%':>9}{'胜率%':>8}{'t':>8}  判定")
    print("-" * 100)

    rows = []
    for name, g in panel.groupby("ind_name"):
        g = g[[factor, ret]].dropna()
        if len(g) < 300:
            continue
        ic = spearman(g[factor].values, g[ret].values)
        p = nonoverlap_perf(g, factor, ret, H)
        if p is None:
            continue
        n_stock = panel[panel.ind_name == name]["n_stock"].iloc[-1]
        ok = (abs(p["t"]) >= 2.0) and (p["ann"] > 0.03) and (p["win"] >= 50)
        verdict = "✅ 可用" if ok else ("🟡 边缘" if abs(p["t"]) >= 1.5 else "❌ 不显著")
        rows.append(dict(ind=name, ic=ic, **p, ok=ok))
        print(f"{name:<10}{int(n_stock):>6}{len(g):>7}{ic:>9.3f}{p['n_rebal']:>6}"
              f"{p['total']*100:>10.2f}{p['ann']*100:>9.2f}{p['mdd']*100:>9.2f}"
              f"{p['win']:>8.1f}{p['t']:>8.2f}  {verdict}")
    print("-" * 100)

    df = pd.DataFrame(rows)
    n_ok = int(df.ok.sum()) if len(df) else 0
    print(f"  「可用」行业数: {n_ok}/{len(df)}")
    if n_ok >= len(df) * 0.6:
        print("  → 多数行业都成立 ⇒ 这是【市场级估值择时】现象，不是券商特有 alpha")
    elif n_ok <= 2:
        print("  → 仅个别行业成立 ⇒ 可能是行业特有，但更可能是多重检验下的偶然")
    else:
        print("  → 部分行业成立 ⇒ 混合，需进一步归因")

    # ---------------- Gate 5: 残差化冗余 ----------------
    print("\n" + "=" * 100)
    print("【Gate 5】冗余检验：行业 PB 分位 ~ 全市场 PB 分位 的 OLS 残差 IC")
    print("=" * 100)
    print(f"{'行业':<10}{'原始IC':>10}{'残差IC':>10}{'与全市场相关':>14}"
          f"{'残差化后':>12}  判定")
    print("-" * 100)
    mc = f"pb_pct{WINDOW}"
    # ⚠️ 必须重命名后再 merge：左表列名就是 factor(="pb_pct750")，
    #    若右表也叫同名列，suffixes=("","_mkt") 只作用于右表，
    #    m[mc] 仍取到【左表自己】的列 → 变成 y~y 自回归，残差=舍入噪音，
    #    Spearman 给噪音排序会产出随机 IC（实测全行业「与全市场相关」都=1.000）。
    mkt_s = mkt[["trade_date", mc]].rename(columns={mc: "mkt_pct"}).dropna()
    for name, g in panel.groupby("ind_name"):
        g = g[["trade_date", factor, ret]].dropna()
        m = g.merge(mkt_s, on="trade_date")
        if len(m) < 300 or "mkt_pct" not in m.columns:
            continue
        ic_raw = spearman(m[factor].values, m[ret].values)
        corr_mkt = spearman(m[factor].values, m["mkt_pct"].values)
        # OLS 残差化：行业分位 = a + b * 全市场分位 + e
        X = np.column_stack([np.ones(len(m)), m["mkt_pct"].values])
        beta = np.linalg.lstsq(X, m[factor].values, rcond=None)[0]
        resid = m[factor].values - X @ beta
        ic_res = spearman(resid, m[ret].values)
        drop = (1 - abs(ic_res) / abs(ic_raw)) * 100 if abs(ic_raw) > 1e-9 else np.nan
        if abs(ic_res) < 0.10:
            verdict = "❌ 冗余（残差几无信息）"
        elif abs(ic_res) < abs(ic_raw) * 0.5:
            verdict = "⚠️ 大半被市场估值解释"
        else:
            verdict = "✅ 有独立增量"
        print(f"{name:<10}{ic_raw:>10.3f}{ic_res:>10.3f}{corr_mkt if np.isfinite(corr_mkt) else np.nan:>14.3f}"
              f"{drop if np.isfinite(drop) else np.nan:>11.0f}%  {verdict}")
    print("-" * 100)
    print("  注：「与全市场相关」接近 1 说明该行业估值分位几乎就是市场估值分位的镜像；")
    print("      残差 IC 接近 0 说明剔除市场估值后没有独立信息 → 判冗余，不单独入库。")

    if len(df):
        df.to_csv(os.path.join(OUT_DIR, "industry_compare.csv"),
                  index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
