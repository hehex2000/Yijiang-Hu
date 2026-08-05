# -*- coding: utf-8 -*-
"""
factor_attribution.py — A股年度调仓策略的 暴露诊断(A) + 收益归因(C)

复用 run_dogs_annual.py 产出的两份 CSV:
  --nav-csv       日净值 CSV (date,value,value_raw)  例: data/results/dogs_annual/backtest_20140101_20260720.csv
  --holdings-csv  调仓持仓快照 CSV (rebalance_date,ts_code)  例: data/results/dogs_annual/holdings_20140101_20260720.csv

(A) 暴露诊断: 每个调仓日对当期持仓算 Value/Momentum/Size/Volatility 的截面 z 分位暴露 + 行业集中度(HHI)
(C) 收益归因 bridge: 用 bulk 加载 daily/daily_basic/index_daily 构建 市场 + 4 风格因子日收益序列(月度再平衡 L-S 组合),
    对组合「原始价」日收益做时序回归得 β/风格载荷/α; 分红贡献用双轨 R=hfq/raw 比值单独拆出。
    年度 bridge = 市场 + 风格 + α + 分红, 并做闭合校验。

输出:
  <out-dir>/exposure_<tag>.csv    各调仓日因子暴露 + 行业集中度
  <out-dir>/attribution_<tag>.csv 逐年级 + 全程 P&L bridge
  控制台打印汇总
"""
import argparse
import os
import sqlite3
import numpy as np
import pandas as pd

import config
DB_PATH = config.DATA["local_db_path"]
SELECTION = getattr(config, "SELECTION", {})


# ----------------------------------------------------------------------------
# 数据加载
# ----------------------------------------------------------------------------
def get_conn():
    return sqlite3.connect(DB_PATH)


def load_nav(nav_csv):
    df = pd.read_csv(nav_csv)
    df["date"] = df["date"].astype(str)
    return df


def load_holdings(holdings_csv):
    df = pd.read_csv(holdings_csv)
    df["rebalance_date"] = df["rebalance_date"].astype(str)
    df["ts_code"] = df["ts_code"].astype(str)
    return df


def load_pivots(start, end):
    """加载窗口内的 close / pe_ttm / total_mv 枢纽表 (date × ts_code)。"""
    conn = get_conn()
    close = pd.read_sql_query(
        "SELECT ts_code, trade_date, close FROM daily WHERE trade_date BETWEEN ? AND ?",
        conn, params=(start, end))
    basic = pd.read_sql_query(
        "SELECT ts_code, trade_date, pe_ttm, total_mv FROM daily_basic "
        "WHERE trade_date BETWEEN ? AND ?",
        conn, params=(start, end))
    conn.close()
    close_p = close.pivot(index="trade_date", columns="ts_code", values="close").astype("float64")
    pe_p = basic.pivot(index="trade_date", columns="ts_code", values="pe_ttm").astype("float64")
    mv_p = basic.pivot(index="trade_date", columns="ts_code", values="total_mv").astype("float64")
    return close_p, pe_p, mv_p


def load_market_returns(market_index, start, end):
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT trade_date, close FROM index_daily WHERE ts_code=? AND trade_date BETWEEN ? AND ? "
        "ORDER BY trade_date",
        conn, params=(market_index, start, end))
    conn.close()
    if df.empty:
        raise RuntimeError(f"市场指数 {market_index} 在库中无数据")
    df["trade_date"] = df["trade_date"].astype(str)
    df = df.set_index("trade_date")["close"].astype("float64")
    return df.pct_change().rename("mkt")


def load_industry_map():
    conn = get_conn()
    df = pd.read_sql_query("SELECT ts_code, industry FROM stock_basic", conn)
    conn.close()
    return dict(zip(df["ts_code"].astype(str), df["industry"].fillna("未知")))


# ----------------------------------------------------------------------------
# 风格因子日收益序列 (月度再平衡 L-S 组合)
# ----------------------------------------------------------------------------
def build_style_factor_returns(close_p, pe_p, mv_p, trade_dates):
    """返回 dict: {value, momentum, size, volatility} -> Series(date->日收益)。"""
    ret_p = close_p.pct_change()
    # 截面特征 (越高越"好"：Value=便宜, Momentum=涨, Size=大, Vol=低波动)
    val_char = (1.0 / pe_p).where(pe_p > 0)            # 高=便宜
    mom_char = close_p / close_p.shift(252) - 1.0      # 12-1 月动量
    size_char = np.log(mv_p.where(mv_p > 0))           # 市值对数
    vol_char = -ret_p.rolling(60).std()                # 低波动=高

    # 动量: 12-1 月; 早期无 252 日历史时用 1 月动量代理, 避免整段 NaN 把该年剔出回归
    mom_long = close_p / close_p.shift(252) - 1.0
    mom_short = close_p / close_p.shift(21) - 1.0
    mom_char = mom_long.where(mom_long.notna(), mom_short)

    chars = {"value": val_char, "momentum": mom_char, "size": size_char, "volatility": vol_char}

    # 月末再平衡点 (按 YYYYMM 分组取每月最后交易日)
    td_series = pd.Series(trade_dates)
    month_ends = td_series.groupby(td_series.str[:6]).tail(1).tolist()

    result = {k: pd.Series(np.nan, index=trade_dates, dtype="float64") for k in chars}

    # 消除残余 NaN (早期/缺口日): 向前填充, 使回归覆盖全样本 (L-S 组合视为持有)
    for k in result:
        result[k] = result[k].ffill()

    for ki, k in enumerate(chars):
        char_p = chars[k]
        for j in range(len(month_ends) - 1):
            as_of = month_ends[j]
            nxt = month_ends[j + 1]
            if as_of not in char_p.index or nxt not in ret_p.index:
                continue
            row = char_p.loc[as_of]
            valid = row.notna() & np.isfinite(row.values)
            valid_codes = [c for c in row.index[valid] if c in ret_p.columns]
            if len(valid_codes) < 60:
                continue
            rank = row.loc[valid_codes].rank(pct=True)
            longs = rank[rank >= 0.7].index
            shorts = rank[rank <= 0.3].index
            if len(longs) < 10 or len(shorts) < 10:
                continue
            days = [d for d in trade_dates if as_of < d <= nxt]
            if not days:
                continue
            sub = ret_p.loc[days, list(longs) + list(shorts)]
            lg = sub[longs].mean(axis=1, skipna=True)
            sh = sub[shorts].mean(axis=1, skipna=True)
            result[k].loc[days] = (lg - sh).values
    return result


# ----------------------------------------------------------------------------
# 归因 (时序回归)
# ----------------------------------------------------------------------------
def ols(y, X):
    """返回 coeffs (含截距), R2。X 已含截距列。"""
    Xt = X.T
    beta, *_ = np.linalg.lstsq(Xt @ X, Xt @ y, rcond=None)
    pred = X @ beta
    resid = y - pred
    tss = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ((resid ** 2).sum() / tss) if tss > 0 else 0.0
    return beta, r2


def run_attribution(nav, factor_rets, market_ret, out_dir, tag):
    nav = nav.copy()
    nav = nav.sort_values("date").reset_index(drop=True)
    nav["port_raw"] = nav["value_raw"].pct_change()
    nav["port_hfq"] = nav["value"].pct_change()
    nav["R"] = nav["value"] / nav["value_raw"]   # 双轨比值 (分红因子累积)

    # 对齐因子序列
    idx = nav["date"]
    aligned = pd.DataFrame({"date": idx})
    aligned["port_raw"] = nav["port_raw"].values
    aligned["port_hfq"] = nav["port_hfq"].values
    aligned["R"] = nav["R"].values
    # 关键: 因子/市场序列必须重索引到 NAV 完整日期轴, 缺失日 ffill→0,
    # 否则 dropna 会删掉组合收益行, 破坏复利闭合 (会把 +50.5% 算成 +80.4%)
    def _align(s):
        return s.reindex(idx).ffill().fillna(0.0).values

    aligned["mkt"] = _align(market_ret)
    for k, s in factor_rets.items():
        aligned[k] = _align(s)
    # 只丢首日 (pct_change 产生的 NaN), 因子已无 NaN
    fac_cols = ["mkt"] + list(factor_rets.keys())
    aligned = aligned.dropna(subset=["port_raw"] + fac_cols).reset_index(drop=True)

    years = sorted(set(d[:4] for d in aligned["date"]))
    rows = []
    # 全程
    for label, sub in [("全程", aligned)] + [(y, aligned[aligned["date"].str[:4] == y]) for y in years]:
        if len(sub) < 30:
            continue
        # 转 log 收益做严谨的区间分解 (每日 log 回归 → 区间因子 log 收益求和 → 转回简单收益乘法闭合)
        def logr(x):
            x = np.asarray(x, dtype="float64")
            x = np.where(x <= -0.9999, -0.9999, x)
            return np.log1p(x)
        lp = logr(sub["port_raw"].values)
        X = np.column_stack([np.ones(len(sub)), logr(sub["mkt"].values)] +
                            [logr(sub[c].values) for c in factor_rets])
        beta, r2 = ols(lp, X)
        alpha_d, beta_mkt = beta[0], beta[1]
        beta_fac = dict(zip(factor_rets.keys(), beta[2:]))

        # 区间因子 log 收益 = 每日 log 收益求和
        FL_mkt = logr(sub["mkt"].values).sum()
        FL = {c: logr(sub[c].values).sum() for c in factor_rets}
        PL = lp.sum()
        mkt_log_c = beta_mkt * FL_mkt
        style_log_c = {c: beta_fac[c] * FL[c] for c in factor_rets}
        style_log_total = sum(style_log_c.values())
        alpha_log = PL - mkt_log_c - style_log_total

        # 转回简单收益
        mkt_contrib = float(np.exp(mkt_log_c) - 1.0)
        style_contrib = {c: float(np.exp(style_log_c[c]) - 1.0) for c in factor_rets}
        style_total = float(np.exp(style_log_total) - 1.0)
        alpha_year = float(np.exp(alpha_log) - 1.0)

        # 分红贡献 (双轨几何): R_end/R_start - 1
        R0 = sub["R"].iloc[0]
        R1 = sub["R"].iloc[-1]
        div_log = float(np.log(R1 / R0)) if (R0 and R0 > 0) else 0.0
        div_contrib = float(np.exp(div_log) - 1.0)

        # 真实区间收益 (几何)
        port_raw_geo = float(np.exp(PL) - 1.0)
        port_hfq_geo = float(np.exp(PL + div_log) - 1.0)

        # 闭合校验: (1+mkt)*(1+style)*(1+alpha)*(1+div) - 1 == hfq_geo
        bridge = (1 + mkt_contrib) * (1 + style_total) * (1 + alpha_year) * (1 + div_contrib) - 1.0
        closure = port_hfq_geo - bridge

        rows.append({
            "区间": label,
            "组合(hfq)": port_hfq_geo,
            "组合(raw)": port_raw_geo,
            "市场β": beta_mkt,
            "市场贡献": mkt_contrib,
            "风格贡献": style_total,
            "Value贡献": style_contrib["value"],
            "Momentum贡献": style_contrib["momentum"],
            "Size贡献": style_contrib["size"],
            "Vol贡献": style_contrib["volatility"],
            "α(选股)": alpha_year,
            "分红贡献": div_contrib,
            "闭合残差": closure,
            "R2": r2,
        })

    out_df = pd.DataFrame(rows)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("\n===== 收益归因 P&L Bridge (年度) =====")
    show = out_df.copy()
    for c in show.columns:
        if c not in ("区间",):
            show[c] = (show[c] * 100).round(2).astype(str) + "%"
    print(show.to_string(index=False))

    # 保存
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"attribution_{tag}.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\n归因结果已保存 → {out_path}")
    return out_df


# ----------------------------------------------------------------------------
# 暴露诊断 (A)
# ----------------------------------------------------------------------------
def exposure_diagnosis(holdings, close_p, pe_p, mv_p, industry_map, out_dir, tag):
    ret_p = close_p.pct_change()
    val_char = (1.0 / pe_p).where(pe_p > 0)
    mom_char = close_p / close_p.shift(252) - 1.0
    size_char = np.log(mv_p.where(mv_p > 0))
    vol_char = -ret_p.rolling(60).std()
    char_map = {"Value": val_char, "Momentum": mom_char, "Size": size_char, "Volatility": vol_char}

    reb_dates = sorted(set(holdings["rebalance_date"]))
    rows = []
    for rd in reb_dates:
        if rd not in close_p.index:
            continue
        codes = holdings[holdings["rebalance_date"] == rd]["ts_code"].tolist()
        # 截面 z 分位 (全市场)
        zrow = {}
        for fname, cp in char_map.items():
            cross = cp.loc[rd]
            mu, sd = cross.mean(), cross.std()
            if sd and np.isfinite(sd) and sd > 0:
                zs = [(c, (cross.get(c, np.nan) - mu) / sd) for c in codes]
                zs = [z for _, z in zs if np.isfinite(z)]
                zrow[fname] = float(np.mean(zs)) if zs else np.nan
            else:
                zrow[fname] = np.nan
        # 行业集中度
        inds = [industry_map.get(c, "未知") for c in codes]
        vc = pd.Series(inds).value_counts(normalize=True)
        hhi = float((vc ** 2).sum())
        top3 = "; ".join(f"{k}({v*100:.0f}%)" for k, v in vc.head(3).items())
        rows.append({
            "调仓日": rd,
            "持仓数": len(codes),
            "Value_z": round(zrow["Value"], 2) if np.isfinite(zrow["Value"]) else None,
            "Momentum_z": round(zrow["Momentum"], 2) if np.isfinite(zrow["Momentum"]) else None,
            "Size_z": round(zrow["Size"], 2) if np.isfinite(zrow["Size"]) else None,
            "Volatility_z": round(zrow["Volatility"], 2) if np.isfinite(zrow["Volatility"]) else None,
            "行业HHI": round(hhi, 3),
            "前3行业": top3,
        })

    out_df = pd.DataFrame(rows)
    print("\n===== 持仓暴露诊断 (逐调仓日) =====")
    print(out_df.to_string(index=False))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"exposure_{tag}.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\n暴露诊断已保存 → {out_path}")
    return out_df


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nav-csv", required=True)
    ap.add_argument("--holdings-csv", required=True)
    ap.add_argument("--market-index", default=None,
                    help="市场因子指数 (默认按 SELECTION.stock_pool 映射, hs300→000300.SH)")
    ap.add_argument("--out-dir", default="data/results/dogs_annual/attribution")
    args = ap.parse_args()

    if args.market_index:
        mkt_idx = args.market_index
    else:
        pool = SELECTION.get("stock_pool", "hs300")
        mkt_idx = {"hs300": "000300.SH", "zz500": "000905.SH", "zz800": "000906.SH",
                   "zz1000": "000852.SH", "all": "000985.SH"}.get(pool, "000300.SH")

    nav = load_nav(args.nav_csv)
    holdings = load_holdings(args.holdings_csv)
    start, end = nav["date"].min(), nav["date"].max()
    tag = f"{start}_{end}"

    print(f"[info] 窗口 {start}→{end}, 市场因子={mkt_idx}")

    print("[info] 加载 daily/daily_basic 枢纽表 ...")
    close_p, pe_p, mv_p = load_pivots(start, end)
    trade_dates = list(close_p.index)
    print(f"[info] {len(trade_dates)} 交易日, {close_p.shape[1]} 只股票")

    print("[info] 构建风格因子日收益 (月度再平衡 L-S) ...")
    style = build_style_factor_returns(close_p, pe_p, mv_p, trade_dates)

    print(f"[info] 加载市场因子 {mkt_idx} 日收益 ...")
    mkt = load_market_returns(mkt_idx, start, end)
    mkt = mkt[mkt.index.isin(trade_dates)].ffill()

    industry_map = load_industry_map()

    print("[info] 运行归因 ...")
    run_attribution(nav, style, mkt, args.out_dir, tag)
    print("[info] 暴露诊断 ...")
    exposure_diagnosis(holdings, close_p, pe_p, mv_p, industry_map, args.out_dir, tag)
    print("\n完成。")


if __name__ == "__main__":
    main()
