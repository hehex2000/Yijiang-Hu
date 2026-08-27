"""
run_diversified_portfolio.py — 组合层低相关分散风控升级 (验证 gated_alpha_report.md §5)

背景
----
gated_alpha_report.md 证明: 利弗莫尔"箱+量能"门控对 alpha 组合能降左尾 CVaR(-22.5%)、
降回撤(+11.3pp), 但代价是 -5.8pp 总收益, 且诚实诊断指出改善几乎全部来自"62%时间不在市场"
的机械降暴露, 而非信号择时。§5 结论: 真正组合层风控升级 = 引入低相关资产(国债/商品/美股)
做分散, 用完整协方差的组合 CVaR 设闸。

本脚本验证 §5: 把"现有 equity alpha"与低相关 sleeve(国债ETF/黄金ETF/货币ETF)做月度再平衡组合,
看是否比"门控减仓到现金"更优(同等降尾下少牺牲收益 / 更高夏普)。

数据
----
- equity alpha sleeve: data/results/livermore/alpha_nav.csv (日频 NAV, 2018-01-02~2025-12-31)
    列: 沪深300 / 价值选股 / 红利低波 / 高股息+成长
    默认 equity=红利低波(P0 唯一真 alpha, 连续 NAV); --equity combo = 三策略等权收益指数
- 低相关 sleeve: etf_daily + etf_adj_factor (后复权 NAV)
    511260.SH 国债10Y | 518880.SH 黄金 | 511990.SH 货币(现金替代)

纪律
----
- 无未来函数: 月度权重用 ≤ 上月末 的滚动波动估计; 组合日收益用权重点乘各 sleeve 日收益
- 后复权 NAV = close * adj_factor(累计), 防除权假跳
- 成本: 默认不计(组合层对比, 与门控报告同口径); --cost 可开单边 bp 近似换手冲击
"""
import argparse
import sqlite3
import numpy as np
import pandas as pd

DB = "D:/tu-shareData/astock_daily.db"
ALPHA_NAV = "data/results/livermore/alpha_nav.csv"

# 低相关 sleeve 定义
LOWCORR = {
    "bond":   "511260.SH",   # 国债10Y ETF
    "gold":   "518880.SH",   # 黄金 ETF
    "cash":   "511990.SH",   # 货币 ETF (现金替代)
}


def load_equity_sleeve(source="divlow"):
    """返回 equity alpha 的日收益指数(Series, 起点=1.0), 索引=trade_date(str)."""
    nav = pd.read_csv(ALPHA_NAV)
    nav = nav.sort_values("trade_date").reset_index(drop=True)
    if source == "divlow":
        col = "红利低波"
    elif source == "value":
        col = "价值选股"
    elif source == "growth":
        col = "高股息+成长"
    elif source == "hs300":
        col = "沪深300"
    elif source == "combo":
        # 三策略等权收益指数(消除不同初始资本尺度)
        idx = (nav["价值选股"] / nav["价值选股"].iloc[0]
               + nav["红利低波"] / nav["红利低波"].iloc[0]
               + nav["高股息+成长"] / nav["高股息+成长"].iloc[0]) / 3.0
        s = idx.rename("equity")
        s.index = nav["trade_date"].astype(str)
        return s
    else:
        raise ValueError(source)
    s = (nav[col] / nav[col].iloc[0]).rename("equity")
    s.index = nav["trade_date"].astype(str)
    return s


def load_etf_adj_nav(con, code):
    """后复权 NAV(Series 起点=1.0), 索引=trade_date(str). 用 etf_adj_factor 累计调整。"""
    df = pd.read_sql(
        f"SELECT trade_date, close FROM etf_daily WHERE ts_code='{code}' ORDER BY trade_date",
        con)
    af = pd.read_sql(
        f"SELECT trade_date, adj_factor FROM etf_adj_factor WHERE ts_code='{code}' ORDER BY trade_date",
        con)
    if df.empty:
        return None
    df["trade_date"] = df["trade_date"].astype(str)
    if not af.empty:
        af["trade_date"] = af["trade_date"].astype(str)
        m = df.merge(af, on="trade_date", how="left").sort_values("trade_date")
        m["adj_factor"] = m["adj_factor"].ffill().bfill()
        # adj_factor 相对首日归一
        m["adj_close"] = m["close"] * (m["adj_factor"] / m["adj_factor"].iloc[0])
    else:
        m = df.copy()
        m["adj_close"] = m["close"]
    s = (m["adj_close"] / m["adj_close"].iloc[0]).rename(code)
    s.index = m["trade_date"].astype(str)
    return s


def build_sleeves(equity_source, use_lowcorr, use_cash_proxy, cost_bp=0.0):
    """返回 {name: 日收益指数 Series}, 已对齐到共同交易日窗口(取 equity 窗口)。"""
    con = sqlite3.connect(DB)
    eq = load_equity_sleeve(equity_source)
    # 低相关 sleeve 选择
    wanted = {}
    if use_cash_proxy:
        wanted["cash"] = LOWCORR["cash"]
    else:
        for k in use_lowcorr:
            wanted[k] = LOWCORR[k]
    navs = {"equity": eq}
    for name, code in wanted.items():
        s = load_etf_adj_nav(con, code)
        if s is None:
            print(f"  ⚠️ {name}({code}) 无 etf_daily 数据, 跳过")
            continue
        navs[name] = s
    con.close()
    # 对齐到 equity 日历: 低相关 sleeve 缺失交易日按"持有"(ffill)处理, 不砍窗口
    eq_idx = navs["equity"].index
    common = sorted(eq_idx)
    out = {}
    for name, s in navs.items():
        if name == "equity":
            out[name] = s.reindex(common)
        else:
            # 缺失日=前收持有; 上市前(bfill)与末尾(ffill)补齐
            out[name] = s.reindex(common).ffill().bfill()
    out["equity"] = out["equity"].dropna()
    # 日收益
    rets = {name: s.pct_change().dropna() for name, s in out.items()}
    returns = pd.DataFrame(rets)
    returns = returns.dropna(how="any")
    if cost_bp > 0:
        # 月度再平衡, 用单边 bp 近似: 对非 equity sleeve 施加换手成本(月度)
        # 简化: 这里成本在组合层按 turnover 后处理, 默认不计
        pass
    return returns, common[0], common[-1]


def monthly_weights(returns, scheme, equity_w, vol_win=60):
    """返回 DataFrame 权重(索引=returns.index), 每行对应一天, 权重在月初设定用≤上月末数据。"""
    names = list(returns.columns)
    idx = returns.index
    # 月初标记
    month = pd.Series(idx, index=idx).str[:6]
    is_month_start = month != month.shift(1)
    weights = pd.DataFrame(0.0, index=idx, columns=names)
    cur_w = None
    # 预计算每个月初之前 60 日波动(年化)
    vol = returns.rolling(vol_win).std() * np.sqrt(252)
    for i, d in enumerate(idx):
        if is_month_start.iloc[i] or cur_w is None:
            if scheme == "static":
                # equity_w 给 equity, 其余均分
                rest = (1 - equity_w) / (len(names) - 1) if len(names) > 1 else 0.0
                cur_w = {n: (equity_w if n == "equity" else rest) for n in names}
            elif scheme == "invvol":
                v = vol.iloc[i - 1] if i > 0 else vol.iloc[0]
                v = v.replace(0, np.nan)
                inv = 1.0 / v
                inv = inv.fillna(inv.mean())
                tot = inv.sum()
                cur_w = {n: (inv[n] / tot) for n in names}
            for n in names:
                weights.at[d, n] = cur_w[n]
        else:
            for n in names:
                weights.at[d, n] = cur_w[n]
    return weights


def portfolio_nav(returns, weights):
    """日组合收益 = 行点乘权重; 累计 NAV。"""
    w = weights.reindex(returns.index).ffill()
    pr = (returns * w).sum(axis=1)
    nav = (1 + pr).cumprod()
    return nav, pr


def metrics(nav, rets_daily, rf_annual=0.0):
    """nav: 日 NAV Series; rets_daily: 日收益 Series。"""
    n = len(rets_daily)
    years = n / 252.0
    total = nav.iloc[-1] / nav.iloc[0] - 1
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1
    vol = rets_daily.std() * np.sqrt(252)
    rf_d = rf_annual / 252.0
    sharpe = (rets_daily.mean() - rf_d) / rets_daily.std() * np.sqrt(252) if rets_daily.std() > 0 else 0.0
    # 回撤
    peak = nav.cummax()
    dd = nav / peak - 1
    mdd = dd.min()
    # CVaR99 (日, 期望短缺)
    q = rets_daily.quantile(0.01)
    cvar = rets_daily[rets_daily <= q].mean()
    return {
        "总收益": total, "年化": cagr, "年化波动": vol, "夏普": sharpe,
        "最大回撤": mdd, "CVaR99(日)": cvar, "年数": years,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--equity", default="divlow",
                    choices=["divlow", "value", "growth", "hs300", "combo"],
                    help="equity alpha sleeve 来源(alpha_nav.csv)")
    ap.add_argument("--lowcorr", default="bond,gold",
                    help="低相关 sleeve, 逗号分隔: bond/gold")
    ap.add_argument("--cash-proxy", action="store_true",
                    help="非 equity 部分用货币ETF(常驻现金)替代债券+黄金, 模拟 gate-to-cash")
    ap.add_argument("--equity-w", type=float, default=0.7,
                    help="static 方案下 equity 权重(默认0.7)")
    ap.add_argument("--schemes", default="static,invvol",
                    help="分配方案: static/invvol 逗号分隔")
    ap.add_argument("--rf-from-cash", action="store_true",
                    help="用货币ETF日收益作无风险利率算夏普(否则 rf=0)")
    args = ap.parse_args()

    use_lowcorr = [x for x in args.lowcorr.split(",") if x in LOWCORR]
    returns, d0, d1 = build_sleeves(
        args.equity, use_lowcorr, args.cash_proxy)
    print(f"窗口 {d0}~{d1} | equity={args.equity} | "
          f"sleeves={list(returns.columns)} | 交易日 {len(returns)}")
    # rf: 始终用货币ETF日收益作无风险利率(若 --rf-from-cash)
    rf_annual = 0.0
    if args.rf_from_cash:
        con = sqlite3.connect(DB)
        cash_nav = load_etf_adj_nav(con, LOWCORR["cash"])
        con.close()
        if cash_nav is not None:
            cash_ret = cash_nav.reindex(returns.index).pct_change().dropna()
            cash_ret = cash_ret[~cash_ret.index.duplicated()]
            rf_annual = cash_ret.mean() * 252
            print(f"无风险利率(年化, 货基): {rf_annual:.4f}")
    print(f"无风险利率(年化): {rf_annual:.4f}")

    schemes = args.schemes.split(",")
    rows = []
    navs = {}
    for sch in schemes:
        w = monthly_weights(returns, sch, args.equity_w)
        nav, pr = portfolio_nav(returns, w)
        navs[sch] = nav
        m = metrics(nav, pr, rf_annual)
        m["方案"] = sch
        rows.append(m)
    # 纯 equity 基线
    w0 = monthly_weights(returns, "static", 1.0)
    nav0, pr0 = portfolio_nav(returns, w0)
    navs["pure_equity"] = nav0
    m0 = metrics(nav0, pr0, rf_annual)
    m0["方案"] = "pure_equity"
    rows.insert(0, m0)

    df = pd.DataFrame(rows).set_index("方案")
    # 相关性矩阵
    corr = returns.corr()
    print("\n=== 组合对照 (equity sleeve=" + args.equity + ") ===")
    print(df.to_string(float_format=lambda x: f"{x:.4f}"))
    print("\n=== 日收益相关性矩阵 ===")
    print(corr.to_string(float_format=lambda x: f"{x:.3f}"))

    # 落盘
    out = "data/results/diversified_portfolio.csv"
    df.to_csv(out)
    corr.to_csv(out.replace(".csv", "_corr.csv"))
    print(f"\n✓ 已保存: {out} | {out.replace('.csv', '_corr.csv')}")


if __name__ == "__main__":
    main()
