# -*- coding: utf-8 -*-
"""
行业ETF切点组合轮动 (Smart Tangency Portfolio 框架复现)
=======================================================
复现 IJFS 2025 "Smart Tangency Portfolio" 的切点组合框架:
  - 滚动 252 天窗口估计 mu(年化收益) 与 Sigma(协方差)
  - 三种效用: MV(均值方差切线组合) / MSV(均值半方差) / CVaR(预留)
  - 波动袖套(volatility sleeve): 组合预测波动 > 目标(18%) 时整体缩放, 余下转货币
  - 三种方案: 静态(30天再平衡) / 动态(波动反馈调风险厌恶lambda) / Blend(50/50)

诚实声明(见批判性复现报告):
  - 原视频未公开 lambda / h 的精确启发式公式(黑盒), 动态方案用"波动反馈"合理近似
  - MSV/CVaR/袖套按论文公开框架实现, 非原 DRL(PPO) 方法
  - 本脚本用于验证"框架能否为己所用", 不代表原文 DRL 效果

数据: etf_daily 表(真实ETF价格, 无adj_factor -> pct_chg/100 作日收益)
成本: narrow=单边20bp(视频口径) / wide=平台真实(佣金+滑点+折溢价近似, 单边~30bp)
"""
import sys, os, argparse, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_monthly_rebalance import get_conn

INITIAL_CAPITAL = 100000.0
RF_ANNUAL = 0.02          # 无风险利率(货币/国债近似), 用于超额收益
VOL_TARGET = 0.18         # 波动袖套目标(年化)
WINDOW = 252              # 滚动估计窗口(交易日)
REBAL_DAYS = 21           # 月频调仓(近似30天再平衡)
MONEY_CODE = "511990.SH"  # 货币基金(袖套余量/全弱避险)

# ── 候选池(仅含 etf_daily 表真实存在的代码, 见数据探查)────────────
# 行业/主题池(对齐视频"12只行业/主题ETF+创业板+黄金"风格, 真实存在)
INDUSTRY_POOL = [
    "512880.SH","512660.SH","512010.SH","159928.SZ","512690.SH","512760.SH",
    "515030.SH","515050.SH","515790.SH","159766.SZ","159915.SZ","518880.SH",
    MONEY_CODE,
]
# 扩展池(全29只, 反过拟合: 避免精选行业造成的生存者偏差)
EXPANDED_POOL = [
    "159766.SZ","159901.SZ","159903.SZ","159915.SZ","159928.SZ","159949.SZ",
    "510050.SH","510210.SH","510300.SH","510330.SH","510500.SH","510880.SH",
    "511010.SH","511260.SH","511990.SH","512010.SH","512100.SH","512660.SH",
    "512690.SH","512760.SH","512880.SH","515030.SH","515050.SH","515790.SH",
    "515800.SH","518880.SH","563300.SH","588000.SH","588190.SH",
]


def load_returns(ts_codes, start, end):
    """读 etf_daily 的 pct_chg 作日收益, 返回对齐的 DataFrame(交易日×代码) 与上市日"""
    con = get_conn()
    first_days = {}
    frames = []
    for code in ts_codes:
        df = pd.read_sql_query(
            "SELECT trade_date, pct_chg FROM etf_daily WHERE ts_code=? "
            "AND trade_date>=? AND trade_date<=? ORDER BY trade_date",
            con, params=(code, start, end))
        if len(df) == 0:
            continue
        first_days[code] = df["trade_date"].iloc[0]
        r = df.copy()
        r["ret"] = r["pct_chg"] / 100.0
        r = r.set_index("trade_date")[["ret"]].rename(columns={"ret": code})
        frames.append(r)
    con.close()
    if not frames:
        return pd.DataFrame(), {}
    R = pd.concat(frames, axis=1).sort_index()
    R = R.ffill().fillna(0.0)   # 缺失日按0收益(ETF极少停牌)
    return R, first_days


def estimate_mu_sigma(R_window, utility):
    """滚动窗口估计 mu(年化) 与 Sigma(年化协方差)
    utility='mv'  -> 标准协方差
    utility='msv' -> 半协方差(下行, 仅 r<mean 计入)
    """
    mu_daily = R_window.mean().values
    cov_daily = R_window.cov().values
    if utility == "msv":
        # 半协方差: 仅低于各自均值的偏差计入
        below = R_window.sub(R_window.mean()) < 0
        semi = (R_window.where(below, 0.0)).cov().values
        cov_daily = (cov_daily + semi) / 2.0  # 对称化
    # Ledoit-Wolf 风格 shrinkage, 防奇异
    n = cov_daily.shape[0]
    shrink = 0.1
    cov_daily = (1 - shrink) * cov_daily + shrink * np.eye(n) * np.mean(np.diag(cov_daily))
    mu = mu_daily * 252.0
    sigma = cov_daily * 252.0
    return mu, sigma


def tangency_weights(mu, sigma, rf=RF_ANNUAL):
    """标准切线组合权重 w = Sigma^-1 (mu-rf) / 1^T Sigma^-1 (mu-rf)
    返回原始未归一化权重(可负, 允许空仓/杠杆近似)
    """
    excess = mu - rf
    try:
        inv = np.linalg.pinv(sigma)
    except Exception:
        return np.zeros_like(mu)
    raw = inv @ excess
    denom = np.sum(raw)
    if abs(denom) < 1e-12:
        return np.zeros_like(mu)
    w = raw / denom
    return w


def apply_sleeve(w, sigma, codes, vol_target=VOL_TARGET, money_idx=None):
    """波动袖套: 组合预测波动 > 目标则整体缩放, 余量转货币
    """
    port_vol = np.sqrt(max(w @ sigma @ w, 0.0))
    if port_vol > vol_target and port_vol > 1e-9:
        scale = vol_target / port_vol
        w = w * scale
    if money_idx is not None:
        w[money_idx] += max(0.0, 1.0 - np.sum(w))
    # 非负约束(不允许做空, 简化)
    w = np.clip(w, 0.0, 1.0)
    s = np.sum(w)
    if s > 1e-9:
        w = w / s
    return w


def run_backtest(R, codes, first_days, utility, mode, cost_one_way, use_sleeve=True,
                 start=None, end=None):
    """主回测
    mode: 'static' 月频固定lambda / 'dynamic' 波动反馈lambda / 'blend' 50/50
    """
    dates = R.index.tolist()
    N = len(codes)
    money_idx = codes.index(MONEY_CODE) if MONEY_CODE in codes else None
    w_static_hist, w_dyn_hist = [], []
    nav = pd.Series(index=dates, dtype=float)
    nav.iloc[0] = 1.0
    w = np.zeros(N)
    if money_idx is not None:
        w[money_idx] = 1.0
    last_rebal = -REBAL_DAYS - 1
    turnovers = []
    for i in range(1, len(dates)):
        # 日收益
        r_day = R.iloc[i].values
        port_ret = float(w @ r_day)
        nav.iloc[i] = nav.iloc[i-1] * (1.0 + port_ret)
        # 调仓判断
        days_since = i - last_rebal
        if days_since >= REBAL_DAYS and i >= WINDOW:
            win = R.iloc[i-WINDOW:i]
            mu, sigma = estimate_mu_sigma(win, utility if utility != "gmv" else "mv")
            if utility == "gmv":   # 最小方差组合: 无收益预测, 只做风险配置
                mu = np.zeros_like(mu)
            # 静态权重
            w_s = tangency_weights(mu, sigma)
            if use_sleeve:
                w_s = apply_sleeve(w_s, sigma, codes, money_idx=money_idx)
            # 动态权重: 风险预算 beta (波动反馈)
            # 修正: 原 lam 标量缩放 mu 会被归一化抵消, dynamic==static;
            # 改为 beta 控制风险资产总权重(波动高->beta低->多持货币)
            vol_recent = np.sqrt(max(w @ sigma @ w, 0.0)) if np.sum(w) > 0 else VOL_TARGET
            beta = 0.5 if vol_recent > VOL_TARGET else 1.0
            w_d = w_s.copy()
            if money_idx is not None:
                risky = w_s.copy(); risky[money_idx] = 0.0
                w_d = risky * beta
                w_d[money_idx] = max(0.0, 1.0 - beta * np.sum(risky))
            else:
                w_d = w_s * beta
            if use_sleeve:
                w_d = apply_sleeve(w_d, sigma, codes, money_idx=money_idx)
            if mode == "static":
                w_new = w_s
            elif mode == "dynamic":
                w_new = w_d
            else:  # blend
                w_new = 0.5 * w_s + 0.5 * w_d
                if use_sleeve:
                    w_new = apply_sleeve(w_new, sigma, codes, money_idx=money_idx)
            # 上市日闸门: 未上市ETF权重归零
            for j, c in enumerate(codes):
                if first_days.get(c, "99999999") > dates[i]:
                    w_new[j] = 0.0
            s = np.sum(w_new)
            if s > 1e-9:
                w_new = w_new / s
            # 成本(换手)
            turnover = float(np.sum(np.abs(w_new - w)))
            turnovers.append(turnover)
            nav.iloc[i] *= (1.0 - turnover * cost_one_way)
            w = w_new
            w_static_hist.append(w_s.copy())
            w_dyn_hist.append(w_d.copy())
            last_rebal = i
    return nav, turnovers


def metrics(nav):
    nav = nav.dropna()
    rets = nav.pct_change().dropna()
    n_days = len(nav)
    years = n_days / 252.0
    total = nav.iloc[-1] / nav.iloc[0] - 1.0
    ann = (nav.iloc[-1] / nav.iloc[0]) ** (1.0 / max(years, 1e-9)) - 1.0
    vol = rets.std() * np.sqrt(252.0)
    sharpe = (ann - RF_ANNUAL) / vol if vol > 1e-9 else 0.0
    peak = nav.cummax()
    mdd = (nav - peak) / peak
    maxdd = mdd.min()
    return dict(total=total, ann=ann, vol=vol, sharpe=sharpe, maxdd=maxdd, years=years,
                nav=nav, rets=rets)


def yearly(nav):
    nav = nav.dropna()
    g = nav.groupby(lambda d: d[:4])
    out = {}
    for y, sub in g:
        t = sub.iloc[-1] / sub.iloc[0] - 1.0
        out[y] = t
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("start", nargs="?", default="20210601")
    p.add_argument("end", nargs="?", default="20260731")
    p.add_argument("--pool", choices=["industry", "expanded"], default="industry")
    p.add_argument("--utility", choices=["mv", "msv", "gmv"], default="mv")
    p.add_argument("--mode", choices=["static", "dynamic", "blend"], default="blend")
    p.add_argument("--cost", choices=["narrow", "wide"], default="narrow")
    p.add_argument("--no-sleeve", action="store_true", help="消融波动袖套")
    p.add_argument("--walkforward", action="store_true", help="分段OOS报告")
    p.add_argument("--bench", default="510300.SH", help="基准ETF代码")
    args = p.parse_args()

    codes = INDUSTRY_POOL if args.pool == "industry" else EXPANDED_POOL
    cost_one_way = 0.002 if args.cost == "narrow" else 0.003
    use_sleeve = not args.no_sleeve

    R, first_days = load_returns(codes, args.start, args.end)
    if R.empty:
        print("NO DATA"); return
    codes = list(R.columns)   # 实际成功加载的代码(维度与收益矩阵一致)
    print("="*60)
    print("Smart Tangency Portfolio 复现")
    print("  池=%d只(%s) | 区间 %s~%s | 效用=%s | 方案=%s" %
          (len(codes), args.pool, args.start, args.end, args.utility, args.mode))
    print("  成本=%s(单边%.1fbp) | 袖套=%s | 窗口=%d天" %
          (args.cost, cost_one_way*1e4, "开" if use_sleeve else "关", WINDOW))
    print("  交易日 %d 天" % len(R))

    nav, turnovers = run_backtest(R, codes, first_days, args.utility, args.mode,
                                  cost_one_way, use_sleeve)
    m = metrics(nav)
    avg_turn = np.mean(turnovers) if turnovers else 0.0
    ann_turn = avg_turn * (252.0 / REBAL_DAYS)
    print("-"*60)
    print("[%s] 累计 %.2f%% | 年化 %.2f%% | 波动 %.2f%% | 夏普 %.2f | 回撤 %.2f%%" %
          (args.mode, m["total"]*100, m["ann"]*100, m["vol"]*100, m["sharpe"], m["maxdd"]*100))
    print("  年化换手(单边)~%.2f次 | 平均单次换手 %.2f%%" % (ann_turn, avg_turn*100))
    yd = yearly(nav)
    print("  年度: " + " ".join("%s:%+.1f%%" % (y, v*100) for y, v in sorted(yd.items())))

    # 基准
    Rb, _ = load_returns([args.bench], args.start, args.end)
    if not Rb.empty:
        nb = (1.0 + Rb.iloc[:, 0]).cumprod()
        mb = metrics(nb)
        print("  基准(%s): 累计 %.2f%% 年化 %.2f%% 夏普 %.2f%%" %
              (args.bench, mb["total"]*100, mb["ann"]*100, mb["sharpe"]))

    if args.walkforward:
        print("-"*60)
        print("[walk-forward] 按年切段, 每段用前段估参, 后段测试(滚动窗口本身即OOS):")
        # 简化: 打印各年度Sharpe
        for y, sub in nav.groupby(lambda d: d[:4]):
            yy = sub.pct_change().dropna()
            if len(yy) > 20:
                sa = yy.mean()*252/np.sqrt(yy.var()*252) if yy.var() > 0 else 0
                print("    %s 区间夏普 %.2f" % (y, sa))


if __name__ == "__main__":
    main()
