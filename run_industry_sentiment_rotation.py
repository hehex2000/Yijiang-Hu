# -*- coding: utf-8 -*-
"""
行业情绪轮动（真实申万一级行业指数版，对标 BV1CZK369EYB 的逻辑复刻）
=====================================================================
数据来源：真实申万一级行业指数（sw_l1_index_daily，Tushare index_daily 拉取）。
  - 基准      = 31 个申万一级行业指数等权（对标视频"28行业等权"）
  - 策略收益  = 每期持有选中 top3 行业指数（等权）
  - 因子信号源：
      F1 站上20日线占比   -> 成分股层面（breadth，指数无此字段；她的逻辑也如此）
      F2 行业RSI(14)      -> 真实行业指数
      F3 换手率强度       -> 真实行业指数成交额/自身20日均
      F4 涨跌停情绪差     -> 成分股层面（指数无此字段；她的逻辑也如此）
      F5 成交额占比       -> 真实行业指数成交额/全市场成交额
  - 综合分 = 0.35*mean(F1,F4) + 0.30*mean(F2,F3) + 0.35*F5（各因子先横截面百分位0-100）
  - 一票否决：综合分 >= 85 直接剔除（主动放弃过热行业）
  - 选股：非否决行业中取 top3，等权持有；每 10 个交易日调仓；成本按换手率计
区间：2016-01-04 ~ 2026-08-07

说明：因子公式/权重/阈值/veto/top3/调仓频率均为"复现逻辑"的假设实现，非视频原版数字。
我们用自己的真实行业指数数据，不追求与视频收益对得上。

== 已修复的回测正确性问题（2026-08-14 代码审查）==
  P0-1 未来函数：调仓日当天不再用当日信号赚当日收益。改用 bisect_left，
        使调仓日 d 当天仍持上一期持仓，次日(rd+1)才切换到新选中行业。
  P0-2 空仓期稀释：NAV 与基准只从 START_STRAT 起算，不含 2015 空仓/基准牛市期，
        CAGR 与 excess 不再被年限拉长而失真。
  P1-3 pct_chg 幻影收益：pct_chg 改为由填充后的 close 重算（pct_change），
        不再对收益率本身 ffill（停牌日不再被复制前一天涨幅）。
  P1-4 成本随换手率：仅首次建仓扣全额 COST；之后按实际换手比例 (changed/(len_h+len_prev)) 计费。
  P1-5 F4 除零：除 cnt 前 replace(0, NaN)，避免 inf/0 传播到综合分。
  P1-6 涨跌停阈值：按板块区分——创业板(30)/科创板(68)=20%，ST/*ST=5%，其余主板=10%
        （注意：ST 状态在 stock_basic.name，不在 ts_code 前缀，故需用 name 判断）。
"""
import sqlite3, bisect, numpy as np, pandas as pd
from datetime import datetime
import config

DB = config.DATA["local_db_path"]
con = sqlite3.connect(DB)
LOOK = "20150101"
START_STRAT = "20160104"
COST = 0.002

# ---------- 1. 真实申万一级行业指数（键统一用 8 位 sw_code）----------
idx = pd.read_sql("SELECT ts_code, trade_date, close, pct_chg, amount FROM sw_l1_index_daily", con)
idx = idx.drop_duplicates(subset=["ts_code", "trade_date"])
idx["trade_date"] = idx["trade_date"].astype(str)
idx["sw_code"] = idx["ts_code"].str[:6]
m = pd.read_sql("SELECT DISTINCT sw_code, sw_name FROM stock_ind_sw_l1", con)
m["sw_code"] = m["sw_code"].str[:6]
code2name = {r.sw_code: r.sw_name for r in m.itertuples()}

# P1-3 修复：由填充后的 close 重算 pct_chg，避免对收益率 ffill 制造幻影收益
idx_close = idx.pivot(index="trade_date", columns="sw_code", values="close").sort_index().ffill()
idxp = idx_close.pct_change() * 100.0
idxp = idxp.fillna(0.0)
idxa = idx.pivot(index="trade_date", columns="sw_code", values="amount").sort_index().ffill().fillna(0.0)
dates = idxp.index.tolist()
sectors = idxp.columns.tolist()
print(f"[init] 真实行业指数数={len(sectors)}, 交易日={len(dates)}")

bench_ret = idxp.mean(axis=1)  # 31行业等权日收益%

# ---------- 2. 指数级因子 F2 / F3 / F5 ----------
def pct_rank(mat):
    return mat.rank(axis=1, pct=True) * 100.0

def rsi(series, n=14):
    s = series.astype(float); d = s.diff()
    g = d.clip(lower=0); l = -d.clip(upper=0)
    ag = g.rolling(n).mean(); al = l.rolling(n).mean()
    rs = ag / al.replace(0, np.nan)
    return 100 - 100/(1+rs)

F2 = pct_rank(pd.DataFrame({c: rsi(idxp[c]) for c in sectors}))
amt_ratio = idxa / idxa.rolling(20).mean().replace(0, np.nan)
F3 = pct_rank(amt_ratio)
tot = idxa.sum(axis=1)
F5 = pct_rank(idxa.div(tot, axis=0))

# ---------- 3. 成分股级因子 F1 / F4（含 P1-6 板块涨跌停阈值）----------
mp = pd.read_sql("SELECT ts_code, sw_code FROM stock_ind_sw_l1", con)
mp["sw_code"] = mp["sw_code"].str[:6]
daily = pd.read_sql(f"SELECT ts_code, trade_date, close, pct_chg FROM daily WHERE trade_date>='{LOOK}'", con)
daily["trade_date"] = daily["trade_date"].astype(str)
daily = daily.merge(mp, on="ts_code", how="inner").sort_values(["sw_code", "trade_date"])
# P1-6 修复：涨跌停阈值按板块；ST/*ST 在 name 字段（非 ts_code 前缀）
nm = pd.read_sql("SELECT ts_code, name FROM stock_basic", con)
daily = daily.merge(nm, on="ts_code", how="left")
daily["is_st"] = daily["name"].fillna("").str.contains("ST")
daily["pre"] = daily["ts_code"].str[:2]
daily["limit"] = np.where(daily["is_st"], 5.0,
                  np.where(daily["pre"].isin(["30", "68"]), 20.0, 10.0))
daily["up"] = (daily["pct_chg"] >= daily["limit"] - 0.5).astype(float)
daily["dn"] = (daily["pct_chg"] <= -(daily["limit"] - 0.5)).astype(float)
daily["ma20"] = daily.groupby("sw_code")["close"].rolling(20).mean().reset_index(level=0, drop=True)
daily["above"] = (daily["close"] > daily["ma20"]).astype(float)
g = daily.groupby(["trade_date", "sw_code"])
F1 = pct_rank(g["above"].mean().unstack("sw_code").reindex(dates))
cnt = g.size().unstack("sw_code").reindex(dates)
upc = g["up"].sum().unstack("sw_code").reindex(dates)
dnc = g["dn"].sum().unstack("sw_code").reindex(dates)
# P1-5 修复：除 cnt 前 replace(0, NaN)，避免 inf/0 传播
F4 = pct_rank(((upc - dnc) / cnt.replace(0, np.nan)))

# ---------- 4. 综合分（veto 前，固定）----------
comp_pre = (0.35*(F1+F4)/2 + 0.30*(F2+F3)/2 + 0.35*F5)

rb_dates = [d for d in dates if d >= START_STRAT][::10]
years = pd.to_datetime(pd.Series(dates), format="%Y%m%d").dt.year.values
yr_of = {d: y for d, y in zip(dates, years)}

def metrics(nav):
    ns = nav.values
    tot = ns[-1]/ns[0] - 1
    d0 = datetime.strptime(nav.index[0], "%Y%m%d"); d1 = datetime.strptime(nav.index[-1], "%Y%m%d")
    yrs = (d1 - d0).days/365.25
    cagr = (ns[-1]/ns[0])**(1/yrs) - 1 if yrs > 0 else 0
    peak = np.maximum.accumulate(ns); mdd = (ns/peak - 1).min()
    return tot, cagr, mdd

# ===================== 以上为共享管道，封装供敏感性脚本复用 =====================
def build_pipeline():
    """返回所有与 VETO 无关的预计算对象。"""
    return dict(idxp=idxp, idxa=idxa, dates=dates, sectors=sectors,
                code2name=code2name, bench_ret=bench_ret,
                F1=F1, F2=F2, F3=F3, F4=F4, comp_pre=comp_pre,
                rb_dates=rb_dates, START_STRAT=START_STRAT, COST=COST)

def run_backtest(VETO, pipe=None):
    """给定否决阈值，返回指标 dict。内含 P0-1/P0-2/P1-4 的修正逻辑。"""
    if pipe is None:
        pipe = build_pipeline()
    comp_pre = pipe["comp_pre"]; idxp = pipe["idxp"]; rb_dates = pipe["rb_dates"]
    dates = pipe["dates"]; bench_ret = pipe["bench_ret"]
    START_STRAT = pipe["START_STRAT"]; COST = pipe["COST"]

    vetoed = (comp_pre >= VETO)
    n_veto = int(vetoed.sum().sum())
    comp = comp_pre.mask(vetoed)
    n_top_removed = 0
    sel = {}
    for rd in rb_dates:
        raw_row = comp_pre.loc[rd].dropna()
        if len(raw_row):
            raw_top = raw_row.sort_values(ascending=False).index[0]
            if bool(vetoed.loc[rd, raw_top]):
                n_top_removed += 1
        row = comp.loc[rd].dropna()
        sel[rd] = row.sort_values(ascending=False).head(3).index.tolist() if len(row) else []

    # P0-1 修复：bisect_left 使调仓日 d 当天仍持上一期，次日(rd+1)才切换新选中行业
    hold_per_date = {}
    for d in dates:
        k = bisect.bisect_left(rb_dates, d) - 1
        hold_per_date[d] = sel[rb_dates[k]] if k >= 0 else []

    # P0-2 修复：净值与基准只从 START_STRAT 起算，不含 2015 空仓/基准牛市期
    strat_dates = [d for d in dates if d >= START_STRAT]
    prev = None; sret = []
    for d in strat_dates:
        h = hold_per_date[d]
        if not h:
            sret.append(0.0); prev = h; continue
        r = idxp.loc[d, h].mean() / 100.0
        # P1-4 修复：成本随换手率；首次建仓扣全额，之后按换手比例计费
        if prev is None:
            r -= COST
        elif h != prev:
            changed = len(set(h) ^ set(prev))
            turn = changed / (len(h) + len(prev))
            r -= COST * turn
        sret.append(r); prev = h
    strat_nav = (1 + pd.Series(sret, index=strat_dates)).cumprod()
    bench_nav = (1 + bench_ret.loc[strat_dates] / 100).cumprod()

    t1, c1, m1 = metrics(strat_nav)
    t2, c2, m2 = metrics(bench_nav)
    s_year = pd.Series(sret, index=strat_dates).groupby([yr_of[d] for d in strat_dates]).apply(lambda x: (1+x).prod()-1)
    return dict(tot=t1, cagr=c1, mdd=m1, excess=c1-c2, n_veto=n_veto,
                n_top_removed=n_top_removed, s2018=s_year.get(2018, np.nan),
                s2022=s_year.get(2022, np.nan), s2015=s_year.get(2015, np.nan),
                strat_nav=strat_nav, bench_nav=bench_nav)

if __name__ == "__main__":
    VETO = 85.0
    res = run_backtest(VETO)
    strat_nav = res["strat_nav"]; bench_nav = res["bench_nav"]
    bt, bc, bm = metrics(bench_nav)
    yrs_total = (datetime.strptime(dates[-1], "%Y%m%d") - datetime.strptime(START_STRAT, "%Y%m%d")).days/365.25
    print(f"\n================ 回测结果（真实申万行业指数·逻辑复刻·已修正） ================")
    print(f"区间        : {START_STRAT} ~ {dates[-1]}  ({yrs_total:.1f}年)")
    print(f"策略总收益  : {res['tot']*100:7.2f}%   年化: {res['cagr']*100:6.2f}%   最大回撤: {res['mdd']*100:6.2f}%")
    print(f"基准总收益  : {bt*100:7.2f}%   年化: {bc*100:6.2f}%   最大回撤: {bm*100:6.2f}%")
    print(f"年化超额    : {res['excess']*100:6.2f}pp")
    print(f"一票否决触发: 共 {res['n_veto']} 次（行业-调仓日）")
    print(f"总收益倍数  : 策略 {res['tot']+1:.2f}x / 基准 {bt+1:.2f}x")
    print("============================================================================")

    out = pd.DataFrame({"date": strat_nav.index,
                        "strat_nav": strat_nav.values,
                        "bench_nav": bench_nav.values})
    out.to_csv("ind_sentiment_equity.csv", index=False)
    print("\n[save] ind_sentiment_equity.csv")
    con.close()
