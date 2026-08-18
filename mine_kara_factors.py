# -*- coding: utf-8 -*-
"""
mine_kara_factors.py  ——  Kara因子挖掘引擎【路线B：保留F1/F2，自挖增量因子补到7因子】
=============================================================================
来源：中信建投《逐鹿Alpha》第30期《量价×基本面因子挖掘统一框架》五层框架的
      *本地可落地实现*（GP/LLM 搜索机制以"枚举候选因子 + IC/增量筛选"替代，
      这是谨慎量化人等价做法，且契合用户"先单因子消融再合入"的纪律）。

五层框架映射：
  L1 原始数据  ：日线(价/量/额)、daily_basic(估值/换手/市值)、fina_indicator(质量/成长)、
                 income(研发/盈利)、adj_factor(复权)、namechange(时点ST)。
  L2 算子集    ：ts_*(ret20/60/120)、zscore、safe_div(收益/分母)、yoy(成长)、
                 组合算子(-pe, roe+cur-debt, 等)。
  L3 横截面标准化：每月对 eligible 宇宙做 cross-sectional zscore。
  L4 中性化/组合 ：候选因子做 规模(log_mv)+行业(stock_basic.industry) 中性化后评IC；
                 相对 F1/F2 正交化得"增量IC"，选增量最强5个补足7因子；等权合成分。
  L5 评估       ：逐月 IC/IR、分组单调、top50月度等权回测(含全收益基准)，对照2因子基线。

保留已修的 P0/P1/P2（复权价、全收益基准连锁、时点ST、NAV断层、IPO交易日、
ann_date严格<t、退市NaN、换手率不ffill、夏普减rf）。

输出：mine_ic.csv(候选因子IC/增量) / mine_equity.csv(2因子 vs 7因子净值) /
      mine_groups.csv(7因子分组) 。
=============================================================================
"""
import sqlite3, bisect, datetime
import numpy as np
import pandas as pd
import config

DB = config.DATA["local_db_path"]
START      = "20140101"
START_LOOK = "20120101"
END        = "20260807"
TOPN       = 50
COST       = 0.002
UNIV_MV_MIN = 2e6
LIST_DAYS_MIN = 252
PIT = True
RF_ANN     = 0.025
HS_DIV_YIELD = 0.022   # 合成全收益基准股息率假设

con = sqlite3.connect(DB)


def zscore(x):
    x = x.astype(float)
    m = x.mean(); s = x.std()
    return (x - m) / (s if s and s == s and s != 0 else 1.0)


# ---------- 1. 交易日 & 月末调仓日 ----------
alldates = [r[0] for r in con.execute(
    f"SELECT DISTINCT trade_date FROM daily WHERE trade_date>='{START_LOOK}' ORDER BY trade_date")]
alldates.sort()
date_idx = {d: i for i, d in enumerate(alldates)}
mm_last = {}
for d in alldates:
    mm_last[d[:6]] = d
rebal = sorted(mm_last.values())
rebal = [d for d in rebal if d >= START and d < END]
print(f"[cal] 交易日 {len(alldates)}；月末调仓日 {len(rebal)} 个 ({rebal[0]}~{rebal[-1]})")


# ---------- 2. 价格/量/换手/估值 矩阵 ----------
print("[load] daily / daily_basic / adj_factor ...")
price = pd.read_sql(
    f"SELECT ts_code, trade_date, close, amount FROM daily WHERE trade_date>='{START_LOOK}'", con)
price["trade_date"] = price["trade_date"].astype(str)
close_p = price.pivot(index="trade_date", columns="ts_code", values="close").reindex(alldates).sort_index()
amount_p = price.pivot(index="trade_date", columns="ts_code", values="amount").reindex(alldates).sort_index()
del price

db = pd.read_sql(
    f"SELECT ts_code, trade_date, turnover_rate, total_mv, pe_ttm, pb, ps_ttm, dv_ratio, dv_ttm, circ_mv "
    f"FROM daily_basic WHERE trade_date>='{START_LOOK}'", con)
db["trade_date"] = db["trade_date"].astype(str)
turn_p = db.pivot(index="trade_date", columns="ts_code", values="turnover_rate").reindex(alldates).sort_index()
mv_p   = db.pivot(index="trade_date", columns="ts_code", values="total_mv").reindex(alldates).sort_index()
pe_p   = db.pivot(index="trade_date", columns="ts_code", values="pe_ttm").reindex(alldates).sort_index()
pb_p   = db.pivot(index="trade_date", columns="ts_code", values="pb").reindex(alldates).sort_index()
ps_p   = db.pivot(index="trade_date", columns="ts_code", values="ps_ttm").reindex(alldates).sort_index()
dv_p   = db.pivot(index="trade_date", columns="ts_code", values="dv_ratio").reindex(alldates).sort_index()
del db

adj = pd.read_sql(
    f"SELECT ts_code, trade_date, adj_factor FROM adj_factor WHERE trade_date>='{START_LOOK}'", con)
adj["trade_date"] = adj["trade_date"].astype(str)
adj_factor_p = adj.pivot(index="trade_date", columns="ts_code", values="adj_factor").reindex(alldates).sort_index().ffill().fillna(1.0)
del adj
adj_close_f   = (close_p.ffill() * adj_factor_p)
adj_close_raw = (close_p * adj_factor_p)

# 动量/反转窗口（复权价，信号腿）
ret5   = adj_close_f / adj_close_f.shift(5)   - 1
ret10  = adj_close_f / adj_close_f.shift(10)  - 1
ret20  = adj_close_f / adj_close_f.shift(20)  - 1
ret60  = adj_close_f / adj_close_f.shift(60)  - 1
ret120 = adj_close_f / adj_close_f.shift(120) - 1
amt20 = amount_p.rolling(20).mean()
turnstd60 = turn_p.rolling(60, min_periods=20).std()
logmv = np.log(mv_p.replace(0, np.nan))
print(f"[load] 矩阵就绪 close{close_p.shape} mv{mv_p.shape} pe{pe_p.shape}")


# ---------- 3. point-in-time 财报映射 ----------
def build_pit_map(sql, valcol, denom=None):
    df = pd.read_sql(sql, con)
    df["ann"] = df["ann_date"].astype(str)
    df[valcol] = pd.to_numeric(df[valcol], errors="coerce")
    if denom is not None:
        df[denom] = pd.to_numeric(df[denom], errors="coerce")
        df["v"] = df[valcol] / df[denom]
    else:
        df["v"] = df[valcol]
    df = df.dropna(subset=["v", "ann"])
    df = df[df["v"] == df["v"]]
    df = df.sort_values(["ts_code", "ann"]).drop_duplicates(["ts_code", "ann"], keep="last")
    out = {}
    for code, g in df.groupby("ts_code"):
        out[code] = (g["ann"].values, g["v"].values)
    return out

print("[load] fina_indicator / income  point-in-time ...")
pit_roe  = build_pit_map("SELECT ts_code, ann_date, roe FROM fina_indicator", "roe")
pit_cur  = build_pit_map("SELECT ts_code, ann_date, current_ratio FROM fina_indicator", "current_ratio")
pit_debt = build_pit_map("SELECT ts_code, ann_date, debt_to_assets FROM fina_indicator", "debt_to_assets")
pit_opy  = build_pit_map("SELECT ts_code, ann_date, op_yoy FROM fina_indicator", "op_yoy")
pit_npy  = build_pit_map("SELECT ts_code, ann_date, netprofit_yoy FROM fina_indicator", "netprofit_yoy")
pit_ory  = build_pit_map("SELECT ts_code, ann_date, or_yoy FROM fina_indicator", "or_yoy")
pit_qsy  = build_pit_map("SELECT ts_code, ann_date, q_sales_yoy FROM fina_indicator", "q_sales_yoy")
pit_gpm  = build_pit_map("SELECT ts_code, ann_date, grossprofit_margin FROM fina_indicator", "grossprofit_margin")
pit_atn  = build_pit_map("SELECT ts_code, ann_date, assets_turn FROM fina_indicator", "assets_turn")
pit_rd   = build_pit_map("SELECT ts_code, ann_date, rd_exp, total_revenue FROM income", "rd_exp", "total_revenue")
pit_cfq  = build_pit_map(
    "SELECT ts_code, ann_date, ocfps, eps FROM fina_indicator WHERE eps IS NOT NULL AND abs(eps)>1e-6", "ocfps", "eps")
pit_opm  = build_pit_map("SELECT ts_code, ann_date, operate_profit, total_revenue FROM income", "operate_profit", "total_revenue")
pit_netm = build_pit_map("SELECT ts_code, ann_date, n_income, total_revenue FROM income", "n_income", "total_revenue")
print(f"       roe{len(pit_roe)} cfq{len(pit_cfq)} rd{len(pit_rd)} opm{len(pit_opm)} netm{len(pit_netm)}")

listd = {r[0]: r[1] for r in con.execute("SELECT ts_code, list_date FROM stock_basic WHERE list_date IS NOT NULL")}
ind_map = {r[0]: r[1] for r in con.execute("SELECT ts_code, industry FROM stock_basic WHERE industry IS NOT NULL")}


# ---------- 4. 时点 ST（同 run_kara_factors） ----------
def load_st_intervals(con):
    cur = con.cursor()
    try:
        n = cur.execute("SELECT COUNT(*) FROM namechange").fetchone()[0]
    except Exception:
        n = 0
    if n > 0:
        df = pd.read_sql("SELECT ts_code, name, start_date, end_date FROM namechange", con)
    else:
        try:
            import tushare as ts, config_tushare
            pro = ts.pro_api(config_tushare.TUSHARE_TOKEN)
            frames, off = [], 0
            while True:
                d = pro.namechange(limit=10000, offset=off)
                if d is None or len(d) == 0:
                    break
                frames.append(d); off += 10000
                if len(d) < 10000:
                    break
            df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
                columns=["ts_code", "name", "start_date", "end_date"])
            df.to_sql("namechange", con, if_exists="replace", index=False)
        except Exception as e:
            print(f"[namechange] 拉取失败({e})，回退当前名ST")
            return None
    df = df[df["name"].astype(str).str.contains("ST", na=False)].copy()
    out = {}
    for code, g in df.groupby("ts_code"):
        ivs = []
        for _, r in g.iterrows():
            s = str(r["start_date"]) if r["start_date"] == r["start_date"] and r["start_date"] is not None else "00000000"
            e = str(r["end_date"]) if r["end_date"] == r["end_date"] and r["end_date"] is not None else "99999999"
            ivs.append((s, e))
        out[code] = sorted(ivs)
    return out

st_intervals = load_st_intervals(con)
st_by_date = {}
if st_intervals:
    for code, ivs in st_intervals.items():
        for (s, e) in ivs:
            lo = bisect.bisect_left(rebal, s)
            hi = bisect.bisect_left(rebal, e)
            for ti in range(lo, hi):
                st_by_date.setdefault(rebal[ti], set()).add(code)
    print(f"[filter] 时点ST: {len(st_intervals)} 只；覆盖 {len(st_by_date)} 调仓日")
else:
    st_by_date = None
st_codes = set(r[0] for r in con.execute("SELECT ts_code FROM stock_basic WHERE name LIKE '%ST%'"))
print(f"[filter] 上市日 {len(listd)}；当前名ST回退 {len(st_codes)}")


def _list_tidx(dstr):
    i = bisect.bisect_right(alldates, dstr) - 1
    return i if i >= 0 else -999
list_tidx = {c: _list_tidx(d) for c, d in listd.items()}


def pit_get(m, code, t):
    if code not in m:
        return np.nan
    anns, vals = m[code]
    i = bisect.bisect_left(anns, t) - 1
    return vals[i] if i >= 0 else np.nan


# ---------- 5. 候选因子库（五层框架·逐鹿Alpha风格，~30个） ----------
# 每个 builder 接收 feat(dict of pd.Series，已对齐 elig)，返回原始得分 Series（可含NaN）。
def CANDIDATES():
    return {
        # ---- 价值 (Value) ----
        "val_ep":      lambda f: -f["pe"],
        "val_bp":      lambda f: -f["pb"],
        "val_sp":      lambda f: -f["ps"],
        "val_div":     lambda f:  f["dv"],
        "val_combo":   lambda f: -f["pe"] - f["pb"] - f["ps"],
        # ---- 质量 (Quality) ----
        "qual_roe":    lambda f:  f["roe"],
        "qual_combo":  lambda f:  f["roe"] + f["cur"] - f["debt"],
        "qual_gpm":    lambda f:  f["gpm"],
        "qual_aturn":  lambda f:  f["aturn"],
        "cashqual":    lambda f:  f["cfq"],
        "lowlev":      lambda f: -f["debt"],
        # ---- 成长 (Growth) ----
        "growth_op":   lambda f:  f["opy"],
        "growth_rev":  lambda f:  f["ory"],
        "growth_combo":lambda f:  f["opy"] + f["ory"] + f["qsy"],
        "earn_yoy":    lambda f:  f["npy"],
        # ---- 动量 (Momentum) ----
        "mom20":       lambda f:  f["ret20"],
        "mom60":       lambda f:  f["ret60"],
        "mom120":      lambda f:  f["ret120"],
        # ---- 反转 (Reversal) ----
        "rev5":        lambda f: -f["ret5"],
        "rev10":       lambda f: -f["ret10"],
        "rev60":       lambda f: -f["ret60"],
        "rev_combo":   lambda f: -f["ret5"] - f["ret20"],
        # ---- 规模 (Size) ----
        "size_small":  lambda f: -f["logmv"],
        # ---- 流动性 (Liquidity) ----
        "liq_turn":    lambda f: -f["turn"],
        "liq_amt":     lambda f: -f["amt20"],
        # ---- 交叉/复合 (Cross) ----
        "qval":        lambda f:  f["roe"] - f["pb"],
        "gpm_value":   lambda f:  f["gpm"] - f["ps"],
        "div_size":    lambda f:  f["dv"] - f["logmv"],
        "opmargin":    lambda f:  f["opm"],
        "netmargin":   lambda f:  f["netm"],
        "qual_growth": lambda f:  f["roe"] + f["ory"],
    }


# ---------- 6. 逐月计算 F1/F2 + 候选因子 ----------
print("[calc] 逐月 F1/F2 + 候选因子 ...")
sel_hist = {}
comp_dates = []
comp2_store = {}      # 基线: z(F1)+z(F2)
comp7_store = {}      # 7因子(选完候选后填充)
fwd_ret = {}
f1_store = {}; f2_store = {}
cand_raw = {name: {} for name in CANDIDATES()}   # 原始得分(未z)
cand_z   = {name: {} for name in CANDIDATES()}   # 截面z得分

for k, t in enumerate(rebal):
    if k + 1 >= len(rebal):
        break
    t_next = rebal[k + 1]
    idi = date_idx[t]
    st_set = st_by_date.get(t, set()) if st_by_date is not None else st_codes
    elig = [c for c in close_p.columns
            if c in list_tidx and list_tidx[c] <= idi - LIST_DAYS_MIN and c not in st_set]
    mv_row = mv_p.loc[t] if t in mv_p.index else None
    elig = [c for c in elig if mv_row is not None and (c in mv_row.index) and (mv_row.get(c, 0) or 0) >= UNIV_MV_MIN]
    if len(elig) < TOPN:
        continue

    # 基础特征
    feat = {}
    feat["pe"] = pd.Series({c: (pe_p.at[t, c] if t in pe_p.index and c in pe_p.columns else np.nan) for c in elig})
    feat["pb"] = pd.Series({c: (pb_p.at[t, c] if t in pb_p.index and c in pb_p.columns else np.nan) for c in elig})
    feat["ps"] = pd.Series({c: (ps_p.at[t, c] if t in ps_p.index and c in ps_p.columns else np.nan) for c in elig})
    feat["dv"] = pd.Series({c: (dv_p.at[t, c] if t in dv_p.index and c in dv_p.columns else np.nan) for c in elig})
    feat["logmv"] = pd.Series({c: (logmv.at[t, c] if t in logmv.index and c in logmv.columns else np.nan) for c in elig})
    feat["turn"] = pd.Series({c: (turn_p.at[t, c] if t in turn_p.index and c in turn_p.columns else np.nan) for c in elig})
    feat["amt20"] = pd.Series({c: (amt20.at[t, c] if t in amt20.index and c in amt20.columns else np.nan) for c in elig})
    feat["turnstd60"] = pd.Series({c: (turnstd60.at[t, c] if t in turnstd60.index and c in turnstd60.columns else np.nan) for c in elig})
    for w in ["ret5","ret10","ret20","ret60","ret120"]:
        feat[w] = pd.Series({c: (globals()[w].at[t, c] if t in globals()[w].index and c in globals()[w].columns else np.nan) for c in elig})
    feat["roe"]  = pd.Series({c: pit_get(pit_roe, c, t) for c in elig})
    feat["cur"]  = pd.Series({c: pit_get(pit_cur, c, t) for c in elig})
    feat["debt"] = pd.Series({c: pit_get(pit_debt, c, t) for c in elig})
    feat["opy"]  = pd.Series({c: pit_get(pit_opy, c, t) for c in elig})
    feat["npy"]  = pd.Series({c: pit_get(pit_npy, c, t) for c in elig})
    feat["ory"]  = pd.Series({c: pit_get(pit_ory, c, t) for c in elig})
    feat["qsy"]  = pd.Series({c: pit_get(pit_qsy, c, t) for c in elig})
    feat["gpm"]  = pd.Series({c: pit_get(pit_gpm, c, t) for c in elig})
    feat["aturn"]= pd.Series({c: pit_get(pit_atn, c, t) for c in elig})
    feat["rd"]   = pd.Series({c: pit_get(pit_rd, c, t) for c in elig})
    feat["cfq"]  = pd.Series({c: pit_get(pit_cfq, c, t) for c in elig})
    feat["opm"]  = pd.Series({c: pit_get(pit_opm, c, t) for c in elig})
    feat["netm"] = pd.Series({c: pit_get(pit_netm, c, t) for c in elig})

    # F1 / F2 （与已验证基线完全一致）
    z_rd = zscore(feat["rd"]); z_amt = zscore(feat["amt20"])
    z_ret = zscore(-feat["ret20"]); z_cfq = zscore(feat["cfq"]); z_tst = zscore(-feat["turnstd60"])
    F1 = z_rd - z_amt
    F2 = z_ret + z_cfq + z_tst
    both = F1.notna() & F2.notna()
    comp2 = pd.Series(np.nan, index=elig)
    comp2[both] = zscore(F1[both]) + zscore(F2[both])
    if comp2.dropna().shape[0] < TOPN:
        continue

    # 候选因子原始 + z
    for name, b in CANDIDATES().items():
        raw = b(feat)
        zv = zscore(raw)
        cand_raw[name][t] = raw
        cand_z[name][t] = zv

    # 前向收益（P2-6 非ffill复权价）
    fwd = {}
    for c in elig:
        c0 = adj_close_raw.at[t, c] if t in adj_close_raw.index else np.nan
        c1 = adj_close_raw.at[t_next, c] if t_next in adj_close_raw.index else np.nan
        fwd[c] = (c1 / c0 - 1) if (c0 == c0 and c0 and c1 == c1 and c1) else np.nan

    sel_hist[t] = None  # 占位，选股在筛完候选后做
    comp_dates.append(t)
    comp2_store[t] = comp2.dropna()
    fwd_ret[t] = pd.Series(fwd)
    f1_store[t] = F1.dropna()
    f2_store[t] = F2.dropna()

print(f"[calc] 有效选股月份: {len(comp_dates)} 个；候选因子 {len(CANDIDATES())} 个")


# ---------- 7. 中性化工具（框架L4） ----------
def neutralize(score, logmv, ind):
    """score ~ [1, logmv, industry dummies] 的OLS残差（规模+行业中性化）"""
    df = pd.concat([score, logmv, ind], axis=1).dropna()
    df.columns = ["s", "lm", "ind"]
    if len(df) < 50:
        return score
    X = pd.get_dummies(df["ind"], drop_first=True).astype(float)
    X = pd.DataFrame({"const": 1.0, "lm": df["lm"].values}).join(X)
    X = X.values.astype(float); y = df["s"].values.astype(float)
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
    except Exception:
        return score
    out = score.copy()
    out.loc[df.index] = resid
    return out


# ---------- 8. IC / IR（含原始、中性化、增量） ----------
print("[eval] IC / IR / 增量 ...")
def calc_ic_from_store(store, neutralize_fn=None):
    ics = []
    for t in comp_dates:
        f = store[t]
        r = fwd_ret[t]
        j = pd.concat([f, r], axis=1).dropna()
        j.columns = ["f", "r"]
        if len(j) < 30:
            continue
        if neutralize_fn is not None:
            j["f"] = neutralize_fn(j["f"], j.index)
        ic = j["f"].corr(j["r"], method="spearman")
        if ic == ic:
            ics.append(ic)
    ics = np.array(ics)
    return ics

def neutralizer_factory(t):
    """为某月 t 构造 neutralizer(score, idx)"""
    lm = pd.Series({c: (logmv.at[t, c] if t in logmv.index and c in logmv.columns else np.nan) for c in fwd_ret[t].index})
    ind = pd.Series({c: ind_map.get(c, "NA") for c in fwd_ret[t].index})
    def fn(score, idx):
        s = score.reindex(idx); m = lm.reindex(idx); i = ind.reindex(idx)
        return neutralize(s, m, i)
    return fn

ic_rows = []
# 基线
_f1 = calc_ic_from_store(f1_store); _f2 = calc_ic_from_store(f2_store)
ic_rows.append(dict(factor="F1_冷门研发股", n=len(_f1),
                    ic_raw=_f1.mean(), ir_raw=_f1.mean()/_f1.std() if _f1.std() else np.nan,
                    ic_neu=np.nan, ir_neu=np.nan, incr_ir=np.nan, tag="base"))
ic_rows.append(dict(factor="F2_质量股反转", n=len(_f2),
                    ic_raw=_f2.mean(), ir_raw=_f2.mean()/_f2.std() if _f2.std() else np.nan,
                    ic_neu=np.nan, ir_neu=np.nan, incr_ir=np.nan, tag="base"))
del _f1, _f2

# 候选
incr_store = {}
for name in CANDIDATES():
    raw_ics = calc_ic_from_store(cand_raw[name])
    # 中性化IC（逐月）
    neu_ics = []
    for t in comp_dates:
        f = cand_raw[name][t]
        r = fwd_ret[t]
        j = pd.concat([f, r], axis=1).dropna(); j.columns = ["f", "r"]
        if len(j) < 30:
            continue
        fn = neutralizer_factory(t)
        j["fn"] = fn(j["f"], j.index)
        ic = j["fn"].corr(j["r"], method="spearman")
        if ic == ic:
            neu_ics.append(ic)
    neu_ics = np.array(neu_ics)
    # 增量IC：相对 F1/F2 正交化后的残差IC
    incr_ics = []
    for t in comp_dates:
        f = cand_z[name][t]
        z1 = zscore(pd.Series(f1_store[t]).reindex(f.index))
        z2 = zscore(pd.Series(f2_store[t]).reindex(f.index))
        df = pd.concat([f, z1, z2], axis=1).dropna()
        df.columns = ["f", "z1", "z2"]
        if len(df) < 30:
            continue
        try:
            X = df[["z1", "z2"]].values.astype(float); y = df["f"].values.astype(float)
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            resid = y - X @ beta
            r = fwd_ret[t].reindex(df.index).dropna()
            jj = pd.DataFrame({"res": resid, "r": r}).dropna()
            ic = jj["res"].corr(jj["r"], method="spearman")
            if ic == ic:
                incr_ics.append(ic)
        except Exception:
            continue
    incr_ics = np.array(incr_ics)
    incr_store[name] = incr_ics
    ic_rows.append(dict(
        factor=name, n=len(raw_ics),
        ic_raw=raw_ics.mean(), ir_raw=raw_ics.mean()/raw_ics.std() if raw_ics.std() else np.nan,
        ic_neu=neu_ics.mean(), ir_neu=neu_ics.mean()/neu_ics.std() if neu_ics.std() else np.nan,
        incr_ir=incr_ics.mean()/incr_ics.std() if incr_ics.std() else np.nan,
        tag="candidate"))

ic_df = pd.DataFrame(ic_rows)
ic_df = ic_df.sort_values("incr_ir", ascending=False, na_position="last")
print(ic_df.to_string(index=False))


# ---------- 9. 选增量最强5个 → 7因子组合 ----------
# 重要：若直接按 incr_ir 排序取前5，会得到 5 个质量/利润率克隆因子
#       (qual_aturn/netmargin/opmargin/qual_roe/qual_combo)，等于把质量重复叠加，
#       既违背"7因子结构"的分散本意，也撞上"naive多因子易过拟合"的纪律。
# 故改为【按类别选冠军→取增量为正的前5】，保证每类最多1个、因子真正分散。
CATEGORY = {
    "val_ep":"价值","val_bp":"价值","val_sp":"价值","val_div":"价值","val_combo":"价值",
    "qual_roe":"质量","qual_combo":"质量","qual_gpm":"质量","qual_aturn":"质量","cashqual":"质量","lowlev":"质量","qval":"质量","qual_growth":"质量",
    "growth_op":"成长","growth_rev":"成长","growth_combo":"成长","earn_yoy":"成长",
    "mom20":"动量","mom60":"动量","mom120":"动量",
    "rev5":"反转","rev10":"反转","rev60":"反转","rev_combo":"反转",
    "size_small":"规模",
    "liq_turn":"流动性","liq_amt":"流动性",
    "gpm_value":"交叉","div_size":"交叉","opmargin":"质量","netmargin":"质量",
}
cand_ic = ic_df[ic_df["tag"] == "candidate"].copy().copy()
cand_ic["cat"] = cand_ic["factor"].map(CATEGORY)
# 每个类别取 incr_ir 最高者
winners = cand_ic.sort_values("incr_ir", ascending=False).groupby("cat", as_index=False).first()
winners = winners[winners["incr_ir"] > 0].sort_values("incr_ir", ascending=False)
NAIVE5 = cand_ic[cand_ic["incr_ir"] == cand_ic["incr_ir"]].sort_values("incr_ir", ascending=False).head(5)["factor"].tolist()
SELECTED = winners.head(5)["factor"].tolist()
print(f"\n[select·分散化] 7因子补全(每类最多1个, incr_ir>0前5): {SELECTED}")
print(f"[select·naive]  若纯按incr_ir取前5(均为质量克隆, 弃用): {NAIVE5}")

# 构建 comp7（与 comp2 同口径：等权z和）
for t in comp_dates:
    s = comp2_store[t].copy()   # 已是 z(F1)+z(F2)（dropna后）
    parts = [s]
    for name in SELECTED:
        zv = cand_z[name][t].reindex(s.index)
        parts.append(zv)
    comp7 = pd.concat(parts, axis=1).dropna().sum(axis=1)
    comp7_store[t] = comp7


# ---------- 10. 分组收益（7因子） ----------
print("[eval] 分组收益 ...")
group_ret = {g: [] for g in range(1, 6)}
for t in comp_dates:
    comp = comp7_store[t]
    r = fwd_ret[t]
    j = pd.concat([comp, r], axis=1).dropna(); j.columns = ["c", "r"]
    if len(j) < 50:
        continue
    try:
        j["g"] = pd.qcut(j["c"], 5, labels=[1, 2, 3, 4, 5])
    except Exception:
        continue
    for g in range(1, 6):
        gr = j[j["g"] == g]["r"]
        if len(gr):
            group_ret[g].append(gr.mean())
group_mean = {g: np.mean(v) if v else np.nan for g, v in group_ret.items()}


# ---------- 11. 回测（comp2 vs comp7，同一引擎） ----------
print("[backtest] top50 月度等权 ...")
def load_hs300_tr():
    try:
        d = pd.read_sql("SELECT trade_date, close FROM index_daily WHERE ts_code='H00300.SH' AND trade_date>='20120101'", con)
        if len(d):
            d["trade_date"] = d["trade_date"].astype(str)
            return d.set_index("trade_date")["close"].sort_index(), "REAL(H00300.SH)"
    except Exception:
        pass
    try:
        import tushare as ts, config_tushare
        pro = ts.pro_api(config_tushare.TUSHARE_TOKEN)
        d = pro.index_daily(ts_code='H00300.SH')
        if d is not None and len(d):
            d["trade_date"] = d["trade_date"].astype(str); d = d[["trade_date", "close"]]
            d.to_sql("index_daily", con, if_exists="append", index=False)
            return d.set_index("trade_date")["close"].sort_index(), "REAL(pulled)"
    except Exception:
        pass
    px = pd.read_sql("SELECT trade_date, close FROM index_daily WHERE ts_code='000300.SH' AND trade_date>='20120101'", con)
    px["trade_date"] = px["trade_date"].astype(str)
    px = px.set_index("trade_date")["close"].sort_index()
    pxm = px.reindex(rebal).dropna()
    trm = pxm / pxm.shift(1) - 1 + HS_DIV_YIELD / 12
    return (1 + trm.fillna(0)).cumprod(), f"SYNTHETIC(+{HS_DIV_YIELD*100:.1f}%/yr)"

hs_tr, bench_src = load_hs300_tr()
print(f"[bench] 全收益基准: {bench_src}")
hs_tr_close = {d: hs_tr[d] for d in hs_tr.index if d in date_idx}
hs_px = pd.read_sql("SELECT trade_date, close FROM index_daily WHERE ts_code='000300.SH' AND trade_date>='20120101'", con)
hs_px["trade_date"] = hs_px["trade_date"].astype(str)
hs_px = hs_px.set_index("trade_date")["close"]
hs_px_close = {d: hs_px[d] for d in hs_px.index if d in date_idx}

def backtest(comp_store):
    strat_ret = []; bench_ret = []; bench_px = []; hold = []
    prev = None
    fi = rebal.index(comp_dates[0])
    for k in range(fi, len(rebal) - 1):
        t = rebal[k]; t_next = rebal[k + 1]
        if t in comp_store:
            cur_scores = comp_store[t]
            top = cur_scores.sort_values(ascending=False).head(TOPN).index.tolist()
            cur = top
        else:
            if prev is None:
                continue
            cur = prev
        rets = []
        for c in cur:
            c0 = adj_close_raw.at[t, c] if (t in adj_close_raw.index and c in adj_close_raw.columns) else np.nan
            c1 = adj_close_raw.at[t_next, c] if (t_next in adj_close_raw.index and c in adj_close_raw.columns) else np.nan
            if c0 == c0 and c0 and c1 == c1 and c1:
                rets.append(c1 / c0 - 1)
        r = np.mean(rets) if rets else 0.0
        if prev is not None:
            changed = len(set(cur) ^ set(prev))
            turn = changed / (len(cur) + len(prev))
            r -= COST * turn
        else:
            r -= COST
        prev = cur
        b0 = hs_tr_close.get(t); b1 = hs_tr_close.get(t_next)
        br = (b1 / b0 - 1) if (b0 and b0 == b0 and b1 and b1 == b1) else 0.0
        p0 = hs_px_close.get(t); p1 = hs_px_close.get(t_next)
        pr = (p1 / p0 - 1) if (p0 and p0 == p0 and p1 and p1 == p1) else 0.0
        strat_ret.append(r); bench_ret.append(br); bench_px.append(pr); hold.append(t)
    return strat_ret, bench_ret, bench_px, hold

def metrics(sr, br, hold):
    nav = (1 + pd.Series(sr, index=hold)).cumprod().values
    bnav = (1 + pd.Series(br, index=hold)).cumprod().values
    tot = nav[-1]/nav[0]-1
    d0 = datetime.datetime.strptime(hold[0], "%Y%m%d"); d1 = datetime.datetime.strptime(hold[-1], "%Y%m%d")
    yrs = (d1-d0).days/365.25
    cagr = (nav[-1]/nav[0])**(1/yrs)-1
    mdd = ((nav/np.maximum.accumulate(nav))-1).min()
    bcagr = (bnav[-1]/bnav[0])**(1/yrs)-1
    btot = bnav[-1]/bnav[0]-1
    bmdd = ((bnav/np.maximum.accumulate(bnav))-1).min()
    rf_m = RF_ANN/12
    sharpe = (((pd.Series(sr)-rf_m).mean()*12)/((pd.Series(sr)-rf_m).std()*np.sqrt(12))) if np.std(sr) else np.nan
    return dict(cagr=cagr, tot=tot, mdd=mdd, bcagr=bcagr, btot=btot, bmdd=bmdd,
                excess=(cagr-bcagr), sharpe=sharpe, yrs=yrs, n=len(hold))

_sr2, _br2, _bp2, _h2 = backtest(comp2_store)
m2 = metrics(_sr2, _br2, _h2)
_sr7, _br7, _bp7, _h7 = backtest(comp7_store)
m7 = metrics(_sr7, _br7, _h7)

print(f"\n================ 对照：2因子基线 vs 7因子(自挖5) ================")
print(f"{'指标':<12}{'2因子':>14}{'7因子':>14}")
print(f"{'年化':<12}{m2['cagr']*100:>13.2f}%{m7['cagr']*100:>13.2f}%")
print(f"{'总收益':<12}{m2['tot']*100:>13.2f}%{m7['tot']*100:>13.2f}%")
print(f"{'最大回撤':<12}{m2['mdd']*100:>13.2f}%{m7['mdd']*100:>13.2f}%")
print(f"{'夏普':<12}{m2['sharpe']:>14.2f}{m7['sharpe']:>14.2f}")
print(f"{'基准年化':<12}{m2['bcagr']*100:>13.2f}%{m7['bcagr']*100:>13.2f}%")
print(f"{'年化超额':<12}{m2['excess']*100:>13.2f}pp{m7['excess']*100:>13.2f}pp")
print(f"{'区间年数':<12}{m2['yrs']:>14.1f}{m7['yrs']:>14.1f}")
print(f"\n7因子构成: F1 + F2 + {SELECTED}")
print(f"分组月均(1差~5好): " + "  ".join(f"G{g}={group_mean[g]*100:5.2f}%" for g in range(1,6)))
print("=====================================================================")


# ---------- 12. 落盘 ----------
eq2r, eq2b, eq2p, eq2h = backtest(comp2_store)
eq7r, eq7b, eq7p, eq7h = backtest(comp7_store)
eq = pd.DataFrame({
    "date": eq2h,
    "nav_2f": (1+pd.Series(eq2r, index=eq2h)).cumprod().values,
    "nav_7f": (1+pd.Series(eq7r, index=eq7h)).cumprod().values,
    "nav_bench_tr": (1+pd.Series(eq2b, index=eq2h)).cumprod().values,
    "ret_2f": eq2r, "ret_7f": eq7r, "ret_bench_tr": eq2b,
})
eq.to_csv("mine_equity.csv", index=False)
ic_df.to_csv("mine_ic.csv", index=False)
pd.DataFrame({"group": list(range(1,6)), "monthly_mean_ret": [group_mean[g] for g in range(1,6)]}).to_csv("mine_groups.csv", index=False)
print("[save] mine_ic.csv / mine_equity.csv / mine_groups.csv")
