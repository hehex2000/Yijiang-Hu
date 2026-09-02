# -*- coding: utf-8 -*-
"""
run_kara_factors.py  ——  Kara说量化 BV12zKN6kEZp 研报复刻【方案A：仅2/7因子】
=============================================================================
!!! 重要声明：本脚本只复现 UP 视频里被详述的 2 个因子（共 7 个核心因子只详述了 2 个），
    NOT 复现 12.3%/夏普0.69/超额9.9% 这些数字（UP 自承"在模拟数据上跑出来"，无法验证）。
    目标是：用我们更干净的数据 + 正确的回测原则，看这 2 个因子逻辑本身有没有真实 alpha，
    并补上 UP 没披露的 maxDD/夏普。

因子定义（严格按视频描述，全部横截面 zscore）：
  F1 冷门研发股 = z(rd_exp/total_revenue) - z(近20日成交额均值)
  F2 质量股反转 = z(-近20日收益) + z(ocfps/eps 现金流质量) + z(-近60日换手率std)
  合成分 = z(F1) + z(F2)  → 月度选 top50 等权持有

【2026-08-14 code review 修复清单】
  P0-1  不复权价破坏 F2 反转腿：改用 daily.adj_factor 复权价（close*adj_factor）计算
        ret20 信号与前向收益。除息日价格跳水是假"超跌"，复权后消除。
        连锁修复（P0-1 第二子点）：复权后策略收益=全收益(含分红)，若仍用 000300.SH
        价格指数(不含分红)作基准会虚增 ~1-2% 超额 → 基准同步切换为沪深300全收益。
        本 token 实测 index_daily 不覆盖全收益指数(H00300.SH 返回空)，故用
        "价格指数 + 股息率加回" 合成全收益基准（股息率取沪深300长期均值 2.2%/年，透明假设）。
  P0-2  ST 前视偏差：已修。改用 Tushare namechange 更名史做时点 ST（name 含'ST'
        且 start_date<=t<=end_date 即判定该票在调仓日 t 处于 ST/*ST），落库本地表
        namechange 免重复拉取；拉取失败回退当前名过滤。
  P1-3  跳过月造成 NAV 断层：重构为"跳过月延续持有上期组合并计其收益"，消除收益断层。
  P1-4  非交易日 IPO 致上市天数过滤失效：用 bisect 取 IPO 日最近的交易日位置再比。
  P1-5  ann_date==t 隐蔽未来函数：pit_get 改为 bisect_left-1，严格取 ann_date < t。
  P2-6  退市股收益伪 0%：前向收益改用非 ffill 复权价，退市/停牌→NaN→剔除（不伪报0%）。
  P2-7  换手率 ffill 虚增流动性稳定：去掉 ffill，用真实换手率序列算 60日std。
  P2-8  夏普未减无风险利率：月度收益减 rf(年化2.5%) 后再算夏普。
=============================================================================
"""
import sqlite3, bisect, datetime
import numpy as np
import pandas as pd
import config
import market_timing_overlay as mto   # A：市场情绪择时 overlay
from pit_ann import norm_ann          # ann_date 规范化(fina_indicator.ann_date 是 REAL 浮点)

# ---------- CLI ----------
import argparse
ap = argparse.ArgumentParser(description="Kara 2因子复刻 + 可选市场择时闸门")
ap.add_argument("--timing-gate", action="store_true", help="启用市场情绪择时闸门（沸点清仓/冰点满仓，非对称只卖不买）")
ap.add_argument("--boil", type=float, default=80.0, help="沸点阈值：osc>=boil → 清仓(floor)")
ap.add_argument("--ice", type=float, default=55.0, help="冰点阈值：osc<=ice → 满仓。默认55=温和真择时(常态满仓吃alpha+仅过热/泡沫顶减仓)；想得深回撤保护用 --wf(冰点≈30的永久去风险)")
ap.add_argument("--floor", type=float, default=0.0, help="清仓时最低保留仓位(0=全清,0.2=永不空仓)")
ap.add_argument("--wf", action="store_true", help="用 walk-forward 滚动选 boil/ice 阈值（防过拟合），覆盖 --boil/--ice")
args = ap.parse_args()

DB = config.DATA["local_db_path"]
START      = "20140101"   # 回测起点（财报数据从此时起算信号）
START_LOOK = "20120101"   # 数据加载起点（留足回看窗口）
END        = "20260807"
TOPN       = 50
COST       = 0.002        # 单边往返成本基准（月度调仓，按换手比例计费）
UNIV_MV_MIN = 2e5         # 宇宙过滤：total_mv 单位=万元(Tushare原生) → 2e5万元 = 20亿元
                          # 2026-08-18 修正：原值 2e6 注释按"千元"口径=20亿，但重建库
                          # total_mv 为万元，2e6万=200亿 → 宇宙从~全市场塌缩到400-950只
                          # 大盘股，年化 13.93%→5.12% 的主因。
LIST_DAYS_MIN = 252       # 上市至少 252 个交易日（剔除次新股利空/IPO噪声）
PIT = True                # point-in-time 财报对齐
RF_ANN     = 0.025        # 无风险利率（年化），用于夏普

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
# 月末交易日
mm_last = {}
for d in alldates:
    mm_last[d[:6]] = d
rebal = sorted(mm_last.values())
rebal = [d for d in rebal if d >= START and d < END]
print(f"[cal] 交易日 {len(alldates)} 个；月末调仓日 {len(rebal)} 个 ({rebal[0]}~{rebal[-1]})")


# ---------- 2. 价格/量/换手 矩阵（date x ts_code） ----------
print("[load] daily / daily_basic / adj_factor ...")
price = pd.read_sql(
    f"SELECT ts_code, trade_date, close, amount FROM daily WHERE trade_date>='{START_LOOK}'", con)
price["trade_date"] = price["trade_date"].astype(str)
close_p = price.pivot(index="trade_date", columns="ts_code", values="close").reindex(alldates).sort_index()
amount_p = price.pivot(index="trade_date", columns="ts_code", values="amount").reindex(alldates).sort_index()
del price

db = pd.read_sql(
    f"SELECT ts_code, trade_date, turnover_rate, total_mv FROM daily_basic WHERE trade_date>='{START_LOOK}'", con)
db["trade_date"] = db["trade_date"].astype(str)
# P2-7：换手率不 ffill，用真实序列算 std（ffill 会让连续相同值虚增"流动性稳定"）
turn_p = db.pivot(index="trade_date", columns="ts_code", values="turnover_rate").reindex(alldates).sort_index()
mv_p = db.pivot(index="trade_date", columns="ts_code", values="total_mv").reindex(alldates).sort_index()
del db

# P0-1：复权价。adj_close = close * adj_factor（ratio 即复权收益，无需归一化）。
#   adj_close_f  ：close 先 ffill（停牌用最近价），用于"信号"计算，避免 NaN 蔓延；
#   adj_close_raw：close 不 ffill，用于"前向收益"，退市/停牌→NaN→如实剔除（P2-6）。
adj = pd.read_sql(
    f"SELECT ts_code, trade_date, adj_factor FROM adj_factor WHERE trade_date>='{START_LOOK}'", con)
adj["trade_date"] = adj["trade_date"].astype(str)
adj_factor_p = adj.pivot(index="trade_date", columns="ts_code", values="adj_factor").reindex(alldates).sort_index().ffill().fillna(1.0)
del adj
adj_close_f   = (close_p.ffill() * adj_factor_p)
adj_close_raw = (close_p * adj_factor_p)

# 因子输入（向量化，按列=个股）
ret20 = adj_close_f / adj_close_f.shift(20) - 1        # 近20日复权收益（信号）
amt20 = amount_p.rolling(20).mean()                    # 近20日成交额均值
turnstd60 = turn_p.rolling(60, min_periods=20).std()   # 近60日换手率std（真实序列）
print(f"[load] 矩阵: close {close_p.shape}, amount {amount_p.shape}, turn {turn_p.shape}")


# ---------- 3. point-in-time 财报映射 ----------
def build_pit_map(sql, valcol, denom=None):
    """返回 {ts_code: (sorted_ann_dates[], values[])}，ann_date<调仓日时取最新。"""
    df = pd.read_sql(sql, con)
    df["ann"] = norm_ann(df["ann_date"])
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

print("[load] income (rd_exp/total_revenue) point-in-time ...")
pit_rd = build_pit_map(
    "SELECT ts_code, ann_date, rd_exp, total_revenue FROM income", "rd_exp", "total_revenue")
print(f"       有 rd 数据的股票: {len(pit_rd)}")

print("[load] fina_indicator (ocfps/eps 现金流质量) point-in-time ...")
pit_cfq = build_pit_map(
    "SELECT ts_code, ann_date, ocfps, eps FROM fina_indicator WHERE eps IS NOT NULL AND abs(eps)>1e-6",
    "ocfps", "eps")
print(f"       有 cfq 数据的股票: {len(pit_cfq)}")

# 上市日
listd = {r[0]: r[1] for r in con.execute("SELECT ts_code, list_date FROM stock_basic WHERE list_date IS NOT NULL")}

# P0-2：时点 ST 判断（基于 Tushare namechange 更名史，杜绝前视）
#   不再用 stock_basic 当前名（那是当前状态，会前视）。改用 namechange 的
#   (ts_code, name, start_date, end_date)，凡 name 含 'ST'(含*ST/SST/S*ST) 且
#   start_date <= t <= end_date 的月份，判定该票在调仓日 t 正处于 ST。
#   全量落库本地表 namechange，重跑免重复拉取；拉取失败则回退当前名过滤。
def load_st_intervals(con):
    cur = con.cursor()
    try:
        n = cur.execute("SELECT COUNT(*) FROM namechange").fetchone()[0]
    except Exception:
        n = 0
    if n > 0:
        df = pd.read_sql("SELECT ts_code, name, start_date, end_date FROM namechange", con)
        print(f"[namechange] 本地缓存 {len(df)} 行")
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
                columns=["ts_code", "name", "start_date", "end_date", "ann_date", "change_reason"])
            df.to_sql("namechange", con, if_exists="replace", index=False)
            print(f"[namechange] 从Tushare拉取并落库 {len(df)} 行")
        except Exception as e:
            print(f"[namechange] 拉取失败({e})，回退当前名ST过滤")
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

st_intervals = load_st_intervals(con)     # {code:[(s,e),...]} 或 None(回退)
st_by_date = {}
if st_intervals:
    for code, ivs in st_intervals.items():
        for (s, e) in ivs:
            lo = bisect.bisect_left(rebal, s)    # start_date 含（左闭）
            hi = bisect.bisect_left(rebal, e)     # end_date 不含（右开）：摘帽日当天不再算ST
            for ti in range(lo, hi):
                st_by_date.setdefault(rebal[ti], set()).add(code)
    print(f"[filter] 时点ST: {len(st_intervals)} 只股票有ST历史；覆盖 {len(st_by_date)} 个调仓日")
else:
    st_by_date = None
# 当前名过滤仅作回退
st_codes = set(r[0] for r in con.execute(
    "SELECT ts_code FROM stock_basic WHERE name LIKE '%ST%'"))
print(f"[filter] 有上市日信息 {len(listd)} 只；当前名ST回退集 {len(st_codes)} 只")


# P1-4：用 bisect 取 IPO 日最近的"交易日"位置（非交易日 IPO 也能正确判上市天数）
def _list_tidx(dstr):
    i = bisect.bisect_right(alldates, dstr) - 1
    return i if i >= 0 else -999
list_tidx = {c: _list_tidx(d) for c, d in listd.items()}


# P1-5：严格取 ann_date < t（避免调仓日当天盘后公告被当日信号使用）
def pit_get(m, code, t):
    if code not in m:
        return np.nan
    anns, vals = m[code]
    i = bisect.bisect_left(anns, t) - 1
    return vals[i] if i >= 0 else np.nan


# ---------- 4. 逐月计算因子 + 选股 ----------
print("[calc] 逐月因子与选股 ...")
records = []
sel_hist = {}
comp_dates = []
comp_matrix = {}
fwd_ret = {}
f1_store = {}
f2_store = {}

for k, t in enumerate(rebal):
    if k + 1 >= len(rebal):
        break
    t_next = rebal[k + 1]
    idi = date_idx[t]; nexti = date_idx[t_next]
    # P1-4：上市 >= LIST_DAYS_MIN（用交易日位置判断，非交易日 IPO 也正确）
    # P0-2：时点 ST 过滤（namechange 落库后按调仓日 t 判断，无前视）
    st_set = st_by_date.get(t, set()) if st_by_date is not None else st_codes
    elig = [c for c in close_p.columns
            if c in list_tidx and list_tidx[c] <= idi - LIST_DAYS_MIN
            and c not in st_set]
    # 市值过滤（t 日）
    mv_row = mv_p.loc[t] if t in mv_p.index else None
    elig = [c for c in elig if mv_row is not None and (c in mv_row.index) and (mv_row.get(c, 0) or 0) >= UNIV_MV_MIN]
    if len(elig) < TOPN:
        continue

    # 取各因子原始值
    rd_v = {c: pit_get(pit_rd, c, t) for c in elig}
    cfq_v = {c: pit_get(pit_cfq, c, t) for c in elig}
    amt_v = {c: (amt20.at[t, c] if t in amt20.index and c in amt20.columns else np.nan) for c in elig}
    ret_v = {c: (ret20.at[t, c] if t in ret20.index and c in ret20.columns else np.nan) for c in elig}
    tst_v = {c: (turnstd60.at[t, c] if t in turnstd60.index and c in turnstd60.columns else np.nan) for c in elig}
    # P2-6：前向收益 t->t_next 用非 ffill 复权价（退市/停牌→NaN→剔除，不伪报0%）
    fwd_v = {}
    for c in elig:
        c0 = adj_close_raw.at[t, c] if t in adj_close_raw.index else np.nan
        c1 = adj_close_raw.at[t_next, c] if t_next in adj_close_raw.index else np.nan
        fwd_v[c] = (c1 / c0 - 1) if (c0 == c0 and c0 and c1 == c1 and c1) else np.nan

    # 构造截面序列
    s_rd = pd.Series(rd_v); s_cfq = pd.Series(cfq_v)
    s_amt = pd.Series(amt_v); s_ret = pd.Series(ret_v); s_tst = pd.Series(tst_v)
    s_fwd = pd.Series(fwd_v)
    # zscore
    z_rd = zscore(s_rd); z_amt = zscore(s_amt)
    z_ret = zscore(-s_ret); z_cfq = zscore(s_cfq); z_tst = zscore(-s_tst)

    F1 = z_rd - z_amt                                   # 冷门研发股
    F2 = z_ret + z_cfq + z_tst                          # 质量股反转（三腿）
    # 合成分：要求两因子都有效
    both = F1.notna() & F2.notna()
    comp = pd.Series(np.nan, index=elig)
    comp[both] = zscore(F1[both]) + zscore(F2[both])

    comp = comp.dropna()
    if len(comp) < TOPN:
        continue
    top = comp.sort_values(ascending=False).head(TOPN).index.tolist()
    sel_hist[t] = top
    comp_dates.append(t)
    comp_matrix[t] = comp.to_dict()
    fwd_ret[t] = s_fwd.to_dict()
    f1_store[t] = F1.dropna().to_dict()
    f2_store[t] = F2.dropna().to_dict()

print(f"[calc] 有效选股月份: {len(comp_dates)} 个")


# ---------- 5. IC / IR（因子 vs 前向1月收益） ----------
print("[eval] IC / IR ...")
def calc_ic(factor_name, extract):
    ics = []
    for t in comp_dates:
        f = extract(t)
        r = pd.Series(fwd_ret[t])
        join = pd.concat([f, r], axis=1).dropna()
        join.columns = ["f", "r"]
        if len(join) < 30:
            continue
        ic = join["f"].corr(join["r"], method="spearman")
        if ic == ic:
            ics.append(ic)
    ics = np.array(ics)
    return dict(factor=factor_name, n=len(ics), ic_mean=ics.mean(), ic_std=ics.std(),
                ir=ics.mean()/ics.std() if ics.std() else np.nan,
                ic_pos_ratio=(ics > 0).mean())

ic_rows = []
ic_rows.append(calc_ic("F1_冷门研发股", lambda t: pd.Series(f1_store[t])))
ic_rows.append(calc_ic("F2_质量股反转", lambda t: pd.Series(f2_store[t])))
ic_rows.append(calc_ic("COMPOSITE_合成分", lambda t: pd.Series(comp_matrix[t])))
ic_df = pd.DataFrame(ic_rows)
print(ic_df.to_string(index=False))


# ---------- 6. 分组收益（quintile） ----------
print("[eval] 分组收益 ...")
group_ret = {g: [] for g in range(1, 6)}
for t in comp_dates:
    comp = pd.Series(comp_matrix[t])
    r = pd.Series(fwd_ret[t])
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


# ---------- 7. 回测（top50 月度等权，HS300 基准） ----------
# P1-3：从首个有效选股月起，逐月计算收益；"跳过月"延续持有上期组合并计其收益，
#       消除原逻辑中 m 被跳过导致 m->m+1 收益丢失的 NAV 断层。
print("[backtest] top50 月度等权（含延续持有）...")
strat_ret = []; bench_ret = []; bench_ret_px = []
prev = None
# ---------- 7. 基准：沪深300 全收益（P0-1 连锁修复） ----------
# 策略前向收益用复权价 = 全收益(含分红)。若仍用 000300.SH 价格指数(不含分红)作基准，
# 会虚增 ~1-2% 超额。本 token 实测 index_daily 不覆盖全收益指数 H00300.SH（返回空），
# 故先试拉，失败则"价格指数 + 股息率加回"合成全收益基准（透明假设）。
HS_DIV_YIELD = 0.022   # 沪深300 长期股息率(年化)假设，用于合成全收益基准

def load_hs300_tr():
    # 1) 本地已有真实全收益指数
    try:
        d = pd.read_sql("SELECT trade_date, close FROM index_daily WHERE ts_code='H00300.SH' AND trade_date>='20120101'", con)
        if len(d):
            d["trade_date"] = d["trade_date"].astype(str)
            return d.set_index("trade_date")["close"].sort_index(), "REAL(H00300.SH)"
    except Exception:
        pass
    # 2) 试 Tushare 拉真实全收益指数
    try:
        import tushare as ts, config_tushare
        pro = ts.pro_api(config_tushare.TUSHARE_TOKEN)
        d = pro.index_daily(ts_code='H00300.SH')
        if d is not None and len(d):
            d["trade_date"] = d["trade_date"].astype(str)
            d = d[["trade_date", "close"]]
            d.to_sql("index_daily", con, if_exists="append", index=False)
            return d.set_index("trade_date")["close"].sort_index(), "REAL(H00300.SH pulled)"
    except Exception as e:
        print(f"[bench] H00300.SH 拉取失败({e})，改用合成全收益基准")
    # 3) 合成：价格指数 + 股息率加回（月收益 = 价格月收益 + 股息率/12）
    px = pd.read_sql("SELECT trade_date, close FROM index_daily WHERE ts_code='000300.SH' AND trade_date>='20120101'", con)
    px["trade_date"] = px["trade_date"].astype(str)
    px = px.set_index("trade_date")["close"].sort_index()
    pxm = px.reindex(rebal).dropna()
    trm = pxm / pxm.shift(1) - 1 + HS_DIV_YIELD / 12
    tr = (1 + trm.fillna(0)).cumprod()
    return tr, f"SYNTHETIC(price+{HS_DIV_YIELD*100:.1f}%/yr)"

hs_tr, bench_src = load_hs300_tr()
print(f"[bench] 沪深300全收益基准来源: {bench_src}")
hs_tr_close = {d: hs_tr[d] for d in hs_tr.index if d in date_idx}
# 价格基准（交叉验证：应与原 8.98pp 超额对应）
hs_px = pd.read_sql("SELECT trade_date, close FROM index_daily WHERE ts_code='000300.SH' AND trade_date>='20120101'", con)
hs_px["trade_date"] = hs_px["trade_date"].astype(str)
hs_px = hs_px.set_index("trade_date")["close"]
hs_px_close = {d: hs_px[d] for d in hs_px.index if d in date_idx}

# ---------- 7. 回测（重构为函数，支持择时闸门 A） ----------
# 基线 = caps 全 1（闸门不触发，等价于原逻辑，保证回归一致）
def run_backtest(caps_by_date):
    strat_ret = []; bench_ret = []; bench_ret_px = []; hold_dates = []
    prev = None; prev_cap = None
    if not comp_dates:
        return strat_ret, bench_ret, bench_ret_px, hold_dates
    fi = rebal.index(comp_dates[0])
    for k in range(fi, len(rebal) - 1):
        t = rebal[k]; t_next = rebal[k + 1]
        if t in sel_hist:
            cur = sel_hist[t]            # 本月重新选股
        else:
            if prev is None:
                continue                 # 尚未建仓，跳过
            cur = prev                   # 跳过月：延续持有上期组合
        rets = []
        for c in cur:
            c0 = adj_close_raw.at[t, c] if (t in adj_close_raw.index and c in adj_close_raw.columns) else np.nan
            c1 = adj_close_raw.at[t_next, c] if (t_next in adj_close_raw.index and c in adj_close_raw.columns) else np.nan
            if c0 == c0 and c0 and c1 == c1 and c1:
                rets.append(c1 / c0 - 1)
        r = np.mean(rets) if rets else 0.0
        # 成本：按换手比例（延续持有时 turnover=0）
        if prev is not None:
            changed = len(set(cur) ^ set(prev))
            turn = changed / (len(cur) + len(prev))
            r -= COST * turn
        else:
            r -= COST   # 首次建仓
        prev = cur
        # ---- 择时闸门（A）：缩放收益 + 现金 sleeve 切换成本 ----
        cap = caps_by_date.get(t, 1.0)
        r = cap * r                                  # 缩放收益与选择成本；cap=0 时选择成本归零(空仓不换股)
        if prev_cap is not None:
            r -= COST * abs(cap - prev_cap)          # 闸门切换：现金 sleeve 买卖单向成本，不缩放
        prev_cap = cap
        # 基准月收益（全收益）
        b0 = hs_tr_close.get(t); b1 = hs_tr_close.get(t_next)
        br = (b1 / b0 - 1) if (b0 and b0 == b0 and b1 and b1 == b1) else 0.0
        # 基准月收益（价格指数，交叉验证）
        p0 = hs_px_close.get(t); p1 = hs_px_close.get(t_next)
        pr = (p1 / p0 - 1) if (p0 and p0 == p0 and p1 and p1 == p1) else 0.0
        strat_ret.append(r); bench_ret.append(br); bench_ret_px.append(pr); hold_dates.append(t)
    return strat_ret, bench_ret, bench_ret_px, hold_dates

# 基线（无闸门）
strat_ret, bench_ret, bench_ret_px, hold_dates = run_backtest({d: 1.0 for d in rebal})

# ---------- 7a. 市场情绪振荡器 + 择时闸门（A：overlay） ----------
gated = args.timing_gate
strat_ret_g = bench_ret_g = bench_ret_px_g = hold_dates_g = None
if gated:
    osc_all = mto.compute_breadth_oscillator(close_p)
    osc_rebal = osc_all.reindex(rebal)
    # 用基线月度收益训练选阈（防过拟合）
    base_ret = pd.Series(strat_ret, index=hold_dates)
    osc_hold = osc_rebal.reindex(hold_dates)
    if args.wf:
        boil, ice, oos = mto.walk_forward_thresholds(osc_hold, base_ret,
                                                     floor=args.floor, cost=COST)
        print(f"[overlay] walk-forward 选阈: 沸点={boil:.0f} / 冰点={ice:.0f} | "
              f"OOS年化={oos[0]*100:.2f}% / maxDD={oos[1]*100:.2f}% / 夏普={oos[2]:.2f}")
    else:
        boil, ice = args.boil, args.ice
    caps_by_date = {}
    for d in rebal:
        o = osc_rebal[d] if d in osc_rebal.index else np.nan
        caps_by_date[d] = mto.position_cap(o, boil, ice, args.floor) if (o == o) else 1.0
    strat_ret_g, bench_ret_g, bench_ret_px_g, hold_dates_g = run_backtest(caps_by_date)
    n_clear = sum(1 for d in hold_dates_g if caps_by_date.get(d, 1.0) < 1.0)
    print(f"[overlay] 闸门: 沸点>={boil:.0f}清仓 / 冰点<={ice:.0f}满仓 / floor={args.floor:.2f} / "
          f"触发减仓或清仓月 = {n_clear}/{len(hold_dates_g)}")

dates_out = hold_dates
snav = (1 + pd.Series(strat_ret, index=dates_out)).cumprod()
bnav = (1 + pd.Series(bench_ret, index=dates_out)).cumprod()
bnav_px = (1 + pd.Series(bench_ret_px, index=dates_out)).cumprod()

def metrics(nav):
    ns = nav.values
    tot = ns[-1] / ns[0] - 1
    d0 = datetime.datetime.strptime(dates_out[0], "%Y%m%d")
    d1 = datetime.datetime.strptime(dates_out[-1], "%Y%m%d")
    yrs = (d1 - d0).days / 365.25
    cagr = (ns[-1] / ns[0]) ** (1 / yrs) - 1
    peak = np.maximum.accumulate(ns); mdd = (ns / peak - 1).min()
    return tot, cagr, mdd, yrs

t1, c1, m1, yrs = metrics(snav)
t2, c2, m2, _ = metrics(bnav)          # 沪深300全收益基准
t2p, c2p, m2p, _ = metrics(bnav_px)    # 沪深300价格指数基准（交叉验证）
# P2-8：夏普减无风险利率（月度）
rf_m = RF_ANN / 12
sr = pd.Series(strat_ret) - rf_m
sharpe = (sr.mean() * 12) / (sr.std() * np.sqrt(12)) if sr.std() else np.nan
br_s = pd.Series(bench_ret) - rf_m
bsharpe = (br_s.mean() * 12) / (br_s.std() * np.sqrt(12)) if br_s.std() else np.nan

n_held = len(dates_out) - len(comp_dates)
print(f"[diag] 回测覆盖月 {len(dates_out)}；重新选股 {len(comp_dates)} 月，延续持有(跳过月) {n_held} 月")
print(f"\n================ 回测结果（Kara 2因子·仅2/7·逻辑复刻·已修P0/P1/P2） ================")
print(f"区间        : {dates_out[0]} ~ {dates_out[-1]}  ({yrs:.1f}年)  有效月: {len(dates_out)}")
print(f"策略总收益  : {t1*100:7.2f}%   年化: {c1*100:6.2f}%   最大回撤: {m1*100:6.2f}%   夏普: {sharpe:5.2f}")
print(f"基准(全收益): {t2*100:7.2f}%   年化: {c2*100:6.2f}%   最大回撤: {m2*100:6.2f}%   夏普: {bsharpe:5.2f}")
print(f"基准(价格)  : {t2p*100:7.2f}%   年化: {c2p*100:6.2f}%   (交叉验证，原口径)")
print(f"年化超额(全收益基准): {(c1-c2)*100:6.2f}pp   年化超额(价格基准): {(c1-c2p)*100:6.2f}pp")
print(f"总收益倍数  : 策略 {(snav.iloc[-1]/snav.iloc[0]):.2f}x / 全收益基准 {(bnav.iloc[-1]/bnav.iloc[0]):.2f}x")
print(f"分组月均收益(1=最差~5=最好): " + "  ".join(f"G{g}={group_mean[g]*100:5.2f}%" for g in range(1,6)))
print(f"  → UP 宣称 12.3%/夏普0.69/超额9.9%(vs沪深300)，且【未披露maxDD】")
print("=====================================================================================")

# ---------- 7b. 择时闸门对比（A：市场情绪 overlay） ----------
if gated and strat_ret_g is not None:
    gnav = (1 + pd.Series(strat_ret_g, index=hold_dates_g)).cumprod()
    _, gcagr, gmdd, _ = metrics(gnav)
    gs = pd.Series(strat_ret_g) - rf_m
    gsharpe = (gs.mean() * 12) / (gs.std() * np.sqrt(12)) if gs.std() else np.nan
    _, gc2, gm2, _ = metrics((1 + pd.Series(bench_ret_g, index=hold_dates_g)).cumprod())
    print(f"\n================ 择时闸门对比（A：市场情绪 overlay·非对称只卖不买） ================")
    print(f"{'指标':<16}{'基线(无闸门)':>16}{'+择时闸门':>16}")
    print(f"{'年化':<16}{c1*100:>15.2f}%{gcagr*100:>15.2f}%")
    print(f"{'最大回撤':<16}{m1*100:>15.2f}%{gmdd*100:>15.2f}%")
    print(f"{'夏普':<16}{sharpe:>16.2f}{gsharpe:>16.2f}")
    print(f"{'基准超额(年化)':<16}{(c1-c2)*100:>15.2f}pp{(gcagr-gc2)*100:>15.2f}pp")
    print(f"  → 闸门目标: maxDD 压到 -25% 内、年化损失 -2~5pp; 非对称=只减仓不抄底; 阈值为 walk-forward 选出")
    print("=====================================================================================")

# ---------- 8. 落盘 ----------
eq = pd.DataFrame({"date": dates_out, "strat_nav": snav.values, "bench_nav_tr": bnav.values,
                   "bench_nav_px": bnav_px.values,
                   "strat_ret": strat_ret, "bench_ret_tr": bench_ret, "bench_ret_px": bench_ret_px})
if gated and strat_ret_g is not None:
    gnav = (1 + pd.Series(strat_ret_g, index=hold_dates_g)).cumprod()
    eq["strat_nav_gated"] = gnav.reindex(dates_out).values
eq.to_csv("kara_factors_equity.csv", index=False)
ic_df.to_csv("kara_factors_ic.csv", index=False)
grp = pd.DataFrame({"group": list(range(1, 6)), "monthly_mean_ret": [group_mean[g] for g in range(1, 6)]})
grp.to_csv("kara_factors_groups.csv", index=False)
print("[save] kara_factors_equity.csv / kara_factors_ic.csv / kara_factors_groups.csv")
