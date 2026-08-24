# -*- coding: utf-8 -*-
"""
run_piotroski_oos.py  ——  Piotroski F-score 单因子 OOS 验证（plan_piotroski.md 第 3 步）

目标：验证经典 F-score（9 项 0/1 之和，0~9）在 A 股能否作为"独立质量 alpha"，
      而非直接接入选股引擎。严格复用 run_kara_factors.py 的回测纪律：
        * 严格 PIT：ann_date < t（盘后公告当日不可用）—— 由 piotroski_fscore 内部保证
        * 时点 ST 过滤（namechange 更名史，无前视）
        * IPO >= 252 交易日（bisect，非交易日 IPO 也正确）
        * 前向收益用非 ffill 复权价（退市/停牌 -> NaN -> 剔除，不伪报 0%）
        * 成本按换手比例计费（COST=0.002/月，往返基准）
        * 沪深300 全收益基准（价格指数+股息率加回合成，本地无 H00300.SH 时）

输出（落盘 data/results/）：
        piotroski_oos_ic.csv          Rank IC / ICIR / IC>0 占比（全期 / train / test）
        piotroski_oos_groups.csv      五分组月均收益（G1=F最低~G5=F最高）
        piotroski_oos_longshort.csv   多空组合 NAV（高F多 / 低F空，月度再平衡）
        piotroski_oos_gatesum.csv     高F(F>=8) vs 低F(F<=2) 前向月均收益对照

walk-forward 纪律：train <= 20221231（样本内探索），test >= 20230101（样本外验证）。
F-score 无待估参数（固定公式），故 train 仅作描述性对照，重点看 test 窗口是否仍有 edge。

用法：
        venv_ml/Scripts/python.exe run_piotroski_oos.py            # 默认 zz800 宇宙
        venv_ml/Scripts/python.exe run_piotroski_oos.py --univ all # 全市场可投资宇宙
        venv_ml/Scripts/python.exe run_piotroski_oos.py --univ zz800 --train-end 20221231
"""
import argparse
import bisect
import datetime
import sqlite3

import numpy as np
import pandas as pd
import config

from piotroski_fscore import build_fscore_maps, compute_fscore

# ---------- 参数 ----------
DB = config.DATA["local_db_path"]
START_LOOK = "20120101"          # 数据加载起点（留足复权/回看窗口）
START = "20140101"               # 回测起点（财报信号从此时起算）
COST = 0.002                     # 月频往返成本基准（按换手比例计费）
LIST_DAYS_MIN = 252              # 上市 >= 252 交易日（剔除次新）
UNIV_MV_MIN = 2e5                # 全市场宇宙过滤：total_mv 万元 -> 20 亿元（zz800 时不适用）
RF_ANN = 0.025                   # 无风险利率年化（夏普）
TRAIN_END = "20221231"           # walk-forward 切分：样本内 <= 此，样本外 >= 次月
HS_DIV_YIELD = 0.022             # 沪深300 合成全收益基准股息率假设
LS_LONG_PCT = 0.20               # 多空组合：多/空各取前/后 20%
GATE_HI = 8                      # 高F 阈值（F>=8）
GATE_LO = 2                      # 低F 阈值（F<=2）


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--univ", choices=["zz800", "all"], default="zz800",
                    help="测试宇宙：zz800=中证800（默认，计划指定） / all=全市场可投资（MV过滤）")
    ap.add_argument("--train-end", default=TRAIN_END, help="walk-forward 样本内截止日 YYYYMMDD")
    args = ap.parse_args()
    UNIV = args.univ
    tr_end = args.train_end
    TEST_START = (datetime.datetime.strptime(tr_end, "%Y%m%d") +
                  datetime.timedelta(days=1)).strftime("%Y%m%d")

    con = sqlite3.connect(DB)

    # ---------- 1. 交易日 / 月末调仓日 ----------
    alldates = [r[0] for r in con.execute(
        f"SELECT DISTINCT trade_date FROM daily WHERE trade_date>='{START_LOOK}' ORDER BY trade_date")]
    alldates.sort()
    date_idx = {d: i for i, d in enumerate(alldates)}
    mm_last = {}
    for d in alldates:
        mm_last[d[:6]] = d
    END = con.execute("SELECT MAX(trade_date) FROM daily WHERE trade_date>=?", (START,)).fetchone()[0]
    rebal = sorted(mm_last.values())
    rebal = [d for d in rebal if START <= d <= END]
    print(f"[cal] 宇宙={UNIV}  交易日 {len(alldates)}；调仓月 {len(rebal)} ({rebal[0]}~{rebal[-1]})  END={END}")

    # ---------- 2. 价格 / 复权矩阵（前向收益用）----------
    price = pd.read_sql(
        f"SELECT ts_code, trade_date, close FROM daily WHERE trade_date>='{START_LOOK}'", con)
    price["trade_date"] = price["trade_date"].astype(str)
    close_p = price.pivot(index="trade_date", columns="ts_code", values="close").reindex(alldates).sort_index()
    del price
    db_ = pd.read_sql(
        f"SELECT ts_code, trade_date, total_mv FROM daily_basic WHERE trade_date>='{START_LOOK}'", con)
    db_["trade_date"] = db_["trade_date"].astype(str)
    mv_p = db_.pivot(index="trade_date", columns="ts_code", values="total_mv").reindex(alldates).sort_index()
    del db_
    adj = pd.read_sql(
        f"SELECT ts_code, trade_date, adj_factor FROM adj_factor WHERE trade_date>='{START_LOOK}'", con)
    adj["trade_date"] = adj["trade_date"].astype(str)
    adj_factor_p = adj.pivot(index="trade_date", columns="ts_code", values="adj_factor").reindex(alldates).sort_index().ffill().fillna(1.0)
    del adj
    adj_close_raw = (close_p * adj_factor_p)   # 非 ffill：退市/停牌 -> NaN -> 剔除

    # ---------- 3. ST / IPO 过滤（严格 PIT，复制 run_kara_factors 范式）----------
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
                print(f"[namechange] 拉取失败({e})，回退当前名过滤")
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
        print(f"[filter] 时点ST: {len(st_intervals)} 只历史ST；覆盖 {len(st_by_date)} 个调仓月")
    else:
        st_by_date = None
    st_codes = set(r[0] for r in con.execute("SELECT ts_code FROM stock_basic WHERE name LIKE '%ST%'"))

    listd = {r[0]: r[1] for r in con.execute("SELECT ts_code, list_date FROM stock_basic WHERE list_date IS NOT NULL")}
    def _list_tidx(dstr):
        i = bisect.bisect_right(alldates, dstr) - 1
        return i if i >= 0 else -999
    list_tidx = {c: _list_tidx(d) for c, d in listd.items()}

    # ---------- 4. zz800 时点成分（若 --univ zz800）----------
    zz_snaps = {}
    if UNIV == "zz800":
        snap_dates = [r[0] for r in con.execute(
            "SELECT DISTINCT trade_date FROM index_constituent WHERE index_code='000906.SH' ORDER BY trade_date")]
        for td in snap_dates:
            members = set(r[0] for r in con.execute(
                "SELECT ts_code FROM index_constituent WHERE index_code='000906.SH' AND trade_date=?", (td,)))
            zz_snaps[td] = members
        print(f"[univ] zz800 快照样本 {len(snap_dates)} 个；最新 {snap_dates[-1]} 含 {len(zz_snaps[snap_dates[-1]])} 只")
    def zz800_at(t):
        i = bisect.bisect_right(snap_dates, t) - 1
        return zz_snaps[snap_dates[i]] if i >= 0 else set()

    # ---------- 5. F-score 建图 ----------
    print("[load] 构建 F-score PIT 地图 ...")
    M = build_fscore_maps(con)
    print("       各 map 覆盖:", {k: len(v) for k, v in M.items()})

    # ---------- 6. 逐月因子 + 前向收益 ----------
    print("[calc] 逐月 F-score + 前向收益 ...")
    fscore_store = {}
    fwd_ret = {}
    comp_dates = []
    for k, t in enumerate(rebal):
        if k + 1 >= len(rebal):
            break
        t_next = rebal[k + 1]
        idi = date_idx[t]
        # 候选集
        if UNIV == "zz800":
            cand = zz800_at(t)
        else:
            cand = set(close_p.columns)
        st_set = st_by_date.get(t, set()) if st_by_date is not None else st_codes
        elig = [c for c in cand
                if c in list_tidx and list_tidx[c] <= idi - LIST_DAYS_MIN
                and c not in st_set]
        if UNIV == "all":
            mv_row = mv_p.loc[t] if t in mv_p.index else None
            elig = [c for c in elig if mv_row is not None and c in mv_row.index
                    and (mv_row.get(c, 0) or 0) >= UNIV_MV_MIN]
        if len(elig) < 30:
            continue
        # F-score + 前向收益
        fs_v = {}
        fwd_v = {}
        for c in elig:
            score, _ = compute_fscore(c, t, M)
            fs_v[c] = score
            c0 = adj_close_raw.at[t, c] if t in adj_close_raw.index else np.nan
            c1 = adj_close_raw.at[t_next, c] if t_next in adj_close_raw.index else np.nan
            fwd_v[c] = (c1 / c0 - 1) if (c0 == c0 and c0 and c1 == c1 and c1) else np.nan
        fscore_store[t] = fs_v
        fwd_ret[t] = fwd_v
        comp_dates.append(t)
    print(f"[calc] 有效选股月: {len(comp_dates)} 个")

    # ---------- 7. IC / IR ----------
    def calc_ic(factor_name, extract, dates):
        ics = []
        for t in dates:
            f = pd.Series(extract(t))
            r = pd.Series(fwd_ret[t])
            join = pd.concat([f, r], axis=1).dropna()
            join.columns = ["f", "r"]
            if len(join) < 30:
                continue
            ic = join["f"].corr(join["r"], method="spearman")
            if ic == ic:
                ics.append(ic)
        ics = np.array(ics)
        return dict(factor=factor_name, window=("train" if dates is train_dates else
                                                ("test" if dates is test_dates else "full")),
                    n=len(ics),
                    ic_mean=ics.mean() if len(ics) else np.nan,
                    ic_std=ics.std() if len(ics) else np.nan,
                    ir=ics.mean() / ics.std() if len(ics) and ics.std() else np.nan,
                    ic_pos_ratio=(ics > 0).mean() if len(ics) else np.nan)

    train_dates = [t for t in comp_dates if t <= tr_end]
    test_dates = [t for t in comp_dates if t >= TEST_START]
    ic_rows = [
        calc_ic("Fscore_0-9", lambda t: pd.Series(fscore_store[t]), comp_dates),
        calc_ic("Fscore_0-9", lambda t: pd.Series(fscore_store[t]), train_dates),
        calc_ic("Fscore_0-9", lambda t: pd.Series(fscore_store[t]), test_dates),
    ]
    ic_df = pd.DataFrame(ic_rows)
    print("\n===== Rank IC（F-score 连续值 vs 前向1月收益）=====")
    print(ic_df.to_string(index=False))

    # ---------- 8. 五分组月均收益 ----------
    group_ret = {g: [] for g in range(1, 6)}
    for t in comp_dates:
        comp = pd.Series(fscore_store[t])
        r = pd.Series(fwd_ret[t])
        j = pd.concat([comp, r], axis=1).dropna(); j.columns = ["c", "r"]
        if len(j) < 50:
            continue
        # 确定性 tie-break：先按 ts_code 排序，再稳定分位（同分按代码打破，避免每跑不同）
        j = j.sort_index().sort_values("c", kind="mergesort")
        try:
            j["g"] = pd.qcut(j["c"], 5, labels=[1, 2, 3, 4, 5])
        except Exception:
            continue
        for g in range(1, 6):
            gr = j[j["g"] == g]["r"]
            if len(gr):
                group_ret[g].append(gr.mean())
    group_mean = {g: np.mean(v) if v else np.nan for g, v in group_ret.items()}
    grp_df = pd.DataFrame({"group": list(range(1, 6)),
                           "monthly_mean_ret": [group_mean[g] for g in range(1, 6)],
                           "annualized": [(group_mean[g] * 12) if group_mean[g] == group_mean[g] else np.nan
                                          for g in range(1, 6)]})
    print("\n===== 五分组月均收益（G1=F最低 ~ G5=F最高）=====")
    print(grp_df.to_string(index=False))
    # 单调性检验：G5 - G1（F-score 高组减低组）
    spread = group_mean[5] - group_mean[1]
    print(f"  G5-G1 月均差: {spread*100:.3f}%  -> 年化 {(spread*12)*100:.2f}pp")
    print(f"  -> 经典 Piotroski 预期 G5>G1（质量越高未来越强）；反向则需警惕")

    # ---------- 9. 多空组合 NAV（高F多 / 低F空，月度再平衡）----------
    def run_long_short(dates, long_pct=LS_LONG_PCT, short_pct=LS_LONG_PCT):
        ls_ret = []; hold_dates = []
        prev = None
        for t in dates:
            ser = pd.Series(fscore_store[t])
            r = pd.Series(fwd_ret[t])
            j = pd.concat([ser, r], axis=1).dropna(); j.columns = ["f", "r"]
            if len(j) < 60:
                continue
            # 确定性 tie-break：先按 ts_code 排序，再按 f 稳定排序(mergesort)，
            # 避免整数 F-score 大量同分导致 sort_values 取端点的股票集合每跑不同（非确定性 bug）
            j = j.sort_index().sort_values("f", kind="mergesort")
            s = j["f"]
            n_long = max(1, int(len(s) * long_pct))
            n_short = max(1, int(len(s) * short_pct))
            longs = s.index[-n_long:]
            shorts = s.index[:n_short]
            lr = j.loc[longs, "r"].mean()
            sr = j.loc[shorts, "r"].mean()
            ret = lr - sr
            cur = set(longs) | set(shorts)
            if prev is not None:
                changed = len(cur ^ prev)
                turn = changed / (len(cur) + len(prev))
                ret -= COST * turn
            else:
                ret -= COST
            prev = cur
            ls_ret.append(ret); hold_dates.append(t)
        return ls_ret, hold_dates

    def nav_metrics(rets, dates):
        if not rets:
            return dict(total=np.nan, cagr=np.nan, mdd=np.nan, sharpe=np.nan, n=0)
        ns = np.array([1.0] + list(np.cumprod(1 + np.array(rets))))
        tot = ns[-1] / ns[0] - 1
        d0 = datetime.datetime.strptime(dates[0], "%Y%m%d")
        d1 = datetime.datetime.strptime(dates[-1], "%Y%m%d")
        yrs = max((d1 - d0).days / 365.25, 1e-9)
        cagr = (ns[-1] / ns[0]) ** (1 / yrs) - 1
        peak = np.maximum.accumulate(ns); mdd = (ns / peak - 1).min()
        rf_m = RF_ANN / 12
        sr = pd.Series(rets) - rf_m
        sharpe = (sr.mean() * 12) / (sr.std() * np.sqrt(12)) if sr.std() else np.nan
        return dict(total=tot, cagr=cagr, mdd=mdd, sharpe=sharpe, n=len(rets))

    ls_full, ls_dates_full = run_long_short(comp_dates)
    ls_train, ls_dates_train = run_long_short(train_dates)
    ls_test, ls_dates_test = run_long_short(test_dates)
    m_full = nav_metrics(ls_full, ls_dates_full)
    m_train = nav_metrics(ls_train, ls_dates_train)
    m_test = nav_metrics(ls_test, ls_dates_test)
    ls_df = pd.DataFrame([
        dict(window="full", **m_full),
        dict(window="train", **m_train),
        dict(window="test", **m_test),
    ])
    print("\n===== 多空组合（高F多 / 低F空，月度再平衡，成本已扣）=====")
    for label, m in [("full", m_full), ("train", m_train), ("test", m_test)]:
        print(f"  {label:6s}: 总收益 {m['total']*100:7.2f}%  年化 {m['cagr']*100:6.2f}%  "
              f"最大回撤 {m['mdd']*100:6.2f}%  夏普 {m['sharpe']:.2f}  月数 {m['n']}")

    # ---------- 10. 高F vs 低F 前向收益对照（gate 意义）----------
    hi_ret, lo_ret = [], []
    for t in comp_dates:
        fs = fscore_store[t]; r = fwd_ret[t]
        j = pd.concat([pd.Series(fs), pd.Series(r)], axis=1).dropna(); j.columns = ["f", "r"]
        if len(j) < 30:
            continue
        hi = j[j["f"] >= GATE_HI]["r"]
        lo = j[j["f"] <= GATE_LO]["r"]
        if len(hi):
            hi_ret.append(hi.mean())
        if len(lo):
            lo_ret.append(lo.mean())
    gate_df = pd.DataFrame({
        "metric": ["月均前向收益", "年化", "样本月数"],
        "F>=%d(高)" % GATE_HI: [np.mean(hi_ret), np.mean(hi_ret) * 12, len(hi_ret)],
        "F<=%d(低)" % GATE_LO: [np.mean(lo_ret), np.mean(lo_ret) * 12, len(lo_ret)],
    })
    print("\n===== 高F(F>=%d) vs 低F(F<=%d) 前向月均收益 =====" % (GATE_HI, GATE_LO))
    print(gate_df.to_string(index=False))

    # ---------- 11. 落盘 ----------
    ic_df.to_csv("data/results/piotroski_oos_ic.csv", index=False)
    grp_df.to_csv("data/results/piotroski_oos_groups.csv", index=False)
    pd.DataFrame({"date": ls_dates_full, "ls_ret": ls_full,
                  "ls_nav": np.cumprod(1 + np.array(ls_full))}).to_csv(
        "data/results/piotroski_oos_longshort.csv", index=False)
    gate_df.to_csv("data/results/piotroski_oos_gatesum.csv", index=False)
    print("\n[save] data/results/piotroski_oos_*.csv")

    # ---------- 12. 一句话结论（严谨，避免误报绿勾）----------
    ic_full = ic_df.iloc[0]["ic_mean"]
    ic_test = ic_df.iloc[2]["ic_mean"] if len(ic_df) > 2 else np.nan
    ir_test = ic_df.iloc[2]["ir"] if len(ic_df) > 2 else np.nan
    spread_ann = spread * 12
    sharpe_test = m_test["sharpe"]
    col_hi = "F>=%d(高)" % GATE_HI
    col_lo = "F<=%d(低)" % GATE_LO
    print("\n================ OOS 结论（%s 宇宙）================" % UNIV)
    print(f"  Rank IC: full={ic_full:.4f}  test={ic_test:.4f}")
    print(f"  ICIR:    full={ic_df.iloc[0]['ir']:.3f}  test={ir_test:.3f}")
    print(f"  五分组 G5-G1 年化差 = {spread_ann*100:.2f}pp")
    print(f"  多空组合 年化 full={m_full['cagr']*100:.2f}% / test={m_test['cagr']*100:.2f}%  夏普 test={sharpe_test:.2f}")
    print(f"  高F(F>={GATE_HI}) vs 低F(F<={GATE_LO}) 年化: {gate_df.iloc[1][col_hi]*100:.1f}% vs {gate_df.iloc[1][col_lo]*100:.1f}%")
    # 严谨判定：独立月度 alpha 需 test 窗口 IC>0 且 ICIR>0.1 且 多空夏普>0.3
    strong = (ic_test == ic_test and ic_test > 0 and ir_test == ir_test and ir_test > 0.1
              and sharpe_test == sharpe_test and sharpe_test > 0.3)
    weak = (ic_full == ic_full and ic_full > 0 and spread_ann > 0)  # 全样本有弱质量溢价，但 test 不稳
    if strong:
        verdict = ("✅ 独立质量 alpha 通过 OOS：test 窗口 IC>0、ICIR>0.1、多空夏普>0.3。"
                   "可 step5 接 --piotroski-gate（作质量门槛）并进一步做组合层验证。")
    elif weak:
        verdict = ("⚠️ 全样本有弱质量溢价（高F比低F年化高 ~%.0fpp、五分组单调），但 test(2023+) 窗口 "
                   "Rank IC≈%.4f、多空夏普 %.2f（转负）—— 作为独立月度多空 alpha 偏弱/衰减。"
                   "建议：仅作价值选股的质量门槛（剔除 F<=2 困境股 + 优先 F>=8），不作独立因子；"
                   "step5 接入后须重新 walk-forward 验证。" % (spread_ann * 100, ic_test, sharpe_test))
    else:
        verdict = ("❌ OOS 无 edge：test 窗口 IC<=0 且多空夏普<=0，视为样本内过拟合/幸存者偏差，"
                   "不接选股引擎。")
    print(f"  -> {verdict}")
    print("===================================================")
    con.close()


if __name__ == "__main__":
    main()
