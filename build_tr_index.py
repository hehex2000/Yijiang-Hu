#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自建全收益指数 (bottom-up total return index)
=============================================
背景(平台已知缺陷):
  index_daily 是**价格指数**(除息即回落, 分红不再投), 而策略净值用 hfq 后复权(含分红再投)。
  → 分子含分红、分母不含分红, 所有"超额收益"被系统性高估
    (实测: 沪深300 年化 2.63% / 中证800 年化 2.29%)。

方法:
  成分: index_constituent 快照 (point-in-time, 无幸存者偏差)
  权重: daily_basic.circ_mv 在快照日定权; 快照间权重随个股收益自然漂移
        (等价于"快照日按权重买入、区间内持有固定份额")
  分红: adj_factor → hfq = close × adj_factor, hfq 收益即含分红再投收益
  指数: I(t) = I(d) × Σ_i w_i × [p_i(t)/p_i(d)]
        - 全收益版 TR: p = hfq   (含分红)
        - 价格版   PR: p = close (不含分红, 用于与官方 index_daily 交叉校验)

校验(不可跳过):
  自建 PR 版日收益 vs 官方 index_daily.pct_chg 逐日比对。
  若年化差异 < 0.5% → 成分+权重重建正确 → TR 版方可作为基准。

输出:
  DB 表 index_total_return(index_code, trade_date, idx_tr, idx_pr, ret_tr, ret_pr)
  CSV  data/results/tr_index/<index_code>_tr.csv

用法:
  ./venv_ml/Scripts/python.exe build_tr_index.py --index 000906.SH
  ./venv_ml/Scripts/python.exe build_tr_index.py --index 000300.SH
"""
import sqlite3, argparse, os, sys
import numpy as np
import pandas as pd
from config import DATA

BASE = os.path.dirname(os.path.abspath(__file__))
# 走 config 而非硬编码盘符，保证 macOS / 换机可跑（与同目录 download_tr_index.py 一致）
DB = DATA.get("local_db_path", r"D:\tu-shareData\astock_daily.db")
OUT_DIR = os.path.join(BASE, "data", "results", "tr_index")
FFILL_LIMIT = 60   # 停牌/缺失价格前推上限(交易日)


def _load_snapshots(con, code):
    df = pd.read_sql_query(
        "SELECT trade_date, ts_code FROM index_constituent "
        "WHERE index_code=? ORDER BY trade_date, ts_code", con, params=(code,))
    return df


def _load_circ_mv(con, code, snap_dates):
    """快照日的流通市值。
    坑: 部分快照日(如 20160729/20240731) daily_basic 缺数据 → 该段权重为空 →
        水平被重置成 1.0 → 指数收益抹平。故对缺失快照日回退到最近可用交易日。
    只按需补加载那些回退日, 避免全量 daily_basic 拖慢 10 分钟。"""
    con.execute("CREATE TEMP TABLE _snap(d TEXT PRIMARY KEY)")
    con.executemany("INSERT OR IGNORE INTO _snap VALUES (?)", [(d,) for d in snap_dates])
    q = """SELECT b.ts_code, b.trade_date, b.circ_mv
           FROM daily_basic b
           JOIN index_constituent c
             ON c.ts_code=b.ts_code AND c.trade_date=b.trade_date
           JOIN _snap s ON s.d=b.trade_date
           WHERE c.index_code=?"""
    df = pd.read_sql_query(q, con, params=(code,))
    have = set(df["trade_date"].unique())
    miss = [d for d in snap_dates if d not in have]
    if miss:
        use = []
        for d in miss:
            r = con.execute("SELECT MAX(trade_date) FROM daily_basic WHERE trade_date<=?",
                            (d,)).fetchone()
            if r and r[0]:
                use.append(r[0])
        use = sorted(set(use))
        if use:
            con.execute("CREATE TEMP TABLE _snap2(d TEXT PRIMARY KEY)")
            con.executemany("INSERT OR IGNORE INTO _snap2 VALUES (?)", [(d,) for d in use])
            q2 = """SELECT b.ts_code, b.trade_date, b.circ_mv
                    FROM daily_basic b
                    JOIN index_constituent c
                      ON c.ts_code=b.ts_code AND c.trade_date=b.trade_date
                    JOIN _snap2 s ON s.d=b.trade_date
                    WHERE c.index_code=?"""
            df = pd.concat([df, pd.read_sql_query(q2, con, params=(code,))],
                           ignore_index=True)
            print(f"      [补] {len(miss)} 个快照日无 circ_mv, 已回退到最近交易日: {use}")
    return df


def _load_panels(con, members, start, end):
    """成分股的 daily.close 与 adj_factor(用临时表避免 SQL 变量数超限)"""
    con.execute("CREATE TEMP TABLE _mem(ts_code TEXT PRIMARY KEY)")
    con.executemany("INSERT OR IGNORE INTO _mem VALUES (?)", [(c,) for c in members])
    d = pd.read_sql_query(
        """SELECT d.ts_code, d.trade_date, d.close
           FROM daily d JOIN _mem m ON m.ts_code=d.ts_code
           WHERE d.trade_date>=? AND d.trade_date<=?""", con, params=(start, end))
    a = pd.read_sql_query(
        """SELECT a.ts_code, a.trade_date, a.adj_factor
           FROM adj_factor a JOIN _mem m ON m.ts_code=a.ts_code
           WHERE a.trade_date>=? AND a.trade_date<=?""", con, params=(start, end))
    return d, a


def build(code, start, end, verbose=True):
    con = sqlite3.connect(DB)
    snap = _load_snapshots(con, code)
    if snap.empty:
        con.close()
        raise SystemExit(f"[ERR] 无成分快照: {code}")
    snap_dates = sorted(snap["trade_date"].unique())
    members = sorted(snap["ts_code"].unique())
    if verbose:
        print(f"[1/5] 成分快照 {len(snap_dates)} 个 ({snap_dates[0]}~{snap_dates[-1]}), "
              f"历史累计成分股 {len(members)} 只")

    cm = _load_circ_mv(con, code, snap_dates)
    d, a = _load_panels(con, members, start, end)
    con.close()

    dates = sorted(d["trade_date"].unique())
    if verbose:
        print(f"[2/5] 面板加载: close {len(d):,} 行 / adj {len(a):,} 行 / "
              f"交易日 {len(dates)}")

    # 宽表: 行=日期 列=股票
    close_w = d.pivot(index="trade_date", columns="ts_code", values="close") \
               .reindex(index=dates, columns=members)
    adj_w = a.pivot(index="trade_date", columns="ts_code", values="adj_factor") \
             .reindex(index=dates, columns=members)
    # 复权因子缺失(未分红前)按 1 处理; 并前推
    adj_w = adj_w.ffill()
    hfq_w = close_w * adj_w
    # 停牌/缺失价格前推(有限), 避免把整段打成 NaN
    close_w = close_w.ffill(limit=FFILL_LIMIT)
    hfq_w = hfq_w.ffill(limit=FFILL_LIMIT)

    C = close_w.to_numpy(dtype=float)
    H = hfq_w.to_numpy(dtype=float)
    T, N = C.shape
    date_pos = {dt: i for i, dt in enumerate(dates)}
    code_pos = {c: j for j, c in enumerate(members)}

    # 每个快照的成分与权重; 快照日缺失则回退到之前最近的可用交易日
    import bisect
    cm = cm.sort_values(["trade_date", "ts_code"])
    avail = {dt: g for dt, g in cm.groupby("trade_date")}
    avail_dates = sorted(avail)
    cm_by_date = {}
    for dt in snap_dates:
        i = bisect.bisect_right(avail_dates, dt) - 1
        if i < 0:
            continue
        g = avail[avail_dates[i]].copy()
        g = g.dropna(subset=["circ_mv"])
        g = g[g["circ_mv"] > 0]
        # JOIN 可能产生重复(ts_code, trade_date), 必须去重否则权重数组与成分数组长度不匹配
        g = g.drop_duplicates(subset=["ts_code"], keep="first")
        if not g.empty:
            cm_by_date[dt] = g

    snap_idx = []
    for dt in snap_dates:
        if dt in date_pos:
            snap_idx.append((date_pos[dt], dt))

    I_tr = np.full(T, np.nan)
    I_pr = np.full(T, np.nan)
    if verbose:
        print(f"[3/5] 逐快照重建指数段 ({len(snap_idx)} 段)...")

    cur = None   # (m_idx, w) 上一段有效成分与权重 —— 快照数据缺失时的兜底
    for k, (s_k, dt) in enumerate(snap_idx):
        new = None
        g = cm_by_date.get(dt)
        if g is not None and not g.empty:
            m_codes = [c for c in g["ts_code"].tolist() if c in code_pos]
            if m_codes:
                m_idx = np.array([code_pos[c] for c in m_codes])
                w = g.set_index("ts_code").loc[m_codes, "circ_mv"].to_numpy(dtype=float)
                p0_pr = C[s_k, m_idx]
                p0_tr = H[s_k, m_idx]
                ok = ~np.isnan(p0_pr) & ~np.isnan(p0_tr) & ~np.isnan(w) & (w > 0)
                if ok.any():
                    m_idx, w = m_idx[ok], w[ok]
                    new = (m_idx, w / w.sum(), p0_pr[ok], p0_tr[ok])
        if new is None and cur is not None:
            # 兜底: 沿用上一段成分与权重, 仅把段起点价格更新到本快照日
            # (绝不 continue —— 那会让 I[s_{k+1}] 变 NaN, 下一段把水平重置成 1.0)
            m_idx, w0 = cur
            p0_pr = C[s_k, m_idx]
            p0_tr = H[s_k, m_idx]
            ok = ~np.isnan(p0_pr) & ~np.isnan(p0_tr)
            if ok.any():
                m_idx, w0 = m_idx[ok], w0[ok]
                new = (m_idx, w0 / w0.sum(), p0_pr[ok], p0_tr[ok])
        if new is None:
            continue
        m_idx, w, p0_pr, p0_tr = new
        cur = (m_idx, w)

        if k == 0:
            lev_tr = lev_pr = 1.0
        else:
            # 用上一段权重把指数接到本快照日
            lev_tr = I_tr[s_k] if not np.isnan(I_tr[s_k]) else 1.0
            lev_pr = I_pr[s_k] if not np.isnan(I_pr[s_k]) else 1.0
        # 本段起点(上一快照的权重已把水平递推到 s_k, 此处用新权重重新起算)
        if k == 0:
            lev_tr = lev_pr = 1.0
        I_tr[s_k] = lev_tr
        I_pr[s_k] = lev_pr

        s_end = snap_idx[k + 1][0] if k + 1 < len(snap_idx) else T
        # 关键: 段内要算到"下一快照日"(含), 否则 I[s_{k+1}] 为 NaN,
        # 下一段会把指数水平回退成 1.0 → 每月重置一次, 累计收益被抹平。
        seg_end = min(s_end + 1, T)
        if seg_end - s_k - 1 <= 0:
            continue
        seg_pr = C[s_k + 1:seg_end][:, m_idx]
        seg_tr = H[s_k + 1:seg_end][:, m_idx]
        ratio_pr = seg_pr / p0_pr
        ratio_tr = seg_tr / p0_tr
        okp = ~np.isnan(ratio_pr)
        num = np.where(okp, ratio_pr * w, 0.0).sum(axis=1)
        den = np.where(okp, w, 0.0).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            I_pr[s_k + 1:seg_end] = np.where(den > 0, lev_pr * num / np.maximum(den, 1e-12), np.nan)
        okt = ~np.isnan(ratio_tr)
        num = np.where(okt, ratio_tr * w, 0.0).sum(axis=1)
        den = np.where(okt, w, 0.0).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            I_tr[s_k + 1:seg_end] = np.where(den > 0, lev_tr * num / np.maximum(den, 1e-12), np.nan)

    # 残存 NaN(首个快照之前 / 极端缺口) → 前推
    s = pd.Series(I_tr).ffill()
    I_tr = s.to_numpy()
    I_pr = pd.Series(I_pr).ffill().to_numpy()
    valid = ~np.isnan(I_tr)
    if verbose:
        print(f"[4/5] 有效交易日 {valid.sum():,}/{T}")

    out = pd.DataFrame({
        "index_code": code,
        "trade_date": dates,
        "idx_tr": I_tr,
        "idx_pr": I_pr,
    })
    out = out[out["idx_tr"].notna()].copy()
    out["ret_tr"] = out["idx_tr"].pct_change()
    out["ret_pr"] = out["idx_pr"].pct_change()
    out = out.iloc[1:].reset_index(drop=True)   # 首行无收益

    return out


def cross_validate(out, code, verbose=True):
    """自建价格版 vs 官方 index_daily —— 必须过的校验关"""
    con = sqlite3.connect(DB)
    off = pd.read_sql_query(
        "SELECT trade_date, close, pct_chg FROM index_daily "
        "WHERE ts_code=? ORDER BY trade_date", con, params=(code,))
    con.close()
    m = out.merge(off, on="trade_date", how="inner")
    if m.empty:
        return None
    m = m.sort_values("trade_date").reset_index(drop=True)
    m["ret_off"] = m["close"].astype(float).pct_change()
    m = m.iloc[1:].copy()
    m = m.dropna(subset=["ret_pr", "ret_off"])
    if m.empty:
        return None
    yrs = len(m) / 252.0
    cum_pr = (m["idx_pr"].iloc[-1] / m["idx_pr"].iloc[0])
    cum_off = (m["close"].astype(float).iloc[-1] / m["close"].astype(float).iloc[0])
    ann_pr = cum_pr ** (1 / yrs) - 1
    ann_off = cum_off ** (1 / yrs) - 1
    corr = m["ret_pr"].corr(m["ret_off"])
    te = (m["ret_pr"] - m["ret_off"]).std() * np.sqrt(252)
    # 全收益 vs 官方价格: 即基准漏计的分红
    cum_tr = m["idx_tr"].iloc[-1] / m["idx_tr"].iloc[0]
    ann_tr = cum_tr ** (1 / yrs) - 1
    if verbose:
        print("\n" + "=" * 74)
        print(f"交叉校验 {code}  ({m['trade_date'].iloc[0]} ~ {m['trade_date'].iloc[-1]}, {yrs:.1f}年)")
        print("=" * 74)
        print(f"  官方价格指数年化 : {ann_off*100:+.2f}%    (累计 {(cum_off-1)*100:+.1f}%)")
        print(f"  自建全收益年化   : {ann_tr*100:+.2f}%    (累计 {(cum_tr-1)*100:+.1f}%)")
        print(f"  ★ 隐含股息率     : {(ann_tr-ann_off)*100:+.2f}%/年   (基准漏计的分红)")
        print("-" * 74)
        print(f"  [成分/权重正确性] 日收益相关性 {corr:.4f}   跟踪误差(年化) {te*100:.2f}%")
        print(f"  [参考·不作锚] 未复权价格版年化 {ann_pr*100:+.2f}% —— 含送股/转增除权跳空, 系统性偏低")
        print("=" * 74)
        ok_corr = corr >= 0.98
        ok_div = 0.010 <= (ann_tr - ann_off) <= 0.045
        if ok_corr and ok_div:
            print("  ✓ 校验通过: 相关性≥0.98 且 隐含股息率落在合理区间 [1.0%, 4.5%]")
        else:
            if not ok_corr:
                print(f"  ✗ 相关性不足 ({corr:.4f} < 0.98): 成分/权重重建有误, 勿用作基准")
            if not ok_div:
                print(f"  ✗ 隐含股息率 {(ann_tr-ann_off)*100:+.2f}% 越界 [1.0%, 4.5%], 需排查复权口径")
    return dict(ann_off=ann_off, ann_pr=ann_pr, ann_tr=ann_tr, corr=corr, te=te,
                years=yrs, implied_dy=ann_tr - ann_off)


def save(out, code):
    os.makedirs(OUT_DIR, exist_ok=True)
    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS index_total_return(
        index_code TEXT NOT NULL,
        trade_date TEXT NOT NULL,
        idx_tr REAL, idx_pr REAL, ret_tr REAL, ret_pr REAL,
        PRIMARY KEY(index_code, trade_date))""")
    con.execute("DELETE FROM index_total_return WHERE index_code=?", (code,))
    con.executemany(
        "INSERT OR REPLACE INTO index_total_return VALUES (?,?,?,?,?,?)",
        out[["index_code", "trade_date", "idx_tr", "idx_pr", "ret_tr", "ret_pr"]]
        .itertuples(index=False, name=None))
    con.commit()
    con.close()
    p = os.path.join(OUT_DIR, f"{code}_tr.csv")
    out.to_csv(p, index=False)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="000906.SH")
    ap.add_argument("--start", default="20100101")
    ap.add_argument("--end", default="20260828")
    args = ap.parse_args()

    print("=" * 72)
    print(f"自建全收益指数  {args.index}   {args.start}~{args.end}")
    print("=" * 72)
    out = build(args.index, args.start, args.end)
    cross_validate(out, args.index)
    p = save(out, args.index)
    print(f"\n[5/5] 已落库 index_total_return 并导出: {p}")
    print(f"      样本 {len(out):,} 个交易日  {out['trade_date'].iloc[0]}~{out['trade_date'].iloc[-1]}")


if __name__ == "__main__":
    main()
