# -*- coding: utf-8 -*-
"""
券商板块「PE 陷阱」因子 —— 数据构建层

背景：B站 BV1Lghg6NEA8 声称「券商 PE 最低 = 业绩顶峰 = 风险最大」，
      2026 年券商 PE 13-15 倍为历史最低。本脚本用平台数据独立复现该说法，
      并把「行业 PE 分位」做成可检验的时序择时因子。

设计要点
--------
1) 这是【时序择时因子】不是【截面选股因子】：
   截面只有一个行业，无法算横截面 IC。检验方式为
   Spearman(分位_t, 未来H日超额收益_t) 的时序相关。

2) 复权口径：板块收益用 hfq（含分红），基准用 bench_index 全收益，
   两端同口径。因子侧（PE/PB）本身不受复权影响。

3) 聚合口径：市值加权「整体法」PE = SUM(total_mv) / SUM(total_mv/pe_ttm)，
   等价于 SUM(总市值)/SUM(归母净利润)，对负/零盈利自动剔除。

4) ⚠️ 已知偏差（Gate 0 标注，见报告）：
   stock_industry 表只有 (ts_code, industry_code) 两列、**无日期字段**，
   即当前快照。回溯历史成分股存在前视偏差 + 幸存者偏差。
   券商退市极少（牌照稀缺），主要风险来自「当时尚未上市」的票被自动排除，
   以及个别票（如东方财富）早期行业归类不同。

用法
----
  python broker_pe_factor.py --build            # 构建并缓存板块序列
  python broker_pe_factor.py --snapshot         # 打印序列快照
"""
import os
import sys
import argparse
import sqlite3

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATA  # noqa: E402

OUT_DIR = os.path.join("data", "results", "broker_pe")
CACHE = os.path.join(OUT_DIR, "sector_daily.csv")

BENCH_CODE = "000906.SH"      # 中证800（000905.SH 才是中证500，勿混）
START = "20100101"
END = "20260831"


def get_conn():
    return sqlite3.connect(DATA.get("local_db_path"))


# ---------------------------------------------------------------- 成分股
def industry_members(con, industry_code):
    """取行业成分股（当前快照，无历史版本 → 含前视偏差，已在文档标注）。"""
    q = "SELECT ts_code FROM stock_industry WHERE industry_code=? ORDER BY ts_code"
    return [r[0] for r in con.execute(q, (industry_code,))]


def industry_name(con, industry_code):
    r = con.execute(
        "SELECT industry_name FROM industry_sector WHERE industry_code=?",
        (industry_code,)).fetchone()
    return r[0] if r else industry_code


# ---------------------------------------------------------------- 面板加载
def load_panel(con, codes, start=START, end=END):
    """逐票加载 close / circ_mv / total_mv / pe_ttm / pb / adj_factor。

    adj_factor 存在整日缺行（2020-2026 有 132 天），必须 ffill + bfill，
    不能用 fillna(1.0)（会造成假跳空）。
    """
    ph = ",".join("?" * len(codes))
    px = pd.read_sql(
        f"SELECT ts_code, trade_date, close, vol, amount FROM daily "
        f"WHERE ts_code IN ({ph}) AND trade_date BETWEEN ? AND ? ORDER BY ts_code, trade_date",
        con, params=list(codes) + [start, end])
    bs = pd.read_sql(
        f"SELECT ts_code, trade_date, total_mv, circ_mv, pe_ttm, pb FROM daily_basic "
        f"WHERE ts_code IN ({ph}) AND trade_date BETWEEN ? AND ? ORDER BY ts_code, trade_date",
        con, params=list(codes) + [start, end])
    aj = pd.read_sql(
        f"SELECT ts_code, trade_date, adj_factor FROM adj_factor "
        f"WHERE ts_code IN ({ph}) AND trade_date <= ? ORDER BY ts_code, trade_date",
        con, params=list(codes) + [end])

    df = px.merge(bs, on=["ts_code", "trade_date"], how="inner")
    df = df.merge(aj, on=["ts_code", "trade_date"], how="left")

    # 按票填充 adj_factor：先 ffill（沿用上一已知因子）再 bfill（上市首日之前）
    df = df.sort_values(["ts_code", "trade_date"])
    df["adj_factor"] = df.groupby("ts_code")["adj_factor"].ffill().bfill()
    df["adj_factor"] = df["adj_factor"].fillna(1.0)
    return df


# ---------------------------------------------------------------- 板块序列
def build_sector_series(df):
    """自建板块全收益指数 + 市值加权估值序列。

    收益率：r_i,t = (close_t * adj_t) / (close_{t-1} * adj_{t-1}) - 1   （hfq，含分红）
    组合收益：用【滞后一期】流通市值加权，避免用当日收盘价定权重的自我实现。
    """
    d = df.sort_values(["ts_code", "trade_date"]).copy()
    d["px_adj"] = d["close"] * d["adj_factor"]
    g = d.groupby("ts_code", sort=False)
    d["ret"] = g["px_adj"].pct_change()
    d["w_lag"] = g["circ_mv"].shift(1)          # 滞后权重

    # 组合收益：逐日加权平均（权重为上一日流通市值）
    def wavg(sub):
        m = sub["ret"].notna() & sub["w_lag"].notna() & (sub["w_lag"] > 0)
        if not m.any():
            return np.nan
        return np.average(sub.loc[m, "ret"], weights=sub.loc[m, "w_lag"])

    port_ret = d.groupby("trade_date", sort=True).apply(wavg)
    port_ret.name = "ret"

    # 估值：整体法市值加权（仅正 pe / 正 pb）
    def agg_pe(sub):
        m = (sub["pe_ttm"] > 0) & (sub["total_mv"] > 0)
        if m.sum() < 3:
            return np.nan
        earn = (sub.loc[m, "total_mv"] / sub.loc[m, "pe_ttm"]).sum()
        return sub.loc[m, "total_mv"].sum() / earn if earn > 0 else np.nan

    def agg_pb(sub):
        m = (sub["pb"] > 0) & (sub["total_mv"] > 0)
        if m.sum() < 3:
            return np.nan
        bv = (sub.loc[m, "total_mv"] / sub.loc[m, "pb"]).sum()
        return sub.loc[m, "total_mv"].sum() / bv if bv > 0 else np.nan

    out = d.groupby("trade_date", sort=True).apply(
        lambda s: pd.Series({
            "pe": agg_pe(s),
            "pb": agg_pb(s),
            "n_stock": s["ts_code"].nunique(),
            "n_pos_pe": int((s["pe_ttm"] > 0).sum()),
            "circ_mv": s["circ_mv"].sum(),
        }))
    out = out.join(port_ret).reset_index()

    out["nav"] = (1 + out["ret"].fillna(0)).cumprod()
    return out


def attach_benchmark(sec, code=BENCH_CODE, start=START, end=END):
    """基准用全收益口径，与 hfq 板块收益同口径。失败则显式报错（不静默降级）。"""
    try:
        import bench_index
        con = get_conn()
        bdf, meta = bench_index.load_benchmark(code, start, end, conn=con,
                                               nav_price_mode="hfq")
        con.close()
        if bdf is None or len(bdf) == 0:
            raise RuntimeError(f"基准 {code} 无数据: {meta}")
        print(f"[基准] {code} mode={meta.get('mode')} "
              f"source={meta.get('source_table')} resolved={meta.get('resolved_code')}")
        bdf = bdf.rename(columns={"close": "bench_close"})
        return sec.merge(bdf[["trade_date", "bench_close"]], on="trade_date", how="left"), meta
    except Exception as e:
        raise RuntimeError(f"无法加载全收益基准 {code}: {e}\n"
                           f"按项目约定，缺数据应补全而非降级到价格指数。")


# ---------------------------------------------------------------- 因子构造
def add_factor(sec, windows=(500, 750, 1000)):
    """时序分位：当前估值在过去 w 日窗口内的百分位（仅用 <=t 数据，无前视）。"""
    sec = sec.sort_values("trade_date").reset_index(drop=True)
    for col in ("pe", "pb"):
        for w in windows:
            r = sec[col].rolling(w, min_periods=int(w * 0.8))
            sec[f"{col}_pct{w}"] = r.apply(
                lambda x: (x < x.iloc[-1]).mean() * 100, raw=False)
    return sec


def add_forward_excess(sec, horizons=(60, 120, 250)):
    """未来 H 日超额收益：板块 hfq 收益 − 基准全收益。

    注意：入口用 T+1（避免用 close[t] 同时定因子和收益）。
    """
    nav = sec["nav"].values
    bc = sec["bench_close"].values
    for H in horizons:
        f_sec = np.full(len(sec), np.nan)
        f_bch = np.full(len(sec), np.nan)
        # 起点 t+1 的净值，终点 t+1+H 的净值 → 收益区间 [t+1, t+1+H]
        if len(nav) > H + 1:
            f_sec[:len(nav) - H - 1] = nav[H + 1:] / nav[1:len(nav) - H] - 1
            f_bch[:len(bc) - H - 1] = bc[H + 1:] / bc[1:len(bc) - H] - 1
        sec[f"fwd_sec_{H}"] = f_sec
        sec[f"fwd_bch_{H}"] = f_bch
        sec[f"fwd_exc_{H}"] = f_sec - f_bch
    return sec


# ---------------------------------------------------------------- 主流程
def build(industry_code="IND_0029", verbose=True):
    os.makedirs(OUT_DIR, exist_ok=True)
    con = get_conn()
    codes = industry_members(con, industry_code)
    name = industry_name(con, industry_code)
    if verbose:
        print(f"[成分] {industry_code} ({name}) 共 {len(codes)} 只")
    if len(codes) == 0:
        con.close()
        raise RuntimeError(f"行业 {industry_code} 无成分股")

    df = load_panel(con, codes)
    con.close()
    if verbose:
        print(f"[面板] {len(df)} 行 | {df.ts_code.nunique()} 票 | "
              f"{df.trade_date.min()} ~ {df.trade_date.max()}")

    sec = build_sector_series(df)
    sec, bmeta = attach_benchmark(sec)
    sec["industry_code"] = industry_code
    sec["industry_name"] = name

    miss = sec["bench_close"].isna().sum()
    if miss:
        print(f"⚠️ 基准缺失 {miss} 日，将前向填充")
        sec["bench_close"] = sec["bench_close"].ffill()

    sec = add_factor(sec)
    sec = add_forward_excess(sec)
    return sec, bmeta


def snapshot(sec):
    print("\n=== 板块序列快照（每年末）===")
    s = sec.copy()
    s["yr"] = s["trade_date"].astype(str).str[:4]
    idx = s.groupby("yr")["trade_date"].idxmax()
    cols = ["trade_date", "n_stock", "pe", "pb", "nav", "pe_pct750", "pb_pct750"]
    t = s.loc[idx, cols].copy()
    t["板块年收益%"] = (s.set_index("trade_date")["nav"].pct_change(250)
                       .reindex(t["trade_date"]).values * 100)
    t = t.round(2)
    print(t.to_string(index=False))

    print("\n=== PE 极值时段（PE 分位 <10 或 >90，750日窗口）===")
    e = s[(s.pe_pct750 < 10) | (s.pe_pct750 > 90)]
    if len(e):
        # 合并连续段
        seg, prev = [], None
        for _, r in e.iterrows():
            if prev is None or r.name - prev > 30:
                seg.append([r.name, r.name])
            else:
                seg[-1][1] = r.name
            prev = r.name
        for a, b in seg:
            sub = s.loc[a:b]
            print(f"  {s.loc[a,'trade_date']} ~ {s.loc[b,'trade_date']}  "
                  f"PE {sub.pe.mean():6.2f} (分位 {sub.pe_pct750.mean():5.1f})  "
                  f"[{len(sub)}日]")

    print("\n=== 最新状态 ===")
    last = s.iloc[-1]
    print(f"  日期 {last.trade_date}  PE {last.pe:.2f} (分位 {last.pe_pct750:.1f})  "
          f"PB {last.pb:.2f} (分位 {last.pb_pct750:.1f})")


def main():
    ap = argparse.ArgumentParser(description="券商 PE 陷阱因子 —— 数据构建层")
    ap.add_argument("--build", action="store_true", help="构建并缓存板块序列")
    ap.add_argument("--snapshot", action="store_true", help="打印序列快照")
    ap.add_argument("--industry", default="IND_0029",
                    help="行业代码（默认 IND_0029 证券）")
    ap.add_argument("--out", default=CACHE, help="输出 CSV 路径")
    args = ap.parse_args()

    if not (args.build or args.snapshot):
        args.build = args.snapshot = True

    if args.build:
        sec, meta = build(args.industry)
        sec.to_csv(args.out, index=False, encoding="utf-8-sig")
        print(f"\n[输出] {args.out}  ({len(sec)} 行)")
        globals()["_SEC"] = sec
    if args.snapshot:
        sec = globals().get("_SEC")
        if sec is None:
            sec = pd.read_csv(args.out, dtype={"trade_date": str})
        snapshot(sec)


if __name__ == "__main__":
    main()
