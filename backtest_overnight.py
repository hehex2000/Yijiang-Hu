# -*- coding: utf-8 -*-
"""
一夜持股法（隔夜持仓）回测引擎
================================
思路：T日尾盘选股 → 以T日收盘价买入 → T+1日开盘价卖出，赚"收盘→次日开盘"价差。
卖点一律用 daily.open 真实开盘价（绕过 get_open_price 的盘后定价 hack）。

本文件分阶段实现：
  v1 (本版): 裸收益 —— 三档选股口径 × (毛收益 / 扣0.35%成本净收益)
  v2 (后续): 封板情景（次日跌停卖不掉强制持有 / 涨停是否立即卖）
  v3 (后续): 卖点扩展（次日收盘 / 持有N日）

费用模型与 run_monthly_rebalance.py 完全一致：
  佣金 0.00025/边(最低5元) + 印花 0.0005(2023-08-28起,卖出) + 滑点 0.001/边
  → 单笔往返 ≈ 0.35%
"""
import sqlite3, sys, os, argparse, math
import numpy as np
import pandas as pd

DB = "D:/tu-shareData/astock_daily.db"

# ---------- 费用模型（与平台一致）----------
COMMISSION_RATE = 0.00025
COMMISSION_MIN = 5.0
SLIPPAGE_RATE = 0.001
STAMP_OLD = 0.001
STAMP_NEW = 0.0005
STAMP_CUT = 20230828

def stamp_rate(td):
    return STAMP_NEW if td >= STAMP_CUT else STAMP_OLD

def trade_net_ret(notional, buy_price, sell_price, buy_td, sell_td):
    """固定名义本金 notional，买→卖一趟的净收益率（含佣金/印花/滑点）。"""
    buy_fill = buy_price * (1 + SLIPPAGE_RATE)
    comm_b = max(notional * COMMISSION_RATE, COMMISSION_MIN)
    shares = int((notional - comm_b) / buy_fill)
    if shares <= 0:
        return None
    sell_fill = sell_price * (1 - SLIPPAGE_RATE)
    gross = shares * sell_fill
    comm_s = max(gross * COMMISSION_RATE, COMMISSION_MIN)
    stamp = gross * stamp_rate(sell_td)
    proceeds = gross - comm_s - stamp
    return proceeds / notional - 1.0

# ---------- 参数 ----------
START = "20150101"
END   = "20260630"
RET_THRESH    = 0.03     # 当日涨幅 > 3%
RANGE_POS_MIN = 0.80     # 收盘位于当日区间上部 20%
VOL_RATIO_MIN = 1.0      # 量比 >= 1（放量）
CIRC_MV_MIN   = 2_000_000   # 流通市值 >= 20亿（daily_basic 单位为千元）
AMOUNT_MIN    = 30_000      # 成交额 >= 3000万（daily.amount 单位为千元）
MIN_AGE_DAYS = 60        # 上市 >= 60 交易日

def load_data(start, end):
    con = sqlite3.connect(DB)
    # 注：daily_basic 在大量交易日为空（稀疏），故全量载入后用 merge_asof 向前取最近一条
    d = pd.read_sql_query(
        "SELECT ts_code, trade_date, open, high, low, close, pre_close, vol, amount "
        "FROM daily WHERE trade_date BETWEEN ? AND ?", con, params=(start, end))
    db_ = pd.read_sql_query(
        "SELECT ts_code, trade_date, circ_mv, volume_ratio FROM daily_basic", con)
    sb = pd.read_sql_query("SELECT ts_code, name, list_date, industry FROM stock_basic", con)
    con.close()
    for c in ["open","high","low","close","pre_close","vol","amount"]:
        d[c] = d[c].astype("float32")
    for c in ["circ_mv","volume_ratio"]:
        db_[c] = db_[c].astype("float32")
    d["trade_date"] = d["trade_date"].astype(int)
    db_["trade_date"] = db_["trade_date"].astype(int)
    d = d.sort_values("trade_date")
    db_ = db_.sort_values("trade_date")
    # as-of 合并：每个交易日向前取最近一条 daily_basic（市值/量比），补齐稀疏缺口
    df = pd.merge_asof(d, db_, on="trade_date", by="ts_code", direction="backward")
    sb["is_st"] = sb["name"].str.contains("ST", na=False).astype(bool)
    sb["list_date"] = pd.to_datetime(sb["list_date"], errors="coerce")
    df = df.merge(sb[["ts_code","name","is_st","list_date","industry"]], on="ts_code", how="left")
    df["is_st"] = df["is_st"].fillna(False).astype(bool)
    return df

def build_signals(df):
    df = df.sort_values(["ts_code","trade_date"]).copy()
    # 基础因子
    df["ret"] = df["close"] / df["pre_close"] - 1.0
    df["range_pos"] = (df["close"] - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan)
    df["white"] = df["close"] > df["open"]
    df["is_limit_up_close"] = (df["high"] == df["close"]) & (df["close"] >= df["pre_close"]*1.095)
    # 上市年龄（交易日数）
    td_min = df.groupby("ts_code")["trade_date"].transform("min")
    df["age_days"] = (df["trade_date"] - td_min) // 1 + 1  # 近似：用首个交易日到现在
    # 实际年龄用 list_date 更准
    ref = pd.to_datetime(df["trade_date"].astype(str), format="%Y%m%d")
    df["age_cal"] = (ref - df["list_date"]).dt.days
    # T+1 的 open / pre_close（真实开盘价，来自 daily.open 直接读取）
    g = df.groupby("ts_code")
    df["next_open"]   = g["open"].shift(-1)
    df["next_pre"]    = g["pre_close"].shift(-1)
    df["next_td"]     = g["trade_date"].shift(-1)
    # 次日开盘是否封板
    df["next_limit_down_open"] = df["next_open"] <= df["next_pre"]*0.905
    df["next_limit_up_open"]   = df["next_open"] >= df["next_pre"]*1.095
    return df

def select(df, mode):
    """返回带 buy_close / sell_open 的交易样本（已过滤不可交易情形）。"""
    df = df.copy()
    # 股票池过滤
    df["eligible"] = (
        (~df["is_st"]) &
        (df["age_cal"] >= 0) & (df["age_cal"] >= 0) &
        (df["circ_mv"] >= CIRC_MV_MIN) &
        (df["amount"] >= AMOUNT_MIN) &
        (~df["is_limit_up_close"]) &          # 收盘已封涨停 → 买不进
        (df["next_open"].notna())            # 有次日开盘价
    )
    # 上市 >= MIN_AGE_DAYS 交易日（用日历天近似，保守）
    df.loc[df["age_cal"] < (MIN_AGE_DAYS*1.5), "eligible"] = False
    vol_ok = df["volume_ratio"].fillna(VOL_RATIO_MIN) >= VOL_RATIO_MIN  # 缺失按中性1.0
    mom = (df["ret"] > RET_THRESH) & (df["range_pos"] > RANGE_POS_MIN) & df["white"] & vol_ok
    if mode == "all":
        df["pick"] = df["eligible"]
    elif mode == "momentum":
        df["pick"] = df["eligible"] & mom
    elif mode == "sector":
        # 板块共振：个股 ret > 同行业当日中位数 ret（仅 eligible 内计算）
        med = df.loc[df["eligible"]].groupby(["trade_date","industry"])["ret"].transform("median")
        df["ind_med"] = np.nan
        df.loc[df["eligible"], "ind_med"] = med
        df["pick"] = df["eligible"] & mom & (df["ret"] > df["ind_med"])
    else:
        raise ValueError(mode)
    out = df[df["pick"]].copy()
    out["gross_ret"] = out["next_open"] / out["close"] - 1.0
    return out

def summarize(out, mode, with_cost):
    n = len(out)
    if n == 0:
        return None
    gr = out["gross_ret"].values
    if with_cost:
        notional = 100000.0
        nets = []
        for _, r in out.iterrows():
            nr = trade_net_ret(notional, r["close"], r["next_open"], int(r["trade_date"]), int(r["next_td"]))
            if nr is not None:
                nets.append(nr)
        nets = np.array(nets)
        mean_ret = nets.mean(); win = (nets > 0).mean()
        label = "净(扣0.35%成本)"
    else:
        mean_ret = gr.mean(); win = (gr > 0).mean()
        label = "毛"
    tdays = out["trade_date"].nunique()
    n_years = (out["trade_date"].max() - out["trade_date"].min()) / 1e4 / 365.0 + 0.2
    trades_per_year = n / max(n_years, 0.5)
    ld = out["next_limit_down_open"].mean() if "next_limit_down_open" in out else np.nan
    lu = out["next_limit_up_open"].mean() if "next_limit_up_open" in out else np.nan
    return {
        "mode": mode, "label": label, "n": n,
        "mean_ret_%": round(mean_ret*100, 4),
        "median_%": round(np.median(gr)*100, 4),
        "win_%": round(win*100, 2),
        "trades/yr": round(trades_per_year, 0),
        "next_limit_down_%": round(ld*100, 3) if not np.isnan(ld) else None,
        "next_limit_up_%": round(lu*100, 3) if not np.isnan(lu) else None,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=START)
    ap.add_argument("--end", default=END)
    ap.add_argument("--modes", default="all,momentum,sector")
    ap.add_argument("--cost", action="store_true", help="扣除0.35%往返成本")
    ap.add_argument("--out", default="data/results/overnight/overnight_summary.csv")
    args = ap.parse_args()

    print(f"加载数据 {args.start}~{args.end} ...")
    df = load_data(args.start, args.end)
    print(f"  日线行数={len(df):,}  股票数={df['ts_code'].nunique():,}")
    df = build_signals(df)
    modes = [m.strip() for m in args.modes.split(",")]
    rows = []
    for m in modes:
        out = select(df, m)
        s = summarize(out, m, args.cost)
        if s:
            rows.append(s)
            print(f"  [{m}] N={s['n']:,}  均值={s['mean_ret_%']:+.4f}%  胜率={s['win_%']:.1f}%  "
                  f"次日跌停开盘={s['next_limit_down_%']}% 次日涨停开盘={s['next_limit_up_%']}%")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"\n已保存: {args.out}")

if __name__ == "__main__":
    main()
