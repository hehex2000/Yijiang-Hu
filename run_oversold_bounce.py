"""
run_oversold_bounce.py - 超跌反弹三状态漏斗策略复现 (BV1Qiuz6WE6Y, UP: 跟着Jim学量化)

忠实复现 Jim "超跌 -> 止跌 -> 反弹" 三状态漏斗, 用显式参数填补视频未给的空白,
套 anti-overfitting 纪律 (分科目成本 / walk-forward / 扩展池对照) 看真 edge。

视频自承 "无稳定优势", 故本脚本定位是 "诚实复现 + 量化其到底有没有 edge",
而非 "做成能赚钱的策略"。复现也无优势 = 预期内; 复现若暴赚 = 先疑过拟合/前视, 不报喜。

数据: D:/tu-shareData/astock_daily.db
  daily        - ts_code, trade_date, open/high/low/close/pre_close/pct_chg/vol/amount
  adj_factor   - ts_code, trade_date, adj_factor (2010 起, 累计因子, 后复权价 = close*adj_factor)
  index_constituent - index_code, ts_code, trade_date, weight (沪深300 时点成分, 2014起, 优先)
  index_weight - 同上, 但 000300 仅 2020 起 (覆盖更晚, 自动降级)
  stock_basic  - ts_code, name, list_date (ST / 上市日过滤)

前视偏差防护 (backtest-lookahead-check):
  - 信号用后复权价算, 灭除权假新低/假放量
  - 信号日 t 生成, t+1 开盘成交, 灭同日(P0-1)成交
  - 沪深300 成分用 index_constituent 时点快照(2014起), 灭幸存者偏差
  - 涨停板 / ST / 上市<120日 过滤
"""
import argparse
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd

# ---- db path (复用 config, 带 fallback) ----
try:
    import config
    DB = config.DATA.local_db_path
except Exception:
    DB = "D:/tu-shareData/astock_daily.db"

# ---- 成本分科目 (wide 默认 = 真实复权口径) ----
COMMISSION_RATE = 0.0003
COMMISSION_MIN = 5.0
STAMP_DUTY_RATE = 0.0005   # 仅卖出
TRANSFER_RATE = 0.00001    # 过户费
SLIPPAGE_WIDE = 0.001      # 单边滑点 (wide)
SLIPPAGE_NARROW = 0.0002   # narrow 单参数近似 (单边20bp 含一切)

# ---- 视频未给的参数, 我们显式定并披露 (anti-overfitting: 不再优化, 固定初值) ----
N1 = 20     # 超跌回看天数
X = 0.15    # 超跌累计跌幅阈值
P = 0.60    # 窗口内下跌日占比
D = 0.12    # 窗口内最大回撤阈值
N2 = 5      # 止跌近窗
N3 = 10     # 止跌对比窗
Q = 0.80    # 量能缩小比例 (vol <= 近窗均值*Q)
Y = 0.05    # 反弹确认涨幅 (价格较 M 日低反弹)
M = 5       # 反弹确认窗口
H = 20      # 固定持有天数
TP = 0.15   # 止盈
SL = -0.08  # 止损
MAX_POS = 20        # 最大持仓数
INIT = 1_000_000.0  # 初始资金
HS300 = "000300.SH"
BUFFER = 60         # 数据预载缓冲天数


def get_conn():
    con = sqlite3.connect(DB)
    con.execute("PRAGMA query_only = ON")
    return con


def load_universe_members(con):
    """返回 {ts_code: list_date}, st_set, 沪深300 时点成分(iw), 选用的表名 src。
    自动选用 000300.SH 覆盖起点更早的表 (index_constituent 2014 vs index_weight 2020)。
    """
    basics = pd.read_sql(
        "SELECT ts_code, name, list_date FROM stock_basic", con)
    members = {r.ts_code: (str(r.list_date) if r.list_date else "19900101")
               for r in basics.itertuples()}
    st_set = {r.ts_code for r in basics.itertuples()
              if r.name and "ST" in str(r.name).upper()}

    # 自动选 000300.SH 覆盖起点更早的表
    best = None
    for t in ["index_constituent", "index_weight"]:
        try:
            r = con.execute(
                f"SELECT MIN(trade_date), MAX(trade_date) FROM {t} "
                f"WHERE index_code='{HS300}'").fetchone()
            if r and r[0]:
                if best is None or r[0] < best[1][0]:
                    best = (t, r)
        except Exception:
            continue
    src = best[0]
    iw = pd.read_sql(
        f"SELECT trade_date, ts_code FROM {src} WHERE index_code='{HS300}'", con)
    iw["trade_date"] = iw["trade_date"].astype(str)
    return members, st_set, iw, src


def build_universe_per_day(all_days, iw):
    """forward-fill 沪深300 成分到每个交易日, 返回 {day: set(codes)}。"""
    snaps = (iw.groupby("trade_date")["ts_code"].apply(set)
             .sort_index())
    snap_days = list(snaps.index)
    uni = {}
    for d in all_days:
        # 最近 <= d 的快照
        cand = [s for s in snap_days if s <= d]
        uni[d] = snaps[cand[-1]] if cand else set()
    return uni


def load_prices(con, codes, start, end):
    """加载日线 + adj_factor, 计算后复权价 hfq = close*adj_factor。"""
    ph = "'" + "','".join(codes) + "'"
    daily = pd.read_sql(
        f"SELECT ts_code, trade_date, open, high, low, close, pre_close, "
        f"pct_chg, vol, amount FROM daily "
        f"WHERE ts_code IN ({ph}) AND trade_date BETWEEN '{start}' AND '{end}'",
        con)
    af = pd.read_sql(
        f"SELECT ts_code, trade_date, adj_factor FROM adj_factor "
        f"WHERE ts_code IN ({ph}) AND trade_date BETWEEN '{start}' AND '{end}'",
        con)
    df = daily.merge(af, on=["ts_code", "trade_date"], how="left")
    df["hfq"] = df["close"] * df["adj_factor"]
    df["trade_date"] = df["trade_date"].astype(str)
    return df


def compute_signals(df, all_days):
    """逐股向量化计算三状态信号, 返回 signals_df (index=all_days, cols=codes, bool)。"""
    sig = {}
    for code, g in df.groupby("ts_code"):
        g = g.set_index("trade_date").reindex(all_days).sort_index()
        hfq = g["hfq"]
        vol = g["vol"]

        # 超跌: 近 N1 日累计跌幅 + 下跌日占比 + 窗口最大回撤
        drop = hfq / hfq.shift(N1) - 1
        down_ratio = (hfq.diff() < 0).rolling(N1).mean()
        roll_min_n1 = hfq.rolling(N1).min()
        maxdd = roll_min_n1 / hfq.shift(N1) - 1
        oversold = (drop <= -X) & (down_ratio >= P) & (maxdd <= -D)

        # 止跌: 近 N2 低点不再明显下移 + 量能缩小(对比下跌段均量, 非反弹日自身)
        # 注: 反弹日量常放大, 若拿反弹日量比自身5日均量会误杀止跌信号 -> 改比下跌段均量
        recent_low = hfq.rolling(N2).min()
        prev_low = hfq.shift(N2).rolling(N3).min()
        low_stable = recent_low >= prev_low * (1 - 0.01)
        decline_vol = vol.shift(N2).rolling(N3).mean()   # 下跌段(前 N2~N2+N3 日)均量
        recent_vol = vol.rolling(N2).mean()              # 近 N2 日均量
        vol_shrink = recent_vol <= decline_vol * Q
        stopped = low_stable & vol_shrink

        # 反弹确认: 价格较近 M 日低反弹 >= Y%
        reb_min = hfq.rolling(M).min()
        rebound = hfq / reb_min - 1 >= Y

        signal = (oversold & stopped & rebound).fillna(False)
        sig[code] = signal
    signals_df = pd.DataFrame(sig, index=all_days).fillna(False)
    return signals_df


def is_limit_up(pct, code):
    if code.startswith("300") or code.startswith("688"):
        return pct >= 19.5
    return pct >= 9.5


def trade_cost(value, is_buy, narrow):
    if narrow:
        return value * SLIPPAGE_NARROW
    comm = max(value * COMMISSION_RATE, COMMISSION_MIN)
    slip = value * SLIPPAGE_WIDE
    if is_buy:
        return comm + slip
    return comm + slip + value * STAMP_DUTY_RATE + value * TRANSFER_RATE


def run_backtest(df, signals_df, uni, members, st_set, all_days,
                 pool_codes, narrow, hold, walkforward):
    # 价格面板
    open_d = df.pivot(index="trade_date", columns="ts_code", values="open")
    close_d = df.pivot(index="trade_date", columns="ts_code", values="close")
    pct_d = df.pivot(index="trade_date", columns="ts_code", values="pct_chg")
    open_d = open_d.reindex(all_days)
    close_d = close_d.reindex(all_days)
    pct_d = pct_d.reindex(all_days)

    positions = {}   # code -> {shares, entry_open, exit_idx}
    cash = INIT
    nav_hist = []
    traded_value = 0.0
    buys = sells = 0

    for i, day in enumerate(all_days):
        u = uni[day] & pool_codes
        # ---- 退出 (用今日 close 判 TP/SL, 用今日 open 判固定持有/出池) ----
        for code in list(positions.keys()):
            pos = positions[code]
            c = close_d.loc[day, code]
            if pd.isna(c):
                continue
            ret = c / pos["entry_open"] - 1
            exit_flag = False
            exit_price = c
            if ret >= TP or ret <= SL:
                exit_flag = True          # TP/SL 日内 close
            elif i >= pos["exit_idx"]:
                exit_flag = True
                exit_price = open_d.loc[day, code]  # 固定持有到期, 次日 open
            elif code not in u:
                exit_flag = True
                exit_price = open_d.loc[day, code]  # 出池
            if exit_flag and not pd.isna(exit_price):
                proceeds = pos["shares"] * exit_price
                cost = trade_cost(proceeds, False, narrow)
                cash += proceeds - cost
                traded_value += proceeds
                sells += 1
                del positions[code]

        # ---- 进入 (信号取 t-1, t 开盘成交) ----
        if i >= 1 and len(positions) < MAX_POS:
            prev = all_days[i - 1]
            for code in u:
                if code in positions:
                    continue
                if not bool(signals_df.loc[prev, code]):
                    continue
                if code in st_set:
                    continue
                # 上市<120日 过滤
                ld = members.get(code, "19900101")
                if int(day) - int(ld) < 120:
                    continue
                # 涨停板 (信号日 close 触板) 买不进
                pc = pct_d.loc[prev, code]
                if not pd.isna(pc) and is_limit_up(pc, code):
                    continue
                o = open_d.loc[day, code]
                if pd.isna(o):
                    continue
                target = INIT / MAX_POS
                cost = trade_cost(target, True, narrow)
                shares = int(target // (o * (1 + (SLIPPAGE_NARROW if narrow else SLIPPAGE_WIDE))))
                if shares <= 0:
                    continue
                pay = shares * o + trade_cost(shares * o, True, narrow)
                if pay > cash:
                    continue
                cash -= pay
                traded_value += shares * o
                buys += 1
                positions[code] = {
                    "shares": shares, "entry_open": o, "exit_idx": i + hold}

        # ---- 盯市 ----
        eq = cash
        for code, pos in positions.items():
            c = close_d.loc[day, code]
            if not pd.isna(c):
                eq += pos["shares"] * c
        nav_hist.append((day, eq))

    return nav_hist, traded_value, buys, sells


def load_benchmark(con, all_days):
    b = pd.read_sql(
        f"SELECT trade_date, close FROM index_daily WHERE ts_code='{HS300}' "
        f"AND trade_date BETWEEN '{all_days[0]}' AND '{all_days[-1]}' ORDER BY trade_date",
        con)
    b["trade_date"] = b["trade_date"].astype(str)
    b = b.set_index("trade_date").reindex(all_days)["close"]
    return b


def stats_from_nav(nav_series, bench_series, all_days):
    nav = nav_series.values.astype(float)
    ret = nav[-1] / nav[0] - 1
    daily = pd.Series(nav).pct_change().dropna()
    ann = (nav[-1] / nav[0]) ** (252 / len(nav)) - 1 if nav[0] > 0 else 0
    vol = daily.std() * np.sqrt(252)
    sharpe = (daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0
    peak = np.maximum.accumulate(nav)
    mdd = (nav - peak) / peak
    maxdd = mdd.min()
    # 年度收益
    yrs = {}
    df = pd.DataFrame({"day": all_days, "nav": nav})
    df["y"] = df["day"].str[:4]
    for y, g in df.groupby("y"):
        yrs[y] = g["nav"].iloc[-1] / g["nav"].iloc[0] - 1
    # 基准
    if bench_series is not None and not bench_series.isna().all():
        b = bench_series.dropna().values.astype(float)
        if len(b) > 1:
            brem = b[-1] / b[0] - 1
            bvol = bench_series.pct_change().dropna().std() * np.sqrt(252)
            bsharpe = (bench_series.pct_change().dropna().mean() /
                       bench_series.pct_change().dropna().std() * np.sqrt(252))
        else:
            brem = bvol = bsharpe = float("nan")
    else:
        brem = bvol = bsharpe = float("nan")
    return dict(cum=ret, ann=ann, vol=vol, sharpe=sharpe, mdd=maxdd,
                yrs=yrs, benc=brem, bvol=bvol, bsharpe=bsharpe)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("start", nargs="?", default="20140101")
    ap.add_argument("end", nargs="?", default="20260821")
    ap.add_argument("--pool", choices=["hs300", "all"], default="hs300")
    ap.add_argument("--cost", choices=["narrow", "wide"], default="wide")
    ap.add_argument("--hold", type=int, default=H)
    ap.add_argument("--walkforward", action="store_true")
    args = ap.parse_args()

    con = get_conn()
    members, st_set, iw, src = load_universe_members(con)
    all_days = [r[0] for r in con.execute(
        "SELECT DISTINCT trade_date FROM daily ORDER BY trade_date")]
    all_days = [d for d in all_days if args.start <= d <= args.end]
    load_start = str(int(args.start) - BUFFER)

    uni = build_universe_per_day(all_days, iw)
    # universe 诊断 + 空窗显式警示
    yrs = sorted({d[:4] for d in all_days})
    print(f"universe source: {src} | iw 覆盖 {iw['trade_date'].min()}~{iw['trade_date'].max()}")
    first_tradable = None
    for y in yrs:
        sz = [len(uni[d]) for d in all_days if d[:4] == y]
        cnt = sz[0] if sz else 0
        print(f"  {y} 沪深300 成分数: {cnt}")
        if cnt > 0 and first_tradable is None:
            first_tradable = y
    if first_tradable and args.start < f"{first_tradable}0101":
        print(f"  ⚠️ 请求起点 {args.start} 早于首个有成分年份 {first_tradable}, "
              f"此前年份 universe 为空、不交易(已明示, 非静默排除)")

    if args.pool == "hs300":
        pool_codes = set(members.keys()) & set(iw["ts_code"].unique())
    else:
        pool_codes = set(members.keys())
    # 加载价格
    df = load_prices(con, sorted(pool_codes), load_start, args.end)
    df = df[df["trade_date"] >= args.start]
    signals_df = compute_signals(df, all_days)

    nav_hist, tv, buys, sells = run_backtest(
        df, signals_df, uni, members, st_set, all_days,
        pool_codes, args.cost == "narrow", args.hold, args.walkforward)
    nav_series = pd.Series([v for _, v in nav_hist], index=[d for d, _ in nav_hist])
    bench = load_benchmark(con, all_days)
    s = stats_from_nav(nav_series, bench, all_days)
    con.close()

    print("=" * 60)
    print("超跌反弹三状态漏斗 复现")
    print(f"  池={args.pool}({len(pool_codes)}只) | 区间 {args.start}~{args.end} "
          f"| 效用窗口 N1={N1}/N2={N2}/N3={N3}/M={M}")
    print(f"  成本={args.cost}({'单边20bp' if args.cost=='narrow' else '分科目'}) "
          f"| 持有={args.hold}日 | 止盈+{TP:.0%}/止损{SL:.0%}")
    print(f"  交易日 {len(all_days)} 天")
    print("-" * 60)
    print(f"  累计 {s['cum']:+.2%} | 年化 {s['ann']:+.2%} | 波动 {s['vol']:.2%} "
          f"| 夏普 {s['sharpe']:.2f} | 回撤 {s['mdd']:+.2%}")
    print(f"  基准(000300.SH): 累计 {s['benc']:+.2%} 波动 {s['bvol']:.2%} "
          f"夏普 {s['bsharpe']:.2f}")
    ann_turn = (tv / 2) / INIT / (len(all_days) / 252) if len(all_days) else 0
    print(f"  成交: 买入{buys}次 卖出{sells}次 | 年化单边换手~{ann_turn:.2f}次")
    print("  年度:", " ".join(f"{y}:{r:+.1%}" for y, r in sorted(s['yrs'].items())))
    if args.walkforward:
        print("-" * 60)
        print("  [walk-forward] 按年切段 (参数固定, 全段即OOS):")
        for y, r in sorted(s['yrs'].items()):
            print(f"    {y}: {r:+.2%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
