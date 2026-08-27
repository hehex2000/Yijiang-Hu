# -*- coding: utf-8 -*-
"""
run_chan_lun_validate.py — 缠论量化验证（平行账本 / 单一变量隔离）
================================================================
方法学(对齐 ablation_livermore_direction.py):
  - 基底 = 股票池等权买入持有 (control, 仓位恒 1)
  - 实验组把各缠论组件作为"仓位闸门"(0/0.5/1) 套到同一基底
  - 只改"用什么信号决定仓位", 退出规则一致 (持有一段) → 隔离单一变量
  - 无未来函数: 信号 T-1 计算, T 开盘执行 (用 close/close 日收益, 仓位用 T-1 信号)
  - 所有股票对齐到统一交易日历 all_dates, 停牌日用最后有效价填充, 停牌期间收益计 0

5 组平行账本:
  control   : 等权满仓持有 (基准)
  G1 中枢回避   : 中枢期(bb分位<th)仓位 0.5
  G2 线段趋势   : 线段 down(MA60 代理) 时空仓  (注: 精确 HH-HL 链见 chan_lun_core.swing_trend)
  G3 背驰闸门   : 近 K 日见底背驰 → 满仓, 否则半仓
  G4 三类买卖点 : 近 K 日见 b1/b2/b3 → 满仓; 见顶背驰/线段转 down → 空仓

运行(本机):
  venv_ml/Scripts/python.exe run_chan_lun_validate.py
  venv_ml/Scripts/python.exe run_chan_lun_validate.py --max-stocks 200
  venv_ml/Scripts/python.exe run_chan_lun_validate.py --start 20180101 --end 20251231

诚实预期(见 docs/livermore_volume_perspective.md 同源逻辑): 缠论结构检测对, 但
方向不可知 → 各闸门大概率不显著改善/甚至拖累基底; 可能仅"中枢回避"在震荡市降回撤。
判定门槛: 年化不降>0.5pp 且 MDD 改善>=2pp 才记为正贡献(对齐 consolidation-filter 计划)。
"""
import os, argparse
import numpy as np
import pandas as pd
import sqlite3

try:
    import config
    DB_PATH = getattr(config, "DB_PATH", None) or getattr(config, "DATA", None)
    if isinstance(DB_PATH, dict):
        DB_PATH = DB_PATH.get("local_db_path")
except Exception:
    DB_PATH = None
if not DB_PATH or not os.path.exists(DB_PATH):
    DB_PATH = r"D:\tu-shareData\astock_daily.db"

import chan_lun_core as CL


def get_trade_dates(start, end):
    c = sqlite3.connect(DB_PATH)
    rs = c.execute(
        "SELECT DISTINCT CAST(trade_date AS TEXT) FROM daily "
        "WHERE trade_date BETWEEN ? AND ? ORDER BY trade_date", (start, end)).fetchall()
    c.close()
    return [r[0] for r in rs]


def load_universe(start, end, min_rows=250, max_stocks=0):
    c = sqlite3.connect(DB_PATH)
    rows = c.execute(
        "SELECT ts_code, COUNT(*) n FROM daily "
        "WHERE trade_date BETWEEN ? AND ? GROUP BY ts_code HAVING n>=?",
        (start, end, min_rows)).fetchall()
    c.close()
    codes = sorted(r[0] for r in rows)
    if max_stocks and len(codes) > max_stocks:
        codes = codes[:max_stocks]
    return codes


def load_aligned(code, alld):
    """返回对齐到 alld 的 open/high/low/close (停牌日 ffill, 首笔前为 NaN)。"""
    c = sqlite3.connect(DB_PATH)
    rs = c.execute(
        "SELECT CAST(trade_date AS TEXT), open, high, low, close FROM daily "
        "WHERE ts_code=? AND trade_date BETWEEN ? AND ? ORDER BY trade_date",
        (code, alld[0], alld[-1])).fetchall()
    c.close()
    if not rs:
        return None
    df = pd.DataFrame(rs, columns=["d", "o", "h", "l", "c"]).set_index("d").reindex(alld)
    if df["c"].notna().sum() < 120:
        return None
    out = dict(
        open=df["o"].ffill().values.astype(float),
        high=df["h"].ffill().values.astype(float),
        low=df["l"].ffill().values.astype(float),
        close=df["c"].ffill().values.astype(float),
    )
    return out


def ma_trend_proxy(close, win=60, look=5):
    """per-bar 趋势代理: close>MA & MA 上升 → up; 否则 down。
    注: 精确线段=HH-HL 链见 chan_lun_core.swing_trend; 此处用 MA 代理降频避免 O(n^2)。"""
    s = pd.Series(close)
    ma = s.rolling(win).mean().values
    up = np.zeros(len(close), dtype=bool)
    for i in range(win + look, len(close)):
        if np.isnan(ma[i]) or np.isnan(ma[i - look]):
            continue
        up[i] = (close[i] > ma[i]) and (ma[i] > ma[i - look])
    return up


def seen_within(flags_bool, K):
    """flags_bool[i]=True 时, 返回 [i-K, i] 任意为 True 的 bool 序列 (信号持续 K 日)。"""
    n = len(flags_bool)
    out = np.zeros(n, dtype=bool)
    for i in range(n):
        lo = max(0, i - K + 1)
        if flags_bool[lo:i + 1].any():
            out[i] = True
    return out


def nav_stats(nav):
    nav = np.asarray(nav, float)
    rets = nav[1:] / nav[:-1] - 1.0
    total = nav[-1] / nav[0] - 1.0
    n_years = max(1.0, len(nav) / 252.0)
    annual = (nav[-1] / nav[0]) ** (1.0 / n_years) - 1.0
    peak = np.maximum.accumulate(nav)
    dd = nav / peak - 1.0
    mdd = float(dd.min())
    sd = rets.std()
    sharpe = float(rets.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0
    return dict(total=total, annual=annual, mdd=mdd, sharpe=sharpe)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="20180101")
    ap.add_argument("--end", default="20251231")
    ap.add_argument("--consol-th", type=float, default=0.25)
    ap.add_argument("--zig-th", type=float, default=0.05)
    ap.add_argument("--hold-window", type=int, default=20, help="信号持续K日(仓位维持)")
    ap.add_argument("--max-stocks", type=int, default=0, help="0=全部(默认)")
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()

    alld = get_trade_dates(a.start, a.end)
    L = len(alld)
    codes = load_universe(a.start, a.end, max_stocks=a.max_stocks)
    print("交易日: %d  股票池: %d 只 (%s~%s)" % (L, len(codes), a.start, a.end))

    groups = ["control", "G1_中枢回避", "G2_线段趋势", "G3_背驰闸门", "G4_买卖点"]
    acc = {g: None for g in groups}
    valid = 0

    for ci, code in enumerate(codes):
        d = load_aligned(code, alld)
        if d is None:
            continue
        close = d["close"]
        if np.isnan(close).any():
            # 首笔前 NaN 用首个有效值填
            first = np.nanargmax(~np.isnan(close))
            close[:first + 1] = close[first]
        ret = close[1:] / close[:-1] - 1.0
        ret = np.nan_to_num(ret, nan=0.0)
        if len(ret) < 2:
            continue

        st = CL.compute_states(d["high"], d["low"], close,
                               zig_th=a.zig_th,
                               consol=dict(win=20, lookback=120, th=a.consol_th))
        con = st["consolidation"]
        bull = np.zeros(L, dtype=bool)
        for i in st["bull_div"]:
            if i < L:
                bull[i] = True
        bear = np.zeros(L, dtype=bool)
        for i in st["bear_div"]:
            if i < L:
                bear[i] = True
        buy = np.zeros(L, dtype=bool)
        for i, _ in st["buy_points_typed"]:
            if i < L:
                buy[i] = True
        trend_up = ma_trend_proxy(close)

        bull_win = seen_within(bull, a.hold_window)
        buy_win = seen_within(buy, a.hold_window)
        bear_win = seen_within(bear, a.hold_window)
        down_win = seen_within(~trend_up, a.hold_window)

        w_control = np.ones(L)
        w_g1 = np.where(con, 0.5, 1.0)
        w_g2 = np.where(trend_up, 1.0, 0.0)
        w_g3 = np.where(bull_win, 1.0, 0.5)
        w_g4 = np.where(buy_win, 1.0, np.where(bear_win | down_win, 0.0, 0.5))

        valid += 1
        for g, w in (("control", w_control), ("G1_中枢回避", w_g1),
                     ("G2_线段趋势", w_g2), ("G3_背驰闸门", w_g3),
                     ("G4_买卖点", w_g4)):
            w_lag = np.concatenate([[0.0], w[:-1]])   # bar0 无信号→0, 之后用 T-1
            contrib = w_lag[1:L] * ret
            acc[g] = contrib if acc[g] is None else acc[g] + contrib

        if (ci + 1) % 100 == 0:
            print("  ... %d/%d" % (ci + 1, len(codes)))

    # 等权聚合: 每日组合收益 = mean(各股当日收益), 再累乘成 NAV
    navs = {}
    for g in groups:
        arr = acc[g] / valid if valid else np.zeros(L - 1)
        navs[g] = np.cumprod(np.concatenate([[1.0], 1.0 + arr]))

    print("\n%s" % ("=" * 78))
    print("  缠论量化验证 | %s~%s | consol_th=%.2f zig_th=%.2f hold=%d"
          % (a.start, a.end, a.consol_th, a.zig_th, a.hold_window))
    print("%s" % ("=" * 78))
    print("  %-12s%10s%9s%9s%8s" % ("组", "总收益", "年化", "最大回撤", "夏普"))
    print("  " + "-" * 50)
    ctrl_total = None
    summary = []
    for g in groups:
        s = nav_stats(navs[g])
        if ctrl_total is None:
            ctrl_total = s["total"]
        print("  %-12s%+9.2f%%%+8.2f%%%+8.2f%%%7.2f"
              % (g, s["total"] * 100, s["annual"] * 100, s["mdd"] * 100, s["sharpe"]))
        summary.append(dict(group=g, **s, excess_ctrl=s["total"] - ctrl_total))
    print("\n  判定(对齐 consolidation-filter 门槛): 年化不降>0.5pp 且 MDD改善>=2pp → 正贡献")
    for r in summary[1:]:
        dp = (-summary[0]["mdd"] * 100) - (-r["mdd"] * 100)  # 回撤改善 = 控制组回撤绝对值 − 实验组回撤绝对值 (>0=改善)
        verdict = "正贡献" if (r["annual"] * 100 - summary[0]["annual"] * 100 > -0.5 and dp >= 2) else "负/中性"
        print("    %-12s 超额vs基准=%+6.2fpp  回撤改善=%+5.1fpp  → %s"
              % (r["group"], r["excess_ctrl"] * 100, dp, verdict))

    if not a.no_save:
        out = "data/results/chan_lun"
        os.makedirs(out, exist_ok=True)
        df = pd.DataFrame(summary)
        csv = "%s/chan_lun_validate_%s_%s.csv" % (out, a.start, a.end)
        df.to_csv(csv, index=False)
        print("\n  CSV → %s" % csv)


if __name__ == "__main__":
    main()
