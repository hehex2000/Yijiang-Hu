# -*- coding: utf-8 -*-
"""
MACD 背离感知策略（平台 MACD/KDJ 能力的"优化补充"，替代淘汰的旧金叉死叉插件）
==========================================================================
严格落地 MACD 科普视频的正确姿态，绝不把金叉/死叉当买卖按钮：

  - 趋势门控（regime gate）：指数(股票池对应)站上 MA200 才做多；跌破 → 空仓。
    （对应视频要点③"趋势和震荡分开看" + 道氏"分季节"）
  - 选股信号：个股出现「底背离」（近 lookback 日两个 pivot 低点，价更低、DIF 更高）
    + 「真出门」价格突破近期 pivot 高点（放量走出区间，对应"出门再跟上"）
    + 非中枢期（布林带宽分位 ≥ 阈值，不在驿站里动手）
    + 价格站上自身 MA(price_ma)（结构向上）
  - 月度调仓、等权持有 top_n；未入选/跌破门槛即卖出。
  - 记账口径与 run_magic_v2 一致（hfq 后复权、买入日因子归一化、卖0.99955/买1.0002）。

用途：作为对照回测的"优化版 MACD 策略"，与买入持有(指数)比较，验证视频方法论
在 A 股能否转化为实盘 edge。结论以实际跑出为准（与背离诊断器互为印证）。

⚠️ KDJ-J 确认门消融结论（ablation_macd_kdj_gate.py, 2014~2026 hs300, 固定参数全样本）：
  - 修复了选股 set 哈希随机顺序导致的「非确定性」bug（同参数净值曾漂移 ±20pp）后，
    结果稳定可复现：baseline 总收益 +145.61%。
  - recover 门（J 由负拐正, N=20, 窗口版）：总收益 −22.10pp、夏普 −0.069、回撤更深 → 拖垮策略。
  - rising_low 门（J 下半区上行）：与 baseline 完全一致（窗口内几乎恒成立=无效过滤）。
  → KDJ-J 作为本策略的「确认门」经严格复现检验**不成立**（冗余且过滤掉有效背离捕捉），
    故 kdj_gate 默认关闭。这恰印证 Jim 第4期：描述器有价值，但当「确认/按钮」叠加到
    已有结构的策略上未必增益，必须 ablation 验证而非想当然。
"""
import sys, os, sqlite3, bisect, argparse
import numpy as np
import pandas as pd
import talib as ta
import run_magic_formula as mf

DB_PATH = "D:/tu-shareData/astock_daily.db"
_POOL_INDEX = {"hs300": "000300.SH", "zz500": "000905.SH",
               "zz800": "000906.SH", "zz1000": "000852.SH"}
SELL_MULT = 0.99955
BUY_MULT = 1.0002

_PX = {}; _FAC = {}; _IDX = {}; _FEAT = {}

def _conn():
    return sqlite3.connect(DB_PATH)

def _load_code(code):
    if code in _PX:
        return
    c = _conn()
    rows = c.execute(
        "SELECT CAST(trade_date AS TEXT), open, high, low, close FROM daily "
        "WHERE ts_code=? ORDER BY trade_date", (code,)).fetchall()
    fr = c.execute(
        "SELECT CAST(trade_date AS TEXT), adj_factor FROM adj_factor "
        "WHERE ts_code=? ORDER BY trade_date", (code,)).fetchall()
    c.close()
    _PX[code] = ([r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows],
                 [r[3] for r in rows], [r[4] for r in rows])
    _FAC[code] = ([r[0] for r in fr], [r[1] for r in fr])

def _factor(code, td):
    _load_code(code)
    dates, facs = _FAC[code]
    i = bisect.bisect_right(dates, td) - 1
    if i >= 0 and facs[i] is not None:
        return float(facs[i])
    for f in facs:
        if f is not None:
            return float(f)
    return None

def _load_index(idx):
    if idx in _IDX:
        return
    c = _conn()
    rows = c.execute(
        "SELECT CAST(trade_date AS TEXT), close FROM index_daily "
        "WHERE ts_code=? ORDER BY trade_date", (idx,)).fetchall()
    c.close()
    dates = [r[0] for r in rows]; closes = [float(r[1]) for r in rows]
    _IDX[idx] = (dates, closes, np.concatenate([[0.0], np.cumsum(closes)]))

def index_close(idx, td):
    _load_index(idx)
    dates, closes, _ = _IDX[idx]
    i = bisect.bisect_right(dates, td) - 1
    return closes[i] if i >= 0 else None

def index_above_ma(idx, td, win=200):
    _load_index(idx)
    dates, closes, csum = _IDX[idx]
    i = bisect.bisect_right(dates, td) - 1
    if i < win - 1:
        return True
    ma = (csum[i + 1] - csum[i + 1 - win]) / win
    return closes[i] >= ma


def build_features(code, start, end, P):
    """返回 dict：dates, idx(date)->i, signal(bool array, 底背离信号在窗口内),
       ma(price_ma 数组), bb_pct 数组, close 数组, breakout 数组, dif 数组。"""
    if code in _FEAT:
        return _FEAT[code]
    _load_code(code)
    dates, opens, highs, lows, closes = _PX[code]
    si = bisect.bisect_left(dates, start)
    ei = bisect.bisect_right(dates, end)
    if ei - si < 200:
        _FEAT[code] = None
        return None
    dates = dates[si:ei]
    highs = np.asarray(highs[si:ei], dtype=float)
    lows = np.asarray(lows[si:ei], dtype=float)
    closes = np.asarray(closes[si:ei], dtype=float)
    n = len(closes)
    if n < 250:
        _FEAT[code] = None
        return None

    dif, _, _ = ta.MACD(closes, fastperiod=P["fast"], slowperiod=P["slow"],
                        signalperiod=P["signal"])
    valid = ~np.isnan(dif)

    # ── KDJ-J 语境确认门（Jim 第4期正确用法：描述器，不预测方向）──
    # 仅取 J 线（=3K-2D，K/D 为 RSV 的不同平滑）。J 承载"价格在区间里移动的
    # 变化速度"，是 RSI/BB 位置都直接给不出的轴。裸 KDJ 与 RSI/BB 高度共线，
    # 故平台只取这一条，且只作"确认门"永不单独当按钮。
    # j_recover   : J 由负区拐头向上 (J_t>0 & J_{t-1}<=0) —— 消融证明最强 edge
    # j_rising_low: J 处下半区且当日上行 (J_t<50 & J_t>J_{t-1}) —— 更宽"回暖"语境
    kdj_n = int(P.get("kdj_n", 20))
    k, d = ta.STOCH(highs, lows, closes,
                   fastk_period=kdj_n, slowk_period=3, slowk_matype=0,
                   slowd_period=3, slowd_matype=0)
    j = 3.0 * k - 2.0 * d
    j_prev = np.roll(j, 1)
    j_prev[0] = np.nan
    j_recover = (j > 0) & (j_prev <= 0)
    j_rising_low = (j < 50) & (j > j_prev)

    s = pd.Series(closes)
    ma = s.rolling(P["price_ma"]).mean().values
    roll_mean = s.rolling(P["bb_win"]).mean()
    roll_std = s.rolling(P["bb_win"]).std()
    width = (roll_std / roll_mean).values

    W = P["pivot_window"]
    sl = pd.Series(lows)
    rmin = sl.rolling(2 * W + 1, center=True, min_periods=2 * W + 1).min()
    pl_mask = (sl == rmin) & (sl < sl.shift(1)) & (sl <= sl.shift(-1))
    pl_idx = np.where(pl_mask.values)[0]
    sh = pd.Series(highs)
    rmax = sh.rolling(2 * W + 1, center=True, min_periods=2 * W + 1).max()
    ph_idx = np.where((sh == rmax).values)[0]

    # 底背离完成标记（在 later pivot low 处）
    bull_div = np.zeros(n, dtype=bool)
    for a in range(1, len(pl_idx)):
        i1, i2 = pl_idx[a - 1], pl_idx[a]
        if i2 - i1 < P["min_gap"] or not valid[i1] or not valid[i2]:
            continue
        if lows[i2] < lows[i1] and dif[i2] > dif[i1]:
            if P["lookback"] and lows[i2] <= np.nanmin(lows[max(0, i2 - P["lookback"]):i2 + 1]):
                bull_div[i2] = True
            elif not P["lookback"]:
                bull_div[i2] = True

    # 信号活跃窗口（近期出现底背离）
    sw = P["signal_window"]
    sig_active = np.zeros(n, dtype=bool)
    for i in range(n):
        lo = max(0, i - sw + 1)
        if bull_div[lo:i + 1].any():
            sig_active[i] = True

    # 突破：收盘突破近期 pivot 高点（真出门）
    ph_high = np.full(n, np.nan)
    for j in ph_idx:
        ph_high[j] = highs[j]
    # 前向填充 pivot 高点，取"截至昨日"的近期高点
    recent_high = pd.Series(ph_high).ffill().values
    breakout = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if not np.isnan(recent_high[i - 1]) and closes[i] > recent_high[i - 1]:
            breakout[i] = True

    # 布林带宽分位
    bb_pct = np.full(n, np.nan)
    lb = P["bb_lookback"]
    for i in range(n):
        lo = max(0, i - lb + 1)
        w = width[lo:i + 1]
        w = w[~np.isnan(w)]
        if len(w) >= max(5, int(lb * 0.5)) and not np.isnan(width[i]):
            bb_pct[i] = float((w < width[i]).mean())

    feat = {
        "dates": dates,
        "idx": {d: i for i, d in enumerate(dates)},
        "sig": sig_active, "ma": ma, "bb_pct": bb_pct,
        "close": closes, "breakout": breakout, "dif": dif,
        "j": j, "j_recover": j_recover, "j_rising_low": j_rising_low,
    }
    _FEAT[code] = feat
    return feat


def run_strategy(start_date, end_date, pool="hs300", capital=100000,
                 top_n=10, regime_filter=True, ma_window=200,
                 fast=12, slow=26, signal=9, pivot_window=10, lookback=60,
                 min_gap=10, price_ma=60, signal_window=20,
                 bb_win=20, bb_lookback=120, bb_th=0.25,
                 kdj_gate=False, kdj_n=20, kdj_gate_mode="recover",
                 kdj_confirm_window=20):
    idx_code = _POOL_INDEX.get(pool, "000300.SH")
    P = dict(fast=fast, slow=slow, signal=signal, pivot_window=pivot_window,
             lookback=lookback, min_gap=min_gap, price_ma=price_ma,
             signal_window=signal_window, bb_win=bb_win, bb_lookback=bb_lookback,
             bb_th=bb_th, kdj_n=kdj_n, kdj_gate=kdj_gate, kdj_gate_mode=kdj_gate_mode)
    c = _conn()
    trade_dates = [r[0] for r in c.execute(
        "SELECT DISTINCT CAST(trade_date AS TEXT) FROM daily "
        "WHERE trade_date BETWEEN ? AND ? ORDER BY trade_date",
        (start_date, end_date)).fetchall()]
    c.close()
    if not trade_dates:
        print("[ERROR] 无交易日")
        return None

    monthly_first = {}
    for td in trade_dates:
        monthly_first.setdefault(td[:6], td)
    monthly_set = set(monthly_first.values())

    print("=" * 72)
    print("  MACD 背离感知策略（趋势门控 + 底背离 + 突破 + 非中枢）· 月度调仓")
    print("=" * 72)
    print(f"  区间: {start_date}~{end_date} | 池: {pool} | 持仓: {top_n} | 资金: {capital:,}")
    print(f"  趋势门控: {'指数MA%d 站上才做多' % ma_window if regime_filter else '关闭(全样本)'}")
    print(f"  选股: 底背离(近{lookback}日) + 突破近期高点 + 价>MA{price_ma} + 非中枢(带宽分位≥{bb_th})")
    print(f"  KDJ-J确认门: {'开(' + kdj_gate_mode + f',N={kdj_n})' if kdj_gate else '关'}  （J回暖才进场，挡掉背离恶化中冲入）")
    print("=" * 72)

    positions = {}   # code -> {shares, buy_factor, last_hfq}
    cash = float(capital)
    daily_vals = []

    def port_value(td, kind="close"):
        tot = cash
        for code, pos in positions.items():
            if kind == "open":
                p = hfq_price_open(code, td, pos)
            else:
                p = hfq_price_close(code, td, pos)
            if p is not None:
                tot += pos["shares"] * p
        return tot

    def hfq_price_close(code, td, pos):
        _load_code(code)
        dates, _, _, _, closes = _PX[code]
        i = bisect.bisect_right(dates, td) - 1
        if i < 0 or closes[i] is None:
            return pos.get("last_hfq")
        f = _factor(code, td)
        p = closes[i] * (f if f else 1.0)
        return p / (pos.get("buy_factor") or 1.0)

    def hfq_price_open(code, td, pos):
        _load_code(code)
        dates, opens, _, _, _ = _PX[code]
        i = bisect.bisect_right(dates, td) - 1
        if i < 0 or opens[i] is None:
            return None
        f = _factor(code, td)
        p = opens[i] * (f if f else 1.0)
        return p / (pos.get("buy_factor") or 1.0)

    def do_sell(code, td, frac=1.0):
        nonlocal cash
        pos = positions[code]
        px = hfq_price_open(code, td, pos)
        if px is None:
            px = pos.get("last_hfq")
        if px is None:
            return
        if frac >= 1.0:
            sh = pos["shares"]
        else:
            sh = int(pos["shares"] * frac / 100) * 100
            if sh <= 0:
                return
            if pos["shares"] - sh < 100:
                sh = pos["shares"]
        cash += sh * px * SELL_MULT
        pos["shares"] -= sh
        if pos["shares"] <= 0:
            del positions[code]

    last_holdings = []
    monthly_hold_counts = []   # 消融用：每月调仓后持仓数（验证"确认门"非靠空仓取巧）
    for i, td in enumerate(trade_dates):
        is_month = td in monthly_set
        if is_month:
            prev_td = trade_dates[i - 1] if i > 0 else td
            regime_ok = (not regime_filter) or index_above_ma(idx_code, prev_td, ma_window)
            if not regime_ok:
                # 跌破趋势闸门 → 清仓
                if positions:
                    for code in list(positions.keys()):
                        do_sell(code, td, 1.0)
                    print(f"  [趋势] {td} 指数跌破MA{ma_window} → 清仓（{len(positions)}→0）")
            else:
                # 选股（信号用 T-1 收盘，T 开盘执行；杜绝未来函数）
                eval_td = prev_td
                const = mf._get_pool_constituents(pool, td)
                const = sorted(const) if const else []   # 确定性：成分股排序，杜绝 set 哈希随机顺序
                cands = []
                for code in const:
                    f = build_features(code, start_date, end_date, P)
                    if f is None:
                        continue
                    ii = f["idx"].get(eval_td)
                    if ii is None or ii < 5:
                        continue
                    if not f["sig"][ii]:
                        continue
                    if np.isnan(f["ma"][ii]) or np.isnan(f["close"][ii]) or f["close"][ii] < f["ma"][ii]:
                        continue
                    if np.isnan(f["bb_pct"][ii]) or f["bb_pct"][ii] < bb_th:
                        continue
                    if not f["breakout"][ii]:
                        continue
                    # ── KDJ-J 语境确认门（Jim 正确用法：只确认，不按钮）──
                    # MACD 定结构（底背离+突破+非中枢），KDJ 做语境确认：
                    # 要求在"背离这段剧情"里 J 曾回暖过（窗口版），而不是要求调仓
                    # 当天 J 必须正在金叉——后者两种罕见信号同日撞上概率为0，会清空策略。
                    # 消融证明 j_recover(N=20) 有显著小 edge；金叉侧无 edge(已退休)。
                    if P.get("kdj_gate"):
                        kw = int(P.get("kdj_confirm_window", 20))
                        lo = max(0, ii - kw + 1)
                        if P.get("kdj_gate_mode") == "rising_low":
                            arr = f["j_rising_low"][lo:ii + 1]
                        else:  # 默认 "recover"
                            arr = f["j_recover"][lo:ii + 1]
                        if not arr.any():
                            continue
                    score = (f["dif"][ii] - f["dif"][ii - 5]) if ii >= 5 else 0.0
                    cands.append((score, code))
                cands.sort(key=lambda x: (-x[0], x[1]))   # 确定性：score 降序，并列按代码升序
                new_set = [code for _, code in cands[:top_n]]
                new_set_set = set(new_set)
                # 卖出移出的
                for code in list(positions.keys()):
                    if code not in new_set_set:
                        do_sell(code, td, 1.0)
                # 买入新的（等权）
                if new_set:
                    val = port_value(td, "open")
                    target = val  # 满仓（regime 已在门控里）
                    per = target * 0.98 / len(new_set)
                    for code in new_set:
                        if code in positions:
                            continue
                        op = hfq_price_open_only(code, td)
                        if op is None or op <= 0:
                            continue
                        bf = _factor(code, td) or 1.0
                        px = op / bf
                        budget = cash * 0.98 / max(1, len(new_set))
                        sh = int(budget / px / 100) * 100
                        cost = sh * px * BUY_MULT
                        if sh <= 0 or cost > cash:
                            continue
                        positions[code] = {"shares": sh, "buy_factor": bf, "last_hfq": px}
                        cash -= cost
                last_holdings = list(positions.keys())
                monthly_hold_counts.append(len(positions))
                if i == 0 or td == monthly_first.get(td[:6]):
                    print(f"  [{td}] 候选 {len(cands)} 入选 {len(new_set)} | 持仓 {len(positions)} | 现金 {cash:,.0f}")

        # 每日估值
        tot = cash
        for code, pos in positions.items():
            p = hfq_price_close(code, td, pos)
            if p is not None:
                pos["last_hfq"] = p
                tot += pos["shares"] * p
        daily_vals.append({"date": td, "value": tot})

    # 末日平仓
    if positions:
        last_td = trade_dates[-1]
        for code in list(positions.keys()):
            do_sell(code, last_td, 1.0)
        daily_vals[-1]["value"] = cash

    rep = _report(daily_vals, trade_dates, capital, idx_code, start_date, end_date, top_n, regime_filter)
    if monthly_hold_counts:
        rep["avg_holdings"] = float(np.mean(monthly_hold_counts))
        rep["months_active"] = int(sum(1 for x in monthly_hold_counts if x > 0))
        rep["total_months"] = len(monthly_hold_counts)
    return rep


def _metrics(vals):
    v = np.array(vals, dtype=float)
    total = v[-1] / v[0] - 1
    n = len(v)
    ann = (v[-1] / v[0]) ** (252.0 / n) - 1 if n > 1 else 0
    cummax = np.maximum.accumulate(v)
    dd = (v - cummax) / cummax
    mdd = float(dd.min())
    j = int(dd.argmin())
    pk = int(np.argmax(v[:j + 1])) if j > 0 else 0
    rets = np.diff(v) / v[:-1]
    sharpe = ((rets.mean() * 252 - 0.025) / (rets.std() * np.sqrt(252))
              if len(rets) > 1 and rets.std() > 0 else 0)
    return total, ann, mdd, sharpe, pk, j


def _yearly(dates, vals, capital):
    out, cur, start_v = {}, None, None
    for d, v in zip(dates, vals):
        y = d[:4]
        if y != cur:
            start_v = capital if cur is None else out[cur][1]
            cur = y
            out[y] = [start_v, v, 1]
        else:
            out[y][1] = v
            out[y][2] += 1
    return out


def hfq_price_open_only(code, td):
    _load_code(code)
    dates, opens, _, _, _ = _PX[code]
    i = bisect.bisect_right(dates, td) - 1
    if i < 0 or opens[i] is None:
        return None
    f = _factor(code, td)
    return opens[i] * (f if f else 1.0)


def _report(daily_vals, trade_dates, capital, bench, start_date, end_date, top_n, regime_filter):
    dates = [d["date"] for d in daily_vals]
    vals = [d["value"] for d in daily_vals]
    total, ann, mdd, sharpe, pk, tr = _metrics(vals)

    b0 = index_close(bench, dates[0])
    b1 = index_close(bench, dates[-1])
    b_total = b1 / b0 - 1 if b0 and b1 else 0

    print(f"\n{'='*72}\n  📊 MACD背离感知策略 vs 买入持有(指数) 对照（hfq）\n{'='*72}")
    print(f"  {'年份':<8}{'策略':>10}{'基准':>10}{'超额':>10}")
    print(f"  {'─'*40}")
    yg = _yearly(dates, vals, capital)
    byg = _yearly(dates, [index_close(bench, d) for d in dates], index_close(bench, dates[0]))
    for y in sorted(yg):
        s0, s1, _ = yg[y]
        sret = s1 / s0 - 1
        if y in byg:
            b0y, b1y, _ = byg[y]
            bret = b1y / b0y - 1
            print(f"  {y:<8}{sret:>+9.2%}{bret:>+9.2%}{sret - bret:>+9.2%}")
    print(f"  {'─'*40}")
    print(f"  {'全程':<7}{total:>+9.2%}{b_total:>+9.2%}{total - b_total:>+9.2%}")

    print(f"\n{'='*72}\n  📈 策略最终汇总\n{'='*72}")
    print(f"  初始资金: {capital:,.0f}  最终资产: {vals[-1]:,.0f}")
    print(f"  总收益: {total:+.2%}  年化: {ann:+.2%}")
    print(f"  最大回撤: {mdd:+.2%}  (峰 {dates[pk]} → 谷 {dates[tr]})")
    print(f"  夏普: {sharpe:.4f}")

    out_dir = "data/results/macd_strategy"
    os.makedirs(out_dir, exist_ok=True)
    csv_path = f"{out_dir}/macd_regime_{start_date}_{end_date}.csv"
    pd.DataFrame(daily_vals).to_csv(csv_path, index=False)
    print(f"\n  日净值 → {csv_path}\n")
    return {"total": total, "annual": ann, "mdd": mdd, "sharpe": sharpe}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="MACD 背离感知策略（趋势门控+底背离+突破）")
    ap.add_argument("start_date", nargs="?", default="20140301")
    ap.add_argument("end_date", nargs="?", default="20260731")
    ap.add_argument("--pool", default="hs300")
    ap.add_argument("--capital", type=int, default=100000)
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--no-regime", action="store_true", help="关闭趋势门控（ ablation ）")
    ap.add_argument("--ma", type=int, default=200)
    ap.add_argument("--price-ma", type=int, default=60)
    ap.add_argument("--signal-window", type=int, default=20)
    ap.add_argument("--lookback", type=int, default=60)
    ap.add_argument("--bb-th", type=float, default=0.25)
    ap.add_argument("--kdj-gate", dest="kdj_gate", action="store_true", default=False,
                    help="开启 KDJ-J 语境确认门（ ablation 证明开启反而拖累净值 ~22pp，默认关）")
    ap.add_argument("--no-kdj-gate", dest="kdj_gate", action="store_false",
                    help="关闭 KDJ-J 确认门（默认）")
    ap.add_argument("--kdj-n", type=int, default=20, help="KDJ N 参数（默认20，消融证明优于9）")
    ap.add_argument("--kdj-gate-mode", choices=["recover", "rising_low"], default="recover",
                    help="确认门模式: recover=J由负拐正(强edge) / rising_low=J处下半区且上行(宽语境)")
    ap.add_argument("--kdj-confirm-window", type=int, default=20,
                    help="确认门回顾窗口：在背离活跃窗口内 J 曾回暖即过（忠实 Jim『看交叉前的位置变化』）")
    a = ap.parse_args()
    run_strategy(a.start_date, a.end_date, pool=a.pool, capital=a.capital,
                 top_n=a.top_n, regime_filter=not a.no_regime, ma_window=a.ma,
                 price_ma=a.price_ma, signal_window=a.signal_window,
                 lookback=a.lookback, bb_th=a.bb_th,
                 kdj_gate=a.kdj_gate, kdj_n=a.kdj_n, kdj_gate_mode=a.kdj_gate_mode,
                 kdj_confirm_window=a.kdj_confirm_window)
