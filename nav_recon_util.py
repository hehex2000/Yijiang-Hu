# -*- coding: utf-8 -*-
"""从 trades CSV 重建日频 NAV 的共享工具（供 ①限价vs市价 / ②活跃税 两个工具复用）。

核心思路：
- 平台回测默认把每笔成交当「市价单」处理（穿越价差 + 平方根冲击），对左侧/逆向/
  价值类策略系统性高估成本——它们本应挂限价单（流动性提供者），几乎不吃冲击、甚至
  赚价差。本工具用同一份 trades 流水，分别在两种成交假设下重建组合净值，隔离「成交
  成本假设」这一单变量。
- 两模型强制使用同一初始资金（market 模型估算），使收益差纯粹来自成本差。
- 复用 run_monthly_rebalance 真实成本函数（佣金/印花税/滑点），不另立炉灶。

标记模式：
  slippage_frac_market(action, amount, ts_code, trade_date)
      -> 平台默认 flat 0.1%；若导入时已设 MFS_SQRT_IMPACT=1，则走平方根冲击。
  slippage_frac_limit(action, amount, ts_code, trade_date, maker_slip=0.0)
      -> 被动限价：不吃 taker 冲击；maker_slip 默认 0（保守：仅消除 taker 成本），
         也可传负值（如 -0.0005）表示吃到 half-spread 的反向收益。
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_monthly_rebalance import (
    get_conn, COMMISSION_RATE, COMMISSION_MIN, SLIPPAGE_RATE,
    stamp_duty_rate, sqrt_impact_slippage, USE_SQRT_IMPACT,
)


def _norm_action(a):
    """归一化成交方向：**必须按前缀匹配**，不能只认精确字符串。

    平台存在带原因后缀的复合标签：``BUY-reentry`` / ``SELL-liquidate`` /
    ``SELL-rebal``（全库 97 个 trades CSV 中 8 个使用）。早期实现只认精确的
    ``BUY`` / ``SELL``，凡不相等就落进调用方的 else 分支当卖出处理，造成两个
    静默错误：
      1. 现金流方向整体反向 → compute_init_cap 算出 min_cash>=0 → init_cap=0
         → 活跃税被显示成毫无意义的 0.00%；
      2. cost_of 里 ``action == "SELL"`` 判定失败 → 卖出漏扣印花税。
    两者都不会报错，只会安静地给出错误的漂亮数字。
    """
    a = str(a).strip().upper()
    if a.startswith("BUY") or a in ("B", "OPEN", "买入", "买"):
        return "BUY"
    if a.startswith("SELL") or a in ("S", "CLOSE", "卖出", "卖"):
        return "SELL"
    return a


# hfq 归一化的固定锚点。必须与 run_daily20_macd.load_closes 的取数起点一致，
# 否则同一只股票在不同窗口会得到不同尺子的后复权价（详见 fetch_closes_bulk 文档）。
HFQ_ANCHOR = 20100101


def detect_price_mode(trades, raw_closes, sample=300, tol=0.02):
    """判断一份 trades CSV 的价格口径是 raw 还是 hfq。

    🔴 为什么必须判断：`活跃税 = 总成本 / (平均净值 × 年数)`，分子取自 trades 的
    price，分母取自用 closes 重建的 NAV。**两者必须是同一把尺子**，否则比值系统性
    失真——hfq 的 trades 价被累计复权因子放大（实测 000012.SZ 20150107：
    hfq 17.141 vs raw 9.01，1.9 倍），而老实现固定取 raw close 建 NAV，
    于是 hfq 策略的活跃税被**系统性高估**（实测日20族 hfq 4.18~5.81% vs raw 3.20~4.46%，
    复权因子随时间增长 → 越到后期高估越多）。hfq 现已是回测默认口径，此偏差影响所有新产物。

    判据：抽样比对 trade price 与当日 raw close 的比值中位数。
    raw 档应恒为 1.00（收盘价成交）；hfq 档为累计复权因子，通常明显 > 1。
    返回值 ("raw"|"hfq", 中位比值, 比对样本数)。
    """
    t = trades.head(sample) if len(trades) > sample else trades
    ratios = []
    for _, r in t.iterrows():
        s = raw_closes.get(str(r["code"]))
        if s is None:
            continue
        try:
            px = s.get(int(r["date"]))
        except Exception:
            px = None
        if px is None or not np.isfinite(px) or px <= 0:
            continue
        tp = float(r["price"])
        if not np.isfinite(tp) or tp <= 0:
            continue
        ratios.append(tp / float(px))
    if len(ratios) < 5:
        return "raw", float("nan"), len(ratios)   # 样本不足，退回 raw（老行为）
    med = float(np.median(ratios))
    # 只有明显偏离 1 才判 hfq；raw 档（比值恒 1）不会被误伤
    return ("hfq" if abs(med - 1.0) > tol else "raw"), med, len(ratios)


def _median_ratio(trades, closes, sample=300):
    """抽样比对 trade price 与当日 close 的比值中位数；样本不足返回 (None, n)。"""
    t = trades.head(sample) if len(trades) > sample else trades
    ratios = []
    for _, r in t.iterrows():
        s = closes.get(str(r["code"]))
        if s is None:
            continue
        try:
            px = s.get(int(r["date"]))
        except Exception:
            px = None
        if px is None or not np.isfinite(px) or px <= 0:
            continue
        tp = float(r["price"])
        if not np.isfinite(tp) or tp <= 0:
            continue
        ratios.append(tp / float(px))
    if len(ratios) < 5:
        return None, len(ratios)
    return float(np.median(ratios)), len(ratios)


def pick_price_mode(trades, closes_raw, closes_hfq, closes_qfq, sample=300):
    """三路检测：在 raw / hfq / qfq 三种收盘价口径中，挑 trade price 与之比
    的中位数最接近 1 的那一种——即 trades 实际使用的口径。

    为什么需要三路：weekly_highdiv_vol 引擎的 qfq 价 = raw×fac_t/fac_last，
    既不等于 raw（比值随 t 变）也不等于 hfq 锚定版（比值≈fac_first/fac_last 常数≠1），
    旧的二选一检测会把它误判成 hfq，导致 trades 与 closes 尺度错配、NAV 失真。
    返回 ("raw"|"hfq"|"qfq", 中位比值, 比对样本数)。
    """
    cands = []
    for mode, px in (("raw", closes_raw), ("hfq", closes_hfq),
                     ("qfq", closes_qfq)):
        if px is None:
            continue
        r, n = _median_ratio(trades, px, sample=sample)
        if r is not None:
            cands.append((mode, abs(r - 1.0), r, n))
    if not cands:
        return "raw", float("nan"), 0
    cands.sort(key=lambda x: x[1])
    return cands[0][0], cands[0][2], cands[0][3]


def slippage_frac_market(action, amount, ts_code, trade_date, **_):
    """市价/taker 滑点占比：默认 flat，开启 sqrt 冲击则按流动性感知。"""
    if USE_SQRT_IMPACT and ts_code:
        frac, _, _ = sqrt_impact_slippage(amount, ts_code, trade_date)
        if frac is not None:
            return frac
    return SLIPPAGE_RATE


def slippage_frac_limit(action, amount, ts_code, trade_date, maker_slip=0.0, **_):
    """限价/maker 滑点占比：被动成交不吃 taker 冲击。"""
    return maker_slip


def cost_of(action, price, shares, trade_date, ts_code, slip_func, **kw):
    """单笔交易成本（佣金 + 滑点 + 印花税[仅卖出]）。"""
    amt = price * shares
    commission = max(amt * COMMISSION_RATE, COMMISSION_MIN)
    slip = amt * slip_func(action, amt, ts_code, trade_date, **kw)
    duty = amt * stamp_duty_rate(trade_date) if action == "SELL" else 0.0
    return commission + slip + duty


def fetch_close(code, start, end):
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT trade_date, close FROM daily WHERE ts_code=? "
        "AND trade_date>=? AND trade_date<=? ORDER BY trade_date",
        conn, params=(code, int(start), int(end)))
    conn.close()
    if len(df) == 0:
        return None
    df["trade_date"] = df["trade_date"].astype(int)
    return df.set_index("trade_date")["close"]


# SQLite 单次查询绑定变量上限留出的安全余量（3.32+ 默认 32766）
_MAX_SQL_PARAMS = 20000


def fetch_closes_bulk(codes, start, end, verbose=False, cache=False, hfq=False,
                      qfq_engine=False):
    """单次查询取回多标的日线收盘，返回 {ts_code: Series(index=trade_date)}。

    hfq=True 时返回**后复权**收盘 = close × adj_factor / 该股首个已知因子，
    口径与 run_daily20_macd.load_closes(hfq=True) 完全一致（ffill→bfill→fillna(1.0)、
    归一化基准取 **20100101** 起的首个因子）。
    🔴 **hfq 的取数起点被强制拉到 HFQ_ANCHOR(20100101)**：归一化基准依赖窗口起点，
       若按各策略自己的 d0 起取，同一只股票在不同文件里会得到不同尺子的 hfq 价，
       与回测引擎写进 trades CSV 的价格对不上（实测 000012.SZ 20150107：
       引擎口径 17.141、按 2015 起算只有 9.01）。**复权归一化基准必须是固定锚点。**

    cache=True 时把结果 pickle 到 ``data/results/.cache/``（该目录已被
    .gitignore 忽略），键为 (标的集合, 日期区间) 的 md5；重跑可跳过 70s 取数。
    默认关闭——缓存是 opt-in，不改变默认行为。改动前请手动删缓存目录。

    性能关键（实测，daily 表 1431 万行，EP中性 2218 标的 / 2020-2026）：
      · 必须写成 ``+ts_code IN (...)``：``+`` 抑制 ts_code 上的主键复合索引
        (ts_code, trade_date)，迫使 SQLite 走 idx_daily_date 做**顺序**区间扫描。
        不加 ``+`` 时会对每个标的做一次索引探查（随机 I/O）——实测
        **70.9s → 20.0s，快 3.5 倍**，返回行数完全相同。
      · **不要分块**：分块会让每块重扫一遍日期区间，实测更慢
        （chunk=500: 158.6s、chunk=1109: 116.6s vs 一次性 70.9s）。
        仅在标的数超过绑定变量上限时才切块（正常策略池远达不到）。
      · **跨策略复用**：多策略共享一次查询远比各查一遍划算——8 策略并集
        2721 标的 / 2010-2026 一次性 43.5s，逐策略合计 200s+。
        故 reconstruct() 支持外部传入 pre_fetched 的 closes。

    目的：避免对大 Universe（上千只）逐标的开连接导致卡死/超时。
    """
    if not codes:
        return {}
    codes = list(dict.fromkeys(codes))  # 去重保序
    # hfq 的归一化基准依赖窗口起点 → 强制用固定锚点，保证与回测引擎同尺子
    if hfq and int(start) > HFQ_ANCHOR:
        start = HFQ_ANCHOR
    start, end = str(int(start)), str(int(end))

    if cache:
        import hashlib
        import pickle
        key = hashlib.md5(("|".join(sorted(codes)) + f"|{start}|{end}|hfq={int(bool(hfq))}|qfq={int(bool(qfq_engine))}")
                          .encode("utf-8")).hexdigest()[:16]
        cdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "data", "results", ".cache")
        cpath = os.path.join(cdir, f"closes_{key}.pkl")
        if os.path.exists(cpath):
            try:
                with open(cpath, "rb") as fh:
                    out = pickle.load(fh)
                if verbose:
                    print(f"  [fetch] 命中缓存 {os.path.basename(cpath)} "
                          f"→ {len(out)} 标的（跳过查询）")
                return out
            except Exception as e:
                print(f"  [fetch] 缓存损坏({e})，重新查询")

    conn = get_conn()
    t0 = time.time()
    frames = []
    try:
        for i in range(0, len(codes), _MAX_SQL_PARAMS):
            chunk = codes[i: i + _MAX_SQL_PARAMS]
            ph = ",".join("?" for _ in chunk)
            if hfq or qfq_engine:
                # 注意：这里**不能**写 +t.ts_code。加了 JOIN 之后 SQLite 不会再拿
                # ts_code 上的复合主键索引做探查，写 + 只会白白丢掉优化器信息。
                q = ("SELECT t.ts_code, t.trade_date, t.close, a.adj_factor "
                     "FROM daily t "
                     "LEFT JOIN adj_factor a "
                     "  ON t.ts_code=a.ts_code AND t.trade_date=a.trade_date "
                     f"WHERE t.ts_code IN ({ph}) "
                     "AND t.trade_date>=? AND t.trade_date<=? "
                     "ORDER BY t.ts_code, t.trade_date")
            else:
                q = ("SELECT ts_code, trade_date, close FROM daily "
                     f"WHERE +ts_code IN ({ph}) AND trade_date>=? AND trade_date<=? "
                     "ORDER BY ts_code, trade_date")
            frames.append(pd.read_sql_query(
                q, conn, params=list(chunk) + [start, end]))
    finally:
        conn.close()

    df = pd.concat(frames, ignore_index=True) if frames else None
    del frames
    if df is None or len(df) == 0:
        return {}
    df["trade_date"] = df["trade_date"].astype(int)
    df["close"] = df["close"].astype(float)
    if hfq or qfq_engine:
        # 与 run_daily20_macd.load_closes 同序：ffill→bfill→fillna(1.0)
        df = df.sort_values(["ts_code", "trade_date"])
        df["adj_factor"] = (df.groupby("ts_code")["adj_factor"]
                              .ffill().bfill().fillna(1.0))
        # hfq：按首因子归一化（锚定 HFQ_ANCHOR 起的第一个因子）。
        # qfq_engine：按**最后因子**归一化 = raw×fac_t/fac_last，与
        #   run_weekly_highdiv_vol 引擎 qfq_close 的 `ref=fs[-1]` 同公式同尺度，
        #   使重建 NAV 与引擎 backtest CSV 逐位可比。
        #   （NAV 比率对每标的常数缩放 invariant，故 fac_last 取窗口内最后因子即可，
        #    不必非取全表最后因子——仅绝对数值尺度不同，比率型指标不受影响。）
        _ref_kind = "last" if qfq_engine else "first"
        ref = df.groupby("ts_code")["adj_factor"].transform(_ref_kind)
        df["close"] = df["close"] * df["adj_factor"] / ref
        df = df.drop(columns=["adj_factor"])
    # 立即转成 Series 字典并释放 DataFrame：ts_code 字符串列占内存大头
    # （实测 7.68M 行 DataFrame 944MB → 转字典后仅数十 MB）
    out = {c: g.set_index("trade_date")["close"]
           for c, g in df.groupby("ts_code", sort=False)}
    if verbose:
        print(f"  [fetch] {len(codes)} 标的 / {start}~{end} → {len(df)} 行, "
              f"{len(out)} 只有数据, {time.time() - t0:.1f}s")

    if cache:
        try:
            os.makedirs(cdir, exist_ok=True)
            with open(cpath, "wb") as fh:
                pickle.dump(out, fh, protocol=4)
            if verbose:
                print(f"  [fetch] 已缓存 → {cpath} "
                      f"({os.path.getsize(cpath) / 1e6:.0f}MB)")
        except Exception as e:
            print(f"  [fetch] 写缓存失败({e})，忽略")

    del df
    return out


def compute_init_cap(trades):
    """用 flat 0.1% 滑点估算所需初始资金（保证现金流非负，确定性）。"""
    cash = 0.0
    min_cash = 0.0
    for _, r in trades.iterrows():
        a = _norm_action(r["action"])
        amt = r["price"] * r["shares"]
        c = cost_of(a, r["price"], r["shares"], int(r["date"]), r["code"],
                    slippage_frac_market)
        if a == "BUY":
            cash -= (amt + c)
        else:
            cash += (amt - c)
        min_cash = min(min_cash, cash)
    return -min_cash if min_cash < 0 else 0.0


def reconstruct(trades, slip_func, init_cap=None, maker_slip=0.0,
                closes=None, **kw):
    """重建日频 NAV 与绩效指标。

    返回 dict：nav(Series), total_return, annualized, max_dd, mean_nav,
               total_cost, years, active_tax_yr, n_trades
    slip_func 可加 maker_slip kw（limit 模型）。

    closes: 预取的 {ts_code: Series(index=trade_date)}。多个策略共用一次
            批量查询时传入（见 fetch_closes_bulk 文档），可把 N 次全表扫描
            降为 1 次；为 None 时自行按本策略的 [d0, d1] 批量取数。
            传入的 closes 日期范围可大于本策略区间，内部会自行裁剪。
    """
    trades = trades.copy()
    trades["date"] = trades["date"].astype(int)
    trades = trades.sort_values("date").reset_index(drop=True)
    d0 = int(trades["date"].min())
    d1 = int(trades["date"].max())

    if init_cap is None:
        init_cap = compute_init_cap(trades)

    # 预取所有标的日线收盘（复权口径用 raw close，两模型一致即可）。
    # 关键：用 fetch_closes_bulk 单次批量查询，绝不逐标的开连接——
    # EP中性 2218 标的逐标的查询会开 2218 次连接并超时。
    if closes is None:
        codes = trades["code"].unique().tolist()
        closes = fetch_closes_bulk(codes, d0, d1)
    else:
        wanted = set(trades["code"].unique().tolist())
        closes = {c: s for c, s in closes.items() if c in wanted}
        # 裁剪到本策略日期区间，避免并集取数时用别处的日期撑大矩阵
        closes = {c: s[(s.index >= d0) & (s.index <= d1)]
                  for c, s in closes.items() if len(s)}
    if not closes:
        return None

    all_dates = sorted(set().union(*[set(s.index) for s in closes.values()]))
    # 逐标的对齐并 ffill（前向）+ bfill（补前导缺口，避免首期持仓无价→NaN 净值）
    close_mat = pd.DataFrame(
        {c: s.reindex(all_dates).ffill().bfill() for c, s in closes.items()}
    ).dropna(axis=1, how="all")

    # ── 按日推进：先处理当日成交，再按当日收盘价估值 ──────────────────
    # 预先把逐笔成交摊平成三个等长数组（现金流 / 股数增量 / 收盘矩阵列号），
    # 主循环只做 O(T+N) 的指针推进 + 一次矩阵行点乘，避免在
    # 「日期数 × 持仓数」的循环里反复 iterrows 与 Series 切片。
    cols = list(close_mat.columns)
    pos = {c: i for i, c in enumerate(cols)}
    mat = close_mat.to_numpy(dtype=float)

    n_tr = len(trades)
    t_cash = np.zeros(n_tr)
    t_qty = np.zeros(n_tr)
    t_col = np.full(n_tr, -1, dtype=np.int64)
    total_cost = 0.0
    n_unpriced = 0
    for k, (_, r) in enumerate(trades.iterrows()):
        a = _norm_action(r["action"])
        amt = r["price"] * r["shares"]
        c = cost_of(a, r["price"], r["shares"], int(r["date"]), r["code"],
                    slip_func, maker_slip=maker_slip, **kw)
        total_cost += c
        qty = abs(r["shares"])
        if a == "BUY":
            t_cash[k] = -(amt + c)
            t_qty[k] = qty
        else:
            t_cash[k] = (amt - c)
            t_qty[k] = -qty
        j = pos.get(r["code"], -1)
        t_col[k] = j
        if j < 0:
            n_unpriced += 1

    # 成交日 → 行情日历下标；成交日若不在日历内（停牌/缺数据）则顺延到下一个
    # 交易日。原实现直接丢弃这类成交（现金与持仓凭空错位），属潜在口径 bug，
    # 这里改为顺延并统计，便于核对。
    _dates_arr = np.asarray(all_dates, dtype=np.int64)
    t_pos = np.searchsorted(_dates_arr, trades["date"].to_numpy(dtype=np.int64))
    n_offcal = int((t_pos >= len(all_dates)).sum())
    t_pos = np.clip(t_pos, 0, len(all_dates) - 1)

    hold_vec = np.zeros(len(cols))
    hold_min = np.zeros(len(cols))   # 逐标的持仓历史最小值，用于流水自洽性校验
    cash = float(init_cap)
    nav_vals = np.empty(len(all_dates))
    p = 0
    for i in range(len(all_dates)):
        while p < n_tr and t_pos[p] <= i:
            cash += t_cash[p]
            j = t_col[p]
            if j >= 0:
                hold_vec[j] += t_qty[p]
                if hold_vec[j] < hold_min[j]:
                    hold_min[j] = hold_vec[j]
            p += 1
        nav_vals[i] = cash + float(mat[i] @ hold_vec)

    # ── 流水自洽性校验 ────────────────────────────────────────────
    # 持仓为负 = 卖出了日志里从未出现过的买入（CSV 不是完整流水，或按快照
    # 差分导出而非逐笔）。这类流水重建出的 NAV/活跃税全是垃圾——实测
    # daily20_divlow 月调 CSV：3953 笔中 2429 笔致持仓转负、208 个标的里
    # 123 个终值为负，却仍能算出"活跃税 19.94%/年"这种像模像样的假数字。
    # 这里显式检出，交给调用方决定是报错还是标记，绝不静默输出。
    neg_hold_codes = int((hold_min < -1e-6).sum())
    min_hold = float(hold_min.min()) if len(hold_min) else 0.0
    nav_min = float(nav_vals.min()) if len(nav_vals) else 0.0

    nav = pd.Series(nav_vals, index=all_dates, dtype=float)
    first, last = nav.iloc[0], nav.iloc[-1]
    # all_dates 为 YYYYMMDD 整数，须转真实日期再算年数（否则会当天数差算出上百"年"）
    _dt = pd.to_datetime([str(int(d)) for d in all_dates], format="%Y%m%d")
    years = (_dt[-1] - _dt[0]).days / 365.25
    # NAV 可能走到 <=0（流水不自洽时），此时收益率/年化在数学上无定义，
    # 返回 NaN 而不是让它炸出 RuntimeWarning 或返回无意义的负数。
    tot = (last / first - 1) if first > 0 else float("nan")
    ann = ((last / first) ** (1 / years) - 1
           if (first > 0 and last > 0 and years > 0) else float("nan"))
    peak = nav.cummax()
    mdd = (nav / peak - 1).min()
    mean_nav = nav.mean()

    # 总成交金额（单边）
    total_traded = float(trades["price"].astype(float)
                         .mul(trades["shares"].astype(float)).abs().sum())
    # 活跃税（年化换手成本率）两种口径，必须成对看：
    #   · 对初始本金：total_cost / (init_cap × years)。本金=最小所需融资额
    #     （≈首日建仓额，实测四策略 -min_cash 与首日买入额相差 <10%），
    #     是不受估值影响的保守上界；但组合若大幅增值，会系统性高估税负。
    #   · 对平均净值：total_cost / (mean_nav × years)。这才是与 B&O
    #     「6.5%/年」可比的口径（B&O 分母是财富，不是初始本金）。
    #     实例：周度高股息对初始本金 16.93%/年，若组合 6.5 年增长数倍，
    #     对平均净值会显著更低——两个数字差多少，就是"增长稀释"了多少。
    active_tax_yr = (total_cost / (init_cap * years)
                     if (init_cap and years > 0) else 0.0)
    active_tax_nav = (total_cost / (mean_nav * years)
                      if (mean_nav and years > 0) else 0.0)
    # 单边换手摩擦成本率（口径无关，跨策略可比）：总成本 / 总成交金额
    round_trip_cost = (total_cost / total_traded) if total_traded > 0 else 0.0

    return dict(
        nav=nav, total_return=tot, annualized=ann, max_dd=mdd,
        mean_nav=float(mean_nav), total_cost=total_cost,
        total_traded=total_traded,
        years=years, active_tax_yr=active_tax_yr, active_tax_nav=active_tax_nav,
        round_trip_cost=round_trip_cost, n_trades=len(trades),
        init_cap=init_cap,
        n_unpriced=n_unpriced, n_offcal=n_offcal,
        neg_hold_codes=neg_hold_codes, min_hold=min_hold, nav_min=nav_min,
    )


def load_trades(path):
    df = pd.read_csv(path)
    df = df.rename(columns={c: c.strip() for c in df.columns})
    need = ["date", "action", "code", "price", "shares"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"{path} 缺列: {miss}；现有列={list(df.columns)}")
    return df
