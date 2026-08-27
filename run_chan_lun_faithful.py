# -*- coding: utf-8 -*-
"""
run_chan_lun_faithful.py — 缠论忠实内核的独立择时回测(可证伪性检验)
================================================================
用 chan_lun_core_faithful.py 的真实几何买卖点(一/二/三类)驱动 long-only 进出场，
与同标的买入持有(BH)对照。目标不是证明有效，而是验证缠论是否可被忠实算子化、
是否能被回测证伪(能跑出干净对照=可证伪；结构塌缩无法构成检验=不可量化)。

规则：
  - 因果回测：逐 Bar 用 data[:t+1] 重算缠论结构；买卖点在"确认完成"的 bar t 首次出现，
    于 open of (t+lag) 执行(lag 默认 1)。绝不在 e_idx+lag 偷跑(段/笔确认需 e_idx 之后的数据)。
  - 信号稳定化：信号须连续出现 >=SETTLE(=2) 个前缀才执行，且同一 (idx,type) 仅执行一次
    (防逐Bar重算时最后一段边界漂移导致的"闪烁"——同一信号反复出现/消失)。二者均仅依赖 <=t 数据。
  - 信号只依赖 ≤t 的数据(无未来函数)；全序列一次性 compute_states + e_idx+lag 执行的旧写法
    会在确认前行动，构成真实未来函数，已弃用。
  - long-only：买入信号(b1/b2/b3)建仓，卖出信号(s1/s2/s3)平仓；无信号持有。
  - 交易计数口径：一律用"实际成交"(受持仓状态门控——满仓时忽略后续买点、空仓时忽略卖点)。
    "走查累计结算信号数"另行披露，二者差额属正常。分段统计同用实际成交，故
    各段交易数之和 == 全期实际成交数（输出内置 [自洽校验] 行硬检查）。
  - 宽成本：单边 0.13%(佣金0.03+印花0.05+滑点0.05)，round-trip 0.26%。
  - 数据护栏：单日 >25% 价格断裂跳过该标的并报告。
  - walk-forward：年度拆解(compute_states 因果，信号不过未来)。

用法：
  venv_ml/Scripts/python.exe run_chan_lun_faithful.py
  venv_ml/Scripts/python.exe run_chan_lun_faithful.py --start 20100101 --lag 1
  venv_ml/Scripts/python.exe run_chan_lun_faithful.py --instruments 510300.SH,510050.SH
"""
import argparse
import os
import sqlite3
import numpy as np
import pandas as pd

from chan_lun_core_faithful import compute_states

DB = r"D:\tu-shareData\astock_daily.db"
COMM_IN = 0.0013
COMM_OUT = 0.0013

DEFAULT_INSTRUMENTS = [
    ("510300.SH", "etf_daily", "沪深300ETF", "20100101"),
    ("510050.SH", "etf_daily", "上证50ETF",  "20100101"),
    ("159915.SZ", "etf_daily", "创业板ETF",  "20110101"),
    ("000905.SH", "index_daily", "中证500指数", "20050101"),
    # 000016.SH(上证50指数) 与 510050.SH(上证50ETF) 同底层。diag_sh50_reversal.py 已诊断确认：
    # 二者为同一篮50只股票的两种表征(日收益相关0.989、归一比例std 1.4%、零断点)，缠论超额在
    # 3/4 重叠时段同向、仅 2012-2015 因刀刃敏感性反转。为免同底层双重计数/自相矛盾，默认仅保留
    # 可交易的 510050 作上证50代表；如需稳健性对照，用
    #   --instruments 510050.SH,etf_daily,上证50ETF,20100101;000016.SH,index_daily,上证50指数,20050101
    # 显式加回。
]


def load_ohlc(code, table, start):
    c = sqlite3.connect(DB)
    rows = c.execute(
        f"SELECT trade_date, open, high, low, close FROM {table} "
        f"WHERE ts_code=? AND trade_date>=? ORDER BY trade_date",
        (code, start)).fetchall()
    c.close()
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close"])
    # 数据护栏：单日 >25% 断裂
    ret = df["close"].pct_change().abs()
    bad = df["date"][ret > 0.25]
    if len(bad):
        print(f"  ⚠ 数据护栏触发 {code}：单日>25%断裂 @ {list(bad)[:3]} -> 跳过")
        return None
    return df


def run_backtest(df, lag=1):
    opens = df["open"].values.astype(float)
    highs = df["high"].values.astype(float)
    lows = df["low"].values.astype(float)
    closes = df["close"].values.astype(float)
    dates = df["date"].values
    n = len(closes)
    # 因果回测核心：逐 Bar 用 data[:t+1] 重算缠论结构，在"信号确认完成"的 bar t
    # 首次出现该买卖点即于 (t+lag) 开盘价执行。绝不在确认前的 e_idx+lag 偷跑。
    # （旧写法：compute_states 一次性全序列 + e_idx+lag 执行，会在段/笔确认前就行动，
    #  构成真实未来函数——信号 e_idx 是结构"终点"，其"确认"需 e_idx 之后的反向结构。）
    #
    # 信号稳定化(防"闪烁")：逐Bar重算时，最后一段(正在形成)的边界随前缀延展漂移，
    # 使近末端的笔反复改变线段归属 -> beichi 标签(b1/s1)与 b3/s3 反复出现/消失。
    # 若直接对每次"新出现"就下单，同一逻辑信号会被重复调度(实测 exec 数爆炸至 16~42
    # 而真实信号仅 2~3)。修正：
    #   1) 信号须连续出现 >=SETTLE 个前缀才执行(过滤 1~数 bar 的瞬时闪烁)；
    #   2) 同一 (idx,type) 信号只执行一次(done 去重，过滤"出现-消失-再现"长周期闪烁)。
    # 两者皆仅依赖 <=t 数据，因果合法。
    SETTLE = 2
    exec_buy = set()
    exec_sell = set()
    persist = {}
    done = set()
    warmup = 120  # 至少需若干 bar 才能成笔/线段，此前跳过
    for t in range(warmup, n):
        st = compute_states(highs[:t + 1], lows[:t + 1], closes[:t + 1])
        cur = set(st["buys"]) | set(st["sells"])
        new_persist = {}
        for sig in cur:
            new_persist[sig] = persist.get(sig, 0) + 1
        persist = new_persist
        if t + lag < n:
            cur_buys = set(st["buys"])
            cur_sells = set(st["sells"])
            for sig in cur:
                if persist[sig] >= SETTLE and sig not in done:
                    if sig in cur_buys:
                        exec_buy.add(t + lag)
                    else:
                        exec_sell.add(t + lag)
                    done.add(sig)

    cash = 1.0
    units = 0.0
    nav = np.empty(n)
    trades = 0
    wins = 0
    entry_nav = None
    trade_buys = []   # 实际成交(建仓)的 bar 序号——受持仓状态门控，非调度集合
    trade_sells = []  # 实际成交(平仓)的 bar 序号
    in_pos = np.zeros(n, dtype=int)  # 每根 bar 收盘是否持仓(1=满仓,0=空仓)，用于判定"有效检验段"
    for t in range(n):
        price = opens[t]
        sold_today = False
        if units > 0 and t in exec_sell:
            proceeds = units * price * (1 - COMM_OUT)
            cash += proceeds
            units = 0
            trades += 1
            trade_sells.append(t)
            if entry_nav is not None and cash > entry_nav:
                wins += 1
            entry_nav = None
            sold_today = True
        # 2.3 修正：同一天若已卖，跳过买（避免无谓的来回换手）
        if units == 0 and (not sold_today) and t in exec_buy:
            if cash > 0:
                units = cash * (1 - COMM_IN) / price
                entry_nav = cash
                cash = 0
                trades += 1
                trade_buys.append(t)
        in_pos[t] = 1 if units > 0 else 0
        nav[t] = cash + units * closes[t]

    total_ret = nav[-1] / nav[0] - 1
    # 最大回撤
    peak = np.maximum.accumulate(nav)
    mdd = (nav - peak) / peak
    mdd = mdd.min()
    # 年度拆解(2.5 修正：跨年衔接——以"上一年末 NAV"为当年基准，而非当年首日 NAV)
    ann = {}
    for t in range(n):
        y = str(dates[t])[:4]
        ann.setdefault(y, []).append(nav[t])
    annual = []
    prev_last = None
    for y in sorted(ann):
        v = ann[y]
        base = prev_last if prev_last is not None else v[0]
        annual.append((y, base, v[-1], v[-1] / base - 1))
        prev_last = v[-1]
    return dict(nav=nav, total_ret=total_ret, mdd=mdd, trades=trades,
                wins=wins, annual=annual, n_bi=len(st["bi"]),
                n_seg=len(st["segments"]), n_buy=len(st["buys"]),
                n_sell=len(st["sells"]), last_dir=st["last_dir"],
                # 分段统计必须用"实际成交 bar"(受持仓门控)，不能用调度集合 exec_*，
                # 否则分段交易数 > 全期总数(逻辑不可能)。见 segmented_report 说明。
                trade_buys=trade_buys, trade_sells=trade_sells, in_pos=in_pos,
                # 透明披露：逐Bar走查中"结算过"的调度信号数(中枢上移会在新 idx 反复结算)，
                # 与实际成交数的差额=被持仓状态门控吞掉的重复同向信号。
                n_settled_buy=len(exec_buy), n_settled_sell=len(exec_sell))


def buy_hold(df):
    opens = df["open"].values.astype(float)
    closes = df["close"].values.astype(float)
    dates = df["date"].values
    n = len(closes)
    # 首根开盘买入，末根收盘卖出(含单边成本)
    nav = np.empty(n)
    units = (1.0 * (1 - COMM_IN)) / opens[0]
    for t in range(n):
        nav[t] = units * closes[t] * (1 if t < n - 1 else (1 - COMM_OUT))
    # 2.2 修正：归一化使 nav[0]=1.0，与策略(起始现金 1.0)口径对齐，避免首日 close≠open 的偏差
    nav = nav / nav[0]
    return nav[-1] / nav[0] - 1, nav


def segmented_report(df, nav, bh_nav, trade_buys, trade_sells, in_pos, split_years):
    """把全期 NAV 按 split_years(4位年)切成不重叠时间段，报告每段缠论择时收益、
    BH 收益、段内实际成交数、是否为"有效检验段"。规则锁定(无逐段调参)→ 用于 OOS 多段证伪。

    两处口径修正(均为此前自造的假象，务必保留)：

    1) 交易数必须用"实际成交 bar"(trade_buys/trade_sells)，即 NAV 主循环里真正建仓/平仓
       的那些 bar。早期版本误传调度集合 exec_buy/exec_sell(逐Bar走查中每个结算过的信号
       都记一个执行 bar，不看持仓状态)，导致分段交易数(16/42/31)远超全期实际成交数
       (5/3/1)——分段是全期子集，逻辑不可能，属纯计数口径 bug。改用实际成交后，
       各段交易数之和 == 全期 trades（切点严格划分 [0,n)）。

    2) BH 段收益必须用 BH 自己的 NAV 切段(bh_nav[b-1]/bh_nav[a])，不能用
       "价格比 × 单边成本"。旧写法给每一段的 BH 都扣一次进出成本，而缠论段内若 0 成交
       则不扣任何成本 -> 制造出"交易0笔却赢 +0.19~0.40pp"的记账不对称假胜
       （曾把 OOS 胜率虚抬到 15/19）。用 bh_nav 切段后，BH 成本只在首尾段各计一次，
       与缠论"只在成交段计成本"完全对称。

    3) 有效检验段：若段内 0 成交且缠论全程满仓，则该段价格路径与 BH 完全相同，
       策略未做任何决策 -> 标记为"无效"(非检验)，不计入胜率。判据：
       valid = (tr > 0) or (段内 in_pos 非全 1)。"""
    dates = df["date"].values
    n = len(nav)
    cuts = [0]
    for y in split_years:
        for i in range(cuts[-1], n):
            if str(dates[i])[:4] >= y:
                cuts.append(i)
                break
    cuts.append(n)
    segs = []
    for s in range(len(cuts) - 1):
        a, b = cuts[s], cuts[s + 1]
        if b - a < 2:
            continue
        chan_ret = nav[b - 1] / nav[a] - 1
        bh_ret = bh_nav[b - 1] / bh_nav[a] - 1
        tr = sum(1 for t in trade_buys if a <= t < b) + \
             sum(1 for t in trade_sells if a <= t < b)
        seg_pos = in_pos[a:b]
        valid = bool(tr > 0 or seg_pos.min() == 0)
        segs.append((a, b, chan_ret, bh_ret, tr, valid))
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="20100101")
    ap.add_argument("--lag", type=int, default=1)
    ap.add_argument("--instruments", default=None,
                    help="code,table,label,start 用分号分隔多组；组内字段逗号分隔")
    ap.add_argument("--split", default=None,
                    help="OOS 多段证伪：逗号分隔切分年(如 2014,2019)，把全期切成不重叠时段对比")
    a = ap.parse_args()

    insts = DEFAULT_INSTRUMENTS
    if a.instruments:
        insts = []
        for grp in a.instruments.split(";"):
            parts = grp.split(",")
            insts.append((parts[0], parts[1] if len(parts) > 1 else "etf_daily",
                          parts[2] if len(parts) > 2 else parts[0],
                          parts[3] if len(parts) > 3 else a.start))

    print("="*70)
    print("缠论忠实内核独立择时回测 (可证伪性检验)")
    print(f"  信号延迟 lag={a.lag} | 单边成本={COMM_IN*100:.2f}% | 起始={a.start}")
    print("="*70)

    summary = []
    oos_rows = []  # (label, seg_label, chan_ret, bh_ret, valid)
    for code, table, label, istart in insts:
        start = istart if istart > a.start else a.start
        df = load_ohlc(code, table, start)
        if df is None:
            print(f"\n[{label}] {code}: 无数据/护栏跳过")
            continue
        r = run_backtest(df, lag=a.lag)
        bh, bh_nav = buy_hold(df)
        rng = f"{df['date'].iloc[0]}~{df['date'].iloc[-1]}"
        print(f"\n--- {label} ({code}) {rng} ---")
        print(f"  缠论择时: 总收益 {r['total_ret']*100:+.2f}% | MDD {r['mdd']*100:+.2f}% | "
              f"实际成交 {r['trades']}笔(胜 {r['wins']})")
        print(f"    末期结构: 笔 {r['n_bi']}/线段 {r['n_seg']}/买点 {r['n_buy']}/卖点 {r['n_sell']}"
              f" | 走查累计结算信号 {r['n_settled_buy']}买/{r['n_settled_sell']}卖"
              f"（差额被持仓状态门控吞掉，属正常：满仓时忽略后续同向买点）")
        print(f"  买入持有: 总收益 {bh*100:+.2f}%")
        print(f"  超额: {(r['total_ret']-bh)*100:+.2f}pp | {'跑赢' if r['total_ret']>bh else '跑输'}")
        if a.split:
            sy = [x.strip() for x in a.split.split(",") if x.strip()]
            segs = segmented_report(df, r["nav"], bh_nav, r["trade_buys"],
                                    r["trade_sells"], r["in_pos"], sy)
            print(f"  OOS 多段(切年 {sy})：")
            seg_tr_sum = 0
            for (a2, b2, cr, br, tr, valid) in segs:
                yy0 = str(df['date'].iloc[a2])[:4]; yy1 = str(df['date'].iloc[b2-1])[:4]
                exc = cr - br
                if not valid:
                    verdict = "无效(0成交且全程满仓=等同BH，不计胜率)"
                elif abs(exc) < 1e-4:
                    verdict = "平"
                else:
                    verdict = "赢" if exc > 0 else "输"
                print(f"    {yy0}-{yy1}: 缠论 {cr*100:+.2f}% | BH {br*100:+.2f}% | "
                      f"超额 {exc*100:+.2f}pp | 交易{tr}笔 {verdict}")
                oos_rows.append((label, f"{yy0}-{yy1}", cr, br, valid))
                seg_tr_sum += tr
            # 自洽性硬校验：分段是全期的划分，各段交易数之和必须等于全期实际成交数
            flag = "✅" if seg_tr_sum == r["trades"] else "❌ 口径不一致"
            print(f"    [自洽校验] 各段交易数之和 {seg_tr_sum} vs 全期实际成交 {r['trades']} {flag}")
        else:
            print(f"  年度: " + " | ".join(f"{y}:{ret*100:+.1f}%" for y, _, _, ret in r["annual"]))
        summary.append((label, code, r, bh))

    # 汇总
    print("\n" + "="*70)
    print("汇总 (缠论择时 vs 买入持有)")
    print("="*70)
    win = sum(1 for _, _, r, bh in summary if r["total_ret"] > bh)
    print(f"{'标的':<14}{'缠论':>10}{'BH':>10}{'超额':>10}")
    for label, code, r, bh in summary:
        print(f"{label:<14}{r['total_ret']*100:>+9.2f}%{bh*100:>+9.2f}%{(r['total_ret']-bh)*100:>+9.2f}pp")
    print(f"\n全期跑赢 {win}/{len(summary)}")
    if a.split:
        # 仅统计"有效检验段"：0成交且全程满仓的段等同 BH、未做决策，不计入胜率
        ow = sum(1 for _, _, cr, br, v in oos_rows if v and cr > br)
        n_valid = sum(1 for _, _, _, _, v in oos_rows if v)
        print(f"OOS 多段: 跑赢 {ow}/{n_valid} 段 "
              f"(有效检验段；无效段 {len(oos_rows)-n_valid} 个已排除，不计胜率；"
              f"共 {len(set(l for l, _, _, _, _ in oos_rows))} 标的 × 多时段)")
        from collections import defaultdict
        per = defaultdict(lambda: [0, 0])
        for label, _, cr, br, v in oos_rows:
            if not v:
                continue
            per[label][0] += (1 if cr > br else 0)
            per[label][1] += 1
        print("  逐标的 OOS 段胜率(仅有效检验段)：")
        for label, (w, t) in per.items():
            print(f"    {label:<14} {w}/{t} 段跑赢")
    print(f"缠论结构是否可量化: "
          f"{'是(已产出笔/线段/买卖点并驱动交易)' if summary else '否'}")


if __name__ == "__main__":
    main()
