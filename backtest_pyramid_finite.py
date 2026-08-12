# -*- coding: utf-8 -*-
"""
金字塔加仓 vs 补仓陷阱 —— 有限资金版(共享资金池 + 少数股票案例演示)
=================================================================
目标: 在「20万共享本金 + 规则化筛选5只股票 + 设定回测期限」的现实约束下,
重跑4种加仓方案(SINGLE/PYRAMID/INVERTED/MARTINGALE), 看是否能下结论。

与万笔同信号版(backtest_pyramid.py)的关系:
  - 万笔版: 每笔信号独立10万, 隔离"加仓方案"单一变量 → 得「普遍结论」(马丁格尔尾部灾难/金字塔非alpha)
  - 本版  : 4方案各跑一个20万共享池账户, 处理同一批(5只股票的)突破信号 → 得「真实感/爆仓案例演示」
  - 关键: 信号定义完全相同(全市场 build_signal 后筛5只, 行业排名仍用全市场), 仅把"无限资金"换成"共享池"

设计要点(保住受控):
  - 同一批5只 + 同一套突破信号 + 同一时间顺序, 4方案各自独立账户, 只变加仓规则 → 仍公平对比
  - 每笔仓位计划总预算 = POS_BUDGET(默认4万, 即池的20%); SINGLE一次性占满, 金字塔/马丁格尔首档40%后递减加
  - 共享池: 加仓需现金足够, 不够就加不动(马丁格尔"加不动"= 真实爆仓约束, 去掉无限资金幻象)
  - 报告看 终值/总收益/最大回撤/是否破产(净值<=0)/笔数, 不只看均净

诚信边界: 本版是「5只股票的案例研究」, 结论受选股与路径影响, 不能推出普遍规律; 普遍规律见万笔版。
"""
import os, argparse
import numpy as np
import pandas as pd
from collections import defaultdict

from backtest_main_rise import (
    load_data, add_base_features, build_signal,
    COMMISSION_RATE, COMMISSION_MIN, SLIPPAGE_RATE, stamp_rate,
)

DB_OUT = "data/results/pyramid"
START = "20190101"
END   = "20260630"

# ---------- 加仓方案(与万笔版一致) ----------
UP_STEP   = 0.05     # 盈利加仓触发: 收盘较上一档入场价 +5%
DOWN_STEP = 0.05     # 补仓触发: 收盘较上一档入场价 -5%
TRAIL     = 0.10     # 跟踪止损: 从持仓最高价回撤 10% 离场(纪律型)
HARD_DOWN = 0.45     # 补仓方案硬止损: 跌破入场价 55% 才离场(=死扛)
MAX_HOLD  = 60       # 封顶持仓交易日
L_DEF, VOL_MULT_DEF = 60, 1.5

SCHEMES = {
    "SINGLE":    dict(sizes=[1.00],                 mode="single", stop="trail", label="一次性建仓(基准, 移动止损)"),
    "PYRAMID":   dict(sizes=[0.40,0.30,0.20,0.10], mode="up",     stop="trail", label="正金字塔40-30-20-10(盈利加+移动止损)"),
    "INVERTED":  dict(sizes=[0.10,0.20,0.30,0.40], mode="up",     stop="trail", label="倒金字塔10-20-30-40(盈利加+移动止损)"),
    "MARTINGALE":dict(sizes=[0.40,0.30,0.20,0.10], mode="down",   stop="hard",  label="补仓/越跌越补40-30-20-10(亏损加+死扛)"),
}

FINITE_CAPITAL = 200000.0   # 共享池本金
POS_BUDGET     = 40000.0    # 每笔仓位计划总预算(=池的20%)


class Portfolio:
    """一个共享资金池账户, 按某方案的加仓规则处理同一批信号流。"""

    def __init__(self, scheme, o, c, h, l, dts, signal_sets, init=FINITE_CAPITAL, pos_budget=POS_BUDGET):
        self.scheme = scheme
        self.o = o; self.c = c; self.h = h; self.dts = dts   # ts_code -> np.array
        self.signal_sets = signal_sets                      # ts_code -> set(li)
        self.cash = init
        self.pos_budget = pos_budget
        self.positions = []          # 持仓列表
        self.trades = []             # 已平仓记录
        self.equity_curve = []       # [(date, equity)]
        self.current_k = {}          # ts_code -> 当日k(用于市值标记)
        self.skipped_entries = 0     # 因现金不足错过开仓的次数
        self.ruined = False          # 净值是否曾 <=0

    # ---- 买卖执行(共享现金) ----
    def _buy(self, notional, price):
        buy_fill = price * (1.0 + SLIPPAGE_RATE)
        comm = max(notional * COMMISSION_RATE, COMMISSION_MIN)
        sh = int((notional - comm) / buy_fill)
        if sh <= 0:
            return 0
        self.cash -= sh * buy_fill + comm
        return sh

    def _sell(self, pos, exit_open, exit_k, reason):
        sell_fill = exit_open * (1.0 - SLIPPAGE_RATE)
        gross = pos["total_shares"] * sell_fill
        comm = max(gross * COMMISSION_RATE, COMMISSION_MIN)
        stamp = gross * stamp_rate(int(self.dts[pos["ts"]][exit_k]))
        proceeds = gross - comm - stamp
        self.cash += proceeds
        net = proceeds / pos["total_invested"] - 1.0
        rec = dict(ts=pos["ts"], entry_date=int(self.dts[pos["ts"]][pos["k0"]]),
                   exit_date=int(self.dts[pos["ts"]][exit_k]), reason=reason,
                   net=net, n_tr=pos["n_tr"])
        self.trades.append(rec)
        self.positions.remove(pos)

    def open_position(self, ts, li):
        """信号在 li, 入场在 open[li+1]。"""
        sizes = self.scheme["sizes"]; mode = self.scheme["mode"]
        init_notional = self.pos_budget * sizes[0]
        if self.cash < init_notional:
            self.skipped_entries += 1
            return False
        k0 = li + 1
        o = self.o[ts]; c = self.c[ts]; h = self.h[ts]; dts = self.dts[ts]
        if k0 >= len(o) - 1:
            return False
        entry_open = o[k0]
        if not np.isfinite(entry_open) or entry_open <= 0:
            return False
        sh = self._buy(init_notional, entry_open)
        if sh <= 0:
            return False
        pos = dict(ts=ts, k0=k0, entry_open=entry_open,
                   deployed=[init_notional], tr_entry=[entry_open], last_ref=entry_open,
                   n_tr=1, mode=mode, stop=self.scheme["stop"],
                   rolling_high=h[k0], end_k=min(k0 + MAX_HOLD, len(o) - 1),
                   total_invested=init_notional, total_shares=sh)
        self.positions.append(pos)
        return True

    def step_day(self, ts, k):
        """对 ts 在日 k 处理其所有持仓的止损/加仓(用 T-1 收盘判定, T 开盘执行)。"""
        for pos in list(self.positions):
            if pos["ts"] != ts or k <= pos["k0"]:
                continue
            prev_c = self.c[ts][k - 1]
            prev_h = self.h[ts][k - 1]
            if np.isfinite(prev_h):
                pos["rolling_high"] = max(pos["rolling_high"], prev_h)
            # 1) 止损/封顶离场
            if pos["stop"] == "trail":
                stop_level = max(pos["tr_entry"][0], pos["rolling_high"] * (1.0 - TRAIL))
                if np.isfinite(prev_c) and prev_c < stop_level:
                    self._sell(pos, self.o[ts][k], k, "trail"); continue
            else:
                if np.isfinite(prev_c) and prev_c < pos["entry_open"] * HARD_DOWN:
                    self._sell(pos, self.o[ts][k], k, "hard"); continue
            # 2) 加仓(仅非一次性且未满档且有现金)
            if pos["mode"] != "single" and pos["n_tr"] < len(self.scheme["sizes"]) and np.isfinite(prev_c):
                sizes = self.scheme["sizes"]
                if pos["mode"] == "up" and prev_c >= pos["last_ref"] * (1.0 + UP_STEP):
                    add = self.pos_budget * sizes[pos["n_tr"]]
                    if self.cash >= add:
                        ap = self.o[ts][k]
                        if np.isfinite(ap) and ap > 0:
                            sh = self._buy(add, ap)
                            if sh > 0:
                                pos["deployed"].append(add); pos["tr_entry"].append(ap)
                                pos["last_ref"] = ap; pos["n_tr"] += 1
                                pos["total_invested"] += add; pos["total_shares"] += sh
                elif pos["mode"] == "down" and prev_c <= pos["last_ref"] * (1.0 - DOWN_STEP):
                    add = self.pos_budget * sizes[pos["n_tr"]]
                    if self.cash >= add:
                        ap = self.o[ts][k]
                        if np.isfinite(ap) and ap > 0:
                            sh = self._buy(add, ap)
                            if sh > 0:
                                pos["deployed"].append(add); pos["tr_entry"].append(ap)
                                pos["last_ref"] = ap; pos["n_tr"] += 1
                                pos["total_invested"] += add; pos["total_shares"] += sh
            # 3) 封顶
            if k >= pos["end_k"]:
                self._sell(pos, self.o[ts][k], k, "maxhold")

    def current_equity(self):
        eq = self.cash
        for pos in self.positions:
            k = self.current_k.get(pos["ts"])
            if k is None:
                continue
            eq += pos["total_shares"] * self.c[pos["ts"]][k] * (1.0 - SLIPPAGE_RATE)
        return eq


def max_drawdown(eq_arr):
    eqs = np.array(eq_arr, dtype=float)
    runmax = np.maximum.accumulate(eqs)
    dd = (eqs - runmax) / runmax
    return dd.min() * 100.0


def screen_five_exante(df_sig, start, pre_years=3):
    """前视-free 选股: 只用回测起点之前 pre_years 年的突破活跃度排序选5只。
    在 t=start 时点, 投资者只能看到 start 之前的数据, 故用 [start-pre_years, start) 的
    突破信号计数排序。这样选股不依赖未来, 但仍是"偏向突破活跃股"的样本(有偏但无前视)。

    返回 (info, pre_start_i)。若估计窗内无信号返回 (None, pre_start_i)。"""
    start_i = int(start)
    pre_start_i = int(f"{int(start[:4]) - pre_years}0101")
    pre = df_sig[(df_sig["trade_date"] >= pre_start_i) & (df_sig["trade_date"] < start_i)]
    cnt = pre.groupby("ts_code")["signal"].sum()
    cnt = cnt[cnt > 0].sort_values(ascending=False)
    if len(cnt) == 0:
        return None, pre_start_i
    top_codes = cnt.head(5).index.tolist()
    info = df_sig[df_sig["ts_code"].isin(top_codes)][["ts_code", "name", "industry"]].drop_duplicates()
    info = info.set_index("ts_code").loc[top_codes].reset_index()
    info["n_pre_signals"] = info["ts_code"].map(cnt)
    return info, pre_start_i


def select_five_neutral(df, start, mode, seed=42):
    """前视-free 中性选股(只用起点当日截面, 无动量偏向), 用于对照实验:
    - random  : 在起点当日 eligible 股票中随机选5只 (纯朴素基线)
    - largecap: 在起点当日 eligible 股票中按流通市值最大选5只 (大蓝筹基线)
    返回 (info, start_i)。"""
    start_i = int(start)
    dts_avail = sorted(df["trade_date"].unique())
    if start_i not in dts_avail:
        start_i = next((x for x in dts_avail if x >= start_i), dts_avail[-1])
    df_start = df[df["trade_date"] == start_i].copy()
    pool = df_start
    if "eligible" in pool.columns:
        pool = pool[pool["eligible"] == True]
    pool = pool.dropna(subset=["ts_code"])
    if mode == "largecap":
        if "circ_mv" in pool.columns:
            pool = pool.dropna(subset=["circ_mv"]).sort_values("circ_mv", ascending=False)
        codes = pool["ts_code"].head(5).tolist()
    else:  # random
        codes_all = pool["ts_code"].unique().tolist()
        rng = np.random.default_rng(seed)
        codes = list(rng.choice(codes_all, size=min(5, len(codes_all)), replace=False))
    info = pool[pool["ts_code"].isin(codes)][["ts_code", "name", "industry"]].drop_duplicates()
    info = info.set_index("ts_code").loc[codes].reset_index()
    info["n_pre_signals"] = np.nan
    return info, int(start_i)


def run_mode(df, df_sig, args, capital, pos_budget, start, mode, codes=None):
    """对单窗口(start) x 单选股模式(mode) 跑4方案, 结果存到 out/start/mode/。"""
    # 1) 选股 (前视-free, 只用起点之前/当日数据)
    if codes:
        sub = df_sig[df_sig["ts_code"].isin(codes)]
        top5 = sub[["ts_code", "name", "industry"]].drop_duplicates()
        cnt = df_sig.groupby("ts_code")["signal"].sum()
        top5["n_signals"] = top5["ts_code"].map(cnt)
        sel_label = "指定codes"
    elif mode == "momentum":
        top5, pre_start_i = screen_five_exante(df_sig, start, args.pre_years)
        codes = top5["ts_code"].tolist()
        sel_label = f"动量(估计窗{pre_start_i}~{start}前5)"
    else:
        top5, sel_start = select_five_neutral(df, start, mode, seed=args.seed)
        codes = top5["ts_code"].tolist()
        sel_label = f"{mode}(起点{sel_start}截面)"
    print(f"[screen][{start}/{mode}] {sel_label}: " + ", ".join(
        f"{r.ts_code}({r.name})" for r in top5.itertuples()))

    # 2) 只保留回测起点之后的信号参与交易, 杜绝前视泄漏
    df5 = df_sig[(df_sig["ts_code"].isin(codes)) & (df_sig["trade_date"] >= int(start))] \
        .sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    # 3) 每只股票提取数组 + 信号索引
    o = {}; c = {}; h = {}; l = {}; dts = {}; signal_sets = {}
    for ts in codes:
        sub = df5[df5["ts_code"] == ts]
        o[ts] = sub["open"].values.astype(float)
        c[ts] = sub["close"].values.astype(float)
        h[ts] = sub["high"].values.astype(float)
        l[ts] = sub["low"].values.astype(float)
        dts[ts] = sub["trade_date"].values.astype(int)
        sig = sub["signal"].values.astype(bool)
        signal_sets[ts] = set(np.where(sig)[0].tolist())
    n_sig = sum(len(v) for v in signal_sets.values())
    print(f"  信号数={n_sig}")

    # 4) 全局交易日 + date->[(ts,k)]
    all_dates = np.unique(df5["trade_date"].values.astype(int))
    all_dates.sort()
    date_map = defaultdict(list)
    for ts in codes:
        for k, d in enumerate(dts[ts]):
            date_map[int(d)].append((ts, k))

    # 5) 4方案各跑一个共享池账户
    results = {}; equity_all = {}
    for name, sc in SCHEMES.items():
        port = Portfolio(sc, o, c, h, l, dts, signal_sets, init=capital, pos_budget=pos_budget)
        eq_series = []
        for d in all_dates:
            todays = date_map.get(int(d), [])
            for ts, k in todays:
                port.current_k[ts] = k
            for ts, k in todays:
                port.step_day(ts, k)
            for ts, k in todays:
                if (k - 1) in signal_sets[ts]:
                    port.open_position(ts, k - 1)
            eq = port.current_equity()
            if eq <= 0:
                port.ruined = True
            eq_series.append((int(d), eq))
        equity_all[name] = eq_series
        nets = np.array([t["net"] for t in port.trades], dtype=float)
        eq_arr = np.array([x[1] for x in eq_series], dtype=float)
        results[name] = dict(
            scheme=name, label=sc["label"],
            final_equity=round(eq_arr[-1], 2),
            total_return_pct=round((eq_arr[-1] / FINITE_CAPITAL - 1) * 100, 2),
            max_dd_pct=round(max_drawdown(eq_arr), 2),
            ruined=port.ruined, n_trades=len(port.trades), skipped_entries=port.skipped_entries,
            win_pct=round((nets > 0).mean() * 100, 2) if len(nets) else 0.0,
            mean_net_pct=round(nets.mean() * 100, 4) if len(nets) else 0.0,
            worst_net_pct=round(nets.min() * 100, 3) if len(nets) else 0.0,
            pct_lt20_pct=round((nets < -0.20).mean() * 100, 2) if len(nets) else 0.0,
        )
        print(f"  {name:10s} 终值={eq_arr[-1]:>12,.0f} 总收益={results[name]['total_return_pct']:+.2f}% "
              f"最大回撤={results[name]['max_dd_pct']:.1f}% 破产={results[name]['ruined']} "
              f"笔数={results[name]['n_trades']} 均净={results[name]['mean_net_pct']:+.3f}% 最差={results[name]['worst_net_pct']:+.1f}%")

    # 6) 保存
    out_dir = os.path.join(args.out, start, mode)
    os.makedirs(out_dir, exist_ok=True)
    top5.to_csv(os.path.join(out_dir, "finite_selected5.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame([results[n] for n in SCHEMES]).to_csv(
        os.path.join(out_dir, "finite_compare.csv"), index=False, encoding="utf-8-sig")
    eq_df = pd.DataFrame({name: [x[1] for x in equity_all[name]] for name in SCHEMES})
    eq_df.insert(0, "trade_date", list(all_dates))
    eq_df.to_csv(os.path.join(out_dir, "finite_equity.csv"), index=False, encoding="utf-8-sig")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=START, help="回测起点, 逗号分隔可多窗口(一次加载全跑)")
    ap.add_argument("--end", default=END)
    ap.add_argument("--L", type=int, default=L_DEF)
    ap.add_argument("--vol-mult", type=float, default=VOL_MULT_DEF)
    ap.add_argument("--capital", type=float, default=FINITE_CAPITAL)
    ap.add_argument("--pos-budget", type=float, default=POS_BUDGET)
    ap.add_argument("--codes", default=None, help="逗号分隔ts_code, 跳过筛选直接用")
    ap.add_argument("--pre-years", type=int, default=3, help="前视-free选股估计窗长度(年)")
    ap.add_argument("--screen-mode", default="momentum",
                    help="选股模式, 逗号分隔可多选: momentum=起点前突破活跃度前5(动量偏向); random=起点当日随机5只; largecap=起点当日流通市值最大5只")
    ap.add_argument("--seed", type=int, default=42, help="random 模式随机种子")
    ap.add_argument("--out", default=DB_OUT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    capital = args.capital
    pos_budget = args.pos_budget
    starts = [s.strip() for s in args.start.split(",") if s.strip()]
    modes = [m.strip() for m in args.screen_mode.split(",") if m.strip()]
    codes = [x.strip() for x in args.codes.split(",") if x.strip()] if args.codes else None

    # 一次加载最早历史, 覆盖所有窗口的前视-free估计窗(多留1年余量)
    sy_min = min(int(s[:4]) for s in starts)
    load_start = f"{sy_min - args.pre_years - 1}0101"
    print(f"[load] {load_start}~{args.end} (回测区间 {starts}, 估计窗前视-free) ...")
    df, idx = load_data(load_start, args.end)
    print(f"  日线行数={len(df):,} 股票数={df['ts_code'].nunique():,}")
    df = add_base_features(df)

    print(f"[signal] L={args.L} VOL_MULT={args.vol_mult} (全市场建信号, 一次) ...")
    df_sig = build_signal(df, idx, args.L, args.vol_mult)

    for start in starts:
        for mode in modes:
            print(f"\n##### 窗口 {start} / 模式 {mode} #####")
            run_mode(df, df_sig, args, capital, pos_budget, start, mode, codes=codes)
    print("\nDONE")


if __name__ == "__main__":
    main()
