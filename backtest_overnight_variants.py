# -*- coding: utf-8 -*-
"""
隔夜持股法 · 两个变体快速证伪
================================
变体 1「只做隔周末」：仅保留 T 日为星期五的买入信号，卖点仍为 next_open / next_close
            （A股周五的次一交易日天然是周一，字段已指向周一，无需特殊处理）。
变体 2「只做强势股」：在 eligible 股票池基础上加中期趋势过滤 close > MA20 > MA60。
            （与现有 momentum「日内强势 ret>3%&高位&放量」区分：这里是中期趋势强势。）

对照维度：mode ∈ {all, momentum, strong} × weekend ∈ {False, True}
情景：base_open(次日开盘卖) / hold_close(次日收盘卖) / forced_ld(次日跌停强制持有) / limitup(涨停开盘)

费用模型与平台一致，单笔往返≈0.35%（净收益按此估算，见 base_open 净 ≈ 毛-0.35%）。

⚠️ 注：本脚本按"合理代理定义"复刻视频两个变体，非逐字复刻原视频参数。
"""
import sys, os, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backtest_overnight as B
import backtest_overnight_scenarios as S

START, END = "20150101", "20260630"


def add_ma(df):
    """在已排序 df 上计算 MA20 / MA60（用于强势股趋势过滤）。"""
    df = df.sort_values(["ts_code", "trade_date"])
    g = df.groupby("ts_code", sort=False)
    df["ma20"] = g["close"].transform(lambda x: x.rolling(20, min_periods=10).mean())
    df["ma60"] = g["close"].transform(lambda x: x.rolling(60, min_periods=30).mean())
    return df


def get_sub(df, mode, weekend):
    """返回一组合格交易样本（已含 ma / next_* 字段）。"""
    if mode == "all":
        sub = B.select(df, "all")
    elif mode == "momentum":
        sub = B.select(df, "momentum")
    elif mode == "strong":
        base = B.select(df, "all")          # eligible 池，保留 ma 列
        sub = base[(base["close"] > base["ma20"]) & (base["ma20"] > base["ma60"])]
    else:
        raise ValueError(mode)
    if weekend:
        td = pd.to_datetime(sub["trade_date"].astype(str), format="%Y%m%d")
        sub = sub[td.dt.weekday == 4]       # 仅星期五买入
    return sub


def run_sub(sub, pm):
    if len(sub) == 0:
        return None
    c = sub["close"].to_numpy(dtype="float64")
    no = sub["next_open"].to_numpy(dtype="float64")
    nc = sub["next_close"].to_numpy(dtype="float64")
    ntd = sub["next_td"].to_numpy()
    codes = sub["ts_code"].to_numpy()
    ld = sub["next_limit_down_open"].to_numpy()
    lu = sub["next_limit_up_open"].to_numpy()

    base = no / c - 1.0
    holdcls = nc / c - 1.0

    forced = base.copy()
    n_forced = 0
    worst = 0.0
    for i in np.where(ld)[0]:
        ret, steps, cap = S.walk_forced_hold(pm, codes[i], int(ntd[i]), c[i])
        if not np.isnan(ret):
            forced[i] = ret
            n_forced += 1
            worst = min(worst, ret)

    lu_idx = np.where(lu)[0]
    lu_sellnow = base[lu_idx] if len(lu_idx) else np.array([])
    lu_holdcls = holdcls[lu_idx] if len(lu_idx) else np.array([])

    return {
        "base_open":      S.scenario_stats(base),
        "hold_close":     S.scenario_stats(holdcls),
        "forced_ld":      S.scenario_stats(forced),
        "forced_ld_n":    n_forced,
        "forced_ld_worst": worst * 100,
        "limitup_n":      len(lu_idx),
        "limitup_sellnow": S.scenario_stats(lu_sellnow),
        "limitup_holdcls": S.scenario_stats(lu_holdcls),
    }


def main():
    t0 = time.time()
    print(f"[load] {START}~{END} ...  注: 按合理代理定义复刻, 非逐字复刻原视频", flush=True)
    df = B.load_data(START, END)
    df = B.build_signals(df)
    df = add_ma(df)
    df = S.add_next(df)
    pm = S.build_price_map(df)
    print(f"[load done] {len(df):,} rows, {time.time()-t0:.1f}s", flush=True)

    rows = []
    for mode in ["all", "momentum", "strong"]:
        for wknd in [False, True]:
            tag = f"{mode}{'_wknd' if wknd else ''}"
            sub = get_sub(df, mode, wknd)
            s = run_sub(sub, pm)
            if s is None:
                print(f"\n=== {tag} : 无样本 ===", flush=True)
                continue
            print(f"\n=== {tag} (N={s['base_open']['n']:,}) ===", flush=True)
            print(f"  base_open  : {s['base_open']['mean']:+.3f}%  胜率{s['base_open']['win']:.1f}%  中位{s['base_open']['median']:+.3f}%")
            print(f"  hold_close : {s['hold_close']['mean']:+.3f}%  胜率{s['hold_close']['win']:.1f}%  中位{s['hold_close']['median']:+.3f}%")
            print(f"  forced_ld  : {s['forced_ld']['mean']:+.3f}%  胜率{s['forced_ld']['win']:.1f}%  强制持有{s['forced_ld_n']}笔  最差{s['forced_ld_worst']:+.1f}%")
            if s['limitup_n']:
                print(f"  涨停开盘 {s['limitup_n']} 笔: 开盘卖 {s['limitup_sellnow']['mean']:+.2f}% vs 收盘 {s['limitup_holdcls']['mean']:+.2f}%")
            for sc in ["base_open", "hold_close", "forced_ld", "limitup_sellnow", "limitup_holdcls"]:
                st = s[sc]
                rows.append(dict(mode=mode, weekend=wknd, scenario=sc, n=st["n"],
                                 mean_ret_pct=round(st["mean"], 4),
                                 median_pct=round(st["median"], 4),
                                 win_pct=round(st["win"], 2),
                                 max_loss_pct=round(st["max_loss"], 2)))
    od = "data/results/overnight"
    os.makedirs(od, exist_ok=True)
    pd.DataFrame(rows).to_csv(f"{od}/variants.csv", index=False, encoding="utf-8-sig")
    print(f"\n[done] 已保存 {od}/variants.csv  ({time.time()-t0:.1f}s)  注: 按合理代理定义复刻, 非逐字复刻原视频")


if __name__ == "__main__":
    main()
