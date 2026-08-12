# -*- coding: utf-8 -*-
"""
真 OOS 实验：用 2018-2023 数据挑 4 只 -> 固定不动 -> 测 2024-2026
============================================================
目的：把"V6 后视镜 4 只篮子(+61.67% 牛市窗口)到底有没有真 alpha"钉死。
方法：
  1. 候选池 = 平台 19 只可交易 ETF + 三大红利 ETF(510880/512890/515080)，
     全部 2018 前上市，为"2023 年底真实可挑的池子"(无未来产品、无幸存者偏差注入)。
  2. in-sample 选择(只用 <=2023-12-31 数据)：按 2018-2023 累计收益排名选前 4；
     另按 2018-2023 夏普排名做稳健性检查。
  3. OOS 固定持有：2024-09-01~2026-08-07(与后视镜V6牛市窗口一致)，
     等权买入持有、不轮动不调仓，平台真实成本(佣0.025%+最低5元+滑点0.1%双向+ETF免印花税)。
  4. 对照：HS300 +42.59% / 平台active -10.78% / 后视镜V6 4只 +61.67%(含答案) /
          候选池22只等权买入持有(被动分散基线)。
"""
import numpy as np
import pandas as pd
from run_monthly_rebalance import get_conn

CANDIDATES = [
    ("510300.SH", "沪深300ETF"), ("510050.SH", "上证50ETF"), ("515800.SH", "中证800ETF"),
    ("510980.SH", "上证指数ETF"), ("510500.SH", "中证500ETF"), ("512100.SH", "中证1000ETF"),
    ("159915.SZ", "创业板ETF"), ("159949.SZ", "创业板50ETF"), ("588000.SH", "科创50ETF"),
    ("512480.SH", "半导体ETF"), ("515030.SH", "新能源车ETF"), ("512010.SH", "医药ETF"),
    ("159928.SZ", "消费ETF"), ("512880.SH", "证券ETF"), ("159920.SZ", "恒生ETF"),
    ("513100.SH", "纳指ETF"), ("518880.SH", "黄金ETF"), ("501018.SH", "原油LOF"),
    ("511010.SH", "国债ETF"),
    ("510880.SH", "红利ETF"), ("512890.SH", "红利低波ETF"), ("515080.SH", "红利ETF"),
]

IN_START, IN_END = "20180101", "20231231"
OOS_START, OOS_END = "20240901", "20260807"
COMMISSION_RATE = 0.00025
COMMISSION_MIN = 5.0
SLIPPAGE_RATE = 0.001
INIT_CAPITAL = 1_000_000.0        # 真实资金规模，避免¥5最低佣失真
N_PICK = 4


def load(code):
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT trade_date, close FROM etf_daily "
        "WHERE ts_code=? AND trade_date BETWEEN ? AND ? ORDER BY trade_date",
        conn, params=(code, "20180101", OOS_END))
    conn.close()
    df["trade_date"] = df["trade_date"].astype(str)
    return df


def metrics_insample(df):
    ins = df[(df.trade_date >= IN_START) & (df.trade_date <= IN_END)]
    if len(ins) < 50:
        return None
    first_c, last_c = ins.iloc[0]["close"], ins.iloc[-1]["close"]
    cum = last_c / first_c - 1.0
    rets = np.log(ins["close"].values[1:] / ins["close"].values[:-1])
    n = len(rets)
    ann_ret = (last_c / first_c) ** (252.0 / n) - 1.0
    vol = rets.std() * np.sqrt(252)
    sharpe = (ann_ret - 0.02) / vol if vol > 0 else 0.0
    return {"cum": cum, "ann_ret": ann_ret, "vol": vol, "sharpe": sharpe}


def oos_hold(df_map, codes):
    """等权买入持有(不调仓)，真实成本。返回 (gross, net, per_list, entry_d, exit_d)。"""
    k = len(codes)
    per_unit = INIT_CAPITAL / k
    gross = 0.0
    net = 0.0
    per = []
    entry_ds, exit_ds = [], []
    for c in codes:
        d = df_map[c]
        oos = d[d.trade_date >= OOS_START]
        if len(oos) < 2:
            return None
        e_d, x_d = oos.iloc[0]["trade_date"], oos.iloc[-1]["trade_date"]
        pe, px = oos.iloc[0]["close"], oos.iloc[-1]["close"]
        ri = px / pe - 1.0
        entry_amt = per_unit
        exit_amt = per_unit * (1 + ri)
        entry_cost = min(max(entry_amt * COMMISSION_RATE, COMMISSION_MIN), entry_amt) + entry_amt * SLIPPAGE_RATE
        exit_cost = min(max(exit_amt * COMMISSION_RATE, COMMISSION_MIN), exit_amt) + exit_amt * SLIPPAGE_RATE
        ri_net = (exit_amt - exit_cost) / (entry_amt - entry_cost) - 1.0
        gross += (1 + ri) / k
        net += (1 + ri_net) / k
        per.append((c, ri, ri_net))
        entry_ds.append(e_d)
        exit_ds.append(x_d)
    return {"gross": gross - 1.0, "net": net - 1.0, "per": per,
            "entry_d": entry_ds[0], "exit_d": exit_ds[0]}


def main():
    print("=" * 72)
    print("真 OOS 实验：2018-2023 选 4 只 -> 固定持有 2024-2026")
    print("=" * 72)
    print(f"候选池: {len(CANDIDATES)} 只 (平台19 + 红利3), 全部 2018 前上市")
    print(f"in-sample 选择窗口: {IN_START}~{IN_END} (只用此区间数据)")
    print(f"OOS 测试窗口:       {OOS_START}~{OOS_END} (与后视镜V6牛市窗口一致)")
    print(f"成本: 佣0.025%+最低5元+滑点0.1%双向, ETF免印花税; 初始{INIT_CAPITAL/10000:.0f}万等权{N_PICK}只")
    print("-" * 72)

    ins, df_map = {}, {}
    for code, name in CANDIDATES:
        df = load(code)
        m = metrics_insample(df)
        if m is None:
            print(f"  [跳过] {code} {name} 数据不足")
            continue
        ins[(code, name)] = m
        df_map[code] = df

    print(f"\n[候选池有效 {len(ins)} 只] 按 2018-2023 累计收益排名(前8):")
    ranked = sorted(ins.items(), key=lambda kv: kv[1]["cum"], reverse=True)
    for i, ((code, name), m) in enumerate(ranked[:8], 1):
        print(f"  {i:2d}. {name:12s} {code}  累计 {m['cum']*100:+7.1f}%  "
              f"年化 {m['ann_ret']*100:+6.1f}%  波动 {m['vol']*100:5.1f}%  夏普 {m['sharpe']:+.2f}")
    ranked_s = sorted(ins.items(), key=lambda kv: kv[1]["sharpe"], reverse=True)

    def show(label, codes):
        names = [dict(CANDIDATES)[c] for c in codes]
        print(f"\n── 选择规则[{label}] 选中: " + ", ".join(f"{n}({c})" for c, n in zip(codes, names)))
        r = oos_hold(df_map, codes)
        if r is None:
            print("  OOS 计算失败(缺价)")
            return
        print(f"  OOS {r['entry_d']}~{r['exit_d']} 固定等权持有:")
        for c, ri, ri_net in r["per"]:
            print(f"    {dict(CANDIDATES)[c]:12s} {c}  毛 {ri*100:+7.2f}%  净 {ri_net*100:+7.2f}%")
        print(f"  >> 组合毛 {r['gross']*100:+7.2f}%   净(扣成本) {r['net']*100:+7.2f}%")

    show("累计收益前4", [c for (c, _), _ in ranked[:N_PICK]])
    show("夏普前4", [c for (c, _), _ in ranked_s[:N_PICK]])

    # 基线：候选池全部等权买入持有(被动分散)
    all_codes = [c for (c, _) in ins.keys()]
    rb = oos_hold(df_map, all_codes)
    if rb:
        print(f"\n── 基线[候选池{len(all_codes)}只等权买入持有]:")
        print(f"  >> 毛 {rb['gross']*100:+7.2f}%   净 {rb['net']*100:+7.2f}%")

    print("\n" + "=" * 72)
    print("对照基准 (同窗口 2024-09-01~2026-08-07):")
    print(f"  后视镜V6 4只(含答案, 全段2018-2026挑):  +61.67%  <-- 要证伪的数字")
    print(f"  沪深300ETF(被动):                      +42.59%")
    print(f"  平台 active(20只广覆盖轮动):           -10.78%")
    print(f"  本实验 OOS选4(2018-2023挑,固定持有):   见上(+38%量级, 且<HS300)")
    print("=" * 72)


if __name__ == "__main__":
    main()
