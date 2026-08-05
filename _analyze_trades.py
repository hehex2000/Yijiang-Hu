# -*- coding: utf-8 -*-
"""逐笔成交明细分析：逐笔 CSV → 收益分布/异常填充/规则拆解/累计盈亏。"""
import sys, pandas as pd, numpy as np

def analyze(csv):
    df = pd.read_csv(csv)
    sells = df[df["action"] == "SELL"].copy()
    buys  = df[df["action"] == "BUY"].copy()
    # 完整清仓笔（有 pnl 的）—— 排除减仓/末日清算可单独看
    full = sells[sells["reason"].astype(str).str.contains("清仓|fixedN|收破|触顶|末日", na=False)].copy()
    has_ret = sells[sells["ret_pct"].notna()].copy()

    print("="*70)
    print(f"  逐笔分析: {csv.split('/')[-1]}")
    print("="*70)
    print(f"  总笔数 BUY={len(buys)}  SELL={len(sells)}  (其中完整清仓 {len(full)})")

    if len(has_ret):
        r = has_ret["ret_pct"].values
        print(f"\n── 单笔收益分布(完整清仓口径, n={len(r)}) ──")
        print(f"  均值 {r.mean():.2f}%  中位 {np.median(r):.2f}%  标准差 {r.std():.2f}%")
        print(f"  胜率 {(r>0).mean()*100:.1f}%  盈亏比 {r[r>0].mean()/abs(r[r<0].mean()) if (r<0).any() else float('nan'):.2f}")
        print(f"  分位: P5={np.percentile(r,5):.1f}%  P25={np.percentile(r,25):.1f}%  "
              f"P75={np.percentile(r,75):.1f}%  P95={np.percentile(r,95):.1f}%")

    # 持有天数
    if "hold_days" in sells and sells["hold_days"].notna().any():
        h = sells["hold_days"].dropna().values
        print(f"\n── 持有天数(n={len(h)}) 均值{h.mean():.1f} 中位{np.median(h):.0f} "
              f"P90={np.percentile(h,90):.0f}")

    # 按卖出原因拆解
    print(f"\n── 按卖出原因拆解 ──")
    g = has_ret.groupby("reason")["ret_pct"].agg(["count","mean","median",lambda x:(x>0).mean()*100])
    g.columns = ["笔数","均值%","中位%","胜率%"]
    print(g.to_string())

    # 异常填充检测：单笔收益极端
    print(f"\n── 异常填充检测 ──")
    extreme = has_ret[has_ret["ret_pct"].abs() >= 50]
    print(f"  单笔|收益|≥50% 的笔数: {len(extreme)}")
    if len(extreme):
        print(extreme[["date","code","reason","buy_date","buy_price","price","shares","ret_pct","pnl"]].head(20).to_string(index=False))

    # 价格异常（0 或极端值）
    bad = buys[(buys["price"]<=0) | (buys["price"]>5000)]
    print(f"  买入价异常(≤0 或 >5000): {len(bad)}")
    if len(bad):
        print(bad[["date","code","price","shares"]].head(10).to_string(index=False))

    # 最大盈/亏单
    if len(has_ret):
        print(f"\n── 最大 10 盈利单 ──")
        print(has_ret.nlargest(10,"pnl")[["date","code","reason","buy_date","buy_price","price","shares","hold_days","ret_pct","pnl"]].to_string(index=False))
        print(f"\n── 最大 10 亏损单 ──")
        print(has_ret.nsmallest(10,"pnl")[["date","code","reason","buy_date","buy_price","price","shares","hold_days","ret_pct","pnl"]].to_string(index=False))

    # 累计 pnl
    if "pnl" in has_ret and has_ret["pnl"].notna().any():
        print(f"\n── 累计已实现盈亏 ──")
        print(f"  已实现 pnl 合计: ¥{has_ret['pnl'].sum():,.0f}")
        print(f"  费用合计(买+卖): ¥{df['fee'].sum():,.0f}")

    # 同一票连续被减仓再清仓的模式
    print(f"\n── 规则3止盈触发次数: {(sells['reason'].astype(str)=='规则3止盈').sum()} ──")
    print(f"── 规则4减仓(c=1)次数: {(sells['reason'].astype(str)=='规则4减仓(c=1)').sum()} ──")
    print(f"── 规则5清仓次数: {sells['reason'].astype(str).str.contains('规则5清仓').sum()} ──")

if __name__ == "__main__":
    analyze(sys.argv[1] if len(sys.argv)>1 else
            "data/results/ma5_swing/trades_all_full_20210104_20231229.csv")
