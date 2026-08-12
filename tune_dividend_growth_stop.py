# -*- coding: utf-8 -*-
"""
高股息+基本面成长 止损消融实验
===============================
同一窗口(20140101~20260720)、同一参数下对比：
  A. 无止损（baseline，原策略）
  B. 15% 硬止损（收盘 < 买入价×0.85 → 收盘卖出）
  C. ATR 动态止损（period14, 入场-ATR×3 起步, 最高价-ATR×3 追踪, 防线只升不降）

输出：总表 + 逐年对比。结论看数据说话。
"""
import pandas as pd
import run_dividend_growth_monthly as rdg

START, END = "20140101", "20260720"
BASE = dict(top_n=10, top_pct=0.10, pe_max=20.0, peg_min=0.08, peg_max=2.0,
            roe_min=3.0, rev_min=5.0, np_min=11.0)
VARIANTS = [
    ("A 无止损(baseline)", {}),
    ("B 15%硬止损", {"stop_loss": 0.15}),
    ("C ATR止损(14/3x)", {"atr_stop": 3.0, "atr_period": 14}),
]

def fmt_pct(x, signed=False):
    if x is None or pd.isna(x):
        return "-"
    return f"{x*100:+.2f}%" if signed else f"{x*100:.2f}%"

rows, yr_rows = [], []
for name, extra in VARIANTS:
    cfg = dict(BASE); cfg.update(extra)
    print(f"\n{'='*80}\n>>> {name}\n{'='*80}")
    r = rdg.run_window(START, END, cfg)
    if r is None:
        rows.append({"变体": name, "结果": "跳过"})
        continue
    mh, mr, mb = r["m_hfq"], r["m_raw"], r["m_b300"]
    rows.append({
        "变体": name,
        "总收益hfq": fmt_pct(mh["total"]),
        "年化hfq": fmt_pct(mh["ann"]),
        "回撤raw": fmt_pct(mr["mdd"]),
        "夏普hfq": f"{mh['sharpe']:.3f}",
        "超额vs沪深300": fmt_pct(mh["total"] - mb["total"], signed=True),
        "止损卖出笔数": r.get("n_stop_sells", 0),
        "期末资产": f"{mh['final']:,.0f}",
    })
    if r.get("m_b932"):
        rows[-1]["中证红利参照"] = fmt_pct(r["m_b932"]["total"])
    for y in sorted(r["y_hfq"]):
        yr_rows.append({
            "变体": name, "年份": y,
            "策略hfq": fmt_pct(r["y_hfq"].get(y, 0), signed=True),
            "沪深300": fmt_pct(r["y_b"].get(y, 0), signed=True),
        })

print("\n\n" + "=" * 90)
print("止损消融总表（20140101~20260720, 初始 200,000, 基准 沪深300 +109.xx% 视运行）")
print("=" * 90)
print(pd.DataFrame(rows).to_string(index=False))

print("\n" + "=" * 90)
print("逐年收益（hfq 口径，%）")
print("=" * 90)
piv = pd.pivot_table(pd.DataFrame(yr_rows), index="年份", columns="变体",
                     values="策略hfq", aggfunc="first")
print(piv.to_string())

out = "data/results/dividend_growth/stop_loss_ablation.csv"
import os
os.makedirs(os.path.dirname(out), exist_ok=True)
pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
print(f"\n总表已保存: {out}")
