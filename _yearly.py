# -*- coding: utf-8 -*-
"""仅做分析：monkeypatch 拦截每日NAV，按年切分算年度盈亏表。不改动任何策略逻辑。"""
import numpy as np
import run_etf_rotation as mod

captured = {}
_orig = mod.compute_reality_discounts

def _wrap(daily_vals, capital, *a, **k):
    captured['daily_vals'] = daily_vals
    return _orig(daily_vals, capital, *a, **k)

mod.compute_reality_discounts = _wrap

START, END = "20200101", "20260807"
res = mod.run_etf_rotation(START, END, method="dual", pool="mixed", verbose=False)
assert captured, "拦截失败"
dv = captured['daily_vals']
dates = [int(d['date']) for d in dv]
vals = np.array([float(d['value']) for d in dv])
capital = mod.INITIAL_CAPITAL
bench = mod.BENCHMARK_CODE
td = mod.get_trade_dates(START, END)

print(f"头条校验: 总收益 {res['total_return']:+.2f}%  年化 {res['annual_return']:+.2f}%  "
      f"最大回撤 {res['max_drawdown']:.2f}%  夏普 {res['sharpe']:.2f}  基准 {res['idx_return']:+.2f}%")

years = sorted(set(d // 10000 for d in dates))
print(f"\n{'年度':<6}{'年末资产':>14}{'年度收益':>12}{'沪深300':>10}{'超额':>10}{'年度最大回撤':>14}")
print("-" * 68)
prev = capital
tot_yr = []
for y in years:
    yr_dates = [d for d in dates if y * 10000 <= d <= y * 10000 + 1231]
    last_d = max(yr_dates)
    end_val = float(vals[dates.index(last_d)])
    yr_ret = end_val / prev - 1.0
    prev = end_val
    # 基准年度
    ty = [int(d) for d in td if y * 10000 <= int(d) <= y * 10000 + 1231]
    b0 = mod.get_etf_price(bench, ty[0])
    b1 = mod.get_etf_price(bench, ty[-1])
    bench_yr = (b1 / b0 - 1) if (b0 and b1) else float('nan')
    # 年度内最大回撤(相对年初)
    idxs = [dates.index(d) for d in yr_dates]
    sub = vals[min(idxs): max(idxs) + 1]
    peak = np.maximum.accumulate(sub)
    dd_within = float(np.min((sub - peak) / peak)) * 100
    ex = yr_ret - (bench_yr if bench_yr == bench_yr else 0)
    tot_yr.append((y, end_val, yr_ret, bench_yr, yr_ret - (bench_yr if bench_yr == bench_yr else 0), dd_within))
    print(f"{y:<6}{end_val:>14,.0f}{yr_ret*100:>+11.2f}%{bench_yr*100:>+9.2f}%{ex*100:>+9.2f}%{dd_within:>13.2f}%")

comp = np.prod([1 + r[2] for r in tot_yr]) - 1
print("-" * 68)
print(f"{'累计':<6}{prev:>14,.0f}{comp*100:>+11.2f}%")
