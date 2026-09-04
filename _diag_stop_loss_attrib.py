# -*- coding: utf-8 -*-
"""归因实验：把 hfq-raw 的 11.5pp/年 差额拆成「股息贡献」与「止损路径效应」两块

四象限：
  (raw , 止损ON )   ← 平台历史口径（含假止损）
  (hfq , 止损ON )   ← 正确口径
  (raw , 止损OFF)
  (hfq , 止损OFF)

判定逻辑：
  股息贡献   ≈ hfq_OFF 年化 − raw_OFF 年化      （纯价格空间，无路径干扰）
  止损路径效应 ≈ (hfq_ON − raw_ON) − (hfq_OFF − raw_OFF)
  若「股息贡献」落在受控实验的 4-7%/年区间 → 价格空间实现正确。
  若「止损路径效应」为正且大 → 说明 raw 口径下的除息假止损在系统性伤害策略，
                              这部分是**真实存在的旧 bug 后果**，不是新引入的偏差。
"""
import sys
import os
import io
import contextlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_monthly_rebalance as m

D0, D1 = "20200102", "20260723"
TOPN = 20


def run(mode, stop):
    m.PRICE_MODE = mode
    m._ADJ_CACHE.clear()
    m._ADJ_REF.clear()
    m.STOP_LOSS = stop
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = m.run_backtest(
            start_date=D0, end_date=D1, top_n=TOPN,
            selection_method="value", stop_loss_pct=stop)
    return res, buf.getvalue()


quad = {}
for mode in ("raw", "hfq"):
    for stop in (0.15, 0.0):
        key = (mode, stop)
        res, log = run(mode, stop)
        quad[key] = res
        print(f"[{mode:3s} 止损{stop:.0%}]  总收益 {res['total_return']:+8.2f}%  "
              f"年化 {res['annual_return']:+7.2f}%  MDD {res['max_drawdown']:+7.2f}%  "
              f"Sharpe {res['sharpe']:+.2f}  交易 {res['trades']:d}", flush=True)

print()
print("=" * 74)
print("拆解（2020-01-02 → 2026-07-23，value top20）")
print("=" * 74)

gap_on = quad[("hfq", 0.15)]["annual_return"] - quad[("raw", 0.15)]["annual_return"]
gap_off = quad[("hfq", 0.0)]["annual_return"] - quad[("raw", 0.0)]["annual_return"]
sl_eff_hfq = quad[("hfq", 0.0)]["annual_return"] - quad[("hfq", 0.15)]["annual_return"]
sl_eff_raw = quad[("raw", 0.0)]["annual_return"] - quad[("raw", 0.15)]["annual_return"]

print(f"  hfq−raw（止损ON ）= {gap_on*100:+.2f} pp/年")
print(f"  hfq−raw（止损OFF）= {gap_off*100:+.2f} pp/年   ← 纯股息贡献（应与受控实验 4-7% 对齐）")
print(f"  止损路径效应      = {(gap_on-gap_off)*100:+.2f} pp/年")
print()
print(f"  止损本身的代价（hfq 口径）= {sl_eff_hfq*100:+.2f} pp/年")
print(f"  止损本身的代价（raw 口径）= {sl_eff_raw*100:+.2f} pp/年")
print()
if 0.03 <= gap_off <= 0.08:
    print("  ✅ 判定：纯股息贡献落在合理区间 → 价格空间实现正确")
else:
    print(f"  ⚠️ 判定：纯股息贡献 {gap_off*100:.2f}%/年 偏离受控实验区间(4-7%) → 需进一步排查")
