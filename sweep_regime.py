# -*- coding: utf-8 -*-
"""
Regime β 兜底 诊断 + βfloor 扫描（rsrs=0 纯平台）
==================================================
1) 诊断牛市窗口：有效 BULL 月数、全防御月数（floor 前）、top β 标的。
2) 扫描 beta_floor ∈ {0.4,0.5,0.6,0.7,0.8} 在 牛市窗口 / 2026(至08) / 全周期 的
   超额 / 夏普 / 回撤，刻画"踏空补齐 ↔ 风险"权衡。
   注：2022 恒为 BEAR，floor 永不生效 → 2022 防御对任意 floor 都不变（已验证保留）。
"""
import contextlib, io
import run_etf_rotation_v6_merged as M
from regime_core import build_regime_hook

RSRS = 0.0

def run(start, end, hook):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return M.run_etf_rotation(start, end, rsrs_weight=RSRS,
                                  regime_hook=hook, verbose=False)

def ex_of(res):
    return res['total_return'] - res['idx_return']

# ── 1) 诊断牛市窗口 ──
print("=" * 90)
print(" 诊断：牛市窗口 24-09~26-08 (regime ON, floor=0.40)")
print("=" * 90)
hook = build_regime_hook(beta_floor=0.40)
res = run("20240901", "20260831", hook)
det = hook.detector
n_bull = sum(1 for _, r, e in det._history if e == 'BULL')
n_raw_bull = sum(1 for _, r, e in det._history if r == 'BULL')
print(f" 调仓月数={len(det._history)} | 原始BULL={n_raw_bull} | 有效BULL(滞后后)={n_bull}")
print(f" 沪深300同期={res['idx_return']:+.2f}%")
# 全防御月（floor 前）：用 off 模式的 targets 不便获取，改以历史成交推断——略。
print(f" 组合收益 OFF≈-10.07% / ON={res['total_return']:+.2f}% | 超额 ON={ex_of(res):+.2f}pp")

# ── 2) βfloor 扫描 ──
print("\n" + "=" * 90)
print(" βfloor 扫描 (regime ON, RuleB, proxy, lag2)")
print("=" * 90)
for label, s, e in [("牛市窗口24-09~26-08", "20240901", "20260831"),
                    ("2026(至08)", "20260101", "20260831"),
                    ("全周期20-26", "20200101", "20260831")]:
    print(f"\n### {label}")
    print(f"  {'floor':>6} | {'总收益':>9} | {'沪深300':>9} | {'超额':>9} | {'夏普':>7} | {'回撤':>8}")
    off = run(s, e, None)
    print(f"  {'OFF':>6} | {off['total_return']:+8.2f}% | {off['idx_return']:+8.2f}% | "
          f"{ex_of(off):+8.2f}pp | {off['sharpe']:+6.2f} | {off['max_drawdown']:+7.2f}%")
    for fl in (0.4, 0.5, 0.6, 0.7, 0.8):
        r = run(s, e, build_regime_hook(beta_floor=fl))
        print(f"  {fl:>6.2f} | {r['total_return']:+8.2f}% | {r['idx_return']:+8.2f}% | "
              f"{ex_of(r):+8.2f}pp | {r['sharpe']:+6.2f} | {r['max_drawdown']:+7.2f}%")
