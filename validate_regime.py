# -*- coding: utf-8 -*-
"""
Regime β 兜底 回测验证（rsrs-weight=0 纯平台，隔离 regime 增量）
==============================================================
对照三件事：
  ① 24-09~26-08 踏空补齐：regime ON 的超额应较 OFF(-53pp) 大幅回升
  ② 2022 防御不被侵蚀：regime ON 的 2022 超额应仍≈+70%（与 OFF 持平）
  ③ 全周期 20-26 夏普/回撤不恶化：ON 的 sharpe/maxDD 不弱于 OFF
每格取 run_etf_rotation 返回的 dict；函数自带打印被重定向丢弃，只输出本表。
"""
import contextlib, io, sys
import run_etf_rotation_v6_merged as M
from regime_core import build_regime_hook

RSRS = 0.0   # 纯平台，隔离 regime 增量

WINDOWS = [
    ("牛市窗口 24-09~26-08", "20240901", "20260831"),
    ("2022 崩盘年",          "20220101", "20221231"),
    ("2024 全年",            "20240101", "20241231"),
    ("2025 全年",            "20250101", "20251231"),
    ("2026(至08)",           "20260101", "20260831"),
    ("全周期 20-26",         "20200101", "20260831"),
]


def run_one(start, end, regime_on):
    hook = None
    if regime_on:
        hook = build_regime_hook()   # 默认 RuleB / floor 0.40 / proxy / lag 2
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = M.run_etf_rotation(start, end, rsrs_weight=RSRS,
                                 regime_hook=hook, verbose=False)
    return res


def fmt(res):
    if res is None:
        return "N/A"
    ex = res["total_return"] - res["idx_return"]
    return (f"{res['total_return']:+7.2f}% | 沪深300 {res['idx_return']:+7.2f}% | "
            f"超额 {ex:+7.2f}pp | 夏普 {res['sharpe']:+.2f} | 回撤 {res['max_drawdown']:+.2f}% | "
            f"交易 {res['trades']}")


print("=" * 120)
print(f" Regime β 兜底 验证  (rsrs-weight={RSRS}, RuleB, βfloor=40%, proxy宽度, 滞后2月)")
print("=" * 120)
print(f"{'窗口':<22} | {'regime OFF':<16} | {'regime ON':<16}")
print("-" * 120)

summary = []
for name, s, e in WINDOWS:
    off = run_one(s, e, False)
    on = run_one(s, e, True)
    summary.append((name, off, on))
    print(f"\n### {name}  ({s}~{e})")
    print(f"  OFF : {fmt(off)}")
    print(f"  ON  : {fmt(on)}")
    if off and on:
        d_ex = (on['total_return'] - on['idx_return']) - (off['total_return'] - off['idx_return'])
        d_sh = on['sharpe'] - off['sharpe']
        d_dd = on['max_drawdown'] - off['max_drawdown']   # 负数=回撤更小=更好
        print(f"  Δ   : 超额 {d_ex:+7.2f}pp | 夏普 {d_sh:+.2f} | 回撤 {d_dd:+.2f}pp")

print("\n" + "=" * 120)
print(" 三件事判定")
print("=" * 120)
# ① 牛市窗口
_, off_bull, on_bull = summary[0]
ex_off = off_bull['total_return'] - off_bull['idx_return']
ex_on = on_bull['total_return'] - on_bull['idx_return']
print(f" ① 踏空补齐 : 牛市窗口超额 OFF={ex_off:+.2f}pp -> ON={ex_on:+.2f}pp "
      f"({'+回升' if ex_on > ex_off + 10 else '未明显改善'})")
# ② 2022
_, off22, on22 = summary[1]
ex22_off = off22['total_return'] - off22['idx_return']
ex22_on = on22['total_return'] - on22['idx_return']
print(f" ② 2022防御 : 超额 OFF={ex22_off:+.2f}pp -> ON={ex22_on:+.2f}pp "
      f"({'保留' if abs(ex22_on - ex22_off) < 5 else '被侵蚀!'})")
# ③ 全周期
_, offall, onall = summary[5]
ok_sh = onall['sharpe'] >= offall['sharpe'] - 0.1
ok_dd = onall['max_drawdown'] <= offall['max_drawdown'] + 2.0
print(f" ③ 全周期    : 夏普 OFF={offall['sharpe']:+.2f} -> ON={onall['sharpe']:+.2f} "
      f"({'不恶化' if ok_sh else '恶化!'}) | "
      f"回撤 OFF={offall['max_drawdown']:+.2f}% -> ON={onall['max_drawdown']:+.2f}% "
      f"({'不恶化' if ok_dd else '恶化!'})")
