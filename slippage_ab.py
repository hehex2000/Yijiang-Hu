"""平方根冲击滑点模型 A/B 验证（单变量隔离）。

对比 flat 0.1% 与 平方根冲击(k·σ·√(Q/ADV)) 在同一篮子个股上的滑点。
输入全为日线（σ_daily、ADV 来自 daily 表），零新数据。

用法：
  ./venv_ml/Scripts/python.exe slippage_ab.py                # 默认 k=1.0
  MFS_SQRT_IMPACT_K=1.5 ./venv_ml/Scripts/python.exe slippage_ab.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_monthly_rebalance import (
    get_conn, estimate_daily_liquidity, sqrt_impact_slippage,
    SLIPPAGE_RATE, get_stock_name, SQRT_IMPACT_LB,
)

# 平台屏蔽：北交所(.BJ)/8xx/920/4xx（投资门槛/老三板）
_EXCLUDED = lambda c: c.endswith(".BJ") or c.startswith(("8", "920", "4"))


def _max_td():
    conn = get_conn()
    r = conn.execute("SELECT MAX(trade_date) FROM daily").fetchone()
    conn.close()
    return int(r[0])


def _small_caps(n=4):
    """取流通市值最小且数据齐备的小票（真实低流动性样本）。"""
    conn = get_conn()
    rows = conn.execute(
        "SELECT ts_code FROM daily_basic WHERE trade_date=? "
        "ORDER BY circ_mv ASC LIMIT 40",
        (_max_td(),)).fetchall()
    conn.close()
    out = []
    for (c,) in rows:
        if _EXCLUDED(c):
            continue
        sigma, adv = estimate_daily_liquidity(c, _max_td())
        if sigma is None or adv <= 0:
            continue
        out.append((c, adv))
        if len(out) >= n:
            break
    return out


def main():
    tc = _max_td()
    k = float(os.environ.get("MFS_SQRT_IMPACT_K", "1.0"))
    print(f"== 平方根冲击滑点 A/B（trade_date={tc}, lookback={SQRT_IMPACT_LB}日, k={k}）==")

    basket = ["600519.SH", "600036.SH", "000858.SZ", "601318.SH",
              "300750.SZ", "002594.SZ"]
    small = _small_caps(4)
    print(f"（低流动性小票样本：{', '.join(c for c, _ in small)}）")

    Q = 1_000_000.0  # 单笔 100 万
    print(f"{'code':<10}{'name':<8}{'ADV(亿)':>9}{'σ_day':>8}{'flat_slip':>10}"
          f"{'sqrt_slip':>10}{'×倍数':>8}")
    rows_out = []
    for c in basket + [c for c, _ in small]:
        sigma, adv = estimate_daily_liquidity(c, tc)
        if sigma is None:
            print(f"{c:<10}{'—':<8}{'NA':>9}{'NA':>8}{'NA':>10}{'NA':>10}{'NA':>8}")
            continue
        frac, _, _ = sqrt_impact_slippage(Q, c, tc)
        flat = Q * SLIPPAGE_RATE
        sq = Q * frac if frac else flat
        name = get_stock_name(c)
        mult = sq / flat if flat else 0
        rows_out.append((c, adv))
        print(f"{c:<10}{name[:6]:<8}{adv / 1e8:>9.2f}{sigma * 100:>7.2f}%"
              f"{flat:>10.0f}{sq:>10.0f}{mult:>7.2f}x")

    # 参与率效应：最低 ADV 小票，不同订单规模
    illiquid = min(rows_out, key=lambda x: x[1])[0] if rows_out else None
    if illiquid:
        print(f"\n== 参与率效应（{illiquid}，订单规模 100万/1000万/1亿）==")
        for q in (1e6, 1e7, 1e8):
            frac, _, _ = sqrt_impact_slippage(q, illiquid, tc)
            slip = q * (frac if frac else SLIPPAGE_RATE)
            sigma, adv = estimate_daily_liquidity(illiquid, tc)
            par = q / adv if adv else 0
            print(f"  Q={q / 1e4:>7.0f}万  Q/ADV={par:>7.3f}  "
                  f"impact={(frac * 100 if frac else 0):>6.2f}%  slip={slip:>10.0f}元")

    print("\n结论：flat 0.1% 对所有票一刀切；平方根冲击让低流动小票滑点自动放大(×数倍)，")
    print("      大盘股反而低于 0.1%（参与率低），且订单越大冲击越大（参与率效应）——")
    print("      正是视频'盘口会消失、实际吃到比看到的差'的日线近似。")


if __name__ == "__main__":
    main()
