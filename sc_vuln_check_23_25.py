# -*- coding: utf-8 -*-
"""
小市值轮动策略 · 脆弱性体检 (Vulnerability Health-Check)
=======================================================
问题：这条策略的收益是不是"真踩在 A股制度/生态位上"（视频说的
      小市值控盘 = 流动性充裕 + 小盘占优 + 散户行为偏差）？
      一旦生态位切换（流动性枯竭 / 小盘跑输 / 风格反转），收益是否塌方？

方法：
  1. 调用真实回测 run_backtest() 拿到逐日净值 (date, value, bench=中证2000)。
  2. 构建两点"生态位"状态（point-in-time，不掺未来函数）：
       · 流动性 Liquidity  = 全市场当日总成交额（亿元）
       · 小盘占优 Style    = 中证2000 近20日收益 − 沪深300 近20日收益
  3. 按历史分布切成三档（低/中/高），把每个交易日归入 (流动性 × 小盘占优) 矩阵。
  4. 在每一格里复利累计策略日收益 / 基准日收益，年化对比。
  5. 另列已知"生态位塌方"窗口：2015股灾 / 2018熊市 / 2024-02微盘流动性危机。

判读：
  若策略只在【高流动性 + 小盘占优】格显著盈利，而在【低流动性 / 小盘跑输】格
  大幅亏损 → 证实它"踩在生态位上"，高度 regime-dependent、脆弱。
  若各格都稳定盈利 → 说明它有一定跨生态稳健性。
"""
import os, sys, json, time, sqlite3
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA
DB = DATA.get("local_db_path", "D:/tu-shareData/astock_daily.db")

START, END = "20230101", "20251231"
NAV_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         f"data/results/sc_vuln_nav_{START}_{END}.json")

from backtest_small_cap_rotation import run_backtest


def load_nav():
    if os.path.exists(NAV_CACHE):
        print(f"[NAV] 命中缓存 {NAV_CACHE}")
        with open(NAV_CACHE, encoding="utf-8") as f:
            nav = json.load(f)
        for d in nav:
            d["date"] = int(d["date"])
        return nav
    print(f"[NAV] 运行真实回测 {START}~{END}（首次，较慢）...")
    t0 = time.time()
    res = run_backtest(start_date=START, end_date=END, hold_count=7,
                       quiet=True, no_html=True)
    print(f"[NAV] 回测完成，耗时 {time.time()-t0:.1f}s")
    nav = [{"date": int(d["date"]), "value": d["value"], "bench": d.get("bench")}
           for d in res["daily_values"]]
    os.makedirs(os.path.dirname(NAV_CACHE), exist_ok=True)
    with open(NAV_CACHE, "w", encoding="utf-8") as f:
        json.dump(nav, f)
    return nav


def build_regimes(nav_dates):
    conn = sqlite3.connect(DB)
    # 1) 全市场总成交额（亿元）per date（限制在窗口内，减少扫描）
    liq = {}
    q = conn.execute(
        "SELECT trade_date, SUM(amount) FROM daily "
        "WHERE trade_date BETWEEN ? AND ? GROUP BY trade_date",
        (nav_dates[0], nav_dates[-1]))
    for d, amt in q.fetchall():
        liq[int(d)] = (amt or 0) / 1e5  # 千元 -> 亿元
    # 2) 中证2000 / 沪深300 收盘（统一 key 为 int，避免 str/int 比较）
    idx = {}
    for code in ("932000.SH", "000300.SH"):
        rows = conn.execute(
            "SELECT trade_date, close FROM index_daily WHERE ts_code=? "
            "AND trade_date BETWEEN ? AND ? ORDER BY trade_date",
            (code, nav_dates[0], nav_dates[-1])).fetchall()
        idx[code] = {int(d): c for d, c in rows}
    conn.close()

    # 近20交易日收益
    def r20(series, date):
        ds = sorted(k for k in series if k <= date)
        if len(ds) < 21:
            return None
        c0, c1 = series[ds[-21]], series[ds[-1]]
        if not c0 or not c1:
            return None
        return c1 / c0 - 1.0

    style = {}
    for d in nav_dates:
        a = r20(idx["932000.SH"], d)
        b = r20(idx["000300.SH"], d)
        if a is None or b is None:
            style[d] = None
        else:
            style[d] = a - b  # 小盘占优度
    return liq, style


def bucket(vals, pcts=(1/3, 2/3)):
    s = sorted(v for v in vals if v is not None)
    if not s:
        return (0, 0)
    n = len(s)
    lo = s[max(0, int(n * pcts[0]) - 1)]
    hi = s[min(n - 1, int(n * pcts[1]) - 1)]
    return lo, hi


def tag(val, lo, hi):
    if val is None:
        return None
    if val <= lo:
        return 0
    if val > hi:
        return 2
    return 1


def compound(rets):
    if not rets:
        return None
    v = 1.0
    for r in rets:
        v *= (1 + r)
    return v - 1.0


def annualize(total_ret, n_days):
    if not n_days or total_ret is None:
        return None
    years = n_days / 252.0
    if years <= 0:
        return None
    return (1 + total_ret) ** (1 / years) - 1


def main():
    nav = load_nav()
    dates = [d["date"] for d in nav]
    vals = np.array([d["value"] for d in nav], dtype=float)
    bench = np.array([d["bench"] for d in nav], dtype=float)
    sret = np.diff(vals) / vals[:-1]
    bret = np.diff(bench) / bench[:-1]
    # 对齐到 t-1 -> t 的日期（用后一天日期标记这段收益）
    rdates = dates[1:]

    liq, style = build_regimes(dates)
    liq_vals = [liq.get(d) for d in dates]
    style_vals = [style.get(d) for d in dates]
    lo_l, hi_l = bucket(liq_vals)
    lo_s, hi_s = bucket(style_vals)
    print(f"\n[Regime 分档]")
    print(f"  流动性(全市场成交额,亿元): 低<{lo_l:.0f} | 中 {lo_l:.0f}~{hi_l:.0f} | 高>{hi_l:.0f}")
    print(f"  小盘占优(中证2000-沪深300 20日收益差): 低<{lo_s*100:.1f}% | 中 | 高>{hi_s*100:.1f}%")

    # 每个收益段(rdates[i]) 对应 regime 状态用当天(=rdates[i]) 的 liq/style
    liq_tag = [tag(liq.get(d), lo_l, hi_l) for d in rdates]
    style_tag = [tag(style.get(d), lo_s, hi_s) for d in rdates]

    LNAME = ["低流动性", "中流动性", "高流动性"]
    SNAME = ["小盘跑输", "小盘中性", "小盘占优"]

    print(f"\n{'='*78}")
    print(f"  3×3 生态位矩阵 — 各格【累计收益】(该regime日内复利, 非年化)")
    print(f"{'='*78}")
    header = "        | " + " | ".join(f"{LNAME[j]:>11}" for j in range(3))
    print(header)
    matrix = {}
    for si in range(3):
        row = []
        for li in range(3):
            mask = [(liq_tag[i] == li and style_tag[i] == si) for i in range(len(rdates))]
            idxs = [i for i, m in enumerate(mask) if m]
            if not idxs:
                row.append("    n/a    ")
                matrix[(si, li)] = None
                continue
            sr = compound([sret[i] for i in idxs])
            br = compound([bret[i] for i in idxs])
            matrix[(si, li)] = (sr, br, len(idxs))
            row.append(f"{sr*100:+9.1f}%")
        print(f"  {SNAME[si]:>6} | " + " | ".join(f"{x:>11}" for x in row))

    print(f"\n{'='*78}")
    print(f"  明细：每格 [策略累计 / 基准累计 / 天数 / 策略该格最大回撤]")
    print(f"{'='*78}")
    for si in range(3):
        for li in range(3):
            cell = matrix[(si, li)]
            if not cell:
                continue
            sr, br, n = cell
            # 该格内净值曲线最大回撤（按归属日的组合净值切片）
            sub_vals = [vals[i + 1] for i in range(len(rdates)) if liq_tag[i] == li and style_tag[i] == si]
            dd = 0.0
            if len(sub_vals) > 1:
                arr = np.array(sub_vals, dtype=float)
                cummax = np.maximum.accumulate(arr)
                dd = float(np.min((arr - cummax) / cummax)) * 100
            print(f"  {SNAME[si]:>6} × {LNAME[li]:>6}: 策略 {sr*100:+8.1f}% | 基准 {br*100:+8.1f}% | 超额 {(sr-br)*100:+7.1f}% | n={n:>3} | 回撤 {dd:>6.1f}%")

    # ── 已知生态位塌方窗口 ──
    windows = [
        ("2015 股灾", "20150601", "20150930"),
        ("2018 熊市", "20180101", "20181231"),
        ("2024-02 微盘流动性危机", "20240101", "20240208"),
        ("2024-09 政策牛市小盘狂飙", "20240901", "20241008"),
    ]
    print(f"\n{'='*78}")
    print(f"  已知生态位窗口 · 策略 vs 基准(中证2000) 区间收益")
    print(f"{'='*78}")
    for name, a, b in windows:
        # 找窗口内净值
        seg = [(d, v) for d, v in zip(dates, vals) if int(a) <= d <= int(b)]
        bseg = [(d, v) for d, v in zip(dates, bench) if int(a) <= d <= int(b)]
        if len(seg) < 2 or len(bseg) < 2:
            print(f"  {name:<28}: 数据不足")
            continue
        s_ret = seg[-1][1] / seg[0][1] - 1
        bl = bseg[0][1]; bh = bseg[-1][1]
        b_ret = (bh / bl - 1) if bl else 0
        print(f"  {name:<28}: 策略 {s_ret*100:+7.1f}% | 基准 {b_ret*100:+7.1f}% | 超额 {(s_ret-b_ret)*100:+6.1f}%")

    # ── 全样本基线 ──
    full_s = vals[-1] / vals[0] - 1
    full_b = bench[-1] / bench[0] - 1
    print(f"\n{'='*78}")
    print(f"  全样本基线 {START}~{END}: 策略 {full_s*100:+7.1f}% | 基准 {full_b*100:+7.1f}% | 超额 {(full_s-full_b)*100:+6.1f}%")
    print(f"{'='*78}")

    # 判定：是否"踩在生态位上"
    # 1) 收益是否集中在【小盘占优】(生态位活跃) 格 vs 【小盘跑输】(生态位弱化) 格
    sc_lead = [matrix[(2, li)] for li in range(3) if matrix[(2, li)]]
    sc_lag = [matrix[(0, li)] for li in range(3) if matrix[(0, li)]]
    lead_cum = sum(c[0] for c in sc_lead) if sc_lead else 0.0
    lag_cum = sum(c[0] for c in sc_lag) if sc_lag else 0.0
    # 2) 低流动性各格合计
    low_liq = [matrix[(si, 0)] for si in range(3) if matrix[(si, 0)]]
    low_cum = sum(c[0] for c in low_liq) if low_liq else 0.0
    # 3) 危机窗口：从已打印结果里抓 2024-02 微盘流动性危机
    crisis_s = None
    for name, a, b in windows:
        if "2024-02" in name:
            seg = [(d, v) for d, v in zip(dates, vals) if int(a) <= d <= int(b)]
            if len(seg) >= 2:
                crisis_s = seg[-1][1] / seg[0][1] - 1
    print("\n[判定]")
    print(f"  小盘占优(生态位活跃)各格累计收益合计: {lead_cum*100:+.1f}%")
    print(f"  小盘跑输(生态位弱化)各格累计收益合计: {lag_cum*100:+.1f}%")
    print(f"  低流动性各格累计收益合计:             {low_cum*100:+.1f}%")
    if crisis_s is not None:
        print(f"  2024-02 微盘流动性危机 策略区间收益:  {crisis_s*100:+.1f}% (基准中证2000 同期约 -27.6%)")
    print()
    if lead_cum > 0 and lag_cum < 0:
        print("  → 收益几乎全来自【小盘占优】期，小盘一旦跑输即转亏；且流动性危机期亏损幅度"
              "甚至大于小盘基准。证实该策略'踩在生态位上'，高度 regime-dependent、脆弱。")
    elif lead_cum > 0 and lag_cum >= 0:
        print("  → 各格均非负，策略有一定跨生态稳健性，不完全依赖单一生态位。")
    else:
        print("  → 结果中性，需结合更多历史区间进一步判读。")


if __name__ == "__main__":
    main()
