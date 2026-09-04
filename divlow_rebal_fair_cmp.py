"""调仓档位公平对比：统一到**同一窗口起点**（最晚首期建仓日）再比。

为什么必须做：
  各档首次建仓日不同（quarter 2020-01-08 / half 2020-06-05 / year 2020-12-14）。
  divlow_nav_replay.py 只是各自截断到自己的首期 → **窗口长度不同**，
  年度档白捡了"错过 2020 年 1-11 月下跌"的好处（季度档那段时间 -7.29%），
  年化/夏普优势里混着**窗口选择偏差**，不能直接当降频/降换手 alpha。

做法：把所有档的 NAV 统一截断到 max(各档首期)=最晚建仓日，同窗口重算指标 + 换手。

用法（参数 = 档位，可带 :cap 表示换手硬上限百分数）：
  venv_ml/Scripts/python.exe divlow_rebal_fair_cmp.py                       # 默认 季度/半年/年度
  venv_ml/Scripts/python.exe divlow_rebal_fair_cmp.py quarter quarter:20     # 季度档 + 20% 硬上限
  venv_ml/Scripts/python.exe divlow_rebal_fair_cmp.py year year:20 year:10
"""
import os
import sys

import numpy as np
import pandas as pd

import run_dividend_low_vol_quality_bt as E

RES = E.RES_DIR
START, END, TOP_N = E.START, E.END, 12
E.PRICE_MODE = "hfq"

FREQS = {"quarter": "季度", "half": "半年", "year": "年度"}
SPECS = sys.argv[1:] or ["quarter", "half", "year"]


def parse(arg):
    """'year' / 'year:20' → (freq, cap_pct)"""
    if ":" in arg:
        f, c = arg.split(":", 1)
        if f not in FREQS:
            raise SystemExit(f"未知频率 {f!r}，可选 {list(FREQS)}")
        return f, int(c)
    if arg not in FREQS:
        raise SystemExit(f"未知频率 {arg!r}，可选 {list(FREQS)}；带硬上限写 'quarter:20'")
    return arg, 0


def mid(freq, cap):
    """档位 → partial 文件名中段（🔴 cap 与 rebal 都必须在名字里，否则静默串档）"""
    return "bk0" + (f"_tc{cap}" if cap else "") + (f"_rb{freq}" if freq != "quarter" else "")


def label(freq, cap):
    return FREQS[freq] + (f"+{cap}%上限" if cap else "") + ("(基线)" if freq == "quarter" and not cap else "")


def load(freq, cap):
    p = os.path.join(RES, f"_official_official_compact_all_{TOP_N}_{mid(freq, cap)}_{START}_{END}_partial.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p, dtype={"rebal_date": str, "ts_code": str}, encoding="utf-8-sig")
    rbs = sorted(df["rebal_date"].unique())
    targets, wmap = [], {}
    for rb in rbs:
        sub = df[df["rebal_date"] == rb]
        targets.append((rb, [str(c) for c in sub["ts_code"]]))
        wmap[rb] = {str(c): float(w) for c, w in zip(sub["ts_code"], sub["weight"])}
    return targets, wmap, rbs


def turn(targets, wmap):
    """换手三分口径：m=每期单边 / ann=年化单边(B&O、活跃税同口径) / two=每期双边(纪要口径)"""
    prev, per = None, []
    for rb, codes in targets:
        cur = wmap[rb]
        if prev is not None:
            per.append(0.5 * sum(abs(cur.get(c, 0.0) - prev.get(c, 0.0)) for c in set(cur) | set(prev)))
        prev = cur
    rbs = [r for r, _ in targets]
    yr = (int(rbs[-1][:4]) - int(rbs[0][:4])) + (int(rbs[-1][4:6]) - int(rbs[0][4:6])) / 12
    m = sum(per) / len(per)
    return m, m * len(per) / yr, m * 2, len(per)


data, order = {}, []
for a in SPECS:
    f, c = parse(a)
    r = load(f, c)
    if r is None:
        print(f"  [MISS] {a} 的 partial 不存在，跳过")
        continue
    targets, wmap, rbs = r
    codes = sorted({cc for _, cs in targets for cc in cs})
    pmap = E.bulk_close_prices(codes, START, END)
    E.EXEC_PMAP.clear()
    E.EXEC_PMAP.update(E.bulk_open_prices(codes, START, END))
    all_dates = E.get_trade_dates(START, END)
    nav = E.run_nav_weighted(targets, wmap, pmap, all_dates, coef_fn=None)
    data[a] = dict(nav=nav, rbs=rbs, turn=turn(targets, wmap), lbl=label(f, c))
    order.append(a)

if not data:
    raise SystemExit("无可用 partial")
if len(order) < 2:
    print("⚠️ 只有一档，无法做对比/配对检验")

# 统一窗口：以最晚的首期建仓日为准
D0 = max(d["rbs"][0] for d in data.values())
print("=" * 118)
print(f"【公平对比：统一窗口 {D0} 起】各档首期建仓日不同，不统一会把「错过下跌」误算成降换手 alpha")
print(f"{'档位':<18}{'首期':>10}{'期数':>6}{'总收益':>10}{'年化':>9}{'最大回撤':>11}"
      f"{'波动':>9}{'夏普':>7}{'卡玛':>7}{'年化单边':>11}{'每期双边':>11}")
for a in order:
    d = data[a]
    nav_t = [(x, v) for x, v in d["nav"] if x >= D0]
    m = E.compute_metrics(nav_t, [x for x, _ in nav_t])
    per, ann, two, n = d["turn"]
    print(f"{d['lbl']:<18}{d['rbs'][0]:>10}{n:>6}{m['total_ret']*100:>9.2f}%{m['ann']*100:>8.2f}%"
          f"{m['max_dd']*100:>10.2f}%{m['vol']*100:>8.2f}%{m['sharpe']:>7.2f}{m['calmar']:>7.2f}"
          f"{ann*100:>10.1f}%{two*100:>10.1f}%")
print("=" * 118)

# 各档自身窗口（对照，看窗口偏差有多大）
print("\n【各档自身窗口（含窗口偏差，勿直接跨档比年化）】")
print(f"{'档位':<18}{'窗口':>22}{'总收益':>10}{'年化':>9}{'夏普':>7}")
for a in order:
    d = data[a]
    nav_t = [(x, v) for x, v in d["nav"] if x >= d["rbs"][0]]
    m = E.compute_metrics(nav_t, [x for x, _ in nav_t])
    print(f"{d['lbl']:<18}{d['rbs'][0]+'~'+d['rbs'][-1]:>22}{m['total_ret']*100:>9.2f}%"
          f"{m['ann']*100:>8.2f}%{m['sharpe']:>7.2f}")

# 同窗口下的配对检验（基线 = 第一个档位）
if len(order) >= 2:
    base_key = order[0]
    print(f"\n【统一窗口 {D0} 起：各档 vs 「{data[base_key]['lbl']}」配对检验（日收益，Newey-West 未调整）】")
    srs = {}
    for a in order:
        nav_t = [(x, v) for x, v in data[a]["nav"] if x >= D0]
        srs[a] = pd.Series([v for _, v in nav_t], index=[x for x, _ in nav_t], dtype=float).pct_change()
    base = srs[base_key]
    print(f"{'档':<18}{'日数':>7}{'日收益差均值(bp)':>18}{'t 值':>9}{'判读':>16}")
    for a in order:
        if a == base_key:
            continue
        diff = (srs[a] - base).dropna()
        tt = diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else float("nan")
        print(f"{data[a]['lbl']:<18}{len(diff):>7}{diff.mean()*1e4:>18.2f}{tt:>9.2f}"
              f"{'显著(|t|>2)' if abs(tt) > 2 else '不显著→噪声':>16}")
    print("\n  🔴 日收益重叠自相关会放大 t 值，本表仅供方向参考；"
          "低频档独立决策次数只有 6~13 次，统计功效本就很低。")
