# -*- coding: utf-8 -*-
"""
Regime 识别信号验证（设计阶段，不改任何策略逻辑）v2
=================================================
对比三种合成规则，聚焦「用户亏钱窗口 2024-09~2026-08」是否触发 β 兜底：
  Rule A (AND)        : BULL = 趋势牛 AND 宽度>=thr        (对称, 宽度脆)
  Rule B (趋势为主)   : BULL = 趋势牛 AND 宽度>=0.25       (宽度仅否决极端背离)
  Rule C (仅趋势)     : BULL = 趋势牛                      (无宽度, 仅作参照)
验证目标:
  - 2022 崩盘年 应≈0% BULL (保留防御, 不破坏 +70% 超额)
  - 用户窗口 2024-09~2026-08 应高占比 BULL (触发 β 兜底治踏空)
"""
import sqlite3
import pandas as pd

DB = r'D:/tu-shareData/astock_daily.db'
SEG = ['000300.SH', '000905.SH', '000852.SH', '399006.SZ',
       '000688.SH', '000016.SH', '399001.SZ', '000985.SH']
START = pd.Timestamp(2020, 1, 1)
END = pd.Timestamp(2026, 8, 31)
USER_WIN = (pd.Timestamp(2024, 9, 1), pd.Timestamp(2026, 8, 31))
MA_LEN = 200


def load():
    con = sqlite3.connect(DB)
    frames = {}
    for c in SEG:
        df = pd.read_sql(
            f"SELECT trade_date,close FROM index_daily WHERE ts_code='{c}' ORDER BY trade_date", con)
        frames[c] = pd.Series(df['close'].values,
                              index=pd.to_datetime(df['trade_date'].astype(str)))
    con.close()
    return frames


def fifth_trading_days(dates):
    mdates = {}
    for d in dates:
        mdates.setdefault((d.year, d.month), []).append(d)
    return [v[4] for k, v in sorted(mdates.items()) if len(v) >= 5]


def ma_bull(series, d, n):
    s = series[:d]
    return bool(series[d] > s.iloc[-n:].mean()) if len(s) >= n else None


def seg_breadth(frames, d):
    cnt = tot = 0
    for c, ser in frames.items():
        s = ser[:d]
        if len(s) < 20:
            continue
        tot += 1
        if ser[d] > s.iloc[-20:].mean():
            cnt += 1
    return cnt / tot if tot else 0.0


def regimes(frames):
    hs = frames['000300.SH']
    dates = [d for d in hs.index if START <= d <= END]
    fb = fifth_trading_days(dates)
    out = []
    for d in fb:
        t = ma_bull(hs, d, MA_LEN)
        br = seg_breadth(frames, d)
        a = (t is True) and (br >= 0.50)
        b = (t is True) and (br >= 0.25)
        c = (t is True)
        out.append((d, t, round(br, 2), a, b, c))
    return out


def pct(rows, lo, hi, col):
    sub = [r for r in rows if lo <= r[0] <= hi]
    if not sub:
        return 0, 0
    return sum(1 for r in sub if r[col]) , len(sub)


def show(label, rows):
    print(f"\n### {label}")
    for lo, hi, name in [(pd.Timestamp(2022,1,1), pd.Timestamp(2022,12,31), '2022崩盘年'),
                         (USER_WIN[0], USER_WIN[1], '用户窗口24-09~26-08'),
                         (pd.Timestamp(2020,1,1), pd.Timestamp(2026,8,31), '全周期20-26')]:
        for col, cname in [(3,'RuleA(AND)'), (4,'RuleB(趋势为主)'), (5,'RuleC(仅趋势)')]:
            n, tot = pct(rows, lo, hi, col)
            print(f"  {name:<16} {cname:<14} BULL {n}/{tot} = {n/tot*100:4.0f}%")


if __name__ == '__main__':
    frames = load()
    rows = regimes(frames)
    print(f"MA_LEN={MA_LEN}  宽度=8大指数收盘>各自MA20 占比")
    show('三种合成规则对比', rows)
    # 季度抽样看 RuleB 行为
    print("\n-- 季度抽样 (RuleB: 趋势牛 AND 宽度>=0.25) --")
    for d, t, br, a, b, c in rows:
        if d.month in (1, 4, 7, 10):
            print(f"  {d.date()} 趋势牛={t} 宽度={br:.0%} -> {'BULL' if b else 'BEAR'}")
