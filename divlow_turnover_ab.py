"""B5 换手率分析：从 official_compact 各 buffer_k 档的 partial sel_log 计算单边换手。
口径：每期单边换手 = 新进入(或退出)只数 / 持仓数；年化 = 各期均值 × 期数/年。
输出各 k 档：年均换手、逐期换手、平均保留带利用率。"""
import glob
import os
import sys

import pandas as pd

RES = "data/results/dividend_low_vol"


def turnover_for(pattern: str, top_n: int = 12) -> pd.DataFrame:
    """换手双口径（🔴 只数口径与资金口径可差 2 倍，必须与基线同口径才可比）：
       one_way_n = (持仓数 - 保留数)/持仓数          只数口径
       one_way_w = 0.5 * Σ|w_t - w_{t-1}|           资金口径（含权重漂移，标准单边换手）"""
    files = sorted(glob.glob(os.path.join(RES, pattern)))
    # 🔴 排除混入的其它组合：查季度档(bk{k})时，bk0_* 也会 glob 到 bk0_rbyear_*（频率档）
    if "_rb" not in pattern:
        files = [f for f in files if "_rb" not in os.path.basename(f)]
    if not files:
        print(f"  [MISS] 无 partial: {pattern}")
        return pd.DataFrame()
    df = pd.read_csv(files[-1], dtype={"rebal_date": str, "ts_code": str}, encoding="utf-8-sig")
    rbs = sorted(df["rebal_date"].unique())
    rows = []
    prev_n = None          # 只数口径：上一期代码集合
    prev_w = None          # 资金口径：上一期 {code: weight}
    for rb in rbs:
        sub = df[df["rebal_date"] == rb]
        cur_n = set(sub["ts_code"])
        cur_w = dict(zip(sub["ts_code"], sub["weight"].astype(float)))
        if prev_n is not None and prev_w is not None:
            kept = len(cur_n & prev_n)
            one_way_n = (top_n - kept) / top_n
            codes = set(cur_w) | set(prev_w)
            one_way_w = 0.5 * sum(abs(cur_w.get(c, 0.0) - prev_w.get(c, 0.0)) for c in codes)
            rows.append({"rebal_date": rb, "kept": kept,
                         "one_way_n": one_way_n, "one_way_w": one_way_w})
        prev_n, prev_w = cur_n, cur_w
    t = pd.DataFrame(rows)
    yr_span = (int(rbs[-1][:4]) - int(rbs[0][:4])) + (int(rbs[-1][4:6]) - int(rbs[0][4:6])) / 12
    n_per_year = len(t) / yr_span if yr_span > 0 else float("nan")
    print(f"\n== {pattern}  期数={len(t)} ({rbs[0]}~{rbs[-1]}) ==")
    print(f"  【资金口径】年化单边换手 = {t['one_way_w'].mean() * n_per_year * 100:.1f}%"
          f"   (每期 {t['one_way_w'].mean() * 100:.1f}% × {n_per_year:.2f} 期/年)")
    print(f"  【只数口径】年化单边换手 = {t['one_way_n'].mean() * n_per_year * 100:.1f}%"
          f"   (每期 {t['one_way_n'].mean() * 100:.1f}% × {n_per_year:.2f} 期/年)")
    print(f"  每期保留: min={int(t['kept'].min())} 中位={int(t['kept'].median())} max={int(t['kept'].max())} / {top_n}")
    print(t.assign(one_way_n=(t["one_way_n"] * 100).round(1),
                   one_way_w=(t["one_way_w"] * 100).round(1)).to_string(index=False))
    return t


if __name__ == "__main__":
    # 参数：纯数字=季度档 buffer_k；year/half/month=频率档（buffer_k=0）
    # 例：divlow_turnover_ab.py 0 6 12 18 36 year half
    FREQS = {"month", "quarter", "half", "year"}
    args = sys.argv[1:] or ["0", "6", "12", "18", "36"]
    for a in args:
        if a in FREQS:
            # quarter 档文件名不带 _rb 后缀
            mid = "bk0" if a == "quarter" else f"bk0_rb{a}"
        else:
            mid = f"bk{a}"
        turnover_for(f"_official_official_compact_all_12_{mid}_*_partial.csv")
