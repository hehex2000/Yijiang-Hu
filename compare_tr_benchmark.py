#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
口径修正前后对照: 策略超额收益 在"价格指数基准" vs "全收益指数基准"下的差异
==========================================================================
平台历史: 策略净值用 hfq(含分红再投), 基准用 index_daily 价格指数(不含分红)
        → 超额收益被系统性高估。本脚本量化"高估了多少", 判断结论是否翻转。

两个必须同时对齐的口径(少对齐一个都会得出错误结论):
  ① 基准侧: 价格指数 → 全收益指数      (影响"虚高" = 旧超额 − 新超额)
  ② NAV  侧: raw(漏分红) → hfq(含分红)  (影响"新口径超额" 本身, 不影响"虚高")
    ⚠️ 虚高 ≈ TR年化 − 价格年化, 与策略 NAV 无关;
       但"新口径超额"高度依赖 NAV —— 用 raw NAV 比全收益基准是
       「策略漏分红 + 基准含分红」双重惩罚, 真超额被系统性低估。
       实测: 红利低波质量复合 2020-2026 低估 3.67pp/年。

用法:
  ./venv_ml/Scripts/python.exe compare_tr_benchmark.py
"""
import sqlite3, os
import pandas as pd
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
DB = r"D:\tu-shareData\astock_daily.db"
OUT = os.path.join(BASE, "data", "results", "tr_index")

# (显示名, NAV文件, 日期列, 策略列, 旧基准列(价格指数), 基准指数代码)
# ⚠️ 策略列必须取 hfq(含分红再投) 那一列/那一份文件。
#    若用 raw NAV 去比全收益基准 = 「策略漏分红 + 基准含分红」双重惩罚,
#    真超额被系统性低估(实测 红利低波质量复合 低估 3.67pp/年)。
#    下方 main() 会自动告警: 存在 _hfq 兄弟文件却仍指向 raw 文件的 case。
CASES = [
    ("红利低波质量复合 2020-2026 [NAV=hfq]",
     "data/results/dividend_low_vol/bt_quality_nav_20200101_20260723_official_compact_hs300_12_hfq.csv",
     "trade_date", "nav_official_compact", "nav_000300.SH", "000300.SH"),
    ("红利低波质量复合 2020-2026 [NAV=raw 旧·仅对照]",
     "data/results/dividend_low_vol/bt_quality_nav_20200101_20260723_official_compact_hs300_12.csv",
     "trade_date", "nav_official_compact", "nav_000300.SH", "000300.SH"),
    ("高股息+基本面成长 2014-2026 [NAV=hfq]",
     "data/results/dividend_growth/nav_20140101_20260720.csv",
     "date", "value_hfq", "bench300", "000300.SH"),
]


def ann(total, ndays):
    yrs = ndays / 252.0
    return (1 + total) ** (1 / yrs) - 1 if yrs > 0 else 0.0


def load_tr(code):
    """优先用 Tushare 官方全收益指数(权威), 无则回退自建版本。
    排除 CNY020(净收益/扣税) 系列——基准口径用税前全收益。"""
    con = sqlite3.connect(DB)
    df = pd.read_sql_query(
        "SELECT trade_date, close AS idx_tr FROM index_tr_official "
        "WHERE local_code=? AND tr_code NOT LIKE '%CNY020%' ORDER BY trade_date",
        con, params=(code,))
    src = "官方"
    if df.empty:
        df = pd.read_sql_query(
            "SELECT trade_date, idx_tr FROM index_total_return "
            "WHERE index_code=? ORDER BY trade_date", con, params=(code,))
        src = "自建"
    con.close()
    return df, src


def dividend_tax_drag():
    """用 中证红利全收益 vs 净收益 量化红利税拖累(视频未扣税的那个坑)"""
    con = sqlite3.connect(DB)
    df = pd.read_sql_query(
        "SELECT trade_date, close, tr_code FROM index_tr_official "
        "WHERE local_code='000922.SH' ORDER BY trade_date", con)
    con.close()
    if df.empty:
        return
    g = df.pivot_table(index="trade_date", columns="tr_code", values="close")
    g = g.dropna()
    if g.shape[1] < 2:
        return
    tr_col = [c for c in g.columns if "CNY020" not in c][0]
    net_col = [c for c in g.columns if "CNY020" in c][0]
    yrs = len(g) / 252.0
    a_tr = (g[tr_col].iloc[-1] / g[tr_col].iloc[0]) ** (1 / yrs) - 1
    a_net = (g[net_col].iloc[-1] / g[net_col].iloc[0]) ** (1 / yrs) - 1
    print("\n" + "=" * 92)
    print("附: 红利税拖累 (中证红利 全收益 vs 净收益, 官方口径)")
    print("=" * 92)
    print(f"  区间 {g.index[0]} ~ {g.index[-1]}  ({yrs:.1f}年)")
    print(f"  全收益(税前) 年化 : {a_tr*100:+.2f}%")
    print(f"  净收益(扣税) 年化 : {a_net*100:+.2f}%")
    print(f"  ★ 红利税拖累     : {(a_tr-a_net)*100:+.2f}%/年")
    print("  (视频用指数全收益口径且声明未扣红利税; 个人实际到手更接近净收益)")


def check_hfq_sibling():
    """告警: 若存在 _hfq 兄弟文件但 case 仍指向 raw 文件, 说明真超额会被低估。
    (除非该 case 名字里已显式标注 raw·仅对照)"""
    bad = []
    for name, path, dcol, scol, bcol, icode in CASES:
        p = os.path.join(BASE, path)
        root, ext = os.path.splitext(p)
        if "_hfq" in root:
            continue
        sib = root + "_hfq" + ext
        if os.path.exists(sib) and "raw" not in name and "对照" not in name:
            bad.append((name, os.path.basename(p), os.path.basename(sib)))
    if bad:
        print("\n[告警] 以下 case 存在 hfq 版 NAV 却仍指向 raw 文件 "
              "(真超额将被双重惩罚低估):")
        for n, a, b in bad:
            print(f"   - {n}\n     现读 {a}\n     应读 {b}")


def main():
    rows = []
    print("=" * 92)
    print("口径修正前后对照: 价格指数基准(旧) vs 全收益指数基准(新)")
    print("=" * 92)
    check_hfq_sibling()

    for name, path, dcol, scol, bcol, icode in CASES:
        p = os.path.join(BASE, path)
        if not os.path.exists(p):
            print(f"[跳过] 不存在: {p}")
            continue
        nav = pd.read_csv(p, encoding="utf-8-sig")
        nav[dcol] = nav[dcol].astype(str)
        nav = nav[[dcol, scol, bcol]].dropna().sort_values(dcol).reset_index(drop=True)
        tr, tr_src = load_tr(icode)
        if tr.empty:
            print(f"[跳过] 无全收益基准 {icode}, 请先跑 build_tr_index.py --index {icode} "
                  f"或 download_tr_index.py")
            continue
        m = nav.merge(tr, left_on=dcol, right_on="trade_date", how="inner")
        if len(m) < 30:
            print(f"[跳过] {name} 与全收益基准重叠交易日仅 {len(m)} 天")
            continue

        s0, s1 = float(m[scol].iloc[0]), float(m[scol].iloc[-1])
        b0, b1 = float(m[bcol].iloc[0]), float(m[bcol].iloc[-1])
        t0, t1 = float(m["idx_tr"].iloc[0]), float(m["idx_tr"].iloc[-1])
        n = len(m)

        r_strat = s1 / s0 - 1
        r_old = b1 / b0 - 1          # 旧基准: 价格指数
        r_new = t1 / t0 - 1          # 新基准: 全收益指数
        a_strat = ann(r_strat, n)
        a_old = ann(r_old, n)
        a_new = ann(r_new, n)
        ex_old = ann((1 + r_strat) / (1 + r_old) - 1, n)
        ex_new = ann((1 + r_strat) / (1 + r_new) - 1, n)

        print(f"\n── {name} ──")
        print(f"  区间 {m[dcol].iloc[0]} ~ {m[dcol].iloc[-1]}  ({n} 个交易日, {n/252:.1f}年)")
        print(f"  策略      : 总 {r_strat*100:+.1f}%   年化 {a_strat*100:+.2f}%")
        print(f"  旧基准(价格指数 {icode}) : 总 {r_old*100:+.1f}%   年化 {a_old*100:+.2f}%")
        print(f"  新基准(全收益   {icode}, {tr_src}) : 总 {r_new*100:+.1f}%   年化 {a_new*100:+.2f}%")
        print(f"  ── 超额收益 ──")
        print(f"  旧口径超额 : {ex_old*100:+.2f}%/年")
        print(f"  新口径超额 : {ex_new*100:+.2f}%/年")
        print(f"  ★ 虚高     : {(ex_old-ex_new)*100:+.2f}pp/年   ({(1+ex_old)**(n/252)/(1+ex_new)**(n/252)*100-100:+.1f}% 累计)")
        flag = ""
        if ex_old > 0 and ex_new < 0:
            flag = "  ⚠️ 结论翻转: 旧口径显示跑赢, 新口径实为跑输"
        elif ex_old > 0 and ex_new > 0 and (ex_old - ex_new) > 0.01:
            flag = "  ⚠️ 超额显著缩水(原超额大部分来自基准口径缺陷)"
        print(flag if flag else "  (方向未变)")

        rows.append(dict(策略=name, 交易日=n, 年数=round(n / 252, 1),
                         策略年化=a_strat, 旧基准年化=a_old, 新基准年化=a_new,
                         旧口径超额=ex_old, 新口径超额=ex_new,
                         虚高pp=ex_old - ex_new,
                         结论翻转="是" if (ex_old > 0 and ex_new < 0) else "否"))

    if rows:
        out = pd.DataFrame(rows)
        p = os.path.join(OUT, "benchmark_restatement.csv")
        out.to_csv(p, index=False, encoding="utf-8-sig")
        print("\n" + "=" * 92)
        print("汇总")
        print("=" * 92)
        print(out.to_string(index=False, formatters={
            "策略年化": lambda x: f"{x*100:+.2f}%", "旧基准年化": lambda x: f"{x*100:+.2f}%",
            "新基准年化": lambda x: f"{x*100:+.2f}%", "旧口径超额": lambda x: f"{x*100:+.2f}%",
            "新口径超额": lambda x: f"{x*100:+.2f}%", "虚高pp": lambda x: f"{x*100:+.2f}pp"}))
        print(f"\n已保存: {p}")
    dividend_tax_drag()


if __name__ == "__main__":
    main()
