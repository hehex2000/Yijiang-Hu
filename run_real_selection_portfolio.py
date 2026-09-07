# -*- coding: utf-8 -*-
"""
真实选股 × 组合级引擎 对照
=====================================================================

背景：此前所有小资金实验（run_small_capital_scan.py）用的都是**随机前 N 只**
（按代码排序截断），而平台实际是**多因子选股**选出的 N 只。用户指出两者不同，
故重跑：用真实选股结果验证，并隔离「选股本身有没有增量」。

四组 2×2 设计：
  选股方式：real（多因子选股 top5） / rand（同池同日按代码排序前 5）
  资金架构：per_stock（默认，资金 1/N 均分） / portfolio（共享资金池，f=auto 2/N）

用法:  venv_ml/Scripts/python.exe run_real_selection_portfolio.py
"""
import sqlite3
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import config  # noqa: E402
import run_backtest as rb  # noqa: E402

START, END = config.GLOBAL["backtest_start"], config.GLOBAL["backtest_end"]
TOP_N = config.GLOBAL["top_n"]
OUT_CSV = Path("data/results/position_sizing/real_selection_ab.csv")


def _codes_from_pool(sel_date: str, n: int):
    """同池同日的随机对照：按代码排序取前 n 只（与 run_small_capital_scan 同口径）。"""
    try:
        pool = rb._get_zz800_from_db(sel_date)
        return sorted(pool["code"].tolist())[:n]
    except Exception as e:  # noqa: BLE001
        print(f"  [WARN] 取池失败: {e}")
        return []


def run_group(tag: str, codes: list, shared: bool):
    """走平台主流程跑一组，返回打印摘要用的 dict。"""
    config.STRATEGIES["mean_reversion"]["portfolio_shared_pool"] = shared
    stocks = pd.DataFrame({"code": codes, "name": [""] * len(codes)})
    print("\n" + "-" * 88)
    print(f"  [{tag}] {'组合级共享池' if shared else '逐票独立(默认)'}｜{len(codes)} 只：{','.join(codes)}")
    print("-" * 88)
    rb.run_backtest(stocks)
    return {"group": tag, "shared": shared, "codes": ",".join(codes)}


def main():
    print("=" * 88)
    print("真实选股 × 组合级引擎 对照")
    print("=" * 88)
    print(f"  区间 {START}~{END}｜池 {config.GLOBAL['stock_pool']}｜top_n={TOP_N}"
          f"｜总资金 {config.BACKTEST['total_capital']:,}")

    # ── 1. 真实多因子选股（慢：zz800 全池因子计算）──
    print("\n[1/3] 执行真实多因子选股 ...")
    sel = rb.run_selection()
    if sel is None or sel.empty:
        print("  [FAIL] 选股返回空，终止")
        return 1
    sel_date = config.SELECTION["date"]
    real_codes = sel["code"].tolist()[:TOP_N]
    cand_codes = sel["code"].tolist()
    print(f"\n  选股日 {sel_date}｜候选 {len(cand_codes)} 只｜取前 {TOP_N} 只：{real_codes}")
    names = dict(zip(sel["code"], sel.get("name", [""] * len(sel))))
    real_named = [f"{c}({names.get(c, '')})" for c in real_codes]
    print(f"  明细：{real_named}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    sel.head(20).to_csv(OUT_CSV.parent / "real_selection_top20.csv",
                        index=False, encoding="utf-8-sig")

    # ── 2. 随机对照组 ──
    print("\n[2/3] 构造随机对照（同池同日，按代码排序前 N 只）...")
    rand_codes = _codes_from_pool(sel_date.replace("-", ""), TOP_N)
    print(f"  随机 {len(rand_codes)} 只：{rand_codes}")

    # 只开 mean_reversion，避免跑全策略矩阵
    for k in config.STRATEGIES:
        config.STRATEGIES[k]["enabled"] = (k == "mean_reversion")
    config.SELECTION["top_n"] = TOP_N
    config.BACKTEST["start_date"] = START
    config.BACKTEST["end_date"] = END

    # ── 3. 四组 A/B ──
    print("\n[3/3] 四组对照（每组走平台主流程 run_backtest）")
    rows = []
    rows.append(run_group("A 真实选股 × 逐票(默认)", real_codes, False))
    rows.append(run_group("B 真实选股 × 组合级", real_codes, True))
    rows.append(run_group("C 随机前N × 逐票(默认)", rand_codes, False))
    rows.append(run_group("D 随机前N × 组合级", rand_codes, True))
    if len(cand_codes) > TOP_N:
        # ⚠️ run_backtest 内部按 SELECTION["top_n"] 凑满即 break（递补设计），
        #    不同步改 top_n 的话传 15 只也只会加载前 5 只（曾因此得到与 B 组相同的数）
        config.SELECTION["top_n"] = len(cand_codes)
        rows.append(run_group(f"E 候选池{len(cand_codes)}只 × 组合级", cand_codes, True))
        config.SELECTION["top_n"] = TOP_N

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print("\n" + "=" * 88)
    print(f"  对照完成。真实选股明细 → {OUT_CSV.parent}/real_selection_top20.csv")
    print("  请从上面各组输出中抄录：总收益 / 年化 / 超额 /（组合级另有 暴露度·f·skip_lot）")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    sys.exit(main())
