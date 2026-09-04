# -*- coding: utf-8 -*-
"""16 个独立回测脚本的 NAV 取价点精确分诊

对每个脚本找出所有"取价点"，按上下文分类：
  NAV  —— 持仓估值 / 成交价 / 组合总值（这些用 raw 就是 bug）
  SIG  —— 涨跌幅 / 动量 / 均线 / 因子（与 NAV 无关，改了会引入新 bug）
  IDX  —— 指数基准
  ?    —— 需人工判读

同时报告：是否 import 主干引擎（是 → 已自动获得 --price-mode hfq 能力）
"""
import os
import re
import sys

FILES = """ab_weight chan_lun_validate daily20_macd darvas darvas_gate ep_neutral
etf_rotation_v6_merged industry_sentiment_rotation limitup_reversion magic_formula
multifactor peg regime_gate_ab shrink_pullback value_backtest macd_timing""".split()

# 取价点模式
PAT_PRICE = re.compile(
    r"(get_price|get_open_price|get_hfq_price|get_close|load_price)\s*\(|"
    r"\[[\"'](close|open|pre_close|high|low)[\"']\]|"
    r"\.(close|open|pre_close)\b(?!\s*\(\s*\))|"
    r"\b(?:row|r|g|df|d|self)\.(close|open)\b|"
    r"SELECT[\s\S]{0,120}?\b(close|open)\b[\s\S]{0,80}?FROM\s+daily|"
    r"FROM\s+daily[\s\S]{0,200}?\b(close|open)\b", re.I)

# NAV 语义关键词（命中 → 判定为净值/成交路径）
NAV_KW = re.compile(
    r"持仓|市值|估值|总资产|组合价值|净值|portfolio|position_value|market_value|"
    r"total_value|equity|nav|shares\s*\*|buy_price|sell_price|成交|exec|"
    r"cost|proceeds|cash|资金|买入价|卖出价|avg_cost|cost_basis", re.I)

# 信号语义关键词
SIG_KW = re.compile(
    r"涨跌幅|动量|momentum|ret\b|return|pct|ma\d|均线|atr|rsi|macd|"
    r"布林|bolling|zscore|信号|signal|因子|factor|ic\b|rank|volatility|波动", re.I)

IDX_KW = re.compile(r"index_daily|benchmark|基准|指数|000300|000906|399006|bench", re.I)


def classify(lines, i, window=6):
    """取第 i 行前后 window 行的上下文做分类"""
    lo, hi = max(0, i - window), min(len(lines), i + window + 1)
    ctx = "\n".join(lines[lo:hi])
    if IDX_KW.search(ctx) and not NAV_KW.search(ctx):
        return "IDX"
    if NAV_KW.search(ctx):
        return "NAV"
    if SIG_KW.search(ctx):
        return "SIG"
    return "?"


def main():
    rows = []
    for name in FILES:
        path = f"run_{name}.py"
        if not os.path.exists(path):
            rows.append((name, "MISSING", 0, 0, 0, 0, 0, 0))
            continue
        src = open(path, encoding="utf-8", errors="replace").read()
        lines = src.split("\n")

        imports_engine = bool(re.search(r"^\s*(import|from)\s+run_monthly_rebalance", src, re.M))
        uses_engine_price = bool(re.search(
            r"(?:^|\.)m\.get_(?:price|open_price)|\bfrom\s+run_monthly_rebalance\s+import[\s\S]{0,200}?get_(?:price|open_price)", src))
        has_adj = "adj_factor" in src

        tally = {"NAV": 0, "SIG": 0, "IDX": 0, "?": 0}
        sites = []
        for i, ln in enumerate(lines):
            if PAT_PRICE.search(ln):
                c = classify(lines, i)
                tally[c] += 1
                sites.append((i + 1, c, ln.strip()[:100]))

        rows.append((name,
                     "engine" if uses_engine_price else ("import" if imports_engine else "standalone"),
                     1 if has_adj else 0,
                     tally["NAV"], tally["SIG"], tally["IDX"], tally["?"], len(sites)))

        # 保存 NAV 判定点的明细，供人工复核
        nav_sites = [s for s in sites if s[1] in ("NAV", "?")]
        if nav_sites:
            with open(f"_triage_{name}.txt", "w", encoding="utf-8") as fh:
                fh.write(f"# {path}  NAV/? 取价点 {len(nav_sites)} 处\n")
                fh.write(f"# import_engine={imports_engine}  adj_factor={has_adj}\n\n")
                for ln_no, c, txt in nav_sites:
                    fh.write(f"L{ln_no:<5} [{c}] {txt}\n")

    print("=" * 108)
    print(f"{'脚本':<34}{'取价来源':<12}{'adj':<5}{'NAV':>5}{'SIG':>5}{'IDX':>5}{'待判':>5}{'合计':>6}")
    print("=" * 108)
    for r in rows:
        if r[1] == "MISSING":
            print(f"{'run_'+r[0]+'.py':<34}⚠️ 文件不存在")
            continue
        print(f"{'run_'+r[0]+'.py':<34}{r[1]:<12}{('有' if r[2] else '无'):<5}"
              f"{r[3]:>5}{r[4]:>5}{r[5]:>5}{r[6]:>5}{r[7]:>6}")
    print("=" * 108)
    print()
    print("取价来源: engine=经主干引擎取价(已获 hfq 能力) | import=import 引擎但可能自己查 daily | standalone=完全独立")
    print("NAV=? 的点已写入 _triage_<name>.txt 供人工复核")


if __name__ == "__main__":
    main()
