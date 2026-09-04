# -*- coding: utf-8 -*-
"""第二层分诊：直接搜「持仓市值 / 成交金额」的充要表达式

NAV 的本质就是 `股数 × 价格`。不管价格从哪个函数来，只要出现这些模式，
该脚本就在算净值，且价格空间决定其是否含分红。

模式：
  shares * price / price * shares / pos['shares'] * px / ...
  portfolio_value / market_value / total_value / equity
  sell_amount / cost = / proceeds
"""
import os
import re

FILES = """ab_weight chan_lun_validate daily20_macd darvas darvas_gate ep_neutral
etf_rotation_v6_merged industry_sentiment_rotation limitup_reversion magic_formula
multifactor peg regime_gate_ab shrink_pullback value_backtest macd_timing""".split()

# 股数 × 价格（NAV 的充要表达式）
MUL = re.compile(
    r"(shares|数量|股数|volume|qty|amount_shares)\s*\*\s*"
    r"([A-Za-z_][\w\.\[\]'\"]*)|"
    r"([A-Za-z_][\w\.\[\]'\"]*)\s*\*\s*(shares|数量|股数)", re.I)

# 组合市值变量
VAL = re.compile(
    r"(portfolio_value|market_value|total_value|equity|nav|total_asset|"
    r"sell_amount|proceeds|pos_value|hold_value|持仓市值|组合价值|总资产)", re.I)

# 自定义价格取数函数（分诊器 v1 漏掉的）
DEFP = re.compile(r"def\s+(get_[\w]*price[\w]*|_?get_px[\w]*|[\w]*_price)\s*\(", re.I)

# 价格来源
SRC_ENGINE = re.compile(r"(?:m\.|mr\.|run_monthly_rebalance\.)?get_(?:open_)?price\s*\(", re.I)
SRC_SQL = re.compile(r"SELECT[\s\S]{0,200}?\b(close|open)\b[\s\S]{0,120}?FROM\s+daily", re.I | re.S)
SRC_ADJ = re.compile(r"adj_factor|hfq|后复权", re.I)


def main():
    print("=" * 116)
    print(f"{'脚本':<34}{'股数×价':>8}{'市值变量':>9}{'自定义取价函数':>16}{'引擎取价':>9}{'SQL取价':>8}{'adj':>5}")
    print("=" * 116)
    summary = {}
    for name in FILES:
        path = f"run_{name}.py"
        if not os.path.exists(path):
            print(f"{path:<34}  ⚠️ 缺失")
            continue
        src = open(path, encoding="utf-8", errors="replace").read()
        lines = src.split("\n")

        mul = len(MUL.findall(src))
        val = len(VAL.findall(src))
        defp = DEFP.findall(src)
        eng = len(SRC_ENGINE.findall(src))
        sql = len(SRC_SQL.findall(src))
        adj = len(SRC_ADJ.findall(src))

        summary[name] = dict(mul=mul, val=val, defp=defp, eng=eng, sql=sql, adj=adj)
        d = ",".join(sorted(set(defp)))[:15] if defp else "-"
        print(f"{path:<34}{mul:>8}{val:>9}{d:>16}{eng:>9}{sql:>8}{adj:>5}")

    print("=" * 116)
    print()
    print("判定：")
    for name, s in summary.items():
        if s["mul"] == 0 and s["val"] == 0:
            verdict = "❌ 不算 NAV（纯信号/因子分析，无需修）"
        elif s["adj"] > 0:
            verdict = "✅ 已含复权（adj_factor/hfq）"
        elif s["eng"] > 0 and s["sql"] == 0 and not s["defp"]:
            verdict = "🟢 全走引擎 → 已自动获得 --price-mode hfq"
        elif s["eng"] > 0 and (s["sql"] > 0 or s["defp"]):
            verdict = "🟡 混合：引擎+自取价 → 需确认 NAV 那一路走哪边"
        else:
            verdict = "🔴 独立实现 raw → 需改"
        print(f"  run_{name}.py".ljust(36) + verdict)


if __name__ == "__main__":
    main()
