"""快速 smoke：验证 --piotroski-blend 连续加权重排生效，且 w=0 == OFF(top5 最便宜)。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.value_stock_selector import select_value_stocks

TD = "20240508"  # 一个近年调仓日

off = select_value_stocks(TD, top_n=5, stock_pool="zz800", piotroski_blend=None)
b0 = select_value_stocks(TD, top_n=5, stock_pool="zz800", piotroski_blend=0.0)
b50 = select_value_stocks(TD, top_n=5, stock_pool="zz800", piotroski_blend=0.5)
b100 = select_value_stocks(TD, top_n=5, stock_pool="zz800", piotroski_blend=1.0)

def codes(df):
    return df["ts_code"].tolist() if not df.empty else []

def fs(df):
    return df["fscore"].tolist() if (not df.empty and "fscore" in df.columns) else []

print(f"OFF        : {codes(off)}  fscore={fs(off)}")
print(f"blend w=0  : {codes(b0)}  fscore={fs(b0)}")
print(f"blend w=0.5: {codes(b50)} fscore={fs(b50)}")
print(f"blend w=1  : {codes(b100)} fscore={fs(b100)}")

print("\n[校验] w=0 应与 OFF 完全相同(纯价值):",
      "PASS" if codes(off) == codes(b0) else f"FAIL off={codes(off)} b0={codes(b0)}")
print("[校验] w=0.5 与 OFF 不同(被F重排):",
      "PASS" if codes(off) != codes(b50) else "FAIL(未重排?)")
print("[校验] w=1 与 OFF 不同(纯F排序):",
      "PASS" if codes(off) != codes(b100) else "FAIL(未重排?)")
