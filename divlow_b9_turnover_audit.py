"""B9 换手/集中度实测：直接从**引擎落盘的选股明细**算，验证实验室预测值。

B9 实验室（`divlow_b9_pool_lab.py`）用 27 期候选池离线估换手，预测 ic0=115.2%。
本脚本用引擎真实产物回算，回答两个问题：
  1. 真实换手是多少？实验室预测准不准？
  2. 取消行业上限后集中度有多高？（每期同行业最大只数 / 最大行业权重 / 银行占比）

口径（与 B5' 报告一致，勿混用）：
  per_one_way  = 0.5 * Σ|Δw|         每期单边
  ann_one_way  = per_one_way * 期数/年  年化单边  ← 与 B&O / 活跃税同口径
"""
import argparse
import os

import pandas as pd


def ann_factor(dates):
    """由调仓日序列推"每年期数"：按实际跨度折算，避免硬编码 4。"""
    d = pd.to_datetime(pd.Series(sorted(set(dates))), format="%Y%m%d")
    if len(d) < 2:
        return 0.0
    span_days = (d.iloc[-1] - d.iloc[0]).days
    if span_days <= 0:
        return 0.0
    return (len(d) - 1) / (span_days / 365.25)


def turnover(sel):
    """sel: 含 rebal_date / ts_code / weight 的 DataFrame → (每期单边, 年化单边)"""
    piv = sel.pivot_table(index="rebal_date", columns="ts_code",
                          values="weight", aggfunc="first").fillna(0.0)
    piv = piv.sort_index()
    dates = list(piv.index)
    per = []
    for i in range(1, len(dates)):
        prev, cur = piv.iloc[i - 1], piv.iloc[i]
        per.append(0.5 * float((cur - prev).abs().sum()))
    af = ann_factor(dates)
    per_mean = sum(per) / len(per) if per else 0.0
    return per_mean, per_mean * af, len(dates), af


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="bt_quality_sel_*.csv")
    ap.add_argument("--labels", default=None, help="短名，逗号分隔")
    args = ap.parse_args()

    labels = args.labels.split(",") if args.labels else [os.path.basename(f) for f in args.files]
    if len(labels) != len(args.files):
        raise SystemExit("[err] --labels 个数与文件数不一致（用逗号分隔）")

    print(f"{'档位':<26}{'期数':>5}{'每年期数':>9}{'每期单边':>10}{'年化单边':>10}"
          f"{'最大行业只数':>13}{'最大行业权重':>13}{'银行占比':>9}")
    print("-" * 97)
    for f, lb in zip(args.files, labels):
        sel = pd.read_csv(f, dtype={"rebal_date": str}, encoding="utf-8-sig")
        sel["rebal_date"] = sel["rebal_date"].astype(str)
        per, ann, nper, af = turnover(sel)

        # 集中度：需要 name/行业。引擎 sel 只有 name，用"每股等权"口径的
        # 「最大同行业只数」需行业映射 → 退而求其次：用 weight 的 HHI + 最大权重
        hhi_max = 0.0
        wmax_max = 0.0
        bank_share_max = 0.0
        for d, g in sel.groupby("rebal_date"):
            w = g["weight"].astype(float)
            hhi_max = max(hhi_max, float((w ** 2).sum()))
            wmax_max = max(wmax_max, float(w.max()))
            # 银行识别：A 股银行股名几乎都以"银行"结尾
            if "name" in g.columns:
                isbank = g["name"].astype(str).str.endswith("银行")
                bank_share_max = max(bank_share_max, float(w[isbank].sum()))
        print(f"{lb:<26}{nper:>5}{af:>9.2f}{per*100:>9.1f}%{ann*100:>9.1f}%"
              f"{hhi_max:>13.3f}{wmax_max*100:>12.1f}%{bank_share_max*100:>8.1f}%")
    print("-" * 97)
    print("HHI = Σw²（等权12只≈0.083；12只全押一类≈0.083 但权重集中则更高）")
    print("银行占比 = 权重加权（名称以'银行'结尾），用于量化'取消上限→全押银行'的程度")


if __name__ == "__main__":
    main()
