"""alpha 风险对比：价值/红利低波/高股息成长 vs 沪深300 基准（2018-2025）
用 risk_metrics 量化"alpha 是否真右偏"——偏度、超额峰度、VaR(CF)、CVaR、尾部比率。

解析加固版：
  - _rows_to_values 兼容 dict 行 / (date,val) tuple 行 / 标量表头行 / 非法值，
    任何无法转成有限 float 的行直接跳过，绝不抛错。
  - risk_summary 的返回键名已与本脚本对齐（excess_kurt / var_9x_cvar / ann_vol）。
  - 单策略 try/except 隔离：一个失败不影响其余，且部分结果也能落 CSV。
  - 进度打印 + 输出目录自动创建。
"""
import io, sys, os
import numpy as np
import pandas as pd

import run_monthly_rebalance as mr
import run_dividend_growth_monthly as dg
from risk_metrics import risk_summary
import logging
logging.disable(logging.WARNING)  # 屏蔽价值选股等模块的 INFO 噪声，只保留本脚本 print

START, END = "20180102", "20251231"
OUT_DIR = "data/results/livermore"
OUT_CSV = os.path.join(OUT_DIR, "alpha_risk_compare.csv")


# ────────────────────────────────────────────────────────────────────────────
#  鲁棒解析层
# ────────────────────────────────────────────────────────────────────────────
_VAL_KEYS = ("value", "nav", "close", "equity", "total", "capital")


def _rows_to_values(seq):
    """从任意 NAV 序列稳健抽取 float 净值数组（长度可变）。
    支持：
      - dict 行：依次尝试 value/nav/close/... 取值
      - (date, val) / (val,) tuple|list 行
      - 标量表头行 / 无法转 float 的行：跳过
    返回一维 np.float 数组（已剔除 NaN/inf）。
    """
    out = []
    if seq is None:
        return np.array([], dtype=float)
    for d in seq:
        v = None
        if isinstance(d, dict):
            for k in _VAL_KEYS:
                if k in d and d[k] is not None:
                    v = d[k]
                    break
        elif isinstance(d, (list, tuple)):
            if len(d) >= 2:
                v = d[1]
            elif len(d) == 1:
                v = d[0]
        else:
            v = d  # 可能已经是标量 float
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue  # 表头 / 非法字符串 → 跳过
        if np.isfinite(fv):
            out.append(fv)
    return np.array(out, dtype=float)


def _seq_to_returns(seq):
    vals = _rows_to_values(seq)
    if len(vals) < 2:
        return np.array([])
    return pd.Series(vals).pct_change().dropna().to_numpy()


def _total_ret_from_values(seq):
    vals = _rows_to_values(seq)
    if len(vals) < 2 or vals[0] == 0:
        return np.nan
    return vals[-1] / vals[0] - 1.0


def _total_ret_from_returns(rets):
    if rets is None or len(rets) == 0:
        return np.nan
    return float(np.prod(1.0 + np.asarray(rets, dtype=float)) - 1.0)


def _annualized_ret_from_returns(rets, periods=252):
    r = np.asarray(rets, dtype=float)
    if len(r) < 2:
        return np.nan
    growth = np.prod(1.0 + r)
    if growth <= 0:
        return -1.0
    return float(growth ** (periods / len(r)) - 1.0)


def _extract_result(res, preferred=None):
    """从策略返回值里挑出 NAV 序列（dict 或 list）。
    优先 preferred，否则按常见键顺序探测。
    """
    if isinstance(res, dict):
        keys = []
        if preferred:
            keys.append(preferred)
        keys += ["daily_values", "nav_raw", "nav", "equity_curve"]
        for k in keys:
            v = res.get(k)
            if isinstance(v, (list, tuple)) and len(v) > 0:
                return v
    if isinstance(res, (list, tuple)) and len(res) > 0:
        return res
    return None


# ────────────────────────────────────────────────────────────────────────────
#  单策略封装（隔离异常）
# ────────────────────────────────────────────────────────────────────────────
class _Sup:
    def __enter__(self):
        self._ = sys.stdout
        sys.stdout = io.StringIO()
        return self
    def __exit__(self, *a):
        sys.stdout = self._


def _safe_risk(returns, label):
    try:
        if returns is None or len(returns) == 0:
            print(f"  [跳过] {label}: 收益序列为空")
            return None
        return risk_summary(returns=returns, label=label)
    except Exception as e:
        print(f"  [跳过] {label}: risk_summary 失败 -> {e!r}")
        return None


def _m(s, key, default=np.nan):
    if not isinstance(s, dict):
        return default
    return s.get(key, default)


def hs300_returns():
    conn = mr.get_conn()
    try:
        df = pd.read_sql(
            f"SELECT trade_date, close FROM index_daily "
            f"WHERE ts_code='000300.SH' AND trade_date BETWEEN '{START}' AND '{END}' "
            f"ORDER BY trade_date", conn)
    finally:
        conn.close()
    df["ret"] = df["close"].pct_change()
    return df["ret"].dropna().to_numpy()


# ────────────────────────────────────────────────────────────────────────────
#  主流程
# ────────────────────────────────────────────────────────────────────────────
def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []  # (name, total_ret, ann_ret, summary_or_None)

    # 1) 沪深300 基准
    print(f"[1/4] 沪深300 基准 {START}~{END} ...", flush=True)
    try:
        r_hs = hs300_returns()
        s = _safe_risk(r_hs, "沪深300(价格指数)")
        rows.append(("沪深300基准", _total_ret_from_returns(r_hs),
                     _annualized_ret_from_returns(r_hs), s))
    except Exception as e:
        print(f"  [跳过] 沪深300: {e!r}")
        rows.append(("沪深300基准", np.nan, np.nan, None))

    # 2) 价值选股
    print(f"[2/4] 价值选股 {START}~{END}（较慢）...", flush=True)
    try:
        with _Sup():
            res = mr.run_backtest(START, END, selection_method="value")
        nav = _extract_result(res, preferred="daily_values")
        s = _safe_risk(_seq_to_returns(nav), "价值选股")
        rows.append(("价值选股", _total_ret_from_values(nav),
                     _annualized_ret_from_returns(_seq_to_returns(nav)), s))
    except Exception as e:
        print(f"  [跳过] 价值选股: {e!r}")
        rows.append(("价值选股", np.nan, np.nan, None))

    # 3) 红利低波
    print(f"[3/4] 红利低波 {START}~{END}...", flush=True)
    try:
        with _Sup():
            res = mr.run_backtest(START, END, selection_method="div_low_vol")
        nav = _extract_result(res, preferred="daily_values")
        s = _safe_risk(_seq_to_returns(nav), "红利低波")
        rows.append(("红利低波", _total_ret_from_values(nav),
                     _annualized_ret_from_returns(_seq_to_returns(nav)), s))
    except Exception as e:
        print(f"  [跳过] 红利低波: {e!r}")
        rows.append(("红利低波", np.nan, np.nan, None))

    # 4) 高股息+成长
    print(f"[4/4] 高股息+成长 {START}~{END}...", flush=True)
    try:
        cfg = dict(top_n=10, top_pct=0.10, pe_max=20.0, peg_min=0.08, peg_max=2.0,
                   roe_min=3.0, rev_min=5.0, np_min=11.0, stop_loss=0.0,
                   atr_stop=0.0, atr_period=14)
        with _Sup():
            res = dg.run_window(START, END, cfg)
        nav = _extract_result(res, preferred="nav_raw")
        s = _safe_risk(_seq_to_returns(nav), "高股息+成长")
        rows.append(("高股息+成长", _total_ret_from_values(nav),
                     _annualized_ret_from_returns(_seq_to_returns(nav)), s))
    except Exception as e:
        print(f"  [跳过] 高股息+成长: {e!r}")
        rows.append(("高股息+成长", np.nan, np.nan, None))

    # ── 汇总表（键名严格对齐 risk_summary 输出） ──
    cols = ["策略", "总收益", "年化收益", "年化波动", "偏度", "超额峰度",
            "VaR95(CF)", "VaR99(CF)", "CVaR95", "CVaR99", "尾部比率"]
    tbl = []
    for name, tot, ann, s in rows:
        if s is None:
            tbl.append([name, "N/A", "N/A", "N/A", "N/A", "N/A",
                        "N/A", "N/A", "N/A", "N/A", "N/A"])
            continue
        m = s  # 已是 dict
        tbl.append([
            name,
            f"{tot*100:+.2f}%" if np.isfinite(tot) else "N/A",
            f"{ann*100:+.2f}%" if np.isfinite(ann) else "N/A",
            f"{_m(m,'ann_vol')*100:.2f}%",
            f"{_m(m,'skew'):+.2f}",
            f"{_m(m,'excess_kurt'):+.2f}",
            f"{_m(m,'var_95_cf')*100:.2f}%",
            f"{_m(m,'var_99_cf')*100:.2f}%",
            f"{_m(m,'var_95_cvar')*100:.2f}%",
            f"{_m(m,'var_99_cvar')*100:.2f}%",
            f"{_m(m,'tail_ratio'):.2f}",
        ])
    df = pd.DataFrame(tbl, columns=cols)
    print("\n" + "=" * 96)
    print(f"alpha 风险对比（{START}~{END}，日收益口径）")
    print("=" * 96)
    print(df.to_string(index=False))
    df.to_csv(OUT_CSV, index=False)
    print(f"\n已保存: {OUT_CSV}")

    # ── 判定：alpha 是否真右偏（偏度>0 且 左尾风险优于沪深300） ──
    print("\n--- 判定：alpha 是否真右偏（偏度>0 且 左尾风险优于沪深300）---")
    hs = None
    for name, _, _, s in rows:
        if name.startswith("沪深300") and s is not None:
            hs = s
            break
    if hs is None:
        print("  [无法判定] 沪深300 基准缺失")
        return
    hs_cvar99 = _m(hs, "var_99_cvar")
    hs_tail = _m(hs, "tail_ratio")
    for name, _, _, s in rows:
        if name.startswith("沪深300") or s is None:
            continue
        skew = _m(s, "skew")
        cvar99 = _m(s, "var_99_cvar")
        tail = _m(s, "tail_ratio")
        skew_ok = np.isfinite(skew) and skew > 0
        cvar_ok = np.isfinite(cvar99) and np.isfinite(hs_cvar99) and cvar99 < hs_cvar99
        print(f"  {name}: 偏度={skew:+.2f}({'✅右偏' if skew_ok else '❌左偏/对称'}) | "
              f"CVaR99={cvar99*100:.2f}% vs 沪深300 {hs_cvar99*100:.2f}% "
              f"({'✅尾风险更低' if cvar_ok else '❌尾风险更高'}) | "
              f"尾部比率={tail:.2f}(沪深300 {hs_tail:.2f})")


if __name__ == "__main__":
    run()
