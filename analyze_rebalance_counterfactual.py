# -*- coding: utf-8 -*-
"""
规则再平衡反事实对照 (task #54)

复用 run_monthly_rebalance.py 引擎，对同一股票池/选股法在多个 --rebalance-freq
下跑回测，聚合出对比表，直接回答视频「努力悖论」：越频繁调仓越赚吗？

依赖引擎改动：run_backtest 新增 rebalance_freq_months 参数（默认1=每月，
3=每季/6=半年/12=每年/999≈买入持有），value 与动量模式均生效。

用法（用户在本机项目根目录执行）:
  python analyze_rebalance_counterfactual.py 20200102 20251231 \
      --selection-method value --stock-pool 000906.SH --top-n 20 \
      --freqs 1 3 6 12 999

  # 带额外引擎参数（如价值模式子策略）:
  python analyze_rebalance_counterfactual.py 20200102 20251231 \
      --selection-method value --stock-pool zz800 --top-n 20 \
      --freqs 1 12 999 --extra "--value-mode pobreak"

输出：各频率档的总收益/年化/回撤/夏普/换手率/交易次数 + 基准净α 对比表。
"""
import argparse
import os
import re
import subprocess
import sys

import sqlite3

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join("D:/tu-shareData", "astock_daily.db")

POOL_TO_INDEX = {
    "zz800": "000906.SH", "zz500": "000905.SH", "hs300": "000300.SH",
    "zz1000": "000852.SH", "sz50": "000016.SH", "all": "000906.SH",
}


def get_conn():
    return sqlite3.connect(DB_PATH)


def benchmark_return(index_code, start, end):
    """取基准指数在 [start,end] 区间的首末收盘价收益。"""
    if not index_code:
        index_code = "000906.SH"
    # 池名(如 zz800)→指数代码(000906.SH)；已是代码则原样返回
    index_code = POOL_TO_INDEX.get(index_code, index_code)
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT close FROM index_daily WHERE ts_code=? AND trade_date>=? ORDER BY trade_date ASC LIMIT 1",
            (index_code, start),
        )
        first = cur.fetchone()
        cur.execute(
            "SELECT close FROM index_daily WHERE ts_code=? AND trade_date<=? ORDER BY trade_date DESC LIMIT 1",
            (index_code, end),
        )
        last = cur.fetchone()
        conn.close()
        if not first or not last:
            return None
        return (float(last[0]) / float(first[0]) - 1.0) * 100.0
    except Exception as e:
        sys.stderr.write(f"[warn] 基准收益计算失败({index_code}): {e}\n")
        return None


def parse_summary(text):
    def f(pattern, flags=0):
        m = re.search(pattern, text, flags)
        return m.group(1) if m else None

    total = f(r"总收益率[:：]\s*([+-]?\d+\.\d+)%")
    annual = f(r"年化收益率[:：]\s*([+-]?\d+\.\d+)%")
    mdd = f(r"最大回撤[:：]\s*([+-]?\d+\.\d+)%")
    sharpe = f(r"夏普比率[:：]\s*(\d+\.\d+)")
    trades = f(r"交易次数[:：]\s*(\d+)")
    ann_trades = f(r"年化交易次数[:：]\s*([\d.]+)\s*次/年")
    turnover = f(r"年化换手率[:：]\s*([\d.]+)x")
    return {
        "total": float(total) if total else None,
        "annual": float(annual) if annual else None,
        "mdd": float(mdd) if mdd else None,
        "sharpe": float(sharpe) if sharpe else None,
        "trades": int(trades) if trades else None,
        "ann_trades": float(ann_trades) if ann_trades else None,
        "turnover": float(turnover) if turnover else None,
    }


def run_one(exe, start, end, freq, base_args, extra):
    cmd = [
        exe, "run_monthly_rebalance.py", start, end,
        "--rebalance-freq", str(freq),
    ] + base_args
    if extra:
        cmd += extra.split()
    # 子进程 stdout 接管道时按系统 locale(中文Windows=GBK)编码，
    # 引擎打印的 emoji(如 ✅ \u2705) 无法编码会 UnicodeEncodeError 崩溃。
    # 强制 PYTHONUTF8=1 让子进程用 UTF-8 输出（与下方 utf-8 解码一致）。
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    proc = subprocess.run(
        cmd, cwd=HERE, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env,
    )
    out = proc.stdout + "\n" + proc.stderr
    if proc.returncode != 0:
        sys.stderr.write(f"[warn] freq={freq} 返回码 {proc.returncode}\n")
        sys.stderr.write(f"[cmd] {' '.join(cmd)}\n")
        sys.stderr.write(f"--- 引擎完整输出 (freq={freq}) ---\n{out.strip()}\n")
    return parse_summary(out), out


def choose_exe(preferred=None):
    """自动挑一个能成功 import 引擎的解释器，规避 sys.executable 指向缺依赖的 Python。"""
    cands = []
    if preferred:
        cands.append(preferred)
    cands.append(sys.executable)
    for c in ("py", "python", "python3"):
        cands.append(c)
    seen = []
    for exe in cands:
        if exe in seen:
            continue
        seen.append(exe)
        try:
            p = subprocess.run(
                [exe, "-c", "import run_monthly_rebalance"],
                cwd=HERE, capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=60,
            )
            if p.returncode == 0:
                sys.stderr.write(f"[info] 选用解释器: {exe}\n")
                return exe
            sys.stderr.write(f"[info] {exe} 不可用 (rc={p.returncode}): {p.stderr.strip()[:160]}\n")
        except Exception as e:
            sys.stderr.write(f"[info] {exe} 探测异常: {e}\n")
    return sys.executable  # 兜底


def main():
    ap = argparse.ArgumentParser(description="规则再平衡反事实对照")
    ap.add_argument("start_date")
    ap.add_argument("end_date")
    ap.add_argument("--selection-method", default="value")
    ap.add_argument("--stock-pool", default=None)
    ap.add_argument("--top-n", type=int, default=None)
    ap.add_argument("--freqs", nargs="+", type=int, default=[1, 3, 6, 12, 999])
    ap.add_argument("--extra", default="", help="附加引擎参数（原样追加）")
    ap.add_argument("--python", default=None, help="显式指定运行引擎的解释器(python/py 路径)；缺省自动探测")
    args = ap.parse_args()

    exe = choose_exe(args.python)

    base_args = []
    if args.selection_method:
        base_args += ["--selection-method", args.selection_method]
    if args.stock_pool:
        base_args += ["--stock-pool", args.stock_pool]
    if args.top_n is not None:
        base_args += ["--top-n", str(args.top_n)]

    bench = benchmark_return(args.stock_pool, args.start_date, args.end_date)
    bench_label = POOL_TO_INDEX.get(args.stock_pool, args.stock_pool) if args.stock_pool and args.stock_pool != "all" else "000906.SH"

    rows = []
    for freq in args.freqs:
        label = {1: "每月", 3: "每季", 6: "半年", 12: "每年", 999: "≈买入持有"}.get(freq, f"{freq}月")
        m, _ = run_one(exe, args.start_date, args.end_date, freq, base_args, args.extra)
        rows.append((freq, label, m))

    # 表头
    hdr = f"{'频率':<10} {'总收益%':>9} {'年化%':>9} {'回撤%':>8} {'夏普':>6} {'换手率x':>9} {'年交易次':>9} {'净α%':>8}"
    print(f"\n规则再平衡反事实对照  [{args.start_date}~{args.end_date}  {args.selection_method}"
          + (f"  池={args.stock_pool}" if args.stock_pool else "")
          + (f"  TOP{args.top_n}" if args.top_n else "") + "]")
    print(f"基准：{bench_label}  区间收益 = {bench:+.2f}%" if bench is not None else f"基准：{bench_label} (无数据)")
    print("-" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for freq, label, m in rows:
        net = (m["total"] - bench) if (m["total"] is not None and bench is not None) else None
        t = lambda v: f"{v:+.2f}" if isinstance(v, float) else "-"
        tt = lambda v: f"{v:.2f}" if isinstance(v, float) else "-"
        print(f"{label:<10} {t(m['total']):>9} {t(m['annual']):>9} {tt(m['mdd']):>8} "
              f"{tt(m['sharpe']):>6} {tt(m['turnover']):>9} {tt(m['ann_trades']):>9} {t(net):>8}")
    print("-" * len(hdr))
    print("净α% = 策略总收益% − 基准区间收益%。换手率/年交易次越高=操作越勤。")
    print("结论判读：若高频档净α ≤ 低频档，则验证视频「努力悖论」——多操作未多赚、只多交成本。\n")


if __name__ == "__main__":
    main()
