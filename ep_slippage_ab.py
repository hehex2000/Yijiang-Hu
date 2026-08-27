import subprocess, os, re, sys

# EP 全G5 滑点 A/B 驱动：flat 0.1% 一刀切 vs 平方根冲击（流动性感知）
# 单变量隔离：两次命令除 env MFS_SQRT_IMPACT 外完全一致
VENV = os.path.join(os.getcwd(), "venv_ml", "Scripts", "python.exe")
SCRIPT = "run_ep_neutral.py"
BASE = ["20200101", "20260803", "--stock-pool", "all", "--slippage", "0.001"]
OUTDIR = "data/results/ep_neutral"
os.makedirs(OUTDIR, exist_ok=True)

PAT = {
    "total_return": r"总收益率：([-+]?\d+\.\d+)%",
    "annual_return": r"年化收益：([-+]?\d+\.\d+)%",
    "max_drawdown": r"最大回撤：([-+]?\d+\.\d+)%",
    "sharpe": r"夏普比率：([-+]?\d+\.\d+)",
    "win_rate": r"胜率：(\d+\.\d+)%",
    "trades": r"交易次数：(\d+)",
}


def run(label, env_extra):
    env = dict(os.environ)
    env.update(env_extra)
    log = os.path.join(OUTDIR, f"ab_{label}.log")
    print(f"[run] {label} -> {log}", flush=True)
    with open(log, "w", encoding="utf-8") as f:
        rc = subprocess.run([VENV, SCRIPT] + BASE, env=env,
                            stdout=f, stderr=subprocess.STDOUT)
    print(f"[run] {label} rc={rc.returncode}", flush=True)
    return log


def parse(log):
    d = {}
    with open(log, encoding="utf-8", errors="replace") as f:
        for line in f:
            for k, pat in PAT.items():
                m = re.search(pat, line)
                if m:
                    d[k] = int(m.group(1)) if k == "trades" else float(m.group(1))
    return d


if __name__ == "__main__":
    la = run("flat", {})
    lb = run("sqrt", {"MFS_SQRT_IMPACT": "1"})
    a, b = parse(la), parse(lb)
    print("\n" + "=" * 78)
    print("  EP 全G5 滑点 A/B：flat 0.1% 一刀切  vs  平方根冲击(流动性感知)")
    print("  区间 2020-01-01 ~ 2026-08-03 | 全A 全G5等权 | 月度调仓 | 开盘成交")
    print("=" * 78)
    print(f"{'指标':<10}{'flat(0.1%)':>16}{'sqrt(冲击)':>16}{'Δ':>16}")
    rows = [
        ("总收益率", "total_return", "%"),
        ("年化收益", "annual_return", "%"),
        ("最大回撤", "max_drawdown", "%"),
        ("夏普比率", "sharpe", ""),
        ("胜率", "win_rate", "%"),
        ("交易次数", "trades", ""),
    ]
    for name, key, unit in rows:
        va, vb = a.get(key), b.get(key)
        if va is None or vb is None:
            print(f"{name:<10}{str(va):>16}{str(vb):>16}")
            continue
        delta = vb - va
        if unit:
            print(f"{name:<10}{va:>15.2f}{unit}{vb:>15.2f}{unit}{delta:>+15.2f}{unit}")
        else:
            print(f"{name:<10}{va:>16}{vb:>16}{delta:>+16.2f}")
    print(f"\n  flat 日志: {la}")
    print(f"  sqrt 日志: {lb}")
