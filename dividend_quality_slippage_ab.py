# -*- coding: utf-8 -*-
# 红利质量 official_compact (hs300 池, top12, 季度, 股息率加权, 行业cap2, overlay默认开)
# 平方根冲击滑点 vs flat 0.1% 一刀切  A/B
# 隔离变量：仅 MFS_SQRT_IMPACT 环境变量；其余参数完全一致
import subprocess, re, sys, os

PY = sys.executable
BASE = ["run_dividend_low_vol_quality_bt.py",
        "--mode", "official_compact", "--pool", "hs300", "--top-n", "12"]
RES = "data/results/dividend_low_vol"


def run_once(tag, env_extra):
    out = os.path.join(RES, f"ab_{tag}.log")
    with open(out, "w", encoding="utf-8") as f:
        subprocess.run([PY] + BASE, env={**os.environ, **env_extra},
                       stdout=f, stderr=subprocess.STDOUT, text=True)
    return out


def parse(path):
    d = {}
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "总收益" in line:
                m = re.findall(r'[-+]?\d+\.\d+', line);  d["总收益%"] = m[0] if m else "?"
            elif "年化" in line:
                m = re.findall(r'[-+]?\d+\.\d+', line);  d["年化%"] = m[0] if m else "?"
            elif "最大回撤" in line:
                m = re.findall(r'[-+]?\d+\.\d+', line);  d["最大回撤%"] = m[0] if m else "?"
            elif "夏普" in line:
                m = re.findall(r'[-+]?\d+\.\d+', line);  d["夏普"] = m[0] if m else "?"
            elif "卡玛" in line:
                m = re.findall(r'[-+]?\d+\.\d+', line);  d["卡玛"] = m[0] if m else "?"
    return d


print(">> FLAT (0.1% 一刀切) 跑...", flush=True)
run_once("flat", {})
print(">> SQRT (平方根冲击 MFS_SQRT_IMPACT=1) 跑...", flush=True)
run_once("sqrt", {"MFS_SQRT_IMPACT": "1"})

f = parse(os.path.join(RES, "ab_flat.log"))
s = parse(os.path.join(RES, "ab_sqrt.log"))
print("\n=== 红利质量 official_compact (hs300, top12, 季度)  flat vs sqrt 冲击 ===")
for k in ["总收益%", "年化%", "最大回撤%", "夏普", "卡玛"]:
    fv, sv = f.get(k, "?"), s.get(k, "?")
    print(f"{k:<10} flat={fv:>10}  sqrt={sv:>10}")
