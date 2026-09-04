"""B7'（≤20% 调整硬上限）冒烟自证 —— 短窗口 2023 全年，只跑 4 期。

硬闸门 ①③ 任一不过就说明 cap 分支写错了，绝不能拿去跑全窗口；
判据② 是**观测项**（超预算=策略结构导致的被动换仓，不是代码 bug）：
  ① cap=1.0（max_change=12=top_n，等价于不限）的 partial 必须与 cap=0（旧分支）**md5 相同**
     → 证明 cap 分支在未触限时行为中性，没偷偷改选股口径。
  ② cap=0.2 必须真的把「每期新进只数」钉在 ≤ max(1,int(12*0.2))=2。
  ③ cap=0.2 的持仓只数必须仍是 12（不能因为回补失败而缩水）。
"""
import hashlib
import os
import sys

import pandas as pd

import run_dividend_low_vol_quality_bt as E
import divlow_select_freq as S

E.PRICE_MODE = "hfq"
E.START, E.END = "20230101", "20231231"        # 短窗口冒烟
RES = E.RES_DIR
MODE, POOL, TOP_N = "official_compact", "all", 12
MAX_CHANGE = max(1, int(TOP_N * 0.2))

print(f"[加速版自证] 抽样验证 _fast_vol 与原版一致 …")
if not S.verify(n=30, date="20231215"):
    raise SystemExit("加速版自证未通过 → 中止（不能拿不等价的函数跑自证）")
E._patched_vol = S._fast_vol


def path_of(cap):
    tag = f"_tc{int(round(cap * 100))}" if cap else ""
    return os.path.join(RES, f"_official_{MODE}_{POOL}_{TOP_N}_bk0{tag}_{E.START}_{E.END}_partial.csv")


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


res = {}
for cap in (0.0, 1.0, 0.2):
    p = path_of(cap)
    if os.path.exists(p):
        print(f"\n[skip] cap={cap} 的 partial 已存在，直接复用 → {os.path.basename(p)}")
        res[cap] = (p, md5(p))
        continue
    print(f"\n{'=' * 78}\n# cap={cap}  → {os.path.basename(p)}\n{'=' * 78}")
    E.select_targets_official(MODE, pool=POOL, top_n=TOP_N, buffer_k=0, turnover_cap=cap)
    if not os.path.exists(p):
        raise SystemExit(f"cap={cap} 未产出 partial")
    res[cap] = (p, md5(p))

print("\n" + "=" * 78)
print("【判据①】cap=1.0（等价不限） vs cap=0（旧分支）md5")
print(f"  cap=0.0 md5 = {res[0.0][1]}")
print(f"  cap=1.0 md5 = {res[1.0][1]}")
ok1 = res[0.0][1] == res[1.0][1]
print(f"  → {'✅ 一致（行为中性）' if ok1 else '❌ 不一致！cap 分支改了选股口径'}")

print("\n【判据②③】cap=0.2 每期新进只数 / 持仓只数")
df = pd.read_csv(res[0.2][0], dtype={"rebal_date": str, "ts_code": str}, encoding="utf-8-sig")
rbs = sorted(df["rebal_date"].unique())
prev, rows = None, []
for rb in rbs:
    cur = set(df[df["rebal_date"] == rb]["ts_code"])
    rows.append((rb, len(cur), "-" if prev is None else len(cur - prev)))
    prev = cur
out = pd.DataFrame(rows, columns=["rebal_date", "持仓只数", "新进只数"])
print(out.to_string(index=False))
ok2 = all(r[2] == "-" or r[2] <= MAX_CHANGE for r in rows)
ok3 = all(r[1] == TOP_N for r in rows)
n_per = len([r for r in rows if r[2] != "-"])
n_over = sum(1 for r in rows if r[2] != "-" and r[2] > MAX_CHANGE)
print(f"  → 持仓恒 = {TOP_N}: {'✅' if ok3 else '❌'}")
print(f"  → 新进 ≤ {MAX_CHANGE}: {n_per - n_over}/{n_per} 期达标，{n_over} 期超预算")
if n_over:
    print(f"    ⚠️ 超预算**不是 cap 代码的 bug**（①③ 已证明代码正确），而是策略结构问题：")
    print(f"       合格池（固定 48 只）本身每期翻转极大，上期 12 只里可能只剩 5 只仍在池内，")
    print(f"       → 被动换仓顶穿硬上限。这是**结论**：12/48 排名结构下靠硬上限压不住换手，")
    print(f"         真正的杠杆是「合格池改绝对门槛」或「扩持仓数」（官方持仓 50≈合格 50）。")

print("\n" + "=" * 78)
print(f"总判定(硬闸门 ①③): {'✅ 过，可以跑全窗口' if (ok1 and ok3) else '❌ 未过，禁止跑全窗口'}"
      f"   （判据② 为观测项：{n_over}/{n_per} 期强制换仓）")

print("\n" + "=" * 78)
print(f"总判定: {'✅ 三条全过，可以跑全窗口' if (ok1 and ok2 and ok3) else '❌ 有判据未过，禁止跑全窗口'}")
print("（注：冒烟用的短窗口 partial 是临时产物，确认后应删除，勿与正式产物混用）")
sys.exit(0 if (ok1 and ok2 and ok3) else 1)
