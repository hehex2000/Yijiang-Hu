"""B8 诊断：最后一段筛子的排序键（股息率 vs 波动率）对「留存率 / 换手」的影响。

背景（2026-09-03 核对官方编制方案时发现的两处偏差）：
  ① 目标指数是 **930955 中证红利低波动100**（100 只、**季度**调仓、**股息率/波动率**加权、
     中证二级行业权重 ≤20%、**无**换手上限）；
     而 B5'/B7' 报告全程对标的是 **H30269 中证红利低波动**（50 只、**年度**调仓、股息率加权、
     单样本 ≤15%、**有**「每次调整 ≤20%」上限）→ 规则书对错了。
  ② 更致命：官方 930955 的选样是「股息率前 300 → **波动率升序**取前 100」，
     **最后一段筛子的排序键是波动率（慢变量）**；
     我们的实现在候选池（波动率最低的 buffer_n 只）上又走 `_cap_industry`，
     而 `_cap_industry` 内部按 **fwd_yield 降序**重排 → 最后一段排序键是**股息率（快变量）**。
     → 方向相反，且用快变量做最终排序 = 每期大翻、换手高。

本脚本**一次选股**同时得到两种键的结果（选股占 95% 耗时，避免跑两遍）：
  包一层 `DividendLowVolSelector._cap_industry`，把每期**行业 cap 前的候选池**录下来；
  跑完后再对每期候选池分别按 fwd_yield / volatility 复算最终持仓。
顺带产出两个之前缺的关键诊断：
  · 候选池（48 只）自身的逐期留存 —— 判断"池子翻转"到底有多严重；
  · 上期持仓有多少仍落在本期候选池内 —— 结构性（不可压缩）换手的下界。

判读：
  · 若 vol 键的留存率显著高于 yield 键 → 换排序键就是真正的杠杆（B8 采纳）；
  · 若两者差不多 → 换手来自更上游（门禁/漏斗宽度），需动 buffer_n 或两段式比例。

用法：
  venv_ml/Scripts/python.exe divlow_b8_key_smoke.py                 # 2023 短窗口 4 期（~8min）
  venv_ml/Scripts/python.exe divlow_b8_key_smoke.py 20210101 20260723
"""
import hashlib
import os
import sys

import pandas as pd

import run_dividend_low_vol_quality_bt as E
from src.dividend_low_vol_selector import DividendLowVolSelector

MODE = "official_compact"
POOL = "all"
TOP_N = 12
IND_CAP = 2

S, V = "fwd_yield", "volatility"          # 两种排序键

# 每期候选池（行业 cap 前，len=buffer_n=48）
POOLS = []
_ORIG_CAP = None        # 保存原版 _cap_industry：复算时必须走原版，否则会污染 POOLS


def _install_recorder():
    global _ORIG_CAP
    _ORIG_CAP = DividendLowVolSelector._cap_industry

    def _rec(self, df, cap, sort_key="fwd_yield"):
        POOLS.append(df.copy())
        return _ORIG_CAP(self, df, cap, sort_key=sort_key)

    DividendLowVolSelector._cap_industry = _rec
    return _ORIG_CAP


def picks_of(pool_df, key, sel_inst):
    """对某一期候选池按 key 复算最终持仓（与裸档口径一致：_cap_industry().head(top_n)）。

    🔴 必须调**原版** _ORIG_CAP：装了 recorder 之后类方法会往 POOLS 里再追加一份，
       复算 n 期会污染列表、让后面的自证假通过。
    """
    capped = _ORIG_CAP(sel_inst, pool_df, IND_CAP, sort_key=key)
    return [str(c) for c in capped["ts_code"].head(TOP_N)]


def series(pick_list):
    """逐期新进 / 留存（第 1 期全为新进，不计入平均）。"""
    new = [len(pick_list[0])]
    for i in range(1, len(pick_list)):
        new.append(len(set(pick_list[i]) - set(pick_list[i - 1])))
    return new


def main():
    start = sys.argv[1] if len(sys.argv) > 1 else "20230101"
    end = sys.argv[2] if len(sys.argv) > 2 else "20231231"
    # 🔴 第三个参数 = 实际跑的键（只影响落盘名/断点续跑，**不影响候选池**：池子与键无关）。
    #    想跑长窗口但同名 partial 已存在时，传 vol 换一个不撞车的落盘名即可。
    run_key = V if (len(sys.argv) > 3 and sys.argv[3] in ("vol", "volatility")) else S
    E.START, E.END = start, end
    E.PRICE_MODE = "hfq"

    # 复用 divlow_select_freq 的加速版波动率（已 80/80 逐位自证）
    import divlow_select_freq as F
    if F.verify(n=30, date="20231215"):
        E._patched_vol = F._fast_vol

    _install_recorder()
    E.MODE_SPECS[MODE]["rebal"] = "quarter"
    print(f"\n{'#' * 74}\n# B8 排序键诊断  {start}~{end}  pool={POOL} top_n={TOP_N} "
          f"ind_cap={IND_CAP}\n{'#' * 74}")
    targets, wmap, sel_log = E.select_targets_official(
        MODE, pool=POOL, top_n=TOP_N, buffer_k=0, turnover_cap=0.0, final_key=run_key)

    # ── 自证：录下来的候选池复算出的 yield 键持仓，必须与实际跑出的逐位一致 ──
    sel = DividendLowVolSelector(E.build_cfg(MODE), None)
    actual = {}
    for rec in sel_log:
        actual.setdefault(str(rec[0]), []).append(str(rec[2]))
    rbs = sorted(actual)
    # ── 落盘**优先于**自检：池子是 45min 跑出来的，绝不能因为自检失败就全丢 ──
    out_dir = os.path.join(E.RES_DIR, "_b8_keydiag")
    os.makedirs(out_dir, exist_ok=True)
    pdir = os.path.join(out_dir, "pools")
    os.makedirs(pdir, exist_ok=True)
    for i, p in enumerate(POOLS):
        d = p.copy()
        d.insert(0, "rebal_date", rbs[i] if i < len(rbs) else f"?{i}")
        d.to_csv(os.path.join(pdir, f"pool_{rbs[i] if i < len(rbs) else i}.csv"),
                 index=False, encoding="utf-8-sig")
    if POOLS:
        print(f"\n[落盘] 候选池 {len(POOLS)} 期 → {pdir}")

    ok = True
    if len(POOLS) == 0:
        # 🔴 首跑踩到：该窗口/参数的 partial 已存在 → select_targets_official 断点续跑，
        #    所有期都被 skip，一次 select_stocks 都没跑 → 自然录不到候选池。
        print("  ❌ 录得 0 期 —— 多半是**命中了已有 partial 断点续跑**（见上方 [resume] 行）。")
        print("     处理：换一个窗口，或把同名 partial 挪走后再跑。")
        return
    if len(POOLS) != len(rbs):
        print("  ❌ 期数不一致，候选池录制有误")
        ok = False
    else:
        for i, p in enumerate(POOLS):
            # 🔴 必须拿**本次实际跑的键** run_key 去比对（首跑踩坑：跑的是 vol 键、
            #    却固定拿 yield 键复算 → 必然不一致，是**检查写错了**不是数据错了）
            if picks_of(p, run_key, sel) != actual[rbs[i]]:
                ok = False
                print(f"  ❌ 第 {i+1} 期复算不一致（比对键={run_key}）")
                print(f"     复算={picks_of(p, run_key, sel)}")
                print(f"     实际={actual[rbs[i]]}")
                break
    print(f"  {'✅ 复算与实际逐位一致，候选池录制可信' if ok else '❌ 不一致，下面的结论不可用'}")
    if not ok:
        return

    py = [picks_of(p, S, sel) for p in POOLS]
    pv = [picks_of(p, V, sel) for p in POOLS]
    ny, nv = series(py), series(pv)
    n = len(POOLS)

    print(f"\n【一、候选池（{len(POOLS[0])} 只）自身逐期留存】—— 池子翻转有多严重")
    ov = [len(POOLS[0])]
    for i in range(1, n):
        a = set(str(c) for c in POOLS[i]["ts_code"])
        b = set(str(c) for c in POOLS[i - 1]["ts_code"])
        ov.append(len(a & b))
    keep_in = [len(set(py[i]) & set(str(c) for c in POOLS[i]["ts_code"])) if i == 0
               else len(set(py[i - 1]) & set(str(c) for c in POOLS[i]["ts_code"]))
               for i in range(n)]
    print(f"  池∩上期池        : {ov[1:]}  平均留存率={sum(ov[1:])/(n-1)/len(POOLS[0]):.1%}")
    print(f"  上期持仓仍在本期池: {keep_in[1:]}  ← 结构性换手下界（这些位置以外必然被迫换）"
          if n > 1 else "")

    print(f"\n【二、两种排序键的逐期新进（越少越好，持仓={TOP_N}）】")
    print(f"  日期          yield键新进   vol键新进   两键重合")
    for i in range(n):
        same = len(set(py[i]) & set(pv[i])) if i < len(py) else 0
        print(f"  {rbs[i] if i < len(rbs) else POOLS[i].iloc[0].get('ts_code','?'):<12}"
              f"  {ny[i]:>8}     {nv[i]:>6}      {same:>5}")
    my = sum(ny[1:]) / (n - 1)
    mv = sum(nv[1:]) / (n - 1)
    print(f"  {'平均':<12}  {my:>8.2f}     {mv:>6.2f}")
    print(f"\n  → yield 键平均留存率 {1 - my / TOP_N:.1%} ／ vol 键平均留存率 {1 - mv / TOP_N:.1%}"
          f"   （差 {(my - mv) / TOP_N:+.1%}）")
    if mv < my - 1e-9:
        print(f"  ✅ vol 键（官方口径）留存更高 → 换排序键是真杠杆，建议采纳 B8")
    else:
        print(f"  ⚠️ vol 键未改善 → 换手来自更上游（门禁/漏斗宽度/buffer_n），B8 不够")

    # 落盘两键持仓，供后续 NAV 重放
    for tag, picks in ((S, py), (V, pv)):
        rows = []
        for i, ps in enumerate(picks):
            for c in ps:
                rows.append((rbs[i] if i < len(rbs) else "", c))
        fp = os.path.join(out_dir, f"picks_{tag}_{start}_{end}.csv")
        pd.DataFrame(rows, columns=["rebal_date", "ts_code"]).to_csv(
            fp, index=False, encoding="utf-8-sig")
        print(f"  → {fp}")


if __name__ == "__main__":
    main()
