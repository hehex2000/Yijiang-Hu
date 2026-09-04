# -*- coding: utf-8 -*-
"""平台流水体检 · 三道闸门一键编排（2026-09-03）

把三道独立闸门串成一条流水线，按**成本从低到高**执行：

  闸门一 流水自洽   run_trades_health_check.py  秒级  （不需要价格数据）
  闸门二 前视筛查v2 diagnose_lookahead_scan.py  ~2min （买入当日超额收益）
  闸门三 活跃税     run_activity_tax_check.py   ~20min（要拉 2000 万行重建 NAV）

设计要点（踩过的坑）：
1. **闸门一零成本先筛全库**——持仓负值/action 前缀/分母异常都不需要价格数据，
   先淘汰掉不自洽的文件，再让昂贵闸门只跑健康文件（实测 133 文件：健康 115 / 坏 16）。
2. **--max 0 语义不统一**：只有闸门二已改为 0=不限，闸门一/三仍是 `files[:0]`=空！
   本脚本对外 --max 0 表示不限，内部统一传 10000 兜底。
3. **闸门二必须用超额口径**（相对同日全市场），绝对值会在"集中建仓+市场普涨"时误伤。

判定：
  🔴 淘汰 = 流水不自洽（后续数据全是垃圾）或 前视嫌疑（结论作废）
  🟡 观察 = 前视 △（样本不足/涨停比失衡）或 活跃税超 B&O 6.5%（对平均净值口径）
  ✅ 合格 = 三闸门全过

用法：
  python run_platform_health_check.py                      # 默认闸门 1+2（快，~3min）
  python run_platform_health_check.py --gate 123           # 全量（含活跃税，~25min）
  python run_platform_health_check.py --root data/results/ep_neutral --gate 12
"""
import argparse
import os
import re
import subprocess
import sys
import time

import pandas as pd

BIG = 10000          # 对内兜底的"不限"
B_O_TH = 0.065       # B&O 活跃税阈值（对平均净值口径）
# 历史副本/归档目录的命名特征。区分它们很关键：**淘汰若全落在这里，
# 说明正式结果目录是干净的**，问题已被处理过，不需要再动。
ARCH_RE = re.compile(r"(_archive_|_bak_|_fixed_\d{8}|_deprecated)")


def _norm(p):
    """把各闸门不同的路径写法统一成 data/results/ 开头的相对路径（正斜杠）。"""
    p = str(p).replace("\\", "/")
    i = p.find("data/results/")
    return p[i:] if i >= 0 else p


def run_gate(k, root, glob, out, extra=None):
    """跑单道闸门，返回 (是否成功, 耗时秒)。"""
    m = str(BIG)          # 对内一律不限，避免 --max 0 在旧脚本里变成空
    if k == "1":
        cmd = [sys.executable, "run_trades_health_check.py",
               "--root", root, "--glob", glob, "--max", m, "--out", out]
        name = "闸门一 流水自洽"
    elif k == "2":
        cmd = [sys.executable, "diagnose_lookahead_scan.py",
               "--scan", root, "--glob", glob, "--max", m, "--out", out]
        name = "闸门二 前视筛查v2"
    else:
        cmd = [sys.executable, "run_activity_tax_check.py",
               "--scan", root, "--glob", glob, "--max", m, "--out", out]
        name = "闸门三 活跃税"
    if extra:
        cmd += extra
    print(f"\n{'=' * 72}\n  执行 {name}\n{'=' * 72}")
    print(f"  $ {' '.join(cmd[1:])}")
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    dt = time.time() - t0
    tail = "\n".join((r.stdout or "").strip().splitlines()[-6:])
    print(f"  退出码 {r.returncode}   耗时 {dt:.0f}s")
    if tail:
        print("  " + tail.replace("\n", "\n  "))
    if r.returncode != 0 and (r.stderr or "").strip():
        print("  [stderr] " + "\n".join(r.stderr.strip().splitlines()[-5:]))
    return r.returncode == 0 and os.path.exists(out), dt


def main():
    ap = argparse.ArgumentParser(description="平台流水体检 · 三道闸门")
    ap.add_argument("--root", default="data/results")
    ap.add_argument("--glob", default="trades_*.csv")
    ap.add_argument("--gate", default="12", help="要跑的闸门，如 12（默认）或 123")
    ap.add_argument("--out-csv", default="data/results/platform_health.csv")
    ap.add_argument("--out-md", default="data/results/platform_health_report.md")
    a = ap.parse_args()

    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    tmp = {k: f"data/results/_gate{k}_{os.getpid()}.csv" for k in a.gate if k in "123"}
    ok = {}
    for k in [c for c in "123" if c in a.gate]:
        ok[k], _ = run_gate(k, a.root, a.glob, tmp[k])

    # ── 汇总 ──────────────────────────────────────────────
    frames = {}
    if ok.get("1"):
        d = pd.read_csv(tmp["1"])
        d["key"] = d["file"].map(_norm)
        frames["1"] = d.set_index("key")
    if ok.get("2"):
        d = pd.read_csv(tmp["2"])
        d["key"] = d["file"].map(_norm)
        frames["2"] = d.set_index("key")
    if ok.get("3"):
        d = pd.read_csv(tmp["3"])
        # 闸门三的 name 是「相对 --scan 根目录」的路径（它自己也靠这个区分子目录
        # 同名文件，见 run_activity_tax_check.py 第 88 行），跑单目录时会退化成
        # basename → 必须先拼回 --root 再归一化，否则与闸门一/二的 key 全部对不上，
        # 活跃税数据会**静默丢失**（不报错，报告里全是"未跑/—"）。
        d["key"] = d["name"].map(lambda n: _norm(os.path.join(a.root, str(n))))
        frames["3"] = d.set_index("key")

    if not frames:
        print("\n所有闸门均未产出，无法汇总")
        return
    keys = sorted(set().union(*[set(f.index) for f in frames.values()]))
    rows = []
    for k in keys:
        rec = {"file": k}
        g1 = frames.get("1")
        g2 = frames.get("2")
        g3 = frames.get("3")
        st1 = str(g1.loc[k, "status"]) if g1 is not None and k in g1.index else ""
        st2 = str(g2.loc[k, "flag"]) if g2 is not None and k in g2.index else ""
        if g3 is not None and k in g3.index:
            at = g3.loc[k, "active_tax_nav"]
            st3 = str(g3.loc[k, "flag"])
        else:
            at, st3 = None, ""
        rec["流水自洽"] = st1 or "—"
        rec["前视v2"] = st2 or "—"
        rec["活跃税"] = st3 or "—"
        rec["活跃税_对净值"] = at
        # 判定（注意顺序：先剔除"非流水/不可用"，否则会把体检产物自己误判成淘汰）
        # 实例：data/results/trades_health_check.csv 以 trades_ 开头，会被 glob 命中，
        #       自噬成"✗ 不可用"；*_fills_as.csv 是成交成本对比产物，非回测流水。
        if st1 and ("非流水" in st1 or "不可用" in st1):
            v = "⚪非流水(跳过)"
        elif st1 and "✓" not in st1:
            v = "🔴淘汰(流水不自洽)"
        elif "⚠" in st2:
            v = "🔴淘汰(前视嫌疑)"
        elif "△" in st2 or (at is not None and pd.notna(at) and float(at) > B_O_TH):
            v = "🟡观察"
        elif st1 or st2:
            v = "✅合格"
        else:
            v = "—"
        rec["判定"] = v
        rows.append(rec)
    df = pd.DataFrame(rows)
    # 🔴 闸门匹配率自检：key 格式对不上时不会报错，只会让那一列全是"—"，
    #    跑 25 分钟才发现活跃税数据全丢了。宁可吵，不可静默。
    allkeys = set(df["file"])
    for g in ("1", "2", "3"):
        if not ok.get(g):
            continue
        hit = sum(1 for k in frames[g].index if k in allkeys)
        tot = len(frames[g])
        if tot and hit < tot * 0.9:
            print(f"  ⚠️ 闸门{g} 匹配率 {hit}/{tot} —— key 格式疑似不一致，"
                  f"该闸门数据可能丢失（样例 {list(frames[g].index[:2])}）")
        else:
            print(f"  ✓ 闸门{g} 匹配 {hit}/{tot}")
    df.to_csv(a.out_csv, index=False, encoding="utf-8-sig")

    skip = int((df["判定"] == "⚪非流水(跳过)").sum())
    eff = df[df["判定"] != "⚪非流水(跳过)"]
    n = len(eff)
    c = eff["判定"].value_counts()
    dead = int(sum(v for k, v in c.items() if k.startswith("🔴")))
    watch = int(c.get("🟡观察", 0))
    good = int(c.get("✅合格", 0))
    print(f"\n{'=' * 72}\n  汇总：{len(df)} 个候选，其中有效流水 {n} 个"
          f"（跳过非流水 {skip} 个）\n{'=' * 72}")
    print(f"  🔴 淘汰 {dead}    🟡 观察 {watch}    ✅ 合格 {good}")

    # ── 分层：正式结果目录 vs 归档/备份目录 ──────────────────
    # 关键视角：68 个淘汰如果全在归档/备份里，说明正式目录是干净的。
    # 只看总数会误以为"平台一堆垃圾"，分层后才看得出问题已被处理。
    df["归档"] = df["file"].str.contains(ARCH_RE)
    live = eff[~eff["file"].str.contains(ARCH_RE)]
    arch = eff[eff["file"].str.contains(ARCH_RE)]
    lc, ac = live["判定"].value_counts(), arch["判定"].value_counts()

    def _cnt(s, pfx):
        return int(sum(v for k, v in s.items() if k.startswith(pfx)))

    ldead, lwatch, lgood = _cnt(lc, "🔴"), _cnt(lc, "🟡"), _cnt(lc, "✅")
    adead, awatch, agood = _cnt(ac, "🔴"), _cnt(ac, "🟡"), _cnt(ac, "✅")
    print(f"\n  【正式结果目录】{len(live)} 个："
          f"🔴 {ldead}  🟡 {lwatch}  ✅ {lgood}")
    print(f"  【归档/备份目录】{len(arch)} 个："
          f"🔴 {adead}  🟡 {awatch}  ✅ {agood}")
    if ldead == 0 and adead:
        print(f"  → 全部 {adead} 个淘汰都落在历史副本/归档目录，"
              f"正式结果目录零淘汰（问题已处理，无需再动）")

    bad = eff[eff["判定"].str.startswith("🔴")]
    if len(bad):
        print(f"\n  淘汰清单（{len(bad)} 个，需处理或归档）：")
        for r in bad.head(30).itertuples():
            print(f"    {r.判定}  {r.file}")
        if len(bad) > 30:
            print(f"    ... 其余 {len(bad) - 30} 个见 CSV")

    # ── Markdown 报告 ─────────────────────────────────────
    with open(a.out_md, "w", encoding="utf-8") as f:
        f.write(f"# 平台流水体检报告\n\n生成时间：{ts} ｜ 范围：`{a.root}/{a.glob}` "
                f"｜ 闸门：{a.gate}\n\n")
        f.write(f"## 一句话结论\n\n共 **{n}** 个有效流水文件"
                f"（另有 {skip} 个非流水产物已跳过）："
                f"**🔴 淘汰 {dead}**（流水不自洽或含前视，结论不可用）｜"
                f"**🟡 观察 {watch}**（样本不足/涨停比失衡/活跃税超 6.5%）｜"
                f"**✅ 合格 {good}**。\n\n")
        f.write("## 判定规则\n\n| 级别 | 触发条件 |\n|---|---|\n")
        f.write("| 🔴 淘汰 | 闸门一流水不自洽（持仓负值/action 异常/分母为 0）——"
                "后续数据全是垃圾；或闸门二前视嫌疑（超额 >0.5% 且超额胜率 >60%）|\n")
        f.write(f"| 🟡 观察 | 闸门二 △（独立买入日 <20 或涨停跌停比失衡）；"
                f"或闸门三活跃税（对平均净值）> {B_O_TH:.1%} |\n")
        f.write("| ✅ 合格 | 三闸门全过 |\n\n")
        f.write("## 正式目录 vs 归档备份（关键分层）\n\n")
        f.write("判定总数会误导：淘汰若全在历史副本里，说明问题已处理过。\n\n")
        f.write("| 范围 | 有效流水 | 🔴 淘汰 | 🟡 观察 | ✅ 合格 |\n|---|---:|---:|---:|---:|\n")
        f.write(f"| **正式结果目录** | {len(live)} | **{ldead}** | {lwatch} | {lgood} |\n")
        f.write(f"| 归档/备份目录 | {len(arch)} | {adead} | {awatch} | {agood} |\n\n")
        if ldead == 0 and adead:
            f.write(f"> ✅ **全部 {adead} 个淘汰都落在历史副本/归档目录，"
                    f"正式结果目录零淘汰**——问题已处理，正式产物可信。\n")
        elif ldead:
            f.write(f"> ⚠️ **正式结果目录有 {ldead} 个淘汰，需要处理**（见下方清单，"
                    f"带 `_archive_`/`_bak_` 路径的可忽略）。\n")
        f.write("\n")
        # ── 策略族健康度（只看正式结果目录）────────────────────
        # 按 data/results/<族> 分组，回答"哪一族的产物能直接引用"。
        if len(live):
            lv = live.copy()

            def _fam(p):
                r = str(p).replace("data/results/", "")
                return r.split("/")[0] if "/" in r else "(根目录)"

            lv["族"] = lv["file"].map(_fam)
            lv["_tax"] = pd.to_numeric(lv["活跃税_对净值"], errors="coerce")

            def _n(s, pfx):
                return int(sum(1 for v in s if str(v).startswith(pfx)))

            rows = []
            for fam, gg in lv.groupby("族"):
                tx = gg["_tax"].dropna()
                rows.append({
                    "族": fam, "文件数": len(gg),
                    "🔴": _n(gg["判定"], "🔴"), "🟡": _n(gg["判定"], "🟡"),
                    "✅": _n(gg["判定"], "✅"),
                    "活跃税中位": f"{tx.median()*100:.2f}%" if len(tx) else "—",
                    "活跃税最大": f"{tx.max()*100:.2f}%" if len(tx) else "—",
                })
            fam_df = pd.DataFrame(rows).sort_values("文件数", ascending=False)
            f.write("## 策略族健康度（仅正式结果目录）\n\n")
            f.write("按 `data/results/<族>` 分组，回答「哪一族的产物能直接引用」。\n\n")
            f.write("| 策略族 | 文件数 | 🔴 淘汰 | 🟡 观察 | ✅ 合格 | 活跃税中位 | 活跃税最大 |\n")
            f.write("|---|---:|---:|---:|---:|---:|---:|\n")
            for r in fam_df.itertuples():
                f.write(f"| `{r.族}` | {r.文件数} | {r._3} | {r._4} | {r._5} | "
                        f"{r.活跃税中位} | {r.活跃税最大} |\n")
            f.write("\n")
            # 🟡 的成色：是"证据不足"还是"真有嫌疑"，两者处置方式完全不同
            # （注意：只能统计 🟡 那部分，把 ✅ 也算进来会让"其他"占大多数，看不出成色）
            wd = lv[lv["判定"].str.startswith("🟡")].copy()
            wd["理由"] = [
                "样本不足(买入日<20)" if "样本不足" in str(v) else
                ("涨停比失衡" if "涨停" in str(v) else "其他")
                for v in wd["前视v2"]
            ]
            wd.loc[pd.to_numeric(wd["活跃税_对净值"], errors="coerce") > B_O_TH,
                   "理由"] = "活跃税>6.5%"
            rc = wd["理由"].value_counts()
            f.write("**🟡 观察的成色**（处置方式完全不同）：\n\n")
            for k, v in rc.items():
                note = ""
                if k == "样本不足(买入日<20)":
                    note = "——**证据不足，不是嫌疑**；但意味着前视筛查对该族**判不了**，需靠别的手段验证"
                elif k == "活跃税>6.5%":
                    note = "——摩擦正在吃掉 alpha，是真问题"
                f.write(f"- {k}：**{int(v)}** 个{note}\n")
            f.write("\n")
        f.write("## 分闸门统计\n\n| 闸门 | 产出 | 说明 |\n|---|---|---|\n")
        f.write(f"| 一 流水自洽 | {'✅' if ok.get('1') else '❌'} | "
                f"不需要价格数据，零成本先筛全库 |\n")
        f.write(f"| 二 前视 v2 | {'✅' if ok.get('2') else '❌'} | "
                f"超额口径（扣同日全市场），样本门槛=独立买入交易日数 |\n")
        f.write(f"| 三 活跃税 | {'✅' if ok.get('3') else '未跑'} | "
                f"对平均净值口径，与 B&O 可比 |\n\n")
        if len(bad):
            f.write(f"## 淘汰清单（{len(bad)} 个）\n\n| 文件 | 判定 | 流水自洽 | 前视v2 |\n"
                    f"|---|---|---|---|\n")
            for r in bad.itertuples():
                f.write(f"| `{r.file}` | {r.判定} | {r.流水自洽} | {r.前视v2} |\n")
            f.write("\n")
        if ok.get("3"):
            tax = eff[pd.to_numeric(eff["活跃税_对净值"], errors="coerce") > B_O_TH]
            if len(tax):
                tax = tax.sort_values("活跃税_对净值", ascending=False)
                f.write(f"## 活跃税超阈值清单（{len(tax)} 个，> {B_O_TH:.1%}/年）\n\n")
                f.write("活跃税 = 总成本 /（平均净值 × 年数），**这是与 B&O 可比的那一路口径**"
                        "（另一路『对初始本金』会因净值增长而稀释，不可直接比）。\n\n")
                f.write("| 文件 | 活跃税(对净值) | 范围 | 判定 |\n|---|---:|---|---|\n")
                for r in tax.itertuples():
                    f.write(f"| `{r.file}` | {float(r.活跃税_对净值) * 100:.2f}% | "
                            f"{'归档' if ARCH_RE.search(str(r.file)) else '**正式**'} | {r.判定} |\n")
                f.write("\n")
                live_tax = int(sum(1 for p in tax["file"]
                                   if not ARCH_RE.search(str(p))))
                f.write(f"> 其中正式结果目录 **{live_tax} 个**、归档备份 "
                        f"{len(tax) - live_tax} 个。\n\n")
        f.write("## 复跑命令\n\n```bash\n")
        f.write(f"# 快速体检（闸门 1+2，约 3 分钟）\n")
        f.write(f"python run_platform_health_check.py --root {a.root} --gate 12\n\n")
        f.write(f"# 全量体检（含活跃税，约 25 分钟）\n")
        f.write(f"python run_platform_health_check.py --root {a.root} --gate 123\n```\n")
    print(f"\n明细：{a.out_csv}\n报告：{a.out_md}")

    for p in tmp.values():
        if os.path.exists(p):
            os.remove(p)


if __name__ == "__main__":
    main()
