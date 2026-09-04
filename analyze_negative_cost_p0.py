# -*- coding: utf-8 -*-
"""P0 元问题验证 —— 「负成本持仓」是心理账户效应，不是财富创造。

来源：B站 BV1G7466MEPN（悦悦笔记）声称 5 种方法可把持仓成本做成负数。
本脚本不评估那 5 法本身的收益，只证一个更根本的命题：

    ┌─────────────────────────────────────────────────────────┐
    │  「持仓成本价」是一个会计标签，不进入任何收益决定式。      │
    │  抽本 ≡ 减仓；成本价的下降精确等于已落袋利润的会计重摊。   │
    └─────────────────────────────────────────────────────────┘

数学核心（解析，无需数据）：
    设初始投入 C0，买入 N0 股 @ K（C0 = N0·K）。
    在价格 P1 卖出 m 股，回收 m·P1，剩余 N1 = N0 − m 股。
    券商口径成本价 = (C0 − m·P1) / N1

    成本价下降量 ΔK = K − (C0 − m·P1)/N1 = m·(P1 − K) / (N0 − m)
    ⟹ ΔK × N1 = m·(P1 − K) = **已实现盈利**

    即：成本价的下降 × 剩余股数 ≡ 已落袋的利润。
    它不是一个新增的财富来源，只是把已实现盈利在会计上重新分摊。

    而总财富 W(t) = shares(t)·P(t) + cash(t)，式中**不出现成本价**。

三个层次：
    L1a 构造性演示：同一时刻、同一价格，成本价可被任意调节（+10 / 0 / −20），
                    而总财富完全相同 → 成本价是纯标签（最硬的一枪）
    L1b 恒等式数值验证：真实序列逐日验证 ΔK×N1 = 已实现盈利，且减仓瞬间
                    财富不变（税前）/ 严格减少（税后成本）
    L2  真实案例：zz800 成分股上复刻视频原规则（涨100%卖50%、再涨25%卖20%），
                    展示「成本价为负」与「总财富更少」并存
    L3  大样本统计：全 zz800 历史成分，抽本 vs 纯持有的财富分布对比

口径：
    - 价格用 hfq（后复权）。raw 会把分红除权误计为下跌，让"落袋为安"看起来正确
      （见 exit-rule-event-study 技能：红利低波案例 raw/hfq 结论符号反转）
    - 取数复用 run_daily20_macd.load_closes（已处理 adj_factor 缺行 ffill
      与首因子归一化两个坑）
    - 成本 RT_COST 单边 0.3%（佣金+滑点+印花），只在卖出侧计

用法：
    python analyze_negative_cost_p0.py                    # 默认 zz800 / 2010-2026
    python analyze_negative_cost_p0.py --step 10 --horizon 250
    python analyze_negative_cost_p0.py --probe            # 小样本快跑
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from run_daily20_macd import load_closes

RT_COST = 0.003          # 单边成本（佣金+滑点+印花），仅卖出侧计
HORIZON = 250            # 事件持有期（交易日）
STEP = 20                # 买入起点采样步长（控制样本重叠）
OUT_DIR = os.path.join("data", "results", "negative_cost")


# ══════════════════════════════════════════════════════════════
# L1a  构造性演示：成本价任意，财富相同
# ══════════════════════════════════════════════════════════════

def demo_cost_is_label(k=10.0, n0=10000.0, p_now=20.0):
    """同一时刻、同一价格 P=20，不同减仓比例下：成本价天差地别，总财富分文不差。

    关键：卖出价 = 当前价 = 20，所以卖出这个动作本身既不创造也不毁灭财富。
    """
    c0 = k * n0                      # 初始投入 10 万
    rows = []
    for label, sell_frac in [("B 纯持有（不卖）", 0.0),
                             ("A1 卖 25%", 0.25),
                             ("A2 卖 50%（视频翻倍抽本）", 0.50),
                             ("A3 卖 75%", 0.75)]:
        m = n0 * sell_frac
        cash = m * p_now
        n1 = n0 - m
        cost = (c0 - cash) / n1 if n1 > 0 else float("nan")
        wealth = n1 * p_now + cash
        rows.append({
            "路径": label,
            "卖出股数": m,
            "剩余股数": n1,
            "落袋现金": cash,
            "持仓成本价": cost,
            "总财富": wealth,
        })
    return pd.DataFrame(rows), c0


# ══════════════════════════════════════════════════════════════
# L1b  恒等式数值验证（真实序列，逐日）
# ══════════════════════════════════════════════════════════════

def simulate_path(px, k, n0=1.0, mult1=2.0, sell1=0.50,
                  mult2=2.5, sell2=0.20, cost=RT_COST, record=False):
    """在真实价格序列上模拟「视频原规则」抽本路径，返回终态与逐日轨迹。

    规则：涨到 mult1×K 卖 sell1；再涨到 mult2×K 卖 sell2（剩余股份的比例）。
    归一化为买 n0=1 股，初始投入 = K。

    返回 dict：
        wealth_trim  终值财富（剩股市值 + 现金，已扣卖出成本）
        wealth_hold  终值财富（纯持有 = 1 股 × 末价）
        cost_basis   终值券商口径成本价
        realized     已实现盈利
        以及逐日轨迹（record=True 时）
    """
    shares = n0
    cash = 0.0
    invested = k * n0
    realized = 0.0
    stage = 0                       # 0=未触发 1=已卖第一次 2=已卖第二次
    traj = []

    for t, p in enumerate(px):
        if not np.isfinite(p) or p <= 0:
            continue
        # 触发判定（按当根 K 线收盘成交，避免用未来价）
        if stage == 0 and p >= mult1 * k:
            m = shares * sell1
            proceeds = m * p * (1 - cost)
            # 口径必须与 cost_basis 一致：成本价按【实际到手现金】算，
            # 故 realized 也必须是【净】已实现盈利，否则恒等式差一个 cost/(1-cost) 因子
            realized += proceeds - m * k
            cash += proceeds
            shares -= m
            stage = 1
        elif stage == 1 and p >= mult2 * k:
            m = shares * sell2
            proceeds = m * p * (1 - cost)
            realized += proceeds - m * k
            cash += proceeds
            shares -= m
            stage = 2

        if record:
            w = shares * p + cash
            cb = (invested - cash) / shares if shares > 0 else float("nan")
            traj.append((t, p, shares, cash, cb, w, realized))

    p_end = px[-1] if np.isfinite(px[-1]) else np.nan
    wealth_trim = shares * p_end + cash
    wealth_hold = n0 * p_end
    cost_basis = (invested - cash) / shares if shares > 0 else float("nan")

    out = {
        "wealth_trim": wealth_trim,
        "wealth_hold": wealth_hold,
        "cost_basis": cost_basis,
        "realized": realized,
        "shares_left": shares,
        "cash": cash,
        "stage": stage,
        "p_end": p_end,
    }
    if record:
        out["traj"] = pd.DataFrame(
            traj, columns=["t", "price", "shares", "cash", "cost_basis", "wealth", "realized"])
    return out


def verify_identity():
    """L1b：用真实序列逐日验证两条恒等式。

    ①  Δcost × 剩余股数 = 已实现盈利（成本价下降 ≡ 落袋利润的会计重摊）
    ②  减仓当日，税前财富不变；扣成本后严格减少
    """
    codes, closes = load_closes(hfq=True)
    # 挑一只长期上涨、能触发两次减仓的票做演示
    checks = []
    picked = None
    for ts in codes:
        s = closes[ts].dropna()
        if len(s) < 600:
            continue
        k = float(s.iloc[0])
        if not np.isfinite(k) or k <= 0:
            continue
        px = s.values
        r = simulate_path(px, k, record=True)
        if r["stage"] >= 2:
            picked = ts
            break
    if picked is None:
        print("  [L1b] 未找到能触发两次减仓的样本，跳过")
        return None

    s = closes[picked].dropna()
    k = float(s.iloc[0])
    r = simulate_path(s.values, k, record=True)
    tr = r["traj"]

    # ① 终态恒等式：Δcost × 剩余股数 vs 已实现盈利
    dk = k - r["cost_basis"]
    lhs = dk * r["shares_left"]
    rhs = r["realized"]
    ident_err = abs(lhs - rhs)

    # ② 减仓当日财富变化：找 shares 发生跳变的日子
    jumps = tr.index[tr["shares"].diff() < 0].tolist()
    jump_rows = []
    for j in jumps:
        prev_w = tr["wealth"].iloc[j - 1]
        cur_w = tr["wealth"].iloc[j]
        prev_p = tr["price"].iloc[j - 1]
        cur_p = tr["price"].iloc[j]
        d_sell = tr["shares"].iloc[j - 1] - tr["shares"].iloc[j]
        gross = d_sell * cur_p
        # 税前理论变化：持有至今 vs 卖出后 —— 仅因价格变动，与卖出无关
        jump_rows.append({
            "减仓日序": int(tr["t"].iloc[j]),
            "价格": cur_p,
            "卖出股数": d_sell,
            "税前财富变化(剔除价格涨跌)": (cur_w - prev_w) - (tr["shares"].iloc[j - 1] * (cur_p - prev_p)),
            "理论值(=-成本)": -gross * RT_COST,
        })

    print("  [L1b] 样本 %s  初始价 %.2f  终态成本价 %.4f  剩余股数 %.4f"
          % (picked, k, r["cost_basis"], r["shares_left"]))
    print("        恒等式 Δcost×剩余股数 = 已实现盈利 ："
          "左 %.6f  右 %.6f  误差 %.2e  → %s"
          % (lhs, rhs, ident_err, "成立" if ident_err < 1e-9 else "不成立"))
    if jump_rows:
        jd = pd.DataFrame(jump_rows)
        max_err = (jd["税前财富变化(剔除价格涨跌)"] - jd["理论值(=-成本)"]).abs().max()
        print("        减仓日财富变化（剔除价格涨跌后）理论 = −成本："
              "最大误差 %.2e → %s" % (max_err, "成立" if max_err < 1e-9 else "不成立"))
    return {"ts": picked, "ident_err": ident_err, "res": r}


# ══════════════════════════════════════════════════════════════
# L3  大样本事件研究
# ══════════════════════════════════════════════════════════════

def run_events(closes, codes, horizon=HORIZON, step=STEP, probe=False,
               mult1=2.0, sell1=0.50, mult2=2.5, sell2=0.20):
    """对每只票、每个采样起点，跑「抽本 vs 纯持有」对照。

    起点采样步长 step 控制事件重叠；horizon 为持有期（交易日）。
    """
    rows = []
    n_codes = 40 if probe else len(codes)
    for ci, ts in enumerate(codes[:n_codes]):
        s = closes[ts]
        if s is None:
            continue
        s = s.dropna()
        px = s.values
        dates = s.index.values
        T = len(px)
        if T < horizon + 5:
            continue
        for i in range(0, T - horizon - 1, step):
            k = px[i]
            if not np.isfinite(k) or k <= 0:
                continue
            seg = px[i + 1: i + 1 + horizon]
            if len(seg) < horizon or not np.isfinite(seg[-1]):
                continue
            r = simulate_path(seg, k, mult1=mult1, sell1=sell1,
                              mult2=mult2, sell2=sell2)
            if not np.isfinite(r["wealth_trim"]) or not np.isfinite(r["wealth_hold"]):
                continue
            rows.append({
                "ts_code": ts,
                "start_date": dates[i],
                "end_date": dates[i + horizon],
                "k": k,
                "p_end": r["p_end"],
                "stage": r["stage"],
                "shares_left": r["shares_left"],
                "cash": r["cash"],
                "realized": r["realized"],
                "cost_basis": r["cost_basis"],
                "cost_basis_pct": r["cost_basis"] / k - 1.0,
                "ret_trim": r["wealth_trim"] / k - 1.0,
                "ret_hold": r["wealth_hold"] / k - 1.0,
            })
        if probe and ci >= 39:
            break
    return pd.DataFrame(rows)


def _winsor(x, p=0.01):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return x
    lo, hi = np.quantile(x, p), np.quantile(x, 1 - p)
    return np.clip(x, lo, hi)


def _tstat(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan")
    sd = x.std(ddof=1)
    return float(x.mean() / (sd / np.sqrt(len(x)))) if sd > 0 else float("nan")


def _fmt(df, cols, nd=4):
    """数值列定宽格式化。

    pandas 3.0 的 to_string(float_format=...) 在 DataFrame 含字符串列时
    会对整表按 object 处理、格式化失效（实测打印出字面量 "%.4f"），故手动转。
    """
    d = df[cols].copy()
    for c in cols:
        if c == "分组":
            continue
        d[c] = pd.to_numeric(d[c], errors="coerce").map(
            lambda v: ("%.*f" % (nd, v)) if np.isfinite(v) else "nan")
    return d.to_string(index=False)


def bucket_table(ev, title=""):
    """按【后续涨跌】分桶：验证抽本的相对表现是否由后续走势机械决定。"""
    buckets = [(None, 0.0, "后续下跌 (≤0)"),
               (0.0, 1.0, "小涨 (0~100%)"),
               (1.0, 3.0, "大涨 (100~300%)"),
               (3.0, None, "暴涨 (>300%)")]
    rows = []
    for lo, hi, lab in buckets:
        m = (ev["ret_hold"] > lo) if lo is not None else (ev["ret_hold"] <= 0)
        if hi is not None:
            m = m & (ev["ret_hold"] <= hi)
        sub = ev[m]
        if len(sub) < 5:
            continue
        rows.append(summarize(sub, lab))
    if rows:
        if title:
            print(title)
        print(_fmt(pd.DataFrame(rows),
                   ["分组", "样本数", "持有收益_中位(%)", "抽本收益_中位(%)",
                    "抽本-持有 差值_中位数(%)", "抽本跑赢比例(%)"], nd=2))
        print()
    return


def summarize(df, label=""):
    """肥尾下必报：中位数 / 缩尾均值 / 胜率 / t 值（对齐 exit-rule-event-study）。"""
    if df.empty:
        return None
    d = df["ret_trim"] - df["ret_hold"]
    dw = _winsor(d.values)
    out = {
        "样本数": len(df),
        "抽本-持有 差值_中位数(%)": np.median(d) * 100,
        "抽本-持有 差值_缩尾均值(%)": dw.mean() * 100,
        "抽本跑赢比例(%)": (d > 0).mean() * 100,
        "t值(原始)": _tstat(d.values),
        "t值(缩尾1%)": _tstat(dw),
        "抽本收益_中位(%)": np.median(df["ret_trim"]) * 100,
        "持有收益_中位(%)": np.median(df["ret_hold"]) * 100,
    }
    if label:
        out["分组"] = label
    return out


# ══════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=STEP)
    ap.add_argument("--horizon", type=int, default=HORIZON)
    ap.add_argument("--probe", action="store_true", help="小样本快跑")
    ap.add_argument("--compare", action="store_true", help="追加 raw 口径对照（自证清单第3条）")
    args = ap.parse_args()

    t_all = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 72)
    print("P0 元问题验证：「负成本」是心理账户效应吗？")
    print("=" * 72)

    # ── L1a 构造性演示 ──
    print("\n【L1a】构造性演示：同一时刻、同一价格，成本价任意调节而总财富不变")
    print("-" * 72)
    print("设定：10 元买入 1 万股（投入 10 万），当前价 20 元，按不同比例卖出")
    print("（卖出价 = 当前价 = 20，故卖出这个动作本身既不创造也不毁灭财富）\n")
    demo, c0 = demo_cost_is_label()
    disp = demo.copy()
    for c in ["卖出股数", "剩余股数", "落袋现金", "总财富"]:
        disp[c] = disp[c].map(lambda v: "%.0f" % v)
    disp["持仓成本价"] = disp["持仓成本价"].map(lambda v: "%.2f" % v)
    print(disp.to_string(index=False))
    print("\n→ 成本价从 %+.2f 一路调到 %+.2f（甚至为负），而总财富列**完全相同**（%.0f）。"
          % (demo["持仓成本价"].iloc[0], demo["持仓成本价"].iloc[-1], demo["总财富"].iloc[0]))
    print("→ 结论：成本价不进入财富决定式，它是纯会计标签。")

    # ── L1b 恒等式数值验证 ──
    print("\n【L1b】恒等式数值验证（真实 zz800 序列，逐日）")
    print("-" * 72)
    verify_identity()

    # ── L3 大样本 ──
    print("\n【L3】大样本事件研究：抽本 vs 纯持有")
    print("-" * 72)
    print("[load] 取 zz800 历史成分收盘（hfq）...", flush=True)
    t0 = time.time()
    codes, closes = load_closes(hfq=True)
    print("       %d 只，耗时 %.1fs" % (len(codes), time.time() - t0), flush=True)

    t0 = time.time()
    df = run_events(closes, codes, horizon=args.horizon, step=args.step, probe=args.probe)
    print("       事件数 %d，耗时 %.1fs" % (len(df), time.time() - t0), flush=True)

    if df.empty:
        print("无有效事件")
        return

    # 自证：stage0（从未翻倍）两条路径必须完全相同，差异恒为 0
    print("\n── 自证组：stage0 从未翻倍 ──")
    s0 = df[df["stage"] == 0]
    if not s0.empty:
        d0 = (s0["ret_trim"] - s0["ret_hold"]).abs().max()
        print("  样本 %d 条。两条路径完全相同，差值绝对值最大 = %.2e → %s"
              % (len(s0), d0, "通过（差异来自取数噪音则此步会暴露）" if d0 < 1e-12 else "异常！"))
        print("  占比 %.1f%% —— 说明翻倍是稀有事件，绝不能混进全样本统计" % (100 * len(s0) / len(df)))

    # 结论组：只统计真实触发过减仓的事件
    ev = df[df["stage"] > 0].copy()
    ev["diff"] = ev["ret_trim"] - ev["ret_hold"]
    print("\n── 结论组：实际触发过减仓的事件（stage>0，n=%d）──" % len(ev))
    s_ev = summarize(ev, "触发减仓")
    for k, v in s_ev.items():
        if k == "分组":
            continue
        print("  %-26s %s" % (k, ("%.4f" % v) if isinstance(v, float) else v))

    # 按减仓档位分组
    print("\n── 按减仓档位分组 ──")
    rows = []
    for st, lab in [(1, "stage1 翻倍卖50%（成本价≈0）"),
                    (2, "stage2 再涨25%卖20%（成本价为负）")]:
        sub = ev[ev["stage"] == st]
        if sub.empty:
            continue
        rows.append(summarize(sub, lab))
    if rows:
        print(_fmt(pd.DataFrame(rows),
                   ["分组", "样本数", "抽本-持有 差值_中位数(%)", "抽本-持有 差值_缩尾均值(%)",
                    "抽本跑赢比例(%)", "t值(原始)", "t值(缩尾1%)"]))

    # 核心检验：抽本 ≡ 减仓。若成立，则抽本的相对表现必须完全由【后续涨跌】决定
    print("\n── 核心检验：抽本 ≡ 减仓，故表现应由后续涨跌决定 ──")
    print("  理论：抽本在高位减了仓 → 后续越涨，抽本越吃亏；后续下跌，抽本占便宜。")
    print("  若此规律成立而【成本价】不出现在任何解释项里，即证它是纯标签。\n")
    bucket_table(ev, "【全部触发减仓的事件】")

    # 分层：stage1「整体跑赢」是规则有效，还是样本构成造成的假象？
    print("── 分层检验：stage1 的「整体跑赢」是规则有效还是后视偏差？──")
    print("  ⚠ stage1 的定义本身就包含「后来没涨到 2.5K」→ 天然富集后续回落的样本。")
    print("  若在每个 stage 内部、按后续涨跌分桶后规律一致，")
    print("  则 stage 之间的整体差异只是【样本构成效应】，不是规则本身的功效。\n")
    for st, lab in [(1, "【stage1 翻倍卖50%】"), (2, "【stage2 再涨25%卖20%】")]:
        sub = ev[ev["stage"] == st]
        if len(sub) >= 20:
            bucket_table(sub, lab)

    # 成本价的解释力（连续变量相关性）
    print("\n── 成本价水平的解释力 ──")
    s2 = ev[ev["stage"] == 2]
    if len(s2) > 10:
        d = s2["diff"].values
        cb = s2["cost_basis_pct"].values
        mask = np.isfinite(d) & np.isfinite(cb)
        if mask.sum() > 10:
            rho, p = spearmanr(cb[mask], d[mask])
            # 对照：后续涨幅的解释力
            fh = s2["ret_hold"].values
            m2 = np.isfinite(d) & np.isfinite(fh)
            rho2, p2 = spearmanr(fh[m2], d[m2])
            print("  Spearman(成本价水平, 抽本超额) = %+.4f  (p=%.3g, n=%d)" % (rho, p, mask.sum()))
            print("  Spearman(后续涨幅  , 抽本超额) = %+.4f  (p=%.3g, n=%d)" % (rho2, p2, m2.sum()))
            print("  → 若前者弱而后者强，即证定价权在【减仓后走势】，不在成本价。")
            print("  注：成本价与超额即便相关也是机械的 —— 成本价由减仓时点/价格决定，")
            print("      它只是减仓强度的代理变量，不是收益的驱动因子。")

    # ── raw 口径对照（exit-rule-event-study 自证清单第 3 条）──
    if args.compare:
        print("\n── raw / hfq 双口径对照 ──")
        print("  自证清单要求双跑：红利类 raw vs hfq 结论可能【符号反转】")
        print("  （红利低波案例：raw 下 +8.94pp，hfq 下 −3.71pp）。")
        print("  本实验的 L1 恒等式不依赖复权，但 L3「抽本 vs 持有」会。\n")
        _, closes_r = load_closes(hfq=False)
        df_r = run_events(closes_r, codes, horizon=args.horizon,
                          step=args.step, probe=args.probe)
        df_r.to_csv(os.path.join(OUT_DIR, "p0_events_%s_raw.csv" % time.strftime("%Y%m%d")),
                    index=False, encoding="utf-8-sig")
        cmp_rows = []
        for lab, d_ in [("hfq 后复权（主口径）", df), ("raw 不复权（对照）", df_r)]:
            e_ = d_[d_["stage"] > 0]
            if e_.empty:
                continue
            r = summarize(e_, lab)
            r["触发减仓事件数"] = len(e_)
            r["触发率(%)"] = 100.0 * len(e_) / len(d_)
            cmp_rows.append(r)
        if cmp_rows:
            print(_fmt(pd.DataFrame(cmp_rows),
                       ["分组", "触发减仓事件数", "触发率(%)",
                        "抽本-持有 差值_中位数(%)", "抽本-持有 差值_缩尾均值(%)",
                        "抽本跑赢比例(%)", "t值(原始)", "t值(缩尾1%)"]))
            print("\n  注：raw 下分红除权被计为下跌 → 持有路径被系统性削平、且翻倍更难触发，")
            print("      故两个口径的【触发率】与【差值】都不同。核心结论看 hfq。")

    # 落盘
    stamp = time.strftime("%Y%m%d")
    out_csv = os.path.join(OUT_DIR, "p0_events_%s%s.csv" % (stamp, "_probe" if args.probe else ""))
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("\n逐事件明细已写入：%s" % out_csv)
    print("总耗时 %.1fs" % (time.time() - t_all))


if __name__ == "__main__":
    main()
