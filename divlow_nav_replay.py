"""B7 缓冲 A/B —— 从持仓日志(partial sel_log)重建四档 NAV 曲线。

背景：run_official_backtest 落盘的 nav/sel 文件名**不含 buffer_k**（已修），
k=0/6/12/18 四档互相静默覆盖，只剩 k=18 一条 NAV。重跑 4 档需 ~2.7h，
而 partial sel_log 带 bk 后缀全部幸存 → 用同一套价格函数 + 同一个
run_nav_weighted 重放即可秒级还原。

自证：k=18 重放结果必须与 run2 日志的 63.79% / 7.82% / -17.76% 吻合，
否则本脚本不可用于其它 k（可能是价格表/权重口径不对）。
"""
import os
import sys

import numpy as np
import pandas as pd

import run_dividend_low_vol_quality_bt as E

RES = E.RES_DIR
START, END = E.START, E.END
TOP_N = 12
# 标签：纯数字=季度档 buffer_k；year/half/month=调仓频率档(buffer_k=0)
FREQS = {"month", "quarter", "half", "year"}
TAGS = sys.argv[1:] or ["0", "6", "12", "18", "36", "half", "year"]


def _mid(tag):
    """标签 → partial 文件名中段（🔴 查季度档时必须排除 _rb 频率档文件）"""
    if tag in FREQS:
        return "bk0" if tag == "quarter" else f"bk0_rb{tag}"
    return f"bk{tag}"


def _label(tag):
    return f"k={tag}" if tag not in FREQS else f"{tag}调仓"

# 引擎实跑公布值（_buffer_ab_run.log / _buffer_ab_run2.log / _buffer_ab_k36.log），用于自证
PUBLISHED = {
    "0": dict(total=49.26, ann=6.30, mdd=-13.07, vol=13.31),
    "18": dict(total=63.79, ann=7.82, mdd=-17.76, vol=13.31),
    "36": dict(total=71.11, ann=8.54, mdd=-16.77, vol=13.08),
    "year": dict(total=80.28, ann=9.41, mdd=-17.36, vol=14.55),
    "half": dict(total=45.27, ann=5.86, mdd=-17.31, vol=14.01),
}
# divlow_turnover_ab.py 实测（年化单边，资金口径）
TURNOVER = {"0": 142.3, "6": 129.8, "12": 117.2, "18": 103.7, "36": 82.2,
            "half": 100.2, "year": 51.1}


def load_sel(k):
    p = os.path.join(RES, f"_official_official_compact_all_{TOP_N}_{_mid(k)}_{START}_{END}_partial.csv")
    if not os.path.exists(p):
        print(f"  [MISS] {p}")
        return None, None, None
    df = pd.read_csv(p, dtype={"rebal_date": str, "ts_code": str}, encoding="utf-8-sig")
    df = df.sort_values(["rebal_date", "ts_code"])
    rbs = sorted(df["rebal_date"].unique())
    targets, wmap = [], {}
    for rb in rbs:
        sub = df[df["rebal_date"] == rb]
        codes = [str(c) for c in sub["ts_code"]]
        targets.append((rb, codes))
        wmap[str(rb)] = {str(c): float(w) for c, w in zip(sub["ts_code"], sub["weight"])}
    print(f"  partial: {len(rbs)} 期 / {len(df)} 行 / {df['ts_code'].nunique()} 只  ({rbs[0]}~{rbs[-1]})")
    return targets, wmap, df


def turnover_of(targets, wmap, top_n=TOP_N):
    """换手**实时**计算（替换原硬编码 TURNOVER 字典——那个只覆盖季度档 k）。

    🔴 换手有三个口径，数值能差 4 倍，拍板前必须说清跟哪个比：
       per_one_way = 0.5·Σ|w_t − w_{t−1}|            每期单边（资金口径，标准）
       ann_one_way = per_one_way × 期数/年             年化单边 ← B&O 摩擦、平台活跃税同口径
       per_two_way = per_one_way × 2                   每期双边 ← 纪要"72.3%"用的口径
    自证：季度档 k=0 → ann_one_way 应 ≈142.3%、per_two_way 应 ≈71.2%（≈纪要 72.3%）。
    """
    prev_n = prev_w = None
    per = []
    for rb, codes in targets:
        cur_n = set(codes)
        cur_w = wmap.get(str(rb), {})
        if prev_n is not None and prev_w is not None:
            kept = len(cur_n & prev_n)
            one_way_n = (top_n - kept) / top_n
            cs = set(cur_w) | set(prev_w)
            one_way_w = 0.5 * sum(abs(cur_w.get(c, 0.0) - prev_w.get(c, 0.0)) for c in cs)
            per.append((one_way_n, one_way_w))
        prev_n, prev_w = cur_n, cur_w
    if not per:
        return None
    rbs = [rb for rb, _ in targets]
    yr = (int(rbs[-1][:4]) - int(rbs[0][:4])) + (int(rbs[-1][4:6]) - int(rbs[0][4:6])) / 12
    n_per_year = len(per) / yr if yr > 0 else float("nan")
    wn = float(np.mean([p[0] for p in per]))
    ww = float(np.mean([p[1] for p in per]))
    return dict(n_per_year=n_per_year, per_one_way=ww, ann_one_way=ww * n_per_year,
                per_two_way=ww * 2, only_count=wn * n_per_year, n_period=len(per))


def _paired(nav_a, rb_a, nav_b, rb_b):
    """两条 NAV 在**调仓间隔**上的期收益（每期=两相邻调仓日），返回 (a, b) 两列 Series。
    只用调仓期收益而不用日收益：日收益重叠自相关会把 t 值放大，配对检验会假显著。"""
    sa = pd.Series([v for _, v in nav_a], index=[d for d, _ in nav_a], dtype=float)
    sb = pd.Series([v for _, v in nav_b], index=[d for d, _ in nav_b], dtype=float)
    # 🔴 用并集而非交集：年度档(12月)与季度档(1/4/7/10)的 rbs 交集可能为空
    rbs = sorted(set(rb_a) | set(rb_b))
    ra = (sa.reindex(rbs).ffill().pct_change().dropna())
    rb = (sb.reindex(rbs).ffill().pct_change().dropna())
    idx = ra.index.intersection(rb.index)
    return ra.reindex(idx), rb.reindex(idx)


def replay(k, pmap_cache):
    targets, wmap, _ = load_sel(k)
    if targets is None:
        return None
    codes = sorted({c for _, cs in targets for c in cs})
    pmap = E.bulk_close_prices(codes, START, END)
    if E.PRICE_MODE == "hfq":
        E.EXEC_PMAP.clear()
        E.EXEC_PMAP.update(E.bulk_open_prices(codes, START, END))
    all_dates = E.get_trade_dates(START, END)
    nav = E.run_nav_weighted(targets, wmap, pmap, all_dates, coef_fn=None)
    rbs = [rb for rb, _ in targets]
    # 🔴 **不要**截断到 rbs[0]：那会把**建仓日当天**的损益一起截掉，
    #    实测使总收益虚高 ~1.8pp（k=0: 49.26→51.08%，year: 80.28→82.49%）→ 自证必失败。
    #    引擎已给低频档插入"期初建仓日"(20200108)，各档起算一致，空仓段只剩
    #    START~建仓日 的 4 天（可忽略）。此处改为**告警**：若空仓段 >20 天才说明调仓日生成有问题。
    _gap = len([d for d, _ in nav if d < rbs[0]])
    if _gap > 20:
        print(f"  ⚠️ 建仓前有 {_gap} 天空仓（NAV 恒=初始资金，年化被稀释）"
              f" → 检查低频档是否漏插期初建仓日")
    nav_t = nav
    m = E.compute_metrics(nav_t, [d for d, _ in nav_t])
    return nav_t, m, [d for d, _ in nav_t], rbs


def main():
    E.PRICE_MODE = "hfq"          # 与 A/B 实跑一致
    out, TOF = {}, {}
    for k in TAGS:
        print(f"\n===== {_label(k)} =====")
        r = replay(k, None)
        if r is None:
            continue
        nav, m, all_dates, rbs = r
        out[k] = (nav, m, all_dates, rbs)
        _tg, _wm, _ = load_sel(k)      # partial 只有几百行，二次读成本可忽略
        TOF[k] = turnover_of(_tg, _wm) if _tg else None
        if TOF.get(k):
            print(f"  换手: 每期单边 {TOF[k]['per_one_way']*100:.1f}%  "
                  f"年化单边 {TOF[k]['ann_one_way']*100:.1f}%  "
                  f"每期双边 {TOF[k]['per_two_way']*100:.1f}%  ({TOF[k]['n_period']} 期)")
        flag = ""
        if k in PUBLISHED:
            p = PUBLISHED[k]
            ok = (abs(m["total_ret"] * 100 - p["total"]) < 0.15
                  and abs(m["ann"] * 100 - p["ann"]) < 0.05
                  and abs(m["max_dd"] * 100 - p["mdd"]) < 0.15)
            flag = "  ✅自证通过(与实跑吻合)" if ok else "  ❌自证失败(重放口径有问题，勿采信)"
            print(f"  [自证] 重放 {m['total_ret']*100:.2f}%/{m['ann']*100:.2f}%/{m['max_dd']*100:.2f}%"
                  f" vs 实跑 {p['total']:.2f}%/{p['ann']:.2f}%/{p['mdd']:.2f}%{flag}")
        print(f"  重放指标: 总收益 {m['total_ret']*100:.2f}%  年化 {m['ann']*100:.2f}%  "
              f"最大回撤 {m['max_dd']*100:.2f}%  波动 {m['vol']*100:.2f}%  "
              f"夏普 {m['sharpe']:.2f}  卡玛 {m['calmar']:.2f}")

    if len(out) < 2:
        return

    # ── 多档对照表 ──
    kind = "调仓频率" if set(TAGS) & FREQS else "B7 缓冲"
    print("\n" + "=" * 118)
    print(f"【{kind} A/B {len(out)} 档对照】官方编制法 compact / 全A / 持仓{TOP_N} / hfq / 无 overlay")
    print(f"  🔴 换手两口径别混用：『年化单边』=每期单边×期数/年（B&O/活跃税同口径）；"
          f"『每期双边』=每期单边×2（纪要 72.3% 用的口径）")
    print(f"{'档位':<12}{'期数':>6}{'总收益':>10}{'年化':>9}{'最大回撤':>11}{'年化波动':>10}"
          f"{'夏普':>7}{'卡玛':>7}{'年化单边':>11}{'每期双边':>11}{'换手降幅':>10}")
    k0 = TAGS[0]                       # 🔴 基线取第一个 tag，不是硬编码 "0"（频率档没有 k=0）
    t0 = (TOF.get(k0) or {}).get("ann_one_way")
    for k in TAGS:
        if k not in out:
            continue
        m = out[k][1]
        tk = TOF.get(k) or {}
        ann, two = tk.get("ann_one_way"), tk.get("per_two_way")
        cut = f"{(ann / t0 - 1) * 100:>9.1f}%" if (ann and t0) else f"{'  -  ':>10}"
        print(f"{_label(k):<12}{tk.get('n_period', 0):>6}{m['total_ret']*100:>9.2f}%{m['ann']*100:>8.2f}%"
              f"{m['max_dd']*100:>10.2f}%{m['vol']*100:>9.2f}%{m['sharpe']:>7.2f}"
              f"{m['calmar']:>7.2f}"
              f"{(f'{ann*100:.1f}%' if ann is not None else '  -  '):>11}"
              f"{(f'{two*100:.1f}%' if two is not None else '  -  '):>11}{cut}")
    print("=" * 118)
    # 自证：季度档 k=0 的实时换手必须复现 divlow_turnover_ab.py 的 142.3%
    if k0 in TURNOVER and t0:
        ok = abs(t0 * 100 - TURNOVER[k0]) < 1.0
        print(f"  [自证] {_label(k0)} 实时换手 年化单边 {t0 * 100:.1f}% vs turnover_ab 实测 "
              f"{TURNOVER[k0]}% → {'✅吻合' if ok else '❌不符，口径有变，勿采信'}")

    # ── NAV 日收益相关性：判定各档收益差是"机制"还是"噪声" ──
    rets = {}
    for k, (nav, m, all_dates, rbs) in out.items():
        s = pd.Series([v for _, v in nav], index=[d for d, _ in nav], dtype=float)
        rets[_label(k)] = s.pct_change()
    R = pd.DataFrame(rets).dropna()
    print(f"\n【各档日收益相关系数矩阵】（样本 {len(R)} 日）")
    print(R.corr().round(4).to_string())
    anns = [out[k][1]["ann"] * 100 for k in out]
    print(f"\n  年化极差 = {max(anns) - min(anns):.2f}pp（{min(anns):.2f}% ~ {max(anns):.2f}%）"
          f"   年化标准差 = {np.std(anns):.2f}pp")
    _cm = R.corr()          # 🔴 列标签对频率档是「half调仓」不是「k=half」，必须用 _label 取
    cors = [_cm.loc[_label(a), _label(b)] for i, a in enumerate(out) for b in list(out)[i + 1:]]
    print(f"  两两相关: min={min(cors):.4f} 中位={np.median(cors):.4f} max={max(cors):.4f}")

    # ── 逐年收益（🔴 yearly_returns 返回的已是百分数，勿再 ×100）──
    print("\n【逐年收益（各档 vs 中证红利低波 000922）】")
    ys = {}
    for k, (nav, m, all_dates, rbs) in out.items():
        ys[_label(k)] = E.yearly_returns([d for d, _ in nav], [v for _, v in nav])
    # 000922 逐年（取自 k=18 实跑落盘的 nav 文件，与 A/B 同口径）
    nb = os.path.join(RES, f"bt_quality_nav_{START}_{END}_official_compact_all_{TOP_N}_hfq.csv")
    y922 = {}
    if os.path.exists(nb):
        b = pd.read_csv(nb, dtype={"trade_date": str}, encoding="utf-8-sig")
        b = b.dropna(subset=["nav_922"])
        y922 = E.yearly_returns(b["trade_date"].tolist(), b["nav_922"].tolist())
    years = sorted(set().union(*[set(v) for v in ys.values()]))
    _klast = TAGS[-1]
    print(f"{'年份':<8}" + "".join(f"{c:>11}" for c in ys) + f"{'红利低波':>12}{f'k{_klast}-k0':>12}")
    for y in years:
        row = f"{y:<8}"
        vals = []
        for c in ys:
            v = ys[c].get(y)
            vals.append(v)
            row += f"{v:>10.2f}%" if v is not None else f"{'  -  ':>11}"
        v9 = y922.get(y)
        row += f"{v9:>11.2f}%" if v9 is not None else f"{'  -  ':>12}"
        a, b2 = vals[0], vals[-1]
        row += f"{b2 - a:>+9.2f}pp" if (a is not None and b2 is not None) else f"{'  -  ':>10}"
        print(row)

    # ── 配对检验：各档 vs k=0 的调仓期收益差是否显著 ──
    # 单次路径上 2pp/年的差，若不随参数单调、且配对 t 不显著 → 判噪声，不作为选档依据
    # 🔴 基线用 TAGS[0] 而非硬编码 "0"：频率档模式下没有 k=0 这个 key，
    #    原写法 `if "0" not in out: return` 会让频率档**静默跳过配对检验与 NAV 落盘**。
    print(f"\n【各档 vs {_label(k0)}(基线) 配对检验（调仓期收益差）】")
    print(f"{'档':<12}{'期数':>6}{'期收益差均值':>14}{'t 值':>9}{'判读':>16}")
    for k in TAGS:
        if k == k0 or k not in out:
            continue
        d0, dk = _paired(out[k0][0], out[k0][3], out[k][0], out[k][3])
        diff = dk - d0
        t = diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else float("nan")
        verdict = "显著(|t|>2)" if abs(t) > 2 else "不显著→噪声"
        print(f"{_label(k):<12}{len(diff):>6}{diff.mean()*100:>13.2f}%{t:>9.2f}{verdict:>16}")

    # ── 落盘各档 NAV ──
    base = None
    for k, (nav, m, all_dates, rbs) in out.items():
        s = pd.Series([v for _, v in nav], index=[d for d, _ in nav], dtype=float)
        if base is None:
            base = s
        p = os.path.join(RES, f"_ab_{_mid(k)}_nav_{START}_{END}.csv")
        s.to_frame(f"nav_{_mid(k)}").to_csv(p, encoding="utf-8-sig")
        print(f"\n  NAV → {p}")
    pd.DataFrame({f"k={k}": pd.Series([v for _, v in out[k][0]],
                                      index=[d for d, _ in out[k][0]], dtype=float)
                  for k in out}).to_csv(
        os.path.join(RES, f"_ab_allk_nav_{START}_{END}.csv"), encoding="utf-8-sig")


if __name__ == "__main__":
    main()
