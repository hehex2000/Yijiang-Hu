# -*- coding: utf-8 -*-
"""
run_downside_switch.py —— 《波段操作法》衍生：下行开关叠加方案
=============================================================================
背景：run_swing_trend.py 真实数据结论——波段策略在两只票上均跑输"一直持有"，
      但【下降段超额 +121pp(茅台)/+124pp(平安)】高度一致稳定。
      即：这套信号作为"独立多空切换"不成立，作为"下行保险"有价值。

方案：底仓 = 一直持有（适合长期看好标的、只求躲大熊市的持有者）。
      开关 = 沿用波段策略的同一组【已定死参数】，不新增任何数字：
        避险离场：dn_cond(20日动量<=-5% 且 高低点双双下移) 连续 K_EXIT=3 日
                  → 次日开盘全仓卖出避险（与波段策略出场规则完全相同）
        回补条件试两版对比（结构性差异，非调参）：
          RULE_A 完整确认版：up_cond 连续 D_CONFIRM=3 日（与波段策略入场相同）
                              —— 保宔回补，转向段可能踏空
          RULE_B 快速回补版：20日动量>0 且 收盘>20日均线
                              —— 复用已有窗口 N_MOM=20，不加新数字；
                                 从"新趋势确认"改为"下跌风险解除"，语义变化

      场景定位：底仓持有者的尾部保险，而非择时增强。评价标准：
        1) maxDD 是否显著低于一直持有（核心卖点）
        2) 年化代价多少（保费）
        3) 避险明细：每次躲掉的跌幅 / 踏空的涨幅，原样记录（错误分类不藏）

回测正确性：与 run_swing_trend.py 同口径——t日收盘信号、t+1开盘执行(含滑点)、
  复权价、涨跌停/停牌顺延<=3日失败记 skipped、夏普减 rf。
"""
import sys
import numpy as np
import pandas as pd

from run_swing_trend import (  # 全部复用，参数一处定死不再改
    N_MOM, K_EXIT, D_CONFIRM, EXEC_WAIT_MAX, LIMIT_PCT, RF_ANN,
    COMMISSION, STAMP_TAX, SLIPPAGE, INITIAL, START_LOOK, SYMBOLS,
    compute_conditions, load_symbol, perf, label_regimes, _try_exec,
)

RULES = ["A", "B"]   # A=完整确认回补, B=快速回补


def run_switch(df: pd.DataFrame, rule: str, initial: float = INITIAL) -> dict:
    """
    下行开关状态机。初始 LONG（首日开盘买入），规则：
      dn_cond 连续 K_EXIT 日 → 次日开盘卖出避险
      rule=="A": up_cond 连续 D_CONFIRM 日 → 次日开盘回补
      rule=="B": 20日动量>0 且 close>MA20 → 次日开盘回补
    """
    n = len(df)
    o = df["open"].values.astype(float)
    c = df["close"].values.astype(float)
    dn = df["dn_cond"].values.astype(bool)
    up = df["up_cond"].values.astype(bool)
    mom = df["mom"].values.astype(float)
    ma20 = pd.Series(c, index=df.index).rolling(N_MOM).mean().values
    dates = df.index

    # 首日开盘建仓（净口径）
    sh = initial / (o[0] * (1 + SLIPPAGE) * (1 + COMMISSION)) if o[0] == o[0] and o[0] > 0 else 0.0
    cash = 0.0
    state = "LONG" if sh > 0 else "FLAT"
    pending = None
    dn_st = up_st = 0
    hedges = []           # 每次避险记录
    skipped = []
    eq = np.full(n, np.nan)
    in_mkt = np.zeros(n, dtype=bool)   # 每日是否持仓（净持仓口径）
    cur = None            # 当前避险段

    for i in range(n):
        # ---- A. 执行挂单（今日开盘）----
        if pending is not None:
            ok, why = _try_exec(i, o, c, pending["act"])
            if ok:
                if pending["act"] == "sell":
                    px = o[i] * (1 - SLIPPAGE)
                    cash = sh * px * (1 - COMMISSION - STAMP_TAX)
                    sh = 0.0
                    state = "FLAT"
                    cur = dict(sell_i=i, sell_date=dates[i])
                else:  # buy 回补
                    px = o[i] * (1 + SLIPPAGE)
                    sh = cash / (px * (1 + COMMISSION))
                    cash = 0.0
                    state = "LONG"
                    if cur is not None:
                        cur["buy_i"] = i
                        cur["buy_date"] = dates[i]
                        cur["days"] = i - cur["sell_i"]
                        # 躲掉的收益：避险期间若继续持有的收益（负=躲掉下跌=成功，
                        # 正=踏空上涨=失败）。用收盘价口径衡量市场，与执行价无关。
                        cur["avoided"] = c[i] / c[cur["sell_i"]] - 1
                        hedges.append(cur)
                        cur = None
                pending = None
            else:
                pending["age"] += 1
                if pending["age"] > EXEC_WAIT_MAX:
                    skipped.append(dict(date=dates[i], act=pending["act"], reason=why))
                    pending = None

        # ---- B. 收盘后信号（无未来函数）----
        up_st = up_st + 1 if up[i] else 0
        dn_st = dn_st + 1 if dn[i] else 0
        if pending is None:
            if state == "LONG" and dn_st >= K_EXIT:
                pending = dict(act="sell", age=0)
            elif state == "FLAT":
                if rule == "A" and up_st >= D_CONFIRM:
                    pending = dict(act="buy", age=0)
                elif rule == "B" and mom[i] == mom[i] and ma20[i] == ma20[i] \
                        and mom[i] > 0 and c[i] > ma20[i]:
                    pending = dict(act="buy", age=0)

        # ---- C. 收盘估值 ----
        eq[i] = cash + sh * c[i]
        in_mkt[i] = sh > 0

    # 期末若仍避险中：原样记录，avoided 用末日收盘
    if cur is not None:
        cur["buy_i"] = n - 1
        cur["buy_date"] = dates[n - 1] + "(期末未回补)"
        cur["days"] = (n - 1) - cur["sell_i"]
        cur["avoided"] = c[n - 1] / c[cur["sell_i"]] - 1
        hedges.append(cur)

    nav = pd.Series(eq, index=dates, name=f"nav_switch_{rule}").dropna()
    return dict(nav=nav, hedges=hedges, skipped=skipped,
                time_in_mkt=float(in_mkt.mean()))


def report_switch(code: str, df: pd.DataFrame, hold_nav: pd.Series, res: dict, rule: str):
    m = perf(res["nav"])
    mh = perf(hold_nav)
    hedges = res["hedges"]
    print(f"\n--- 下行开关 RULE_{rule}（{'up_cond确认3日回补' if rule == 'A' else '动量>0且站上20日线回补'}） ---")
    print(f"总收益 {m['tot']*100:8.1f}%  年化 {m['cagr']*100:6.2f}%  "
          f"最大回撤 {m['mdd']*100:7.2f}%  夏普 {m['sharpe']:5.2f}  "
          f"在场比例 {res['time_in_mkt']*100:5.1f}%  避险 {len(hedges)} 次")
    print(f"vs 一直持有 : 年化差 {(m['cagr']-mh['cagr'])*100:+6.2f}pp  "
          f"回撤差 {(m['mdd']-mh['mdd'])*100:+6.2f}pp（负=回撤更低）")
    if hedges:
        ok = [h for h in hedges if h["avoided"] < 0]   # 躲掉下跌
        bad = [h for h in hedges if h["avoided"] >= 0]  # 踏空
        print(f"避险明细（原样记录，不删除）：成功 {len(ok)} 次 / 踏空 {len(bad)} 次")
        print(f"{'卖出日':>9s}{'回补日':>17s}{'避险天数':>6s}{'期间市场':>9s}  归因")
        for h in sorted(hedges, key=lambda x: x["sell_i"]):
            tag = "躲掉下跌" if h["avoided"] < 0 else "踏空上涨"
            print(f"{h['sell_date']:>9s}{str(h['buy_date']):>17s}{h['days']:6d}"
                  f"{h['avoided']*100:8.1f}%  {tag}")
        if ok:
            print(f"成功避险平均躲掉 {np.mean([h['avoided'] for h in ok])*100:.1f}%；"
                  f"平均避险天数 {np.mean([h['days'] for h in ok]):.0f}")
        if bad:
            print(f"踏空平均代价 {np.mean([h['avoided'] for h in bad])*100:.1f}%；"
                  f"平均空仓天数 {np.mean([h['days'] for h in bad]):.0f}")
    for s in res["skipped"]:
        print(f"  未执行 {s['date']} {s['act']}: {s['reason']}")


def main():
    import sqlite3
    import config
    con = sqlite3.connect(config.DATA["local_db_path"])
    for code in SYMBOLS:
        raw = load_symbol(con, code)
        if len(raw) < 300:
            print(f"[skip] {code} 数据不足")
            continue
        cond = compute_conditions(raw)
        warm = max(N_MOM, 25)
        cond = cond.iloc[warm:]
        dates = cond.index
        c = cond["close"].values.astype(float)

        # 基准：一直持有（同口径：首日开盘买入含成本，期末收盘市值）
        o = cond["open"].values.astype(float)
        sh_h = INITIAL / (o[0] * (1 + SLIPPAGE) * (1 + COMMISSION))
        hold_nav = pd.Series(sh_h * c, index=dates, name="nav_hold")

        print(f"\n================ 下行开关叠加 {code} ================")
        print(f"区间 {dates[0]} ~ {dates[-1]}  交易日 {len(dates)}")
        mh = perf(hold_nav)
        print(f"一直持有   : 总收益 {mh['tot']*100:8.1f}%  年化 {mh['cagr']*100:6.2f}%  "
              f"最大回撤 {mh['mdd']*100:7.2f}%  夏普 {mh['sharpe']:5.2f}")

        # 原波段策略作参照（同一组参数的独立择时版）
        from run_swing_trend import run_swing_backtest, classify_exits
        swing = run_swing_backtest(cond)
        ms = perf(swing["nav_net"])
        print(f"波段策略参照: 总收益 {ms['tot']*100:8.1f}%  年化 {ms['cagr']*100:6.2f}%  "
              f"最大回撤 {ms['mdd']*100:7.2f}%  夏普 {ms['sharpe']:5.2f}")

        results = {}
        for rule in RULES:
            res = run_switch(cond, rule)
            results[rule] = res
            report_switch(code, cond, hold_nav, res, rule)

        # 分阶段对比：持有 vs 两版开关
        regimes = label_regimes(cond)
        print("\n--- 跨阶段验证（年化收益）---")
        rows = []
        rn = {r: results[r]["nav"].pct_change() for r in RULES}
        for st in ["上升段", "下降段", "震荡段", "转向段", "混合"]:
            mask = (regimes == st).reindex(dates).fillna(False)
            if mask.sum() < 20:
                continue
            hold_a = hold_nav.pct_change()[mask].mean() * 252
            row = dict(阶段=st, 天数=int(mask.sum()), 持有=f"{hold_a*100:7.2f}%")
            for rule in RULES:
                a = rn[rule][mask].mean() * 252
                row[f"开关{rule}"] = f"{a*100:7.2f}%"
                row[f"{rule}超额"] = f"{(a-hold_a)*100:+7.2f}pp"
            rows.append(row)
        if rows:
            print(pd.DataFrame(rows).to_string(index=False))

        # 落盘
        safe = code.replace(".", "_")
        out = pd.DataFrame({"date": dates, "nav_hold": hold_nav.values})
        for rule in RULES:
            out[f"nav_switch_{rule}"] = results[rule]["nav"].reindex(dates).values
            if results[rule]["hedges"]:
                pd.DataFrame(results[rule]["hedges"]).to_csv(
                    f"switch_hedges_{rule}_{safe}.csv", index=False)
        out.to_csv(f"switch_equity_{safe}.csv", index=False)
        print(f"[save] switch_equity_{safe}.csv"
              + "".join(f" / switch_hedges_{r}_{safe}.csv" for r in RULES))
    con.close()


def selftest():
    """合成四段行情验证：下降段应空仓避险，转向段 RULE_B 回补快于 RULE_A。"""
    rng = np.random.default_rng(7)
    n = 1500
    drifts = np.concatenate([
        np.full(300, 0.0015), np.full(300, 0.0),
        np.full(300, -0.0018), np.full(300, 0.0), np.full(300, 0.002),
    ])
    ret = drifts + rng.normal(0, 0.02, n)
    close = 100 * np.cumprod(1 + ret)
    idx = pd.bdate_range("2015-01-01", periods=n).strftime("%Y%m%d")
    op = close * (1 + rng.normal(0, 0.004, n))
    hi = np.maximum(op, close) * (1 + np.abs(rng.normal(0, 0.006, n)))
    lo = np.minimum(op, close) * (1 - np.abs(rng.normal(0, 0.006, n)))
    df = pd.DataFrame({"open": op, "high": hi, "low": lo, "close": close}, index=idx)
    cond = compute_conditions(df).iloc[30:]
    o = cond["open"].values.astype(float)
    c = cond["close"].values.astype(float)
    sh_h = INITIAL / (o[0] * (1 + SLIPPAGE) * (1 + COMMISSION))
    hold_nav = pd.Series(sh_h * c, index=cond.index)
    mh = perf(hold_nav)
    print(f"[selftest] 一直持有: 总{mh['tot']*100:.1f}% 年化{mh['cagr']*100:.2f}% 回撤{mh['mdd']*100:.2f}%")
    for rule in RULES:
        res = run_switch(cond, rule)
        m = perf(res["nav"])
        print(f"[selftest] RULE_{rule}: 总{m['tot']*100:.1f}% 年化{m['cagr']*100:.2f}% "
              f"回撤{m['mdd']*100:.2f}% 在场{res['time_in_mkt']*100:.0f}% 避险{len(res['hedges'])}次")
    print("[selftest] 预期：两版开关回撤显著低于持有；下降段(第600-900日)在场比例低")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
