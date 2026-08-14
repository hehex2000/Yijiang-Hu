# -*- coding: utf-8 -*-
"""
缩量回踩逐笔分析（深挖 spb_trades_*.csv）
=========================================
配对 BUY/SELL -> 逐笔收益；按 exit reason 分组算胜率/分布；
重点回答用户两个问题：
  1) 胜率分布（分 reason 看）
  2) 破位止损(stop_breakout)是否"系统性亏在跌停 trapped"
     -> 信号日 T 收盘生成信号，T+1 开盘执行；
        隔夜缺口 gap = open_exec / close_T - 1（这部分损失是止损逻辑看不见的）
        + 执行日是否触跌停(low <= pre_close*0.9 主板 / *0.8 创业板)
用法：
  analyze_spb_trades.py <trades_csv> [<out_enriched_csv>]
"""
import os, sys, sqlite3
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_monthly_rebalance import get_conn, calc_fee

DB = "D:/tu-shareData/astock_daily.db"


def _limit_down_price(pre_close, code):
    """主板/创业板/科创板跌停价（四舍五入到分）。
    - 创业板(300/301)/科创板(688)：±20% → 0.80
    - 北交所(8xx/4xx)：±30% → 0.70（本平台已屏蔽，兜底）
    - 主板：±10% → 0.90
    """
    if code.startswith("30"):      # 创业板 20%
        f = 0.80
    elif code.startswith("688"):   # 科创板 20%
        f = 0.80
    elif code.startswith("8") or code.startswith("4"):  # 北交所 30%
        f = 0.70
    else:                           # 主板 10%
        f = 0.90
    return round(pre_close * f, 2)


def load_daily(code):
    """返回 trade_date -> (pre_close, open, high, low, close) 字典。"""
    con = get_conn()
    df = pd.read_sql_query(
        "SELECT trade_date, pre_close, open, high, low, close FROM daily "
        "WHERE ts_code=? ORDER BY trade_date ASC", con, params=(code,))
    con.close()
    out = {}
    for r in df.itertuples(index=False):
        out[int(r.trade_date)] = (r.pre_close, r.open, r.high, r.low, r.close)
    return out


def analyze(trades_csv, out_csv=None):
    tr = pd.read_csv(trades_csv, dtype={"code": str})
    tr["date"] = tr["date"].astype(int)
    # 配对：每 code 按序 FIFO（策略单标的单仓，无加仓）
    pairs = []
    open_buy = {}   # code -> buy row
    for _, r in tr.iterrows():
        if r["action"] == "BUY":
            open_buy[r["code"]] = r
        else:  # SELL
            b = open_buy.pop(r["code"], None)
            if b is None:
                continue
            pairs.append((b, r))
    print(f"成交行 {len(tr)} | 配对成功 {len(pairs)} 笔（未配对卖出 {len(tr)-2*len(pairs)} 行残留/挂单）")

    # 预载所需 code 的日线（仅信号日/执行日周边）
    need = sorted({p[0]["code"] for p in pairs} | {p[1]["code"] for p in pairs})
    daily = {c: load_daily(c) for c in need}

    def prev_trade_date(d):
        return d - 1  # 近似：T+1 执行，信号日=T=执行日-1 交易日。A股连续，足够

    rows = []
    for b, s in pairs:
        code = b["code"]
        buy_px = float(b["price"]); sell_px = float(s["price"])
        shares = int(b["shares"])
        buy_fee = calc_fee("buy", buy_px, shares)
        sell_fee = calc_fee("sell", sell_px, shares)
        net = (sell_px - buy_px) * shares - buy_fee - sell_fee
        gross_ret = (sell_px / buy_px - 1) * 100
        reason = s["reason"]
        # 隔夜缺口：信号日(T=执行日-1)收盘 vs 执行日开盘
        td = int(s["date"]); t_prev = td - 1
        d_exec = daily[code].get(td)
        d_prev = daily[code].get(t_prev)
        gap = np.nan; touched_ld = False; opened_ld = False; exec_low = np.nan
        if d_exec is not None:
            pre_close_exec, op, hi, lo, cl = d_exec
            exec_low = lo
            if d_prev is not None:
                close_T = d_prev[4]   # 信号日收盘
                if close_T and close_T > 0:
                    gap = (sell_px / close_T - 1) * 100   # 用的是 T+1 开盘=卖出价
            # 触跌停判断（用执行日 pre_close）
            if pre_close_exec and pre_close_exec > 0:
                ld_px = _limit_down_price(pre_close_exec, code)
                if lo <= ld_px + 1e-6:
                    touched_ld = True
                if abs(op - ld_px) <= max(0.02, ld_px*0.002):
                    opened_ld = True
        rows.append({
            "code": code, "name": b["name"], "buy_date": int(b["date"]),
            "sell_date": td, "reason": reason, "buy_px": buy_px,
            "sell_px": sell_px, "shares": shares, "gross_ret%": round(gross_ret, 2),
            "net_pnl": round(net, 1), "overnight_gap%": round(gap, 2) if not np.isnan(gap) else np.nan,
            "exec_low": exec_low, "touched_limit_down": touched_ld,
            "opened_limit_down": opened_ld,
        })
    res = pd.DataFrame(rows)

    # ── 分组胜率分布 ──
    print("\n" + "=" * 72)
    print(f"逐笔分析：{os.path.basename(trades_csv)}")
    print("=" * 72)
    print(f"总配对交易: {len(res)}  整体胜率: { (res['gross_ret%']>0).mean()*100:.1f}%"
          f"  平均收益: {res['gross_ret%'].mean():+.2f}%  中位: {res['gross_ret%'].median():+.2f}%")
    print("-" * 72)
    print(f"{'reason':<14}{'n':>5}{'胜率%':>8}{'均值%':>9}{'中位%':>9}{'最佳%':>9}{'最差%':>9}")
    for rs in ["take_profit", "stop_breakout", "stop_structure", "max_hold", "end"]:
        sub = res[res["reason"] == rs]
        if len(sub) == 0:
            continue
        print(f"{rs:<14}{len(sub):>5}{sub['gross_ret%'].gt(0).mean()*100:>7.1f}"
              f"{sub['gross_ret%'].mean():>+8.2f}{sub['gross_ret%'].median():>+8.2f}"
              f"{sub['gross_ret%'].max():>+8.2f}{sub['gross_ret%'].min():>+8.2f}")

    # ── 破位止损 trapped 专查 ──
    sb = res[res["reason"] == "stop_breakout"]
    print("-" * 72)
    print(f"【破位止损 trapped 专查】 n={len(sb)}")
    if len(sb) > 0:
        print(f"  胜率: {(sb['gross_ret%']>0).mean()*100:.1f}%  均值: {sb['gross_ret%'].mean():+.2f}%  中位: {sb['gross_ret%'].median():+.2f}%")
        print(f"  隔夜缺口(信号日收->执行日开) 均值: {sb['overnight_gap%'].mean():+.2f}%  中位: {sb['overnight_gap%'].median():+.2f}%")
        print(f"    其中 缺口<=-5% 的占比: {(sb['overnight_gap%']<=-5).mean()*100:.1f}%")
        print(f"    其中 缺口<=-9% 的占比: {(sb['overnight_gap%']<=-9).mean()*100:.1f}%")
        print(f"  执行日触跌停(low<=跌停价) 笔数: {sb['touched_limit_down'].sum()} ({(sb['touched_limit_down'].mean()*100):.1f}%)")
        print(f"  执行日开盘即跌停 笔数: {sb['opened_limit_down'].sum()} ({(sb['opened_limit_down'].mean()*100):.1f}%)")
        # 对比：止盈的笔隔夜缺口（应接近0或正）
        tp = res[res["reason"] == "take_profit"]
        if len(tp) > 0:
            print(f"  [对照]止盈隔夜缺口 均值: {tp['overnight_gap%'].mean():+.2f}% <- 破位组显著更负说明 T+1 缺口吃掉利润")

    if out_csv:
        res.to_csv(out_csv, index=False, encoding="utf-8")
        print(f"\n已写出逐笔明细: {out_csv}")
    return res


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: analyze_spb_trades.py <trades_csv> [out_csv]")
        sys.exit(1)
    tcsv = sys.argv[1]
    ocsv = sys.argv[2] if len(sys.argv) > 2 else None
    analyze(tcsv, ocsv)
