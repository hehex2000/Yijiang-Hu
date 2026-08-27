# -*- coding: utf-8 -*-
"""
利弗莫尔策略 · 方向确认 ablation
================================
触发(体积坍缩) = 关键点突破(breakout_confirmed) + 量能确认(vol_confirmed) + 横盘压缩(squeeze_pass)
对每一个"体积坍缩触发"的开仓, 用一组候选【方向确认信号】作为额外闸门, 分桶比较:

  组 control(无附加) : 仅体积坍缩触发即开仓 (现状基线, 对应假突破 58.7%)
  C1 板块动量        : 个股5日动量在行业内排名前30%
  C2 低波            : 个股20日已实现波动率行业内底部30%
  C3 突破前趋势      : 突破前20日个股自身收益>0 (最小阻力方向)
  C4 量能潮 OBV      : OBV 20日斜率为正 (吸筹方向确认)
  C5 突破幅度>1%     : 收盘高于关键点>1% (排除勉强过线假突破)

每组独立记账(平行账本, 单次数据扫描), 输出:
  交易笔数 / 假突破率 / 命中率 / 净值总收益 / 年化 / 最大回撤 / 夏普 / 持仓占比 / 超额HS300

判据: 若某候选组假突破率显著 <50% 且净值转正, 则该方向确认信号有救, 策略值得重构;
      否则检测器本身也是噪声, 整个框架该放弃。

口径: 所有信号 T-1 收盘判定 / T 开盘执行(无未来函数); hfq 空间算突破位与高低点, 原始价估值。
      退出规则在组间完全一致(仅开仓闸门不同), 隔离单一变量。
"""
import sys
import os
import argparse
import numpy as np
import pandas as pd

import run_livermore_v2 as L
from run_monthly_rebalance import get_trade_dates, calc_fee, COMMISSION_RATE, STAMP_DUTY_RATE, SLIPPAGE_RATE, COMMISSION_MIN

CAPITAL = 1_000_000.0
INDEX_MARKET = "000300.SH"
INDEX_BENCH_1 = "000300.SH"
INDEX_BENCH_2 = "000906.SH"
UNIV_INDEX = "000906.SH"
RES_DIR = "data/results/livermore"


# ────────────────────────────────────────────────────────────
#  信号预计算(全部 T-1 空间, 避免未来函数)
# ────────────────────────────────────────────────────────────
def build_ablation_signals(all_dates, open_r, high_r, low_r, close_r, pre_close_r, vol_r, adj_d, industry_map, cfg):
    codes = list(open_r.columns)
    idx = open_r.index.intersection(all_dates)
    open_raw = open_r.reindex(all_dates)
    high_r = high_r.reindex(all_dates)
    low_r = low_r.reindex(all_dates)
    close_raw = close_r.reindex(all_dates)
    pre_close_raw = pre_close_r.reindex(all_dates)
    vol_r = vol_r.reindex(all_dates)
    adj_d = adj_d.reindex(all_dates)
    adj_f = adj_d.ffill().fillna(1.0)

    close_h = close_raw * adj_f
    high_h = high_r * adj_f

    look = int(cfg["lookback"])
    key_level = high_h.rolling(look, min_periods=look).max().shift(1)
    breakout_ready = (close_h > key_level) & key_level.notna()
    breakout_ready = breakout_ready.fillna(False)
    confirm_days = int(cfg["confirm_days"])
    breakout_confirmed = breakout_ready.rolling(confirm_days, min_periods=confirm_days).sum().eq(confirm_days).fillna(False)

    vol_win = int(cfg["vol_win"])
    vol_ma = vol_r.rolling(vol_win, min_periods=vol_win).mean().shift(1)
    vol_confirmed = (vol_r > vol_ma * 1.5) & vol_ma.notna()
    vol_confirmed = vol_confirmed.fillna(False)

    # 横盘压缩(体积坍缩前置)
    box_len = int(cfg["box_len"])
    box_width_thr = float(cfg["box_width"])
    box_hi = close_h.rolling(box_len, min_periods=box_len).max()
    box_lo = close_h.rolling(box_len, min_periods=box_len).min()
    box_mid = close_h.rolling(box_len, min_periods=box_len).mean()
    box_width = (box_hi - box_lo) / box_mid
    squeeze_pass = (box_width.shift(1) <= box_width_thr) & box_width.notna()
    squeeze_pass = squeeze_pass.fillna(False)

    base_trigger = breakout_confirmed & vol_confirmed & squeeze_pass

    # ── 候选方向确认信号 ──
    mom5 = close_h / close_h.shift(5) - 1.0
    sector_mom = L._sector_rank(mom5, codes, industry_map, cfg["sector_top_pct"])

    daily_ret = close_h / close_h.shift(1) - 1.0
    realized_vol = daily_ret.rolling(20, min_periods=20).std().shift(1)
    lowvol = L._sector_rank(-realized_vol, codes, industry_map, cfg["sector_top_pct"])

    pre_trend = (close_h / close_h.shift(20) - 1.0 > 0) & close_h.notna()
    pre_trend = pre_trend.fillna(False)

    close_diff = close_h.diff()
    obv = (close_diff.apply(np.sign) * vol_r).cumsum()
    obv_slope = (obv / obv.shift(20) - 1.0 > 0) & obv.notna()
    obv_slope = obv_slope.fillna(False)

    breakout_size = ((close_h - key_level) / key_level > 0.01) & key_level.notna()
    breakout_size = breakout_size.fillna(False)

    # 退出用矩阵(hfq 空间)
    ma_per = int(cfg["ma_period"])
    ma = close_h.rolling(ma_per, min_periods=ma_per).mean()
    ma_below = ((close_h < ma) & ma.notna()).fillna(False)
    exit_pct = float(cfg["exit_pct"])
    below_key = ((close_h < key_level * (1.0 - exit_pct)) & key_level.notna()).fillna(False)

    return dict(
        open_raw=open_raw, close_raw=close_raw, pre_close_raw=pre_close_raw,
        close_h=close_h, high_h=high_h, key_level=key_level,
        base_trigger=base_trigger,
        masks=dict(
            C1_sector_mom=sector_mom, C2_lowvol=lowvol, C3_pre_trend=pre_trend,
            C4_obv=obv_slope, C5_breakout_size=breakout_size,
        ),
        ma_below=ma_below, below_key=below_key,
    )


# ────────────────────────────────────────────────────────────
#  平行账本回测(单次扫描, 6 组同时记账)
# ────────────────────────────────────────────────────────────
def run_ablation(start, end, cfg, group_names):
    all_dates = get_trade_dates(start, end)
    if len(all_dates) < 60:
        print(f"  [跳过] {start}-{end} 交易日不足"); return None

    univ_snaps = L.load_universe_dates(end)
    snap_dates = [s[0] for s in univ_snaps]
    import bisect
    def univ_at(d):
        i = bisect.bisect_right(snap_dates, d) - 1
        return univ_snaps[i][1] if i >= 0 else set()

    all_codes_set = set()
    for _, s in univ_snaps:
        all_codes_set |= s
    all_codes = sorted(all_codes_set)
    if not all_codes:
        print(f"  [跳过] {start}-{end} 无成分数据"); return None

    open_r, high_r, low_r, close_r, pre_close_r, vol_r, adj_d = L.load_panels(all_codes, start, end)
    industry_map = L.load_industry()
    sig = build_ablation_signals(all_dates, open_r, high_r, low_r, close_r, pre_close_r, vol_r, adj_d, industry_map, cfg)

    # 市场门控(HS300>ma20)
    idx_close = L.load_index(INDEX_MARKET, start, end)
    mkt = idx_close.reindex(all_dates)
    mkt_ma20 = mkt.rolling(20, min_periods=20).mean()
    bull = ((mkt > mkt_ma20) & mkt_ma20.notna()).fillna(False).astype(bool)

    open_raw = sig["open_raw"]; close_raw = sig["close_raw"]; pre_close_raw = sig["pre_close_raw"]
    close_h = sig["close_h"]; high_h = sig["high_h"]; key_level = sig["key_level"]
    ma_b = sig["ma_below"]; bk = sig["below_key"]
    base = sig["base_trigger"]
    masks = sig["masks"]

    n = len(all_dates)
    cols = list(base.columns)
    code2idx = {c: j for j, c in enumerate(cols)}
    base_arr = base.values.astype(bool)
    close_h_arr = close_h.values.astype(float)
    high_h_arr = high_h.values.astype(float)
    kl_arr = key_level.values.astype(float)
    ma_arr = ma_b.values.astype(bool)
    bk_arr = bk.values.astype(bool)

    # 每日触发集合(base, T-1) 与候选掩码集合
    base_sets = [set(np.compress(base_arr[i], cols)) for i in range(n)]
    mask_sets = {}
    for name in group_names:
        if name == "control":
            mask_sets[name] = None
            continue
        m = masks[name].reindex(all_dates)
        m_arr = m.values.astype(bool)
        mask_sets[name] = [set(np.compress(m_arr[i], cols)) for i in range(n)]

    # 初始化平行账本
    books = {}
    for name in group_names:
        books[name] = dict(cash=CAPITAL, positions={}, pending=set(), nav=[], trades=[])

    stop_loss = float(cfg["stop_loss"])
    trailing_stop = float(cfg["trailing_stop"])
    exit_pct = float(cfg["exit_pct"])
    max_hold = int(cfg["max_hold"])
    market_exit = bool(cfg.get("market_exit", True))

    def close_trade(bk, code, sell_price, d, exit_type, entry_idx):
        pos = bk["positions"][code]
        sh = pos["shares"]
        if sh <= 0:
            return
        proceeds = sell_price * sh - calc_fee("sell", sell_price, sh, d)
        bk["cash"] += proceeds
        cost = pos["entry_open"] * sh + calc_fee("buy", pos["entry_open"], sh, pos["entry_date"])
        ret = proceeds / cost - 1.0
        # 假突破标记: 入场后第1/2日(if available)均未创出新高
        false_flag = False
        if entry_idx >= 0 and entry_idx + 2 < n:
            eh = high_h_arr[entry_idx, code2idx[code]]
            d1 = high_h_arr[entry_idx + 1, code2idx[code]]
            d2 = high_h_arr[entry_idx + 2, code2idx[code]]
            if not pd.isna(eh) and not pd.isna(d1) and not pd.isna(d2):
                false_flag = (d1 <= eh and d2 <= eh)
        bk["trades"].append(dict(entry_date=pos["entry_date"], exit_date=d, code=code,
                                 ret=ret, false_breakout=false_flag, exit_type=exit_type))
        bk["positions"].pop(code, None)

    for i in range(n):
        d = all_dates[i]
        univ = univ_at(d)
        if not univ:
            for name in group_names:
                books[name]["nav"].append((d, books[name]["cash"]))
            continue

        bull_prev = bool(bull.iloc[i - 1]) if i >= 1 else False
        force_exit_all = market_exit and (not bull_prev)
        ma_i = ma_sets_prev = set(np.compress(ma_arr[i - 1], cols)) if i >= 1 else set()

        for name in group_names:
            bk = books[name]
            positions = bk["positions"]

            # 1) 卖出执行(前一日跌停封死重试)
            sell_exec = []
            for code in list(bk["pending"]):
                if code not in positions:
                    bk["pending"].discard(code); continue
                op = open_raw.iloc[i].get(code); cl = close_raw.iloc[i].get(code); pc = pre_close_raw.iloc[i].get(code)
                if op is None or pd.isna(op) or cl is None or pd.isna(cl) or pc is None or pd.isna(pc):
                    bk["pending"].discard(code); continue
                if op <= pc * 0.901:
                    if cl <= pc * 0.901:
                        continue
                    sell_exec.append((code, cl))
                else:
                    sell_exec.append((code, op))
            bk["pending"].clear()

            # 2) 持仓退出判定(用 T-1 收盘信号, T 开盘执行)
            prev_i = i - 1
            for code, pos in list(positions.items()):
                if code not in univ:
                    last_c = pos.get("last_close")
                    if last_c is None or pd.isna(last_c):
                        last_c = pos.get("entry_open", 0.0)
                    sell_exec.append((code, last_c)); continue
                if prev_i < 0 or code not in code2idx:
                    continue
                j = code2idx[code]
                ch = close_h_arr[prev_i, j]; hh = high_h_arr[prev_i, j]
                pos["high_water_mark"] = max(pos["high_water_mark"], ch, hh if not pd.isna(hh) else ch)
                exit_now = False
                # 假突破: 入场后第1/2日未创新高
                entry_idx = all_dates.index(pos["entry_date"]) if pos["entry_date"] in all_dates else -1
                if entry_idx >= 0 and i >= entry_idx + 3:
                    d1 = high_h_arr[entry_idx + 1, j] if entry_idx + 1 < n else np.nan
                    d2 = high_h_arr[entry_idx + 2, j] if entry_idx + 2 < n else np.nan
                    eh = high_h_arr[entry_idx, j]
                    if not pd.isna(d1) and not pd.isna(d2) and not pd.isna(eh) and d1 <= eh and d2 <= eh:
                        exit_now = True
                if not exit_now and trailing_stop > 0 and pos["high_water_mark"] > 0 and ch < pos["high_water_mark"] * (1 - trailing_stop):
                    exit_now = True
                if not exit_now and stop_loss > 0 and not pd.isna(pos["entry_hfq"]) and ch < pos["entry_hfq"] * (1 - stop_loss):
                    exit_now = True
                if not exit_now and not pd.isna(pos["key_hfq"]) and ch < pos["key_hfq"] * (1 - exit_pct):
                    exit_now = True
                if not exit_now and code in ma_i:
                    exit_now = True
                if force_exit_all:
                    exit_now = True
                if exit_now:
                    op = open_raw.iloc[i].get(code); cl = close_raw.iloc[i].get(code); pc = pre_close_raw.iloc[i].get(code)
                    if op is not None and not pd.isna(op) and cl is not None and not pd.isna(cl) and pc is not None and not pd.isna(pc):
                        if op <= pc * 0.901:
                            if cl <= pc * 0.901:
                                bk["pending"].add(code)
                            else:
                                sell_exec.append((code, cl))
                        else:
                            sell_exec.append((code, op))

            for code, price in sell_exec:
                if code in positions:
                    close_trade(bk, code, price, d, "退出", all_dates.index(positions[code]["entry_date"]) if positions[code]["entry_date"] in all_dates else -1)

            # 3) 新开仓(base_trigger & 候选闸门, T-1 信号 T 开盘执行)
            can_open = bull_prev
            if can_open and prev_i >= 0:
                cand_set = base_sets[prev_i] & univ
                if mask_sets[name] is not None:
                    cand_set = cand_set & mask_sets[name][prev_i]
                cand = sorted(cand_set - set(positions) - bk["pending"])
                def _px(c):
                    p = close_raw.iloc[i].get(c)
                    if p is None or pd.isna(p):
                        p = positions[c].get("last_close") or 0.0
                    return p
                equity = bk["cash"] + sum(_px(c) * positions[c]["shares"] for c in positions)
                slots = max_hold - len(positions)
                if slots > 0 and cand:
                    take = cand[:slots]
                    per_val = equity / max_hold
                    for code in take:
                        op = open_raw.iloc[i].get(code); cl = close_raw.iloc[i].get(code); pc = pre_close_raw.iloc[i].get(code)
                        if op is None or pd.isna(op) or op <= 0 or cl is None or pd.isna(cl) or pc is None or pd.isna(pc):
                            continue
                        if op >= pc * 1.099:
                            if cl >= pc * 1.099:
                                continue
                            buy_price = cl
                        else:
                            buy_price = op
                        if pd.isna(buy_price) or buy_price <= 0:
                            continue
                        sh = int(per_val / (buy_price * (1 + COMMISSION_RATE + SLIPPAGE_RATE)) / 100) * 100
                        if sh <= 0:
                            sh = 100
                        cost = buy_price * sh + calc_fee("buy", buy_price, sh, d)
                        if pd.isna(cost) or cost > bk["cash"]:
                            continue
                        bk["cash"] -= cost
                        j = code2idx.get(code)
                        positions[code] = dict(shares=sh, entry_open=buy_price,
                                               entry_hfq=close_h_arr[prev_i, j] if j is not None else np.nan,
                                               key_hfq=kl_arr[prev_i, j] if j is not None else np.nan,
                                               high_water_mark=close_h_arr[prev_i, j] if j is not None else np.nan,
                                               entry_date=d, last_close=buy_price)

        # 4) 估值
        for name in group_names:
            bk = books[name]
            mv = bk["cash"]
            held = 0
            for code, pos in bk["positions"].items():
                c = close_raw.iloc[i].get(code)
                if c is None or pd.isna(c):
                    c = pos.get("last_close")
                if c is not None and not pd.isna(c):
                    mv += pos["shares"] * c
                    pos["last_close"] = c
                    held += 1
            bk["nav"].append((d, mv))

    # ── 指标 ──
    def metrics(vals):
        vals = np.array(vals, dtype=float)
        tot = vals[-1] / vals[0] - 1
        years = (pd.Timestamp(all_dates[-1]) - pd.Timestamp(all_dates[0])).days / 365.25
        ann = (vals[-1] / vals[0]) ** (1 / years) - 1 if years > 0 else 0
        peak = np.maximum.accumulate(vals)
        mdd = (vals / peak - 1).min()
        rets = np.diff(vals) / vals[:-1]
        vol = rets.std() * np.sqrt(252) if len(rets) > 1 else 0
        sharpe = (rets.mean() * 252 - 0.02) / vol if (vol > 0 and len(rets) > 1) else 0
        return dict(total=tot, ann=ann, mdd=mdd, sharpe=sharpe)

    b1 = L.load_index(INDEX_BENCH_1, start, end).reindex(all_dates)
    b2 = L.load_index(INDEX_BENCH_2, start, end).reindex(all_dates)
    def bench_nav(series):
        fv = series.first_valid_index()
        if fv is None:
            return np.array([CAPITAL] * len(all_dates))
        base = series[fv]
        nav = series / base * CAPITAL
        nav = nav.where(nav.notna(), CAPITAL)  # 前置 NaN 视为持平于基准起点, 避免污染 vals[0]
        return nav.ffill().values
    mb1 = metrics(bench_nav(b1))
    mb2 = metrics(bench_nav(b2))

    out = []
    for name in group_names:
        bk = books[name]
        nav_vals = [v for _, v in bk["nav"]]
        m = metrics(nav_vals)
        tr = pd.DataFrame(bk["trades"])
        n_tr = len(tr)
        false_rate = (tr["false_breakout"].sum() / n_tr) if n_tr else float("nan")
        hit = (tr[tr["ret"] > 0].shape[0] / n_tr) if n_tr else float("nan")
        out.append(dict(group=name, trades=n_tr,
                        false_breakout=f"{false_rate*100:.1f}%" if n_tr else "-",
                        hit_rate=f"{hit*100:.1f}%" if n_tr else "-",
                        total_return=f"{m['total']*100:+.2f}%",
                        ann=f"{m['ann']*100:+.2f}%",
                        mdd=f"{m['mdd']*100:.2f}%",
                        sharpe=f"{m['sharpe']:.2f}",
                        excess_hs300=f"{(m['total']-mb1['total'])*100:+.2f}pp"))
    return out, mb1, mb2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="20180101")
    ap.add_argument("--end", default="20251231")
    ap.add_argument("--lookback", type=int, default=20)
    ap.add_argument("--confirm-days", type=int, default=2)
    ap.add_argument("--vol-win", type=int, default=5)
    ap.add_argument("--box-len", type=int, default=20, help="横盘压缩窗口(日); 体积坍缩触发前置")
    ap.add_argument("--box-width", type=float, default=0.15, help="箱宽阈值(相对)")
    ap.add_argument("--sector-top-pct", type=float, default=0.30)
    ap.add_argument("--exit-pct", type=float, default=0.03)
    ap.add_argument("--ma-period", type=int, default=20)
    ap.add_argument("--stop-loss", type=float, default=0.05)
    ap.add_argument("--max-hold", type=int, default=5)
    ap.add_argument("--trailing-stop", type=float, default=0.12)
    ap.add_argument("--no-market-exit", action="store_true")
    ap.add_argument("--candidates", default="all", help="子集: C1,C2,C3,C4,C5 或 all")
    args = ap.parse_args()

    cfg = dict(lookback=args.lookback, confirm_days=args.confirm_days, vol_win=args.vol_win,
               box_len=args.box_len, box_width=args.box_width, sector_top_pct=args.sector_top_pct,
               exit_pct=args.exit_pct, ma_period=args.ma_period, stop_loss=args.stop_loss,
               max_hold=args.max_hold, trailing_stop=args.trailing_stop,
               market_exit=not args.no_market_exit)

    all_groups = ["control", "C1_sector_mom", "C2_lowvol", "C3_pre_trend", "C4_obv", "C5_breakout_size"]
    label_map = {"control": "control(无附加)", "C1_sector_mom": "C1板块动量", "C2_lowvol": "C2低波",
                 "C3_pre_trend": "C3突破前趋势", "C4_obv": "C4量能潮OBV", "C5_breakout_size": "C5突破幅度>1%"}
    if args.candidates != "all":
        sel = [g for g in all_groups if g == "control" or g.split("_")[0] in args.candidates.split(",")]
        group_names = sel if sel else all_groups
    else:
        group_names = all_groups

    print(f"体积坍缩 ablation | {args.start}-{args.end} | box_len={args.box_len} box_width={args.box_width} "
          f"lookback={args.lookback} confirm={args.confirm_days} vol_win={args.vol_win}")
    print(f"触发 = 关键点突破 + 量能确认 + 横盘压缩 | 候选闸门: {[label_map[g] for g in group_names]}\n")

    res, mb1, mb2 = run_ablation(args.start, args.end, cfg, group_names)
    if res is None:
        return

    rows = []
    for r in res:
        rows.append([r["group"], r["trades"], r["false_breakout"], r["hit_rate"],
                     r["total_return"], r["ann"], r["mdd"], r["sharpe"], r["excess_hs300"]])
    df = pd.DataFrame(rows, columns=["组", "笔数", "假突破率", "命中率", "净值总收益", "年化", "最大回撤", "夏普", "超额HS300"])
    # 用中文组名
    df["组"] = df["组"].map(lambda g: label_map.get(g, g))
    print(df.to_string(index=False))
    print(f"\n基准 HS300: 总收益 {mb1['total']*100:+.2f}% | 中证800: 总收益 {mb2['total']*100:+.2f}%")
    print("[判据] 若某候选组 假突破率<50% 且 净值总收益转正 → 方向确认有效, 策略值得重构; 否则检测器本身也是噪声。")

    os.makedirs(RES_DIR, exist_ok=True)
    out_path = os.path.join(RES_DIR, "ablation_direction.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nCSV 已写: {out_path}")


if __name__ == "__main__":
    main()
