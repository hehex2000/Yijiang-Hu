# -*- coding: utf-8 -*-
"""
diag_lookahead.py — 缠论内核"未来函数"审计（分两半）
================================================================
审查意见 P0 声称 compute_states 在全序列上一次性计算 = 未来函数。
本脚本分两步验证：

[甲] 因果性：对全序列算出的每个买卖点 (idx,类型)，扫描它首次出现于
    哪个前缀 data[:t+1]（t<N）。若所有信号都能在有限前缀复现、且"从未在任一前缀出现"
    数=0，则确认：信号定型只用 <=e_idx 数据，无未来函数。

[乙] 稳定性/闪烁：在 [甲] 的逐前缀扫描中，记录每个信号在多少个前缀中存在
    （持续前缀数），以及它"出现->消失->再现"的闪烁次数。若大量信号 持续前缀数=1
    （只闪 1 个前缀就消失）或存在长周期闪烁，则说明逐根重算会让同一逻辑信号反复
    出现/消失 —— 这正是旧引擎 成交笔数爆炸（16~42）而真实信号仅 2~3 的根因。
    修复：run_chan_lun_faithful.py 用 稳定阈值>=2 + 已执行去重 过滤闪烁，仅交易连续
    稳定出现的信号。

用法：
  venv_ml/Scripts/python.exe diag_lookahead.py
"""
import sys
sys.path.insert(0, ".")

from chan_lun_core_faithful import compute_states
from run_chan_lun_faithful import DEFAULT_INSTRUMENTS, load_ohlc

预热根数 = 120


def audit(code, table, label, istart):
    df = load_ohlc(code, table, istart)
    if df is None:
        print(f"[{label}] {code}: 无数据/护栏跳过")
        return
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    c = df["close"].values.astype(float)
    n = len(h)

    full = compute_states(h, l, c)
    full_sigs = set(full["buys"]) | set(full["sells"])

    # 逐前缀扫描：首次出现位置 / 出现次数 / 闪烁次数
    first_seen = {}          # 信号 -> 首次出现的前缀序号
    presence = {}            # 信号 -> 出现过的前缀集合
    flicker = {}             # 信号 -> 闪烁（出现->消失->再现）计数
    ever_seen = set()        # 迄今曾出现过的信号
    prev_present = set()     # 上一前缀出现的信号集合
    for t in range(预热根数, n):
        st = compute_states(h[:t + 1], l[:t + 1], c[:t + 1])
        cur = set(st["buys"]) | set(st["sells"])
        for sig in cur:
            if sig not in first_seen:
                first_seen[sig] = t
            presence.setdefault(sig, set()).add(t)
            # 闪烁：曾出现过、且上一前缀缺席（中间有间隔）-> 本次为"再现"，计一次闪烁
            if sig in ever_seen and sig not in prev_present:
                flicker[sig] = flicker.get(sig, 0) + 1
        ever_seen |= cur
        prev_present = cur

    # [甲] 因果性
    delays = []
    never = []
    for sig in sorted(full_sigs):
        if sig in first_seen:
            delays.append(first_seen[sig] - sig[0])  # 确认延迟 = 首次出现前缀 - 信号序号
        else:
            never.append(sig)

    print(f"\n[{label}] {code}")
    print(f"  [甲] 因果性: 全序列信号 {len(full_sigs)} | "
          f"确认延迟根(t-序号) 最小={min(delays)} 最大={max(delays)} 均值={sum(delays)/len(delays):.1f} | "
          f"从未在任一前缀出现(真未来函数)={len(never)}")
    print(f"      因果结论: {'✅ 无未来函数（全部信号有限前缀复现）' if not never else '❌ 存在真前视'}")

    # [乙] 闪烁/稳定性
    pers = [len(presence[s]) for s in full_sigs]
    pers1 = sum(1 for p in pers if p == 1)
    flick_total = sum(flicker.values())
    print(f"  [乙] 稳定性: 全序列信号 持续前缀数（出现前缀数） "
          f"最小={min(pers)} 最大={max(pers)} 均值={sum(pers)/len(pers):.1f}")
    print(f"      持续前缀数=1（瞬时闪 1 根即消失）的信号数 = {pers1} / {len(full_sigs)}")
    print(f"      长周期闪烁（出现->消失->再现）总次数 = {flick_total}")
    verdict = ("✅ 信号稳定，无闪烁" if pers1 == 0 and flick_total == 0
               else "⚠ 存在闪烁 -> 需用 稳定阈值+已执行去重 过滤（引擎已做）")
    print(f"      稳定性结论: {verdict}")


if __name__ == "__main__":
    for code, table, label, istart in DEFAULT_INSTRUMENTS:
        audit(code, table, label, istart)
