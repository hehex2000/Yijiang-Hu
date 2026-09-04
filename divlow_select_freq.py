"""只跑**选股**、生成指定调仓频率的 partial（供 divlow_nav_replay.py 秒级重放 NAV）。

为什么不用 run_dividend_low_vol_quality_bt.py 跑全量：
  该脚本在 select 之前会调 `_preload_pool_prices("all")` → 全A分红股日线一次性入内存
  （写入全局 `_PRICE`，约 4GB）。而 `_PRICE` **只被 ffill_price 读取、只服务 NAV 估值**，
  与选股无关（ffill_price 内部还有按需单股查询兜底，见第 326 行）。
  在内存受限环境下跑全量会被 OOM/SIGTERM 干掉（实测 2m43s Terminated，日志只有头部）。
  → 本脚本跳过预加载，只跑 select；NAV 交给 divlow_nav_replay.py 用 bulk_close_prices(hfq) 重放。

自证：对 quarter 档跑本脚本，产出的 partial 必须与既有
  `_official_official_compact_all_12_bk0_{START}_{END}_partial.csv` 逐位一致（md5 相同），
  否则说明跳过预加载改变了选股行为，本脚本不可用。

用法：
  venv_ml/Scripts/python.exe divlow_select_freq.py half year
  venv_ml/Scripts/python.exe divlow_select_freq.py year cap=20      # 年度调仓 + 每次调整≤20% 硬上限(B7')
  venv_ml/Scripts/python.exe divlow_select_freq.py quarter cap=20   # 季度调仓 + 同上（用于拆解"降频 vs 硬上限"各自贡献）

  # B8：最后一段筛子的排序键（官方 930955 = volatility；旧行为 = fwd_yield）
  venv_ml/Scripts/python.exe divlow_select_freq.py quarter key=vol
  venv_ml/Scripts/python.exe divlow_select_freq.py quarter key=vol cap=20
"""
import sys

import numpy as np
import pandas as pd

import run_dividend_low_vol_quality_bt as E

MODE = "official_compact"
POOL = "all"
TOP_N = 12
WIN = 252                     # 与 build_cfg 的 volatility_window 一致


# ─────────────────────────────────────────────────────────────────────
#  加速版 _patched_vol（只在本脚本 monkey-patch，不改原引擎文件）
# ─────────────────────────────────────────────────────────────────────
def _fast_vol(self, ts_code, trade_date, window=None):
    """`E._patched_vol` 的按需分支加速版：只取最近 window+1 根 K 线。

    原版按需分支拉**全历史**（`SELECT ... WHERE ts_code=? AND trade_date<=?`，可达数千行），
    但后面 `closes[-(window+1):]` 只用最后 window+1 个 → 多余行纯浪费（全A池 2000+ 只 × 每期）。

    等价性论证（window=252）：
      原：closes = [s[d] for d in sorted(s) if d <= trade_date]，再 closes[-(253):]
      新：ORDER BY trade_date DESC LIMIT 253，reverse 成升序 → 同为"最后 253 个"
      样本量门槛 len(closes) < max(int(252*0.6), 60) = 151：
        历史 >253 行 → 原 len=N(≥253)、新 len=253，两侧都 ≥151，通过；
        历史 ≤253 行 → 两侧 len 相同，判定一致。
      closes[-(window+1):] 对新版是 no-op。→ **逐位等价**（并由 verify() 抽样实证）。
    """
    if window is None:
        window = self.volatility_window
    s = E._PRICE.get(ts_code)
    if not s:
        conn = E.get_conn()
        df = pd.read_sql_query(
            "SELECT trade_date, close FROM daily WHERE ts_code=? AND trade_date<=? "
            "ORDER BY trade_date DESC LIMIT ?",
            conn, params=(ts_code, trade_date, int(window) + 1))
        conn.close()
        if len(df) == 0:
            return None
        closes = list(df["close"].astype(float))[::-1]      # 降序 → 升序
    else:
        closes = [s[d] for d in sorted(s) if d <= trade_date]
    if len(closes) < max(int(window * 0.6), 60):
        return None
    closes = closes[-(window + 1):]
    rets = np.diff(closes) / closes[:-1]
    return float(np.std(rets) * np.sqrt(252))


def verify(n=80, date="20251215"):
    """抽样自证：同一 (code, date) 分别用原版与加速版算波动率，逐位比对。

    🔴 必须显式清 _PRICE 缓存，否则第二版会命中第一版写入的缓存、走同一分支，
       测出来"必然相同"，是假通过。
    """
    conn = E.get_conn()
    df = pd.read_sql_query(
        "SELECT DISTINCT ts_code FROM daily WHERE trade_date=? ORDER BY ts_code LIMIT ?",
        conn, params=(date, n))
    conn.close()
    codes = [str(c) for c in df["ts_code"]]
    if not codes:
        print("[verify] 无样本，跳过")
        return False
    bad, both_none = 0, 0
    for c in codes:
        E._PRICE.pop(c, None)
        v_old = E._patched_vol(None, c, date, window=WIN)   # 显式传 window → 不访问 self
        E._PRICE.pop(c, None)
        v_new = _fast_vol(None, c, date, window=WIN)
        E._PRICE.pop(c, None)
        if v_old is None and v_new is None:
            both_none += 1
            continue
        if v_old is None or v_new is None or abs(v_old - v_new) > 1e-12:
            bad += 1
            if bad <= 3:
                print(f"  [差异] {c}: 原={v_old} 新={v_new}")
    ok = bad == 0
    print(f"[verify] 抽样 {len(codes)} 只 @ {date}：一致 {len(codes)-bad-both_none} "
          f"/ 双 None {both_none} / 不一致 {bad} → {'✅通过，启用加速版' if ok else '❌失败，回退原版'}")
    return ok


def main():
    # 参数：频率（必给/默认 half year）；可选 cap=20 表示「每次调整 ≤20% 硬上限」(B7')，
    #       写 cap=0.2 或 cap=20 都行（>1 视为百分数）
    _caps, freqs, key = None, [], "fwd_yield"
    for a in sys.argv[1:]:
        if a.startswith("cap="):
            v = float(a.split("=", 1)[1])
            _caps = v / 100.0 if v > 1 else v
        elif a.startswith("key="):
            key = a.split("=", 1)[1]        # fwd_yield(默认) | volatility(官方 930955 口径)
            assert key in ("fwd_yield", "volatility"), key
        else:
            freqs.append(a)
    freqs = freqs or ["half", "year"]
    CAP = _caps or 0.0
    E.PRICE_MODE = "hfq"          # 选股本身与价格口径无关，保持与 A/B 同设置
    # 加速版自证通过才 patch；不通过则沿用原版（慢但行为不变）
    if verify():
        E._patched_vol = _fast_vol
    for f in freqs:
        if f not in E.REBAL_SPECS:
            print(f"[skip] 未知频率 {f!r}，可选 {list(E.REBAL_SPECS)}")
            continue
        # 🔴 必须改 MODE_SPECS 本体：select_targets_official 内部读的是它（同 PRICE_MODE 那类坑）
        E.MODE_SPECS[MODE]["rebal"] = f
        print(f"\n{'#' * 70}\n# {f}（{E.REBAL_SPECS[f][1]}）  pool={POOL} top_n={TOP_N} "
              f"turnover_cap={CAP or '关'}  final_key={key}\n{'#' * 70}")
        targets, wmap, sel_log = E.select_targets_official(
            MODE, pool=POOL, top_n=TOP_N, buffer_k=0, turnover_cap=CAP, final_key=key)
        n_code = len({r[2] for r in sel_log})
        print(f"[done] {f}: {len(targets)} 期 / {len(sel_log)} 行 / {n_code} 只不同标的")


if __name__ == "__main__":
    main()
