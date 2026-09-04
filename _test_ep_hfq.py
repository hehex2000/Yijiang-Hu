# -*- coding: utf-8 -*-
"""杠杆点B 回归测试：run_ep_neutral / run_multifactor 的 hfq 计价空间不变量。

对照已知案例：300853.SZ 在 20210610 送转，raw 收盘 72.36→37.55（-48.11%），
真实经济损失（hfq）仅 -6.09%。raw 口径凭空记了 42pp 亏损。
"""
import sys
import run_ep_neutral as ep

OK = True


def check(name, cond, detail=""):
    global OK
    tag = "PASS" if cond else "FAIL"
    if not cond:
        OK = False
    print(f"[{tag}] {name}" + (f"  {detail}" if detail else ""))


def run_case(mode, code, d0, d1):
    """在指定口径下取 d0/d1 两日收盘，返回 (p0, p1, ret)。"""
    ep.reset_price_cache()
    ep.PRICE_MODE = mode
    p0 = ep._px(code, d0, "close")
    p1 = ep._px(code, d1, "close")
    return p0, p1, (p1 / p0 - 1 if p0 else float("nan"))


print("=" * 68)
print("杠杆点B：run_ep_neutral hfq 计价空间测试")
print("=" * 68)

# ── 1. raw 模式：adj 必须为 None，价格等于 daily 原始值 ──
ep.reset_price_cache()
ep.PRICE_MODE = "raw"
ep._ensure_day("20210609")
rec = ep._DAY_PX["20210609"]["300853.SZ"]
check("1. raw 模式 adj=None（不查询、不缩放）", rec[7] is None, f"adj={rec[7]}")
check("1b. raw 模式 _scale 原样返回", ep._scale("300853.SZ", 55.23, None) == 55.23)

# ── 2. hfq 模式：首日严格等于 raw（归一化正确性）──
# 起点取【除权前一日】20210609，隔离区间内其它涨跌，只看除权当日效应
D0, D1 = "20210609", "20210610"
p0_raw, p1_raw, ret_raw = run_case("raw", "300853.SZ", D0, D1)
p0_hfq, p1_hfq, ret_hfq = run_case("hfq", "300853.SZ", D0, D1)
check("2. hfq 首日 == raw 首日（归一化基准正确）",
      abs(p0_hfq - p0_raw) < 1e-9, f"hfq={p0_hfq:.4f} raw={p0_raw:.4f}")

# ── 3. 送转日：hfq 跌幅应远小于 raw ──
print(f"\n  300853.SZ {D0}→{D1}  raw {p0_raw:.2f}→{p1_raw:.2f} ({ret_raw:+.2%})")
print(f"  300853.SZ {D0}→{D1}  hfq {p0_hfq:.2f}→{p1_hfq:.2f} ({ret_hfq:+.2%})")
check("3. raw 记录巨额假亏损（<-40%）", ret_raw < -0.40, f"{ret_raw:+.2%}")
check("3b. hfq 跌幅远小（>-15%，真实损失）", ret_hfq > -0.15, f"{ret_hfq:+.2%}")
check("3c. 凭空亏损 > 30pp", (ret_hfq - ret_raw) > 0.30, f"{(ret_hfq - ret_raw)*100:.2f}pp")

# ── 4. pct_chg 永不缩放（涨跌停判定不受口径影响）──
ep.reset_price_cache()
ep.PRICE_MODE = "hfq"
pct_hfq = ep._ensure_day(D1)["300853.SZ"][6]
ep.reset_price_cache()
ep.PRICE_MODE = "raw"
pct_raw = ep._ensure_day(D1)["300853.SZ"][6]
check("4. pct_chg 在两种口径下一致（涨跌停判定不受污染）",
      pct_hfq == pct_raw, f"hfq={pct_hfq} raw={pct_raw}")

# ── 5. adj_factor 整表缺行日必须 ffill，不能掉回 1.0 ──
con = ep.get_conn()
row = con.execute(
    "SELECT d.trade_date FROM (SELECT DISTINCT trade_date FROM daily "
    "WHERE trade_date BETWEEN '20200101' AND '20260723') d "
    "LEFT JOIN (SELECT DISTINCT trade_date FROM adj_factor "
    "WHERE trade_date BETWEEN '20200101' AND '20260723') a ON d.trade_date=a.trade_date "
    "WHERE a.trade_date IS NULL AND d.trade_date > '20210610' ORDER BY d.trade_date LIMIT 1"
).fetchone()
miss_day = row[0] if row else None
if miss_day:
    ep.reset_price_cache()
    ep.PRICE_MODE = "hfq"
    # 先取一个正常日建立 ffill 基线
    prev = con.execute(
        "SELECT MAX(trade_date) FROM daily WHERE trade_date < ?", (miss_day,)).fetchone()[0]
    ep._px("000651.SZ", prev, "close")
    ep._px("000651.SZ", miss_day, "close")
    a_miss = ep._DAY_PX[miss_day]["000651.SZ"][7]
    a_prev = ep._DAY_PX[prev]["000651.SZ"][7]
    check("5. 缺行日 ffill 继承前值（不掉回 1.0）",
          a_miss is not None and abs(a_miss - a_prev) < 1e-9,
          f"缺行日 {miss_day} adj={a_miss:.6f} vs 前日 {prev} adj={a_prev:.6f}；"
          f"若误用 fillna(1.0) 会得到 1.0（{a_prev:.1f}× 假跳空）")
else:
    print("[SKIP] 5. 未找到 20210610 之后的整表缺行日")
con.close()

# ── 6. 高股息蓝筹：hfq 年化应高于 raw（分红贡献为正且量级合理 2~10%/年）──
code, y0, y1 = "600036.SH", "20200102", "20260723"
r0, r1, _ = run_case("raw", code, y0, y1)
h0, h1, _ = run_case("hfq", code, y0, y1)
yrs = 6.5
ann_raw = (r1 / r0) ** (1 / yrs) - 1
ann_hfq = (h1 / h0) ** (1 / yrs) - 1
gap = (1 + ann_hfq) / (1 + ann_raw) - 1
print(f"\n  {code} {y0}→{y1}: raw {ann_raw:+.2%}/年  hfq {ann_hfq:+.2%}/年  差 {gap:+.2%}/年")
check("6. 分红贡献为正", gap > 0, f"{gap:+.2%}/年")
check("6b. 量级合理（0~12%/年，超出说明有 bug）", 0 < gap < 0.12, f"{gap:+.2%}/年")

print("=" * 68)
print("结果:", "全部 PASS ✅" if OK else "存在 FAIL ❌")
sys.exit(0 if OK else 1)
