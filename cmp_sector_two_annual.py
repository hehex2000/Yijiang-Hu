"""同引擎同期对比 v2：逐年在【完全相同成本】下拆解，消除口径差异。
两配置：对方(20/60) vs 我方taskD(60/250)，均跑已修复引擎。
分别用 3bp 与 20bp 两种成本，各自独立跑一遍，逐年抓 策略/基准/超额。
"""
import io
import re
import contextlib
import csv
import os
import run_sector_rotation as m

CONFIGS = [
    ("对方(20/60)", 20, 60, 60),
    ("我方taskD(60/250)", 60, 60, 250),
]
COSTS = [("3bp", 0.0003), ("20bp", 0.002)]

YE_RE = re.compile(
    r"\s(\d{4}):\s*策略\s*([+\-]?\d+\.?\d*)%\s*\|\s*基准\s*([+\-]?\d+\.?\d*)%\s*\|\s*超额\s*([+\-]?\d+\.?\d*)pp"
)


def run_cfg(name, mom, mt, mb, cost):
    m.MOM_W, m.MA_TREND, m.MA_BENCH = mom, mt, mb
    m.COST_RATE = cost
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        m.main()
    out = buf.getvalue()
    # 全周期
    def grab(pat):
        mm = re.search(pat, out)
        return float(mm.group(1)) if mm else None
    s = re.search(r"轮动策略\s+([+\-]?\d+\.?\d*)%\s+([+\-]?\d+\.?\d*)%\s+([+\-]?\d+\.?\d*)%\s+([\d.]+)", out)
    b = re.search(r"等权基准\s+([+\-]?\d+\.?\d*)%\s+([+\-]?\d+\.?\d*)%\s+([+\-]?\d+\.?\d*)%\s+([\d.]+)", out)
    ex = re.search(r"超额:\s*([+\-]?\d+\.?\d*)pp", out)
    full = dict(
        strat_tot=s.group(1), bench_tot=b.group(1),
        excess=float(ex.group(1)) if ex else None,
    )
    # 逐年
    annual = {}
    for mm in YE_RE.finditer(out):
        y = mm.group(1)
        annual[y] = dict(strat=float(mm.group(2)), bench=float(mm.group(3)),
                         excess=float(mm.group(4)))
    return full, annual


def main():
    data = {}   # data[cost_name][cfg_name] = (full, annual)
    for cname, cr in COSTS:
        data[cname] = {}
        for name, mom, mt, mb in CONFIGS:
            full, annual = run_cfg(name, mom, mt, mb, cr)
            data[cname][name] = (full, annual)

    years = [str(y) for y in range(2010, 2027)]

    for cname, _ in COSTS:
        print("=" * 78)
        print(f"同成本口径：{cname}  （两配置都按 {cname} 跑）")
        print("=" * 78)
        o_full, o_ann = data[cname]["对方(20/60)"]
        w_full, w_ann = data[cname]["我方taskD(60/250)"]
        print(f"\n[全周期] 对方 总收益={o_full['strat_tot']} 基准={o_full['bench_tot']} 超额={o_full['excess']:+.2f}pp"
              f" | 我方 总收益={w_full['strat_tot']} 基准={w_full['bench_tot']} 超额={w_full['excess']:+.2f}pp")
        print(f"{'年份':<6}{'对方策略':<10}{'对方基准':<10}{'对方超额':<10}"
              f"{'我方策略':<10}{'我方基准':<10}{'我方超额':<10}{'谁赢':<8}")
        win_o = win_w = 0
        for y in years:
            oa = o_ann.get(y, {})
            wa = w_ann.get(y, {})
            os_ = f"{oa.get('strat',0):+.1f}%" if oa else "-"
            ob_ = f"{oa.get('bench',0):+.1f}%" if oa else "-"
            oe_ = f"{oa.get('excess',0):+.1f}pp" if oa else "-"
            ws_ = f"{wa.get('strat',0):+.1f}%" if wa else "-"
            wb_ = f"{wa.get('bench',0):+.1f}%" if wa else "-"
            we_ = f"{wa.get('excess',0):+.1f}pp" if wa else "-"
            if oa and wa:
                if oa['excess'] > wa['excess']:
                    win_o += 1; winner = "对方"
                elif wa['excess'] > oa['excess']:
                    win_w += 1; winner = "我方"
                else:
                    winner = "持平"
            else:
                winner = "-"
            print(f"{y:<6}{os_:<10}{ob_:<10}{oe_:<10}{ws_:<10}{wb_:<10}{we_:<10}{winner:<8}")
        print(f"\n逐年超额胜场（{cname}）：对方 {win_o} 年 / 我方 {win_w} 年")

    # 落盘：两张同口径逐年表 + 全周期
    out_dir = "data/results/sector_rotation"
    os.makedirs(out_dir, exist_ok=True)
    for cname, _ in COSTS:
        o_full, o_ann = data[cname]["对方(20/60)"]
        w_full, w_ann = data[cname]["我方taskD(60/250)"]
        fn = f"{out_dir}/cmp_annual_samecost_{cname}.csv"
        with open(fn, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["year", "other_strat", "other_bench", "other_excess_pp",
                        "our_strat", "our_bench", "our_excess_pp"])
            for y in years:
                oa = o_ann.get(y, {}); wa = w_ann.get(y, {})
                w.writerow([y, oa.get('strat'), oa.get('bench'), oa.get('excess'),
                            wa.get('strat'), wa.get('bench'), wa.get('excess')])
            w.writerow(["FULL", o_full['strat_tot'], o_full['bench_tot'], o_full['excess'],
                        w_full['strat_tot'], w_full['bench_tot'], w_full['excess']])
        print(f"\n[save] {fn}")
    print("\n[done]")


if __name__ == "__main__":
    main()
