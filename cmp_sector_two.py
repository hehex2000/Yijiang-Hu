"""同引擎同期对比：我方(taskD: MOM_W=60/MA_BENCH=250) vs 对方(20/60)。
两配置均跑在已修复的 run_sector_rotation 引擎上，同期 2010-01-04~2026-07-31。
成本做 2x2 隔离（参数效应 vs 成本效应）。
"""
import io
import re
import contextlib
import csv
import run_sector_rotation as m

CONFIGS = [
    ("对方(20/60)", 20, 60, 60),
    ("我方taskD(60/250)", 60, 60, 250),
]
COSTS = [("20bp", 0.002), ("3bp", 0.0003)]


def parse_main(globals_cfg):
    m.MOM_W, m.MA_TREND, m.MA_BENCH = globals_cfg[1], globals_cfg[2], globals_cfg[3]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        m.main()
    out = buf.getvalue()
    row = {}

    def grab(pat, cast=float, grp=1):
        mm = re.search(pat, out)
        return cast(mm.group(grp)) if mm else None

    s = re.search(r"轮动策略\s+([+\-]?\d+\.?\d*%)\s+([+\-]?\d+\.?\d*%)\s+([+\-]?\d+\.?\d*%)\s+([\d.]+)", out)
    if s:
        row['strat_tot'], row['strat_cagr'], row['strat_mdd'], row['strat_sharpe'] = \
            s.group(1), s.group(2), s.group(3), float(s.group(4))
    b = re.search(r"等权基准\s+([+\-]?\d+\.?\d*%)\s+([+\-]?\d+\.?\d*%)\s+([+\-]?\d+\.?\d*%)\s+([\d.]+)", out)
    if b:
        row['bench_tot'], row['bench_cagr'], row['bench_mdd'], row['bench_sharpe'] = \
            b.group(1), b.group(2), b.group(3), float(b.group(4))
    ex = re.search(r"超额:\s*([+\-]?\d+\.?\d*)pp", out)
    row['excess_pp'] = float(ex.group(1)) if ex else None
    cm = re.search(r"持币月数:\s*\d+\s*\(([\d.]+)%\)", out)
    row['cash_pct'] = float(cm.group(1)) if cm else None
    tm = re.search(r"交易笔数:\s*(\d+)", out)
    row['trades'] = int(tm.group(1)) if tm else None
    return row, out


def main():
    table = []
    annuals = {}
    for name, mom, mt, mb in CONFIGS:
        for cname, cr in COSTS:
            m.COST_RATE = cr
            row, out = parse_main((name, mom, mt, mb))
            table.append({
                'config': name, 'cost': cname,
                'strat_tot': row['strat_tot'], 'strat_cagr': row['strat_cagr'],
                'strat_mdd': row['strat_mdd'], 'strat_sharpe': row['strat_sharpe'],
                'bench_tot': row['bench_tot'], 'excess_pp': row['excess_pp'],
                'cash_pct': row['cash_pct'], 'trades': row['trades'],
            })
            if name not in annuals:
                annuals[name] = {}
            annuals[name][cname] = out

    # 打印摘要表
    hdr = f"{'配置':<18}{'成本':<6}{'策略总收益':<12}{'年化':<10}{'MDD':<10}{'Sharpe':<8}{'等权基准':<12}{'超额pp':<9}{'持币%':<8}{'交易':<6}"
    print("=" * len(hdr))
    print("板块轮动 同期对比（2010-01-04 ~ 2026-07-31，同修正引擎）")
    print("=" * len(hdr))
    print(hdr)
    for r in table:
        print(f"{r['config']:<18}{r['cost']:<6}{r['strat_tot']:<12}{r['strat_cagr']:<10}"
              f"{r['strat_mdd']:<10}{r['strat_sharpe']:<8}{r['bench_tot']:<12}"
              f"{r['excess_pp']:<+9.2f}{r['cash_pct']:<8.1f}{r['trades']:<6}")

    # 逐年拆解（取两配置各自的"原生成本"口径：对方20bp / 我方3bp）
    print("\n--- 逐年收益拆解（对方@20bp / 我方taskD@3bp）---")
    years = [str(y) for y in range(2010, 2027)]
    print(f"{'年份':<6}{'对方策略':<10}{'对方基准':<10}{'对方超额':<10}{'我方策略':<10}{'我方基准':<10}{'我方超额':<10}")
    for y in years:
        def annual(out, y):
            mm = re.search(rf"\s{y}:\s*策略\s+([+\-]?\d+\.?\d*%)\s*\|\s*基准\s+([+\-]?\d+\.?\d*%)\s*\|\s*超额\s+([+\-]?\d+\.?\d*)pp", out)
            return (mm.group(1), mm.group(2), mm.group(3) + 'pp') if mm else ('-', '-', '-')
        os_, ob_, oe_ = annual(annuals["对方(20/60)"]["20bp"], y)
        ms_, mb_, me_ = annual(annuals["我方taskD(60/250)"]["3bp"], y)
        print(f"{y:<6}{os_:<10}{ob_:<10}{oe_:<10}{ms_:<10}{mb_:<10}{me_:<10}")

    # 写 CSV
    import os
    os.makedirs("data/results/sector_rotation", exist_ok=True)
    with open("data/results/sector_rotation/cmp_two_configs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "cost", "strat_tot", "strat_cagr", "strat_mdd",
                    "strat_sharpe", "bench_tot", "excess_pp", "cash_pct", "trades"])
        for r in table:
            w.writerow([r['config'], r['cost'], r['strat_tot'], r['strat_cagr'],
                        r['strat_mdd'], r['strat_sharpe'], r['bench_tot'],
                        r['excess_pp'], r['cash_pct'], r['trades']])
    print("\n[save] data/results/sector_rotation/cmp_two_configs.csv")


if __name__ == "__main__":
    main()
