"""对比：修复后(去掉下限) 4%对称网格 —— 无过滤 vs 开启趋势过滤。
验证冻结是否消失，并给出可信对照数。"""
import sys, os, io, contextlib, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_grid_backtest import run_grid_backtest

def run(label, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run_grid_backtest(ts_code="510300.SH", start_date="20180102", end_date="20260703", **kw)
    out = buf.getvalue()
    # 抽取结果段
    res = {}
    for line in out.splitlines():
        for key in ("总收益率", "年化收益率", "最大回撤", "夏普比率", "网格交易次数", "胜率", "沪深300ETF涨幅", "超额收益"):
            if line.strip().startswith(key):
                res[key] = line.strip()
    # 统计 2025/2026 成交笔数 + 最后成交日
    yr = {"2025": 0, "2026": 0}
    last_trade = None
    for line in out.splitlines():
        m = re.search(r"(📤 卖出|📥 买入) (\d{8})", line)
        if m:
            d = m.group(2)
            if d.startswith("2025"): yr["2025"] += 1; last_trade = d
            elif d.startswith("2026"): yr["2026"] += 1; last_trade = d
    print(f"\n########## {label} ##########")
    for k in ("总收益率","年化收益率","最大回撤","夏普比率","网格交易次数","胜率","沪深300ETF涨幅","超额收益"):
        if k in res: print("  " + res[k])
    print(f"  2025成交: {yr['2025']}笔 | 2026成交: {yr['2026']}笔 | 最后成交日: {last_trade}")
    return out

run("4%对称 · 无过滤(验证冻结消失)", grid_pct=0.04, mode="symmetric", trend_filter=False, ma_window=250)
run("4%对称 · 开启趋势过滤(推荐·可信对照)", grid_pct=0.04, mode="symmetric", trend_filter=True, ma_window=250)
