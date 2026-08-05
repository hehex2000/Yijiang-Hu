# -*- coding: utf-8 -*-
"""通用增量补数：把任意指数的日行情从「本地库现有最大交易日 +1」续接到最新(默认今天)。
采用 INSERT OR REPLACE(主键 ts_code,trade_date) 幂等写入，绝不 DELETE 已有数据，可安全重复跑。
典型用途：补齐对比表所依赖的跟踪指数序列(000698 科创100 / 932000 中证2000 / 000985 中证全指 等)。

用法:
    python download_index_daily.py --ts-code 000985.SH
    python download_index_daily.py --ts-code 932000.SH --tushare-code 932000.CSI
    python download_index_daily.py --ts-code 000698.SH
依赖: tushare(1.4.29), pandas, config.DATA
"""
import sys, os, argparse, sqlite3, time
from datetime import datetime, timedelta
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import tushare as ts
from config import DATA

DB_PATH = DATA.get("local_db_path", r"D:\tu-shareData\astock_daily.db")
TOKEN = DATA.get("tushare_token", "")

COLS = ["ts_code", "trade_date", "close", "open", "high", "low",
        "pre_close", "change", "pct_chg", "vol", "amount"]


def get_pro():
    if not TOKEN:
        print("[错误] config.DATA.tushare_token 为空，无法调用 Tushare")
        sys.exit(1)
    ts.set_token(TOKEN)
    return ts.pro_api()


# ---- 限速 + 重试（与 download_etf_adj.py 同源，卡在 200/分钟以内）----
_CALLS = []          # 滑动窗口调用时间戳
_MAX_CALLS = 175     # 60 秒内最多调用次数（< 180，留余量）
_WINDOW = 60.0


def _ratelimit():
    now = time.time()
    while _CALLS and now - _CALLS[0] >= _WINDOW:
        _CALLS.pop(0)
    if len(_CALLS) >= _MAX_CALLS:
        sleep_for = _WINDOW - (now - _CALLS[0]) + 0.5
        if sleep_for > 0:
            print(f"    [限速] 已达 {_MAX_CALLS}/分钟，休眠 {sleep_for:.1f}s")
            time.sleep(sleep_for)
        now = time.time()
        while _CALLS and now - _CALLS[0] >= _WINDOW:
            _CALLS.pop(0)
    _CALLS.append(time.time())


def _is_ratelimit_err(ex):
    s = str(ex)
    return ("每分钟" in s) or ("每分钟" in s.lower()) or ("rate" in s.lower()) \
        or ("freq" in s.lower()) or ("超限" in s) or ("频率" in s)


def _call_with_retry(fn, **kwargs):
    delays = [5, 10, 20, 40, 60, 90]
    for i, d in enumerate(delays):
        _ratelimit()
        try:
            return fn(**kwargs)
        except Exception as ex:
            if _is_ratelimit_err(ex) and i < len(delays) - 1:
                print(f"    [限流重试 {i+1}/{len(delays)}] {ex}  休眠 {d}s")
                time.sleep(d)
                continue
            raise
    return None


def last_date(conn, ts_code):
    r = conn.execute("SELECT MAX(trade_date) FROM index_daily WHERE ts_code=?",
                    (ts_code,)).fetchone()
    return r[0]


def download(pro, ts_code, tushare_code, start, end):
    print(f"[补数] {ts_code} (Tushare={tushare_code}) {start}~{end}")
    sy, ey = int(start[:4]), int(end[:4])
    frames = []
    for y in range(sy, ey + 1):
        s = max(start, f"{y}0101")
        e = f"{y}1231"
        if e > end:
            e = end
        if s > e:
            continue
        df = None
        # 空结果也重试：index_daily 接口偶发瞬时返回空 DataFrame（不报错）
        for _att in range(3):
            try:
                df = _call_with_retry(pro.index_daily, ts_code=tushare_code,
                                      start_date=s, end_date=e)
            except Exception as ex:
                print(f"    [失败重试] {y}: {ex}")
                df = None
            if df is not None and not df.empty:
                break
            print(f"    [空/重试] {y} 第{_att+1}次为空，8s 后重试")
            time.sleep(8)
        if df is None or df.empty:
            print(f"    [跳过] {y}: 多次重试仍为空")
            continue
        frames.append(df)
    if not frames:
        print("    [空] 无数据返回（可能接口权限/网络问题）")
        return 0
    big = pd.concat(frames, ignore_index=True)
    big["ts_code"] = ts_code
    big["trade_date"] = big["trade_date"].astype(int)
    big = big[[c for c in COLS if c in big.columns]]
    conn = sqlite3.connect(DB_PATH)
    conn.executemany(
        "INSERT OR REPLACE INTO index_daily "
        "(ts_code,trade_date,close,open,high,low,pre_close,change,pct_chg,vol,amount) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        big.values.tolist(),
    )
    conn.commit()
    conn.close()
    print(f"    [完成] 写入/更新 {len(big)} 行 ({big['trade_date'].min()}~{big['trade_date'].max()})")
    return len(big)


def main():
    ap = argparse.ArgumentParser(description="增量补指数日行情到 index_daily")
    ap.add_argument("--ts-code", required=True, help="本地存储代码 如 000985.SH")
    ap.add_argument("--tushare-code", default=None,
                    help="Tushare 真实代码(默认同 ts-code)，如 000698.SH / 932000.CSI")
    ap.add_argument("--end", default=datetime.now().strftime("%Y%m%d"),
                    help="结束日期 YYYYMMDD(默认今天)")
    ap.add_argument("--start-override", default=None,
                    help="强制起始日期(默认:库内现有 max+1 天)")
    args = ap.parse_args()
    tushare_code = args.tushare_code or args.ts_code
    pro = get_pro()
    conn = sqlite3.connect(DB_PATH)
    last = last_date(conn, args.ts_code)
    conn.close()
    if args.start_override:
        start = args.start_override
    elif last:
        d = datetime.strptime(str(last), "%Y%m%d").date() + timedelta(days=1)
        start = d.strftime("%Y%m%d")
    else:
        start = "20100101"
    print(f"  库内现有最大交易日={last}，本次从 {start} 续接至 {args.end}")
    if int(start) > int(args.end):
        print("  [跳过] 库内数据已晚于结束日，无需补。")
        return
    download(pro, args.ts_code, tushare_code, start, args.end)


if __name__ == "__main__":
    main()
