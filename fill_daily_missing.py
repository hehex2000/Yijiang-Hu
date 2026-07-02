"""
补全 daily 表缺失的日线数据
按交易日遍历，发现某天数据不够全时就重新下载
使用 INSERT OR REPLACE，重复运行不会产生重复行
"""
import sys, os, time, sqlite3
import pandas as pd

sys.path.insert(0, r"C:\Users\99395\workbuddy\Tushare-Downloader")
import tushare as ts
import config as dl_config
import utils

DB_PATH = r"D:\tu-shareData\astock_daily.db"
TOTAL_STOCKS = 5200

ts.set_token(dl_config.TUSHARE_TOKEN)
pro = ts.pro_api()


def get_date_coverage(conn):
    rows = conn.execute(
        "SELECT trade_date, COUNT(*) AS cnt FROM daily GROUP BY trade_date ORDER BY trade_date"
    ).fetchall()
    return {r[0]: r[1] for r in rows}


def download_date(trade_date):
    try:
        utils.rate_limiter.wait()
        df = pro.daily(trade_date=trade_date)
        if df is None or df.empty:
            return 0
        fields = ["ts_code", "trade_date", "open", "high", "low", "close",
                  "pre_close", "change", "pct_chg", "vol", "amount"]
        rows_data = df[fields].values.tolist()
        utils.bulk_insert("daily", fields, rows_data)
        return len(rows_data)
    except Exception as e:
        print(f"    [ERR] {trade_date}: {e}")
        return 0


def main(start_year=None, end_year=None):
    conn = sqlite3.connect(DB_PATH)
    coverage = get_date_coverage(conn)
    all_dates = sorted(coverage.keys())

    if not all_dates:
        print("[ERROR] daily 表无数据")
        conn.close()
        return

    print(f"daily 表交易日范围: {all_dates[0]} ~ {all_dates[-1]}")
    print(f"总交易日数: {len(all_dates)}")

    if start_year is None:
        start_year = all_dates[0][:4]
    if end_year is None:
        end_year = all_dates[-1][:4]

    target_dates = [d for d in all_dates if start_year <= d[:4] <= end_year]
    print(f"检查范围: {start_year}年 ~ {end_year}年 ({len(target_dates)} 个交易日)\n")

    low_coverage = [(d, c) for d, c in sorted(coverage.items(), key=lambda x: x[1])
                    if c < TOTAL_STOCKS * 0.7 and start_year <= d[:4] <= end_year]

    print(f"数据不完整（<70%）的交易日: {len(low_coverage)} 个\n")

    total_filled = 0
    total_dates = 0
    total_missing = sum(TOTAL_STOCKS - c for _, c in low_coverage)

    for i, (trade_date, cnt) in enumerate(low_coverage):
        missing = TOTAL_STOCKS - cnt
        print(f"  [{i+1}/{len(low_coverage)}] {trade_date}: 现有 {cnt} 只, 缺失 {missing} 只")

        rows = download_date(trade_date)
        if rows > 0:
            total_filled += rows
            total_dates += 1
            print(f"    [OK] 补入 {rows} 行")
        else:
            print(f"    [SKIP] 未下载到数据")

        if (i + 1) % 10 == 0:
            print(f"\n  进度: {i+1}/{len(low_coverage)} 天, "
                  f"已补 {total_filled:,} 行 / 共缺 {total_missing:,} 行\n")

    new_total = conn.execute("SELECT COUNT(*) FROM daily").fetchone()[0]
    conn.close()

    print(f"\n{'='*60}")
    print(f"  补全完成!")
    print(f"  处理 {total_dates} 个交易日, 补入 {total_filled:,} 行")
    print(f"  daily 表总行数: {new_total:,}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="补全 daily 表缺失的日线数据")
    parser.add_argument("--start-year", type=str, default=None)
    parser.add_argument("--end-year", type=str, default=None)
    args = parser.parse_args()
    main(start_year=args.start_year, end_year=args.end_year)
