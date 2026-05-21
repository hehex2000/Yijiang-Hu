#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从本地数据库选股 - 沪深300 Top N
使用核心5因子（价值、成长、质量、动量、低波动）
数据源: D:\tu-shareData\astock_daily.db

支持命令行参数：
  --start-date: 开始日期（用于计算因子）
  --end-date: 结束日期
  --top-n: 选出Top N只股票

示例：
  python select_top10_hs300.py --start-date 20190101 --end-date 20191231 --top-n 5
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger
import sys
import os
import argparse

# ========== 命令行参数 ==========
parser = argparse.ArgumentParser(description='沪深300选股 - 核心5因子')
parser.add_argument('--start-date', type=str, default='20190101', help='开始日期（用于计算因子，格式:YYYYMMDD）')
parser.add_argument('--end-date', type=str, default='20191231', help='结束日期（格式:YYYYMMDD）')
parser.add_argument('--top-n', type=int, default=5, help='选出Top N只股票')
args = parser.parse_args()

# 使用命令行参数（可被命令行覆盖）
START_DATE = args.start_date
END_DATE = args.end_date
TOP_N = args.top_n

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


# ========== 配置 ==========
# Windows路径（Git Bash访问Windows D盘）
DB_PATH = r"D:\tu-shareData\astock_daily.db"
# 如果在Mac上运行，使用以下路径
# DB_PATH = r"/Users/huyijiang/tu-sharedata/data/astock_daily.db"

def get_db_connection():
    """获取数据库连接"""
    if not os.path.exists(DB_PATH):
        raise Exception(f"数据库文件不存在: {DB_PATH}")
    return sqlite3.connect(DB_PATH)


def get_hs300_components() -> list:
    """
    获取沪深300成分股代码列表
    如果没有存储，则从AkShare获取
    """
    try:
        import akshare as ak
        logger.info("从AkShare获取沪深300成分股...")
        df = ak.index_stock_cons_csindex(symbol="000300")
        # 获取成分券代码（如 "600000"）
        codes = df['成分券代码'].astype(str).str.replace(r'\.(SH|SZ)$', '', regex=True).tolist()
        logger.info(f"✓ 获取到 {len(codes)} 只沪深300成分股")
        return codes
    except Exception as e:
        logger.error(f"获取沪深300成分股失败: {e}")
        # 返回空列表，后续会处理
        return []


def get_stock_history(conn, code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    从本地数据库获取股票历史行情（应用复权因子）
    """
    ts_code = _convert_code_to_ts_format(code)
    
    # 1. 获取日线数据（未复权）
    query_daily = """
    SELECT trade_date, open, high, low, close, vol
    FROM daily
    WHERE ts_code = ? AND trade_date >= ? AND trade_date <= ?
    ORDER BY trade_date
    """
    
    df_daily = pd.read_sql_query(query_daily, conn, params=(ts_code, start_date, end_date))
    
    if df_daily is None or len(df_daily) == 0:
        return None
    
    # 2. 获取复权因子
    query_adj = """
    SELECT trade_date, adj_factor
    FROM adj_factor
    WHERE ts_code = ? AND trade_date >= ? AND trade_date <= ?
    ORDER BY trade_date
    """
    
    df_adj = pd.read_sql_query(query_adj, conn, params=(ts_code, start_date, end_date))
    
    # 3. 如果有权因子，应用它
    if df_adj is not None and len(df_adj) > 0:
        # 合并数据
        df_merged = pd.merge(df_daily, df_adj, on='trade_date', how='left')
        
        # 用ffill填充缺失的复权因子（向前填充）
        df_merged['adj_factor'] = df_merged['adj_factor'].ffill()
        
        # 检查是否有有效的复权因子
        if not pd.isna(df_merged['adj_factor'].iloc[0]):
            # 前复权：调整后的价格 = 未调整价格 × (当日复权因子 / 最新复权因子)
            latest_adj = df_merged['adj_factor'].iloc[-1]
            
            for col in ['open', 'high', 'low', 'close']:
                df_merged[col] = df_merged[col] * (df_merged['adj_factor'] / latest_adj)
            
            df_merged = df_merged[['trade_date', 'open', 'high', 'low', 'close', 'vol']]
            df_merged.columns = ['日期', '开盘', '最高', '最低', '收盘', '成交量']
        else:
            # 复权因子全部为NaN，使用未复权数据
            df_merged = df_merged[['trade_date', 'open', 'high', 'low', 'close', 'vol']]
            df_merged.columns = ['日期', '开盘', '最高', '最低', '收盘', '成交量']
    else:
        # 没有复权因子，使用未复权数据
        df_merged = df_daily
        df_merged.columns = ['日期', '开盘', '最高', '最低', '收盘', '成交量']
    
    return df_merged


def get_valuation_data(conn, code: str) -> dict:
    """
    从本地数据库获取股票估值数据（最新）
    返回字典：{PE_ttm, PB, PS_ttm, dividend_yield, market_cap}
    """
    ts_code = _convert_code_to_ts_format(code)
    
    query = """
    SELECT pe_ttm, pb, ps_ttm, dv_ttm, total_mv
    FROM daily_basic
    WHERE ts_code = ?
    ORDER BY trade_date DESC
    LIMIT 1
    """
    
    cursor = conn.cursor()
    cursor.execute(query, (ts_code,))
    row = cursor.fetchone()
    cursor.close()
    
    if row is None:
        return {}
    
    return {
        'pe_ttm': row[0],
        'pb': row[1],
        'ps_ttm': row[2],
        'dv_ttm': row[3],
        'total_mv': row[4]
    }


def calculate_factors(conn, code: str, start_date: str, end_date: str) -> dict:
    """
    计算单只股票的所有因子
    返回：因子字典
    """
    factors = {"code": code}
    
    # 获取历史行情
    hist_data = get_stock_history(conn, code, start_date, end_date)
    if hist_data is None or len(hist_data) == 0:
        return None
    
    # 获取估值数据
    val_data = get_valuation_data(conn, code)
    
    # ========== 1. 价值因子 ==========
    # VF1: 市盈率（反向）
    factors["VF1_PE"] = val_data.get('pe_ttm', np.nan)
    
    # VF2: 市净率（反向）
    factors["VF2_PB"] = val_data.get('pb', np.nan)
    
    # VF3: 市销率（反向）
    factors["VF3_PS"] = val_data.get('ps_ttm', np.nan)
    
    # VF6: 股息率（正向）
    factors["VF6_dividend_yield"] = val_data.get('dv_ttm', np.nan)
    
    # VF4_PEG 和 VF5_EV_EBITDA 需要财务数据，本地DB中没有，暂时设为NaN
    factors["VF4_PEG"] = np.nan
    factors["VF5_EV_EBITDA"] = np.nan
    
    # ========== 2. 动量因子 ==========
    close_prices = hist_data['收盘'].values
    
    # MF1: 1个月收益率
    if len(close_prices) >= 20:
        factors["MF1_return_1m"] = (close_prices[-1] / close_prices[-20]) - 1
    else:
        factors["MF1_return_1m"] = np.nan
    
    # MF2: 3个月收益率
    if len(close_prices) >= 60:
        factors["MF2_return_3m"] = (close_prices[-1] / close_prices[-60]) - 1
    else:
        factors["MF2_return_3m"] = np.nan
    
    # MF3: 6个月收益率
    if len(close_prices) >= 120:
        factors["MF3_return_6m"] = (close_prices[-1] / close_prices[-120]) - 1
    else:
        factors["MF3_return_6m"] = np.nan
    
    # MF4: 12个月收益率
    if len(close_prices) >= 250:
        factors["MF4_return_12m"] = (close_prices[-1] / close_prices[-250]) - 1
    else:
        factors["MF4_return_12m"] = np.nan
    
    # MF5: 相对强度（简化，用12个月收益率代替）
    factors["MF5_relative_strength"] = factors.get("MF4_return_12m", np.nan)
    
    # ========== 3. 低波动因子 ==========
    # 计算日收益率
    returns = np.diff(close_prices) / close_prices[:-1]
    
    # LVF1: 历史波动率（年化）
    if len(returns) >= 20:
        daily_vol = np.std(returns[-20:])
        factors["LVF1_hist_vol"] = daily_vol * np.sqrt(252)
    else:
        factors["LVF1_hist_vol"] = np.nan
    
    # LVF3: 下行波动率
    if len(returns) >= 20:
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            factors["LVF3_downside_vol"] = np.std(downside_returns) * np.sqrt(252)
        else:
            factors["LVF3_downside_vol"] = 0.0
    else:
        factors["LVF3_downside_vol"] = np.nan
    
    # LVF5: VaR（风险价值）
    if len(returns) >= 20:
        factors["LVF5_VAR"] = abs(np.percentile(returns, 5))
    else:
        factors["LVF5_VAR"] = np.nan
    
    # LVF2_beta 和 LVF4_idiosyncratic_vol 需要市场收益率，暂时设为NaN
    factors["LVF2_beta"] = np.nan
    factors["LVF4_idiosyncratic_vol"] = np.nan
    
    # ========== 4. 流动性因子（换手率/量比）==========
    try:
        ts_code = _convert_code_to_ts_format(code)
        query_liquidity = """
        SELECT turnover_rate, turnover_rate_f, volume_ratio
        FROM daily_basic
        WHERE ts_code = ? AND trade_date >= ? AND trade_date <= ?
        ORDER BY trade_date
        """
        df_liq = pd.read_sql_query(query_liquidity, conn, params=(ts_code, start_date, end_date))
        if df_liq is not None and len(df_liq) > 0:
            # LF1: 平均换手率（20日均值，正向——换手率适中为好，这里用排名处理）
            turnover_valid = df_liq['turnover_rate'].dropna()
            if len(turnover_valid) > 0:
                factors["LF1_turnover_rate"] = turnover_valid.tail(20).mean()
            else:
                factors["LF1_turnover_rate"] = np.nan

            # LF2: 换手率稳定性（20日换手率标准差的倒数，正向——越稳定越好）
            if len(turnover_valid) >= 20:
                std_turnover = turnover_valid.tail(20).std()
                factors["LF2_turnover_stability"] = 1.0 / (1.0 + std_turnover) if std_turnover > 0 else 1.0
            else:
                factors["LF2_turnover_stability"] = np.nan

            # LF3: 最新量比（正向）
            if 'volume_ratio' in df_liq.columns:
                latest_vr = df_liq['volume_ratio'].dropna()
                if len(latest_vr) > 0:
                    factors["LF3_volume_ratio"] = float(latest_vr.iloc[-1])
                else:
                    factors["LF3_volume_ratio"] = np.nan
            else:
                factors["LF3_volume_ratio"] = np.nan
        else:
            factors["LF1_turnover_rate"] = np.nan
            factors["LF2_turnover_stability"] = np.nan
            factors["LF3_volume_ratio"] = np.nan
    except Exception as e:
        logger.error(f"计算流动性因子失败 {code}: {e}")
        factors["LF1_turnover_rate"] = np.nan
        factors["LF2_turnover_stability"] = np.nan
        factors["LF3_volume_ratio"] = np.nan

    # 添加股票名称
    factors["name"] = get_stock_name(conn, code)
    factors["market_cap"] = val_data.get('total_mv', np.nan)
    
    # 保存实际收益率（用于输出）
    if 'MF1_return_1m' in factors and not pd.isna(factors['MF1_return_1m']):
        factors["MF1_return_1m_actual"] = factors['MF1_return_1m']
    else:
        factors["MF1_return_1m_actual"] = np.nan
    
    return factors


def get_stock_name(conn, code: str) -> str:
    """获取股票名称"""
    ts_code = _convert_code_to_ts_format(code)
    query = "SELECT name FROM stock_basic WHERE ts_code = ?"
    cursor = conn.cursor()
    cursor.execute(query, (ts_code,))
    row = cursor.fetchone()
    cursor.close()
    
    if row:
        return row[0]
    return ""


def _convert_code_to_ts_format(code: str) -> str:
    """将简单代码转换为Tushare格式（如 000001 -> 000001.SZ）"""
    if '.' in code:
        return code
    
    # 上海：60xxxx, 50xxxx, 51xxxx
    # 深圳：00xxxx, 30xxxx
    if code.startswith('6') or code.startswith('5'):
        return f"{code}.SH"
    else:
        return f"{code}.SZ"


def normalize_factors(df_factors: pd.DataFrame) -> pd.DataFrame:
    """
    因子归一化（z-score）
    正向因子：值越大越好
    反向因子：值越小越好
    """
    # 正向因子（值越大越好）
    positive_factors = [
        "VF6_dividend_yield",
        "MF1_return_1m", "MF2_return_3m", "MF3_return_6m", "MF4_return_12m",
        "MF5_relative_strength",
        # 流动性因子（正向）
        "LF1_turnover_rate",      # 平均换手率（流动性越好）
        "LF2_turnover_stability", # 换手率稳定性（越稳定越好）
        "LF3_volume_ratio",        # 量比（放量越好）
    ]
    
    # 反向因子（值越小越好）
    negative_factors = [
        "VF1_PE", "VF2_PB", "VF3_PS", "VF4_PEG", "VF5_EV_EBITDA",
        "LVF1_hist_vol", "LVF2_beta", "LVF3_downside_vol", 
        "LVF4_idiosyncratic_vol", "LVF5_VAR"
    ]
    
    df_norm = df_factors.copy()
    
    # 归一化正向因子（z-score）
    for factor in positive_factors:
        if factor in df_norm.columns:
            values = df_norm[factor].astype(float)
            mean_val = values.mean()
            std_val = values.std()
            if std_val > 0:
                df_norm[factor] = (values - mean_val) / std_val
            else:
                df_norm[factor] = 0
    
    # 归一化反向因子（z-score，然后取负）
    for factor in negative_factors:
        if factor in df_norm.columns:
            values = df_norm[factor].astype(float)
            mean_val = values.mean()
            std_val = values.std()
            if std_val > 0:
                df_norm[factor] = -(values - mean_val) / std_val  # 取负，使方向一致
            else:
                df_norm[factor] = 0
    
    return df_norm


def calculate_total_score(df_norm: pd.DataFrame) -> pd.DataFrame:
    """
    计算总分（所有可用因子的等权平均）
    """
    # 所有因子列（排除code, name, market_cap）
    factor_cols = [col for col in df_norm.columns 
                   if col not in ['code', 'name', 'market_cap']]
    
    # 计算总分（忽略NaN）
    df_norm['total_score'] = df_norm[factor_cols].mean(axis=1, skipna=True)
    
    return df_norm


def select_top_stocks(conn,
                      start_date: str,
                      end_date: str,
                      top_n: int = 5) -> pd.DataFrame:
    """
    选股核心逻辑（可调用函数）
    从沪深300中根据多因子打分选出Top N股票

    Args:
        conn: SQLite数据库连接
        start_date: 因子计算开始日期（格式: YYYYMMDD）
        end_date: 因子计算结束日期（格式: YYYYMMDD）
        top_n: 选出前N只股票

    Returns:
        DataFrame，columns=[code, name, market_cap, total_score, VF1_PE, ...]
        如果失败返回空DataFrame
    """
    logger.info(f"select_top_stocks: {start_date} ~ {end_date}, top_n={top_n}")

    # 1. 获取沪深300成分股
    hs300_codes = get_hs300_components()
    if len(hs300_codes) == 0:
        logger.error("未能获取沪深300成分股，返回空结果")
        return pd.DataFrame()

    # 2. 计算所有股票的因子
    all_factors = []
    for code in hs300_codes:
        try:
            factors = calculate_factors(conn, code, start_date, end_date)
            if factors is not None:
                all_factors.append(factors)
        except Exception as e:
            logger.error(f"计算因子失败 {code}: {e}")

    if len(all_factors) == 0:
        logger.error("没有成功计算任何股票的因子，返回空结果")
        return pd.DataFrame()

    # 3. 转换为DataFrame
    df_factors = pd.DataFrame(all_factors)

    # 4. 因子归一化
    df_norm = normalize_factors(df_factors)

    # 5. 计算总分
    df_score = calculate_total_score(df_norm)

    # 6. 排序，选出Top N
    df_top = df_score.sort_values('total_score', ascending=False).head(top_n)

    logger.info(f"✓ 选股完成: {len(df_top)} 只")
    return df_top


def main():
    """主函数（命令行入口）"""
    print("=" * 70)
    print("沪深300选股 - 核心5因子")
    print("=" * 70)
    print(f"\n数据库: {DB_PATH}")
    print(f"时间范围: {START_DATE} 到 {END_DATE}")
    print(f"选出Top: {TOP_N} 只股票\n")
    
    # 1. 连接数据库
    logger.info("连接本地数据库...")
    try:
        conn = get_db_connection()
        logger.info("✓ 数据库连接成功")
    except Exception as e:
        logger.error(f"数据库连接失败: {e}")
        return
    
    # 2. 获取沪深300成分股
    print("\n[1/4] 获取沪深300成分股...")
    hs300_codes = get_hs300_components()
    
    if len(hs300_codes) == 0:
        logger.error("未能获取沪深300成分股，退出")
        conn.close()
        return
    
    print(f"  ✓ 获取到 {len(hs300_codes)} 只成分股\n")
    
    # 3. 计算所有股票的因子
    print(f"[2/4] 计算因子（共 {len(hs300_codes)} 只股票）...")
    all_factors = []
    failed = []
    
    for i, code in enumerate(hs300_codes):
        try:
            factors = calculate_factors(conn, code, START_DATE, END_DATE)
            if factors is not None:
                all_factors.append(factors)
            
            if (i + 1) % 50 == 0:
                print(f"  进度: {i+1}/{len(hs300_codes)}")
        except Exception as e:
            logger.error(f"计算因子失败 {code}: {e}")
            failed.append(code)
    
    if len(all_factors) == 0:
        logger.error("没有成功计算任何股票的因子，退出")
        conn.close()
        return
    
    print(f"  ✓ 成功计算 {len(all_factors)} 只股票的因子")
    if len(failed) > 0:
        print(f"  ⚠ 失败 {len(failed)} 只: {failed[:5]}")
    print()
    
    # 4. 转换为DataFrame
    df_factors = pd.DataFrame(all_factors)
    
    # 5. 因子归一化
    print("[3/4] 因子归一化（z-score）...")
    df_norm = normalize_factors(df_factors)
    print("  ✓ 归一化完成\n")
    
    # 6. 计算总分
    print("[4/4] 计算总分...")
    df_score = calculate_total_score(df_norm)
    print("  ✓ 总分计算完成\n")
    
    # 7. 排序，选出Top N
    df_top = df_score.sort_values('total_score', ascending=False).head(TOP_N)
    
    # 8. 输出结果
    print("=" * 70)
    print(f"沪深300 Top {TOP_N} 股票（按总分排序）")
    print("=" * 70 + "\n")
    
    for idx, row in df_top.iterrows():
        print(f"{row['code']}  {row['name']:10}  总分: {row['total_score']:.4f}  "
              f"市值: {row['market_cap']:.0f}万")
        
        # 打印主要因子（实际值，不是归一化的z-score）
        # PE和PB显示实际值
        print(f"  VF1_PE:{row.get('VF1_PE_actual', row['VF1_PE']):.2f}  "
              f"VF2_PB:{row.get('VF2_PB_actual', row['VF2_PB']):.2f}  "
              f"MF1_ret1m:{row.get('MF1_return_1m_actual', row['MF1_return_1m']):.2%}  "
              f"LVF1_vol:{row.get('LVF1_hist_vol_actual', row['LVF1_hist_vol']):.2%}")
        print()
    
    # 9. 保存到CSV
    output_file = os.path.join(project_root, 'data', 'results', 
                              f'hs300_top{TOP_N}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_top.to_csv(output_file, index=False, encoding='utf-8-sig')
    print("=" * 70)
    print(f"结果已保存: {output_file}")
    print("=" * 70)
    
    conn.close()


if __name__ == "__main__":
    main()
