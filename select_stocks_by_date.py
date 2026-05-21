"""
多因子选股 - 支持指定日期（历史回看）
在指定日期，使用历史数据计算因子，选出Top N股票
"""

import sys
import os
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger

# 添加项目根目录到路径
project_root = os.path.join(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from src.data_fetcher import DataFetcher
from src.factor_calculator import FactorCalculator
from src.factor_processor import FactorProcessor
from src.stock_selector import StockSelector


def select_stocks_by_date(
    select_date: str,
    top_n: int = 5,
    lookback_years: int = 1,
    enable_quality: bool = False,
    enable_momentum: bool = False,
    enable_technical: bool = False
):
    """
    在指定日期，使用历史数据计算因子，选出Top N股票
    
    Args:
        select_date: 选股日期（格式：'20200101'）
        top_n: 选择Top N股票
        lookback_years: 回溯几年数据计算因子（默认1年）
        enable_quality: 是否启用质量因子
        enable_momentum: 是否启用动量因子
        enable_technical: 是否启用技术因子
        
    Returns:
        选中的股票DataFrame
    """
    logger.info("="*60)
    logger.info(f"开始选股：日期 {select_date}，Top {top_n}")
    logger.info("="*60)
    
    # 1. 计算数据起始日期（回溯）
    select_date_dt = datetime.strptime(select_date, "%Y%m%d")
    start_date_dt = select_date_dt - timedelta(days=365 * lookback_years)
    start_date = start_date_dt.strftime("%Y%m%d")
    end_date = select_date  # 使用选股日期作为结束日期
    
    logger.info(f"数据范围：{start_date} 至 {end_date}")
    
    # 2. 初始化数据获取器（使用本地 SQLite 数据库）
    db_path = "D:/tu-shareData/astock_daily.db"
    
    try:
        from src.data_fetcher import SQLiteDataFetcher
        data_fetcher = SQLiteDataFetcher(db_path=db_path)
        logger.info(f"✓ 使用本地数据库: {db_path}")
    except Exception as e:
        logger.error(f"无法连接本地数据库: {e}")
        logger.info("尝试使用 Tushare API...")
        # 备用：使用 Tushare API
        from src.data_fetcher import DataFetcher
        ts_token = "761165a821532fe625262d6b33e144b9859a887c004acb981c319b"
        data_fetcher = DataFetcher(
            primary_source="tushare",
            tushare_token=ts_token,
            use_akshare_backup=True,
            use_tushare_backup=False
        )
    
    # 3. 获取沪深300成分股（使用最新数据，假设成分股变化不大）
    logger.info("[1/5] 获取股票池...")
    stock_pool = data_fetcher.get_hs300_components()
    logger.info(f"✓ 获取到 {len(stock_pool)} 只股票")
    
    # 4. 初始化因子计算器
    factor_calculator = FactorCalculator(
        enable_quality=enable_quality,
        enable_momentum=enable_momentum,
        enable_technical=enable_technical
    )
    
    # 5. 计算因子（使用指定日期范围的历史数据）
    logger.info("[2/5] 计算因子...")
    
    factors_list = []
    total = len(stock_pool)
    
    for idx, (_, row) in enumerate(stock_pool.iterrows(), 1):
        code = row['code']
        
        if idx % 50 == 0:
            logger.info(f"  进度：{idx}/{total}")
        
        try:
            # 计算单只股票的因子（传入日期范围）
            factors = factor_calculator.calculate_single_stock_factors(
                code, data_fetcher, 
                start_date=start_date, 
                end_date=end_date
            )
            
            if factors:
                factors_list.append(factors)
                
        except Exception as e:
            logger.warning(f"  计算 {code} 因子失败: {e}")
            continue
    
    if len(factors_list) == 0:
        logger.error("没有成功计算任何股票的因子，退出")
        return None
    
    factors_df = pd.DataFrame(factors_list)
    logger.info(f"✓ 因子计算完成: {len(factors_df)} 只股票")
    
    # 6. 处理因子（清洗、标准化、打分）
    logger.info("[3/5] 处理因子...")
    factor_processor = FactorProcessor()
    processed_df = factor_processor.process(factors_df)
    logger.info(f"✓ 因子处理完成")
    
    # 7. 执行选股
    logger.info(f"[4/5] 执行选股（Top {top_n}）...")
    stock_selector = StockSelector()
    selected_df = stock_selector.select(processed_df, top_n=top_n)
    logger.info(f"✓ 选股完成: {len(selected_df)} 只股票")
    
    # 8. 打印结果
    logger.info("[5/5] 输出结果...")
    print("\n" + "="*70)
    print(f"选股结果（{select_date}，Top {top_n}）：")
    print("="*70)
    
    display_cols = ["rank", "code", "name", "total_score"]
    display_cols = [col for col in display_cols if col in selected_df.columns]
    print(selected_df[display_cols].head(top_n).to_string(index=False))
    print("="*70 + "\n")
    
    # 9. 保存结果
    output_dir = os.path.join(project_root, "data", "results")
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"selection_{select_date}_top{top_n}_{timestamp}.xlsx"
    output_path = os.path.join(output_dir, filename)
    
    selected_df.to_excel(output_path, index=False, engine='xlsxwriter')
    logger.info(f"✓ 结果已保存到: {output_path}")
    
    return selected_df, output_path


if __name__ == "__main__":
    # 配置日志
    logger.add("selection_by_date.log", rotation="500 MB", encoding="utf-8")
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>"
    )
    
    # 选股参数
    SELECT_DATE = "20200101"  # 选股日期
    TOP_N = 5  # 选择Top 5
    
    # 执行选股
    result = select_stocks_by_date(
        select_date=SELECT_DATE,
        top_n=TOP_N,
        lookback_years=1,  # 使用1年历史数据
        enable_quality=False,  # 是否启用质量因子
        enable_momentum=False,  # 是否启用动量因子
        enable_technical=False  # 是否启用技术因子
    )
    
    if result:
        selected_df, output_path = result
        print(f"\n✓ 选股完成！结果已保存到: {output_path}")
        print(f"选中的股票代码：{selected_df['code'].tolist()}")
    else:
        print("\n✗ 选股失败")
