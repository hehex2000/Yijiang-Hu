#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
价值投资策略 - 选股脚本
使用方法：python run_value_selection.py
配置修改：编辑 config.py 中的 VALUE_STRATEGY 节
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.data_fetcher import DataFetcher
from src.value_stock_selector import ValueStockSelector
from loguru import logger


def main():
    """主函数"""
    # 1. 读取配置
    value_config = config.VALUE_STRATEGY
    data_config = config.DATA
    
    logger.info("=" * 60)
    logger.info("价值投资策略 - 选股系统")
    logger.info("=" * 60)
    logger.info(f"选股日期: {value_config.get('date')}")
    logger.info(f"股票池: {value_config.get('stock_pool')}")
    logger.info("=" * 60)
    
    # 2. 初始化数据获取器
    logger.info("[1/4] 初始化数据获取器...")
    data_fetcher = DataFetcher(
        primary_source=data_config["primary_source"],
        tushare_token=data_config["tushare_token"],
        use_akshare_backup=data_config["use_akshare_backup"],
        use_tushare_backup=data_config["use_tushare_backup"],
        local_db_path=data_config["local_db_path"],
    )
    
    # 3. 初始化价值选股器
    logger.info("[2/4] 初始化价值选股器...")
    selector = ValueStockSelector(
        config=value_config,
        data_fetcher=data_fetcher,
    )
    
    # 4. 执行选股
    logger.info("[3/4] 执行选股...")
    result_df = selector.select_stocks(
        date=value_config.get("date"),
        top_n=value_config.get("top_n", 0),
    )
    
    if len(result_df) == 0:
        logger.warning("未找到符合条件的股票！")
        return
    
    logger.info(f"✓ 选股完成，共找到 {len(result_df)} 只符合条件的股票")
    
    # 5. 导出结果
    logger.info("[4/4] 导出结果...")
    output_path = selector.export_to_csv(result_df)
    logger.info(f"✓ 结果已保存到: {output_path}")
    
    # 6. 打印前N只股票
    print("\n" + "=" * 60)
    print("选股结果（前20只）")
    print("=" * 60)
    print(result_df.head(20).to_string(index=False))
    

if __name__ == "__main__":
    main()
