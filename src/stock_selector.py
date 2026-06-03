"""
选股核心模块 - StockSelector Class
负责最终筛选和结果导出
"""

import pandas as pd
import os
from typing import Optional
from datetime import datetime
from loguru import logger


class StockSelector:
    """选股器"""
    
    def __init__(self, config: Optional[dict] = None):
        """
        初始化选股器
        
        Args:
            config: 配置字典（可选）
        """
        self.config = config or {}
        
        # 默认配置
        self.top_n = self.config.get("top_n", 50)
        self.min_score = self.config.get("min_score", 0.0)
        self.output_dir = self.config.get("output_dir", "data/results")
        
        logger.info(f"StockSelector initialized (top_n={self.top_n})")
    
    def select(self, factors_df: pd.DataFrame, 
               top_n: Optional[int] = None,
               min_score: Optional[float] = None) -> pd.DataFrame:
        """
        执行选股（筛选TOP N股票）
        
        Args:
            factors_df: 处理后的因子DataFrame（已打分和排名）
            top_n: 选择TOP N股票（可选，使用配置值）
            min_score: 最小得分阈值（可选，使用配置值）
            
        Returns:
            筛选后的DataFrame
        """
        top_n = top_n or self.top_n
        min_score = min_score or self.min_score
        
        logger.info(f"Selecting top {top_n} stocks (min_score={min_score})...")
        
        df = factors_df.copy()
        
        # 0. 过滤 current_price 为 NaN 的行（选股日停牌/无数据，无法交易）
        if "current_price" in df.columns:
            before = len(df)
            df = df[df["current_price"].notna()]
            filtered = before - len(df)
            if filtered > 0:
                logger.warning(f"Filtered {filtered} stocks with NaN current_price (suspended on selection date)")
        
        # 1. 按综合得分排序（如果还没有排序）
        if "total_score" in df.columns:
            df = df.sort_values("total_score", ascending=False).reset_index(drop=True)
            df["rank"] = df.index + 1
        
        # 2. 筛选：最小得分阈值
        if min_score > 0 and "total_score" in df.columns:
            df = df[df["total_score"] >= min_score]
            logger.info(f"After min_score filter: {len(df)} stocks")
        
        # 3. 选择TOP N
        if top_n > 0:
            df = df.head(top_n)
            logger.info(f"Selected top {len(df)} stocks")
        
        return df
    
    def export_to_csv(self, df: pd.DataFrame, 
                       filename: Optional[str] = None,
                       output_dir: Optional[str] = None) -> str:
        """
        导出结果到CSV文件（简化格式）
        
        Args:
            df: 要导出的DataFrame
            filename: 文件名（可选，自动生成，格式: multi-YYYYMM-selection.csv）
            output_dir: 输出目录（可选，使用配置值）
            
        Returns:
            保存的文件路径
        """
        """
        导出结果到CSV文件（简化格式）
        
        Args:
            df: 要导出的DataFrame
            filename: 文件名（可选，自动生成）
            output_dir: 输出目录（可选，使用配置值）
            
        Returns:
            保存的文件路径
        """
        output_dir = output_dir or self.output_dir
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成文件名（格式: multi-YYYYMM-selection.csv）
        if filename is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m")
            filename = f"multi-{timestamp}-selection.csv"
        
        output_path = os.path.join(output_dir, filename)
        
        # 重命名列为中文
        rename_map = {
            'rank': '排名',
            'code': '股票代码',
            'name': '股票名称',
            'market_cap': '市值',
            'current_price': '最近收盘价',
            'total_score': '总得分'
        }
        
        # 只重命名存在的列
        actual_rename = {k: v for k, v in rename_map.items() if k in df.columns}
        export_df = df.rename(columns=actual_rename)
        
        # 按顺序排列输出列
        desired_order = list(actual_rename.values())
        export_cols = [c for c in desired_order if c in export_df.columns]
        
        export_df = export_df[export_cols]
        
        # 将市值转换为亿元（如果不是 already in 亿元）
        if '市值' in export_df.columns:
            # 如果市值 > 1e6，认为是"元"，需要转换为"亿元"
            if export_df['市值'].max() > 1e6:
                export_df['市值'] = export_df['市值'] / 1e8
            # 格式化为字符串（保留两位小数）
            export_df['市值'] = export_df['市值'].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
        
        # 确保'股票代码'列为字符串格式（保留前导零）
        if '股票代码' in export_df.columns:
            export_df['股票代码'] = export_df['股票代码'].astype(str)
        
        # 导出到CSV（使用utf-8-sig编码，支持中文）
        export_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        
        logger.info(f"✓ Results exported to CSV: {output_path}")
        
        return output_path
    
    def export_to_excel(self, df: pd.DataFrame,
                         filename: Optional[str] = None,
                         output_dir: Optional[str] = None,
                         sheet_name: str = "Selection") -> str:
        """
        导出结果到Excel文件（简化格式）
        
        Args:
            df: 要导出的DataFrame
            filename: 文件名（可选，自动生成）
            output_dir: 输出目录（可选，使用配置值）
            sheet_name: sheet名称
            
        Returns:
            保存的文件路径
        """
        output_dir = output_dir or self.output_dir
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成文件名
        if filename is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m")
            filename = f"selection_{timestamp}.xlsx"
        
        output_path = os.path.join(output_dir, filename)
        
        # 重命名列为中文
        rename_map = {
            'rank': '排名',
            'code': '股票代码',
            'name': '股票名称',
            'market_cap': '市值',
            'current_price': '最近收盘价',
            'total_score': '总得分'
        }
        
        # 只重命名存在的列
        actual_rename = {k: v for k, v in rename_map.items() if k in df.columns}
        export_df = df.rename(columns=actual_rename)
        
        # 按顺序排列输出列
        desired_order = list(actual_rename.values())
        export_cols = [c for c in desired_order if c in export_df.columns]
        
        export_df = export_df[export_cols]
        
        # 将市值转换为亿元（如果不是 already in 亿元）
        if '市值' in export_df.columns:
            # 如果市值 > 1e6，认为是"元"，需要转换为"亿元"
            if export_df['市值'].max() > 1e6:
                export_df['市值'] = export_df['市值'] / 1e8
        
        # 确保'股票代码'列为字符串格式（保留前导零）
        if '股票代码' in export_df.columns:
            export_df['股票代码'] = export_df['股票代码'].astype(str).str.zfill(6)
        
        # 导出到Excel
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            export_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # 设置'市值'列为数字格式（保留两位小数）
            if '市值' in export_df.columns:
                worksheet = writer.sheets[sheet_name]
                col_idx = export_df.columns.get_loc('市值') + 1  # Excel is 1-based
                for row in range(2, len(export_df) + 2):
                    cell = worksheet.cell(row=row, column=col_idx)
                    if cell.value is not None:
                        try:
                            cell.number_format = '0.00"亿元"'
                            cell.value = float(cell.value)
                        except:
                            pass
            
            # 设置'股票代码'列为文本格式（保留前导零）
            if '股票代码' in export_df.columns:
                worksheet = writer.sheets[sheet_name]
                # 找到'股票代码'列的索引（Excel是1-based，且不使用index）
                col_idx = export_df.columns.get_loc('股票代码') + 1
                
                # 设置整列为文本格式
                for row in range(2, len(export_df) + 2):  # 从第2行开始（第1行是表头）
                    cell = worksheet.cell(row=row, column=col_idx)
                    cell.number_format = '@'  # @ 表示文本格式
                    # 确保值是字符串且保留前导零
                    if cell.value is not None:
                        cell.value = str(cell.value).zfill(6)
        
        logger.info(f"✓ Results exported to Excel: {output_path}")
        
        return output_path
    
    def print_top_stocks(self, df: pd.DataFrame, n: int = 10):
        """
        打印TOP N股票（格式：股票代码、股票名称、市值、最近收盘价、总得分）
        
        Args:
            df: 选股结果DataFrame
            n: 打印前N只股票
        """
        print("\n" + "="*70)
        print(f"TOP {min(n, len(df))} 股票:")
        print("="*70)
        
        # 重命名列为中文
        rename_map = {
            'rank': '排名',
            'code': '股票代码',
            'name': '股票名称',
            'market_cap': '市值',
            'current_price': '最近收盘价',
            'total_score': '总得分'
        }
        
        # 只重命名存在的列
        actual_rename = {k: v for k, v in rename_map.items() if k in df.columns}
        if actual_rename:
            display_df = df.rename(columns=actual_rename)
        else:
            display_df = df
        
        # 只显示存在的列（中文名）
        display_cols = list(actual_rename.values())
        display_cols = [col for col in display_cols if col in display_df.columns]
        
        # 打印前，将市值转换为亿元（如果不是 already in 亿元）
        df_display = display_df[display_cols].head(n).copy()
        if '市值' in df_display.columns:
            # 如果市值 > 1e6，认为是"元"，需要转换为"亿元"
            if df_display['市值'].max() > 1e6:
                df_display['市值'] = df_display['市值'] / 1e8
            # 格式化为字符串（保留两位小数，加"亿元"后缀）
            df_display['市值'] = df_display['市值'].map(lambda x: f"{x:.2f}亿" if pd.notna(x) else "")
        
        # 打印
        if display_cols:
            # 设置 Pandas 显示选项，避免科学计数法
            pd.set_option('display.float_format', lambda x: '%.2f' % x)
            print(df_display.to_string(index=False))
        else:
            print(df.head(n).to_string(index=False))
        
        print("="*70 + "\n")


if __name__ == "__main__":
    # 测试代码
    from loguru import logger
    import sys
    import os
    
    # 添加src目录到路径
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    
    from src.data_fetcher import DataFetcher
    from src.factor_calculator import FactorCalculator
    from src.factor_processor import FactorProcessor
    
    # 初始化日志
    logger.add("stock_selector_test.log", rotation="500 MB")
    
    print("\n" + "="*70)
    print("多因子选股系统 - 测试")
    print("="*70 + "\n")
    
    # 1. 初始化所有模块
    print("[1/5] 初始化模块...")
    data_fetcher = DataFetcher(use_tushare=False)
    factor_calculator = FactorCalculator()
    factor_processor = FactorProcessor()
    stock_selector = StockSelector(config={"top_n": 20})
    print("✓ 模块初始化完成\n")
    
    # 2. 获取沪深300成分股（测试用前10只）
    print("[2/5] 获取股票池...")
    hs300 = data_fetcher.get_hs300_components()
    test_codes = hs300["code"].head(10).tolist()
    print(f"✓ 测试股票池: {len(test_codes)} 只股票\n")
    
    # 3. 计算因子
    print("[3/5] 计算因子...")
    factors_df = factor_calculator.calculate_all_factors(test_codes, data_fetcher)
    print(f"✓ 因子计算完成: {len(factors_df)} 只股票\n")
    
    # 4. 处理因子（清洗、标准化、打分）
    print("[4/5] 处理因子...")
    processed_df = factor_processor.process(factors_df)
    print(f"✓ 因子处理完成\n")
    
    # 5. 选股和导出
    print("[5/5] 执行选股...")
    selected_df = stock_selector.select(processed_df, top_n=10)
    print(f"✓ 选股完成: {len(selected_df)} 只股票\n")
    
    # 打印TOP 10
    stock_selector.print_top_stocks(selected_df, n=10)
    
    # 导出结果
    output_path = stock_selector.export_to_csv(selected_df)
    print(f"✓ 结果已保存到: {output_path}")
    
    print("\n" + "="*70)
    print("测试完成!")
    print("="*70 + "\n")
