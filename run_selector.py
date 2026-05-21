"""
通用多因子选股入口脚本
使用方法:
    python run_selector.py                          # 使用默认配置
    python run_selector.py --config config/selection_config.yaml  # 使用指定配置
"""

import sqlite3
import pandas as pd
import yaml
import argparse
import os
import sys
from loguru import logger
from typing import List, Dict

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(__file__))

# 导入src模块
from src.factor_calculator import FactorCalculator
from src.factor_processor import FactorProcessor
from src.stock_selector import StockSelector


# ==================== 数据获取函数 ====================
def get_all_a_stocks(db_path: str, list_date_before: str) -> List[str]:
    """获取所有A股代码列表"""
    conn = sqlite3.connect(db_path)
    
    query = f"""
        SELECT ts_code FROM stock_basic
        WHERE list_date <= '{list_date_before}'
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # 转换为简单格式（000063，不是000063.SZ）
    codes = []
    for ts_code in df['ts_code'].tolist():
        if ts_code.endswith('.SZ'):
            codes.append(ts_code[:6])
        elif ts_code.endswith('.SH'):
            codes.append(ts_code[:6])
    
    logger.info(f"获取到 {len(codes)} 只A股")
    return codes


def get_value_factors(code: str, db_path: str, date: str) -> Dict:
    """
    获取价值因子（从SQLite数据库）
    因子: PE、PB、PS、股息率
    """
    conn = sqlite3.connect(db_path)
    
    query = f"""
        SELECT d.pe, d.pb, d.ps, d.dv_ratio, s.name, d.total_mv
        FROM daily_basic d
        JOIN stock_basic s ON d.ts_code = s.ts_code
        WHERE d.ts_code = (
            SELECT CASE 
                WHEN SUBSTR('{code}', 1, 1) IN ('6') THEN '{code}.SH'
                ELSE '{code}.SZ'
            END
        )
        AND d.trade_date = '{date}'
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if len(df) == 0:
        return None
    
    factors = {}
    
    # 1. 市盈率（PE）-- 越低越好
    if not pd.isna(df['pe'].iloc[0]) and df['pe'].iloc[0] > 0:
        factors['pe'] = -df['pe'].iloc[0]  # 负值表示越低越好
    
    # 2. 市净率（PB）-- 越低越好
    if not pd.isna(df['pb'].iloc[0]) and df['pb'].iloc[0] > 0:
        factors['pb'] = -df['pb'].iloc[0]
    
    # 3. 市销率（PS）-- 越低越好
    if not pd.isna(df['ps'].iloc[0]) and df['ps'].iloc[0] >= 0:
        factors['ps'] = -df['ps'].iloc[0]
    
    # 4. 股息率 -- 越高越好
    if not pd.isna(df['dv_ratio'].iloc[0]) and df['dv_ratio'].iloc[0] > 0:
        factors['dv_ratio'] = df['dv_ratio'].iloc[0]
    
    # 保存股票名称和市值
    if not pd.isna(df['name'].iloc[0]):
        factors['name'] = df['name'].iloc[0]
    
    if not pd.isna(df['total_mv'].iloc[0]):
        factors['market_cap'] = df['total_mv'].iloc[0] / 10000  # 转换为亿元
    
    # PE和PB至少有一个有效才返回
    return factors if ('pe' in factors or 'pb' in factors) else None


def get_momentum_factors(code: str, db_path: str, base_date: str) -> Dict:
    """
    计算动量因子（从SQLite daily表）
    因子: 1月收益率、3月收益率、6月收益率、12月收益率
    """
    conn = sqlite3.connect(db_path)
    try:
        # 需要足够历史数据（约250个交易日 = 约1年）
        from datetime import datetime, timedelta
        base = datetime.strptime(base_date, '%Y%m%d')
        start_1y = (base - timedelta(days=400)).strftime('%Y%m%d')

        ts_code = _to_ts_code(code)
        query = f"""
            SELECT trade_date, close
            FROM daily
            WHERE ts_code = '{ts_code}'
              AND trade_date BETWEEN '{start_1y}' AND '{base_date}'
            ORDER BY trade_date
        """
        df = pd.read_sql_query(query, conn)
        conn.close()

        if len(df) < 20:
            return None

        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df = df.dropna(subset=['close']).reset_index(drop=True)
        prices = df['close'].values

        factors = {}
        n = len(prices)

        # 1月收益率（约20个交易日）
        if n >= 20:
            factors['ret_1m'] = (prices[-1] / prices[-20]) - 1
        else:
            factors['ret_1m'] = np.nan

        # 3月收益率（约60个交易日）
        if n >= 60:
            factors['ret_3m'] = (prices[-1] / prices[-60]) - 1
        else:
            factors['ret_3m'] = np.nan

        # 6月收益率（约120个交易日）
        if n >= 120:
            factors['ret_6m'] = (prices[-1] / prices[-120]) - 1
        else:
            factors['ret_6m'] = np.nan

        # 12月收益率（约250个交易日）
        if n >= 250:
            factors['ret_12m'] = (prices[-1] / prices[-250]) - 1
        else:
            factors['ret_12m'] = np.nan

        return factors if not all(pd.isna(list(factors.values()))) else None

    except Exception as e:
        conn.close()
        logger.debug(f"动量因子计算失败 {code}: {e}")
        return None


def get_technical_factors(code: str, db_path: str, base_date: str) -> Dict:
    """
    计算技术因子（从SQLite daily表）
    因子: MA多头、20日成交量比、换手率（如有）
    """
    conn = sqlite3.connect(db_path)
    try:
        from datetime import datetime, timedelta
        base = datetime.strptime(base_date, '%Y%m%d')
        start_60d = (base - timedelta(days=120)).strftime('%Y%m%d')

        ts_code = _to_ts_code(code)
        query = f"""
            SELECT trade_date, close, vol
            FROM daily
            WHERE ts_code = '{ts_code}'
              AND trade_date BETWEEN '{start_60d}' AND '{base_date}'
            ORDER BY trade_date
        """
        df = pd.read_sql_query(query, conn)
        conn.close()

        if len(df) < 20:
            return None

        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df = df.dropna(subset=['close']).reset_index(drop=True)
        prices = df['close'].values

        factors = {}

        # MA多头: MA5 > MA10 > MA20 则得分为1，否则为0
        if len(prices) >= 20:
            ma5 = np.mean(prices[-5:])
            ma10 = np.mean(prices[-10:])
            ma20 = np.mean(prices[-20:])
            factors['ma_bullish'] = 1.0 if (ma5 > ma10 and ma10 > ma20) else 0.0
        else:
            factors['ma_bullish'] = np.nan

        # 成交量比: 当日成交量 / 20日均量
        if 'vol' in df.columns and len(df) >= 20:
            vols = pd.to_numeric(df['vol'], errors='coerce')
            vols = vols.dropna().values
            if len(vols) >= 20 and np.mean(vols[-20:]) > 0:
                factors['vol_ratio'] = vols[-1] / np.mean(vols[-20:])
            else:
                factors['vol_ratio'] = np.nan
        else:
            factors['vol_ratio'] = np.nan

        return factors if not all(pd.isna(list(factors.values()))) else None

    except Exception as e:
        conn.close()
        logger.debug(f"技术因子计算失败 {code}: {e}")
        return None


def get_volatility_factors(code: str, db_path: str, base_date: str) -> Dict:
    """
    计算低波动因子（从SQLite daily表）
    因子: 历史波动率（反向）、下行波动率（反向）、VaR（反向）
    """
    conn = sqlite3.connect(db_path)
    try:
        from datetime import datetime, timedelta
        base = datetime.strptime(base_date, '%Y%m%d')
        start_6m = (base - timedelta(days=300)).strftime('%Y%m%d')

        ts_code = _to_ts_code(code)
        query = f"""
            SELECT trade_date, close
            FROM daily
            WHERE ts_code = '{ts_code}'
              AND trade_date BETWEEN '{start_6m}' AND '{base_date}'
            ORDER BY trade_date
        """
        df = pd.read_sql_query(query, conn)
        conn.close()

        if len(df) < 20:
            return None

        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df = df.dropna(subset=['close']).reset_index(drop=True)
        prices = df['close'].values

        # 计算日收益率
        returns = np.diff(prices) / prices[:-1]

        factors = {}

        # 1. 历史波动率（年化，越低越好，存为负值）
        if len(returns) >= 20:
            daily_vol = np.std(returns[-20:])
            annualized_vol = daily_vol * np.sqrt(252)
            factors['hist_vol'] = -annualized_vol  # 负值表示越低越好
        else:
            factors['hist_vol'] = np.nan

        # 2. 下行波动率（越低越好，存为负值）
        if len(returns) >= 20:
            downside = returns[returns < 0]
            if len(downside) > 0:
                downside_vol = np.std(downside) * np.sqrt(252)
                factors['downside_vol'] = -downside_vol
            else:
                factors['downside_vol'] = 0.0  # 没有下跌，最好
        else:
            factors['downside_vol'] = np.nan

        # 3. VaR（5%分位数，取绝对值，越低越好，存为负值）
        if len(returns) >= 20:
            var_95 = np.percentile(returns, 5)
            factors['var_95'] = -(-var_95)  # 负值表示越低（绝对值小）越好
        else:
            factors['var_95'] = np.nan

        return factors if not all(pd.isna(list(factors.values()))) else None

    except Exception as e:
        conn.close()
        logger.debug(f"波动因子计算失败 {code}: {e}")
        return None


def _to_ts_code(code: str) -> str:
    """将纯数字代码转为Tushare格式"""
    code = str(code).strip()
    if code.endswith(('.SZ', '.SH')):
        return code
    if code.startswith('6'):
        return f"{code}.SH"
    elif code.startswith(('0', '3')):
        return f"{code}.SZ"
    return code


def calculate_factors_from_sqlite(config: Dict) -> pd.DataFrame:
    """
    从SQLite数据库计算因子
    根据配置选择启用的因子类别
    """
    db_path = config['data']['db_path']
    base_date = config['data']['base_date']

    # 获取股票池
    print("\n[1/4] 获取股票池...")
    stocks = get_all_a_stocks(db_path, base_date)
    print(f"✓ 股票池: {len(stocks)} 只股票\n")

    # 计算因子
    print("[2/4] 计算因子...")
    results = []

    for i, code in enumerate(stocks):
        if i % 50 == 0:
            print(f"  进度: {i}/{len(stocks)}...")

        result = {'code': code}

        # 价值因子
        if config['factors']['value']:
            vf = get_value_factors(code, db_path, base_date)
            if vf:
                result.update(vf)

        # 动量因子
        if config['factors'].get('momentum', False):
            mf = get_momentum_factors(code, db_path, base_date)
            if mf:
                result.update(mf)

        # 技术因子
        if config['factors'].get('technical', False):
            tf = get_technical_factors(code, db_path, base_date)
            if tf:
                result.update(tf)

        # 低波动因子
        if config['factors'].get('volatility', False):
            vf = get_volatility_factors(code, db_path, base_date)
            if vf:
                result.update(vf)

        # 至少有个价值因子才保留（可调整）
        if len(result) > 1:
            results.append(result)

    if len(results) == 0:
        print("错误：没有计算出任何有效因子！")
        return pd.DataFrame()

    factors_df = pd.DataFrame(results)
    print(f"✓ 因子计算完成: {len(factors_df)} 只股票\n")

    return factors_df


# ==================== 因子标准化 ====================
def normalize_factors(factors_df: pd.DataFrame) -> pd.DataFrame:
    """标准化因子（z-score）"""
    normalized = factors_df.copy()
    
    # 对每列进行z-score标准化
    for col in factors_df.columns:
        if col in ['code', 'name']:
            continue
        mean_val = factors_df[col].mean()
        std_val = factors_df[col].std()
        if std_val > 0:
            normalized[col] = (factors_df[col] - mean_val) / std_val
        else:
            normalized[col] = 0
    
    return normalized


# ==================== 主流程 ====================
def run_selection(config: Dict):
    """
    执行选股主流程
    
    Args:
        config: 配置字典（从YAML加载）
    """
    logger.info("="*50)
    logger.info("开始多因子选股...")
    logger.info(f"基准日期: {config['data']['base_date']}")
    logger.info("="*50)
    
    # 1. 计算因子（从SQLite）
    factors_df = calculate_factors_from_sqlite(config)
    if len(factors_df) == 0:
        return
    
    # 2. 标准化因子
    print("[3/4] 标准化因子...")
    
    # 创建副本用于标准化
    normalized_df = factors_df.copy()
    
    # 只对因子列进行标准化（不对code、name、market_cap标准化）
    cols_to_normalize = [col for col in normalized_df.columns 
                         if col not in ['code', 'name', 'market_cap']]
    
    for col in cols_to_normalize:
        mean_val = normalized_df[col].mean()
        std_val = normalized_df[col].std()
        if std_val > 0:
            normalized_df[col] = (normalized_df[col] - mean_val) / std_val
        else:
            normalized_df[col] = 0
    
    print(f"✓ 因子标准化完成\n")
    
    # 3. 计算综合得分（根据启用的因子类别）
    print("[3.5/4] 计算综合得分...")
    
    # 确定要使用的因子列（根据配置和实际存在的列）
    value_factor_cols = ['pe', 'pb', 'ps', 'dv_ratio']
    momentum_factor_cols = ['ret_1m', 'ret_3m', 'ret_6m', 'ret_12m']
    technical_factor_cols = ['ma_bullish', 'vol_ratio']
    volatility_factor_cols = ['hist_vol', 'downside_vol', 'var_95']

    # 根据启用的因子类别，选择要加总的列
    score_cols = []
    if config['factors']['value']:
        score_cols.extend([col for col in normalized_df.columns if col in value_factor_cols])
    if config['factors'].get('momentum', False):
        score_cols.extend([col for col in normalized_df.columns if col in momentum_factor_cols])
    if config['factors'].get('technical', False):
        score_cols.extend([col for col in normalized_df.columns if col in technical_factor_cols])
    if config['factors'].get('volatility', False):
        score_cols.extend([col for col in normalized_df.columns if col in volatility_factor_cols])

    # 等权重加总
    if len(score_cols) > 0:
        # 去除重复列（如果有）
        score_cols = list(dict.fromkeys(score_cols))
        normalized_df['total_score'] = normalized_df[score_cols].sum(axis=1)
        print(f"✓ 综合得分计算完成（使用因子: {score_cols}）\n")
    else:
        print("警告：没有启用的因子，total_score设为0\n")
        normalized_df['total_score'] = 0
    
    # 4. 选股
    print("[4/4] 执行选股...")
    selector_config = {
        'top_n': config['selection']['top_n'],
        'min_score': config['selection']['min_score'],
        'output_dir': config['output']['output_dir']
    }
    
    selector = StockSelector(config=selector_config)
    selected_df = selector.select(normalized_df)
    
    # 打印结果
    selector.print_top_stocks(selected_df, n=config['selection']['top_n'])
    
    # 5. 导出结果
    output_format = config['output']['format']
    output_dir = config['output']['output_dir']
    filename = config['output']['filename']
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    
    if output_format == 'csv':
        selector.export_to_csv(selected_df, filename=filename)
    elif output_format == 'excel':
        selector.export_to_excel(selected_df, filename=filename)
    
    logger.info(f"✓ 选股完成！结果已保存到: {output_path}")
    
    return selected_df


# ==================== 主程序 ====================
def main():
    """主程序"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='多因子选股系统')
    parser.add_argument('--config', type=str, default='config/selection_config.yaml',
                        help='配置文件路径（YAML格式）')
    args = parser.parse_args()
    
    # 加载配置文件
    if not os.path.exists(args.config):
        print(f"错误：配置文件不存在: {args.config}")
        return
    
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 初始化日志
    logger.remove()
    logger.add(sys.stdout, level=config['logging']['level'],
               format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")
    
    if config['logging']['save_to_file']:
        log_file = config['logging']['log_file']
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        logger.add(log_file, rotation="500 MB")
    
    # 执行选股
    run_selection(config)


if __name__ == '__main__':
    main()
