with open('data_fetcher.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 要替换的旧代码（第425-458行附近）
old = """        # 2. 使用AkShare单只股票查询（高效，不需要拉全市场数据）
        try:
            df = ak.stock_individual_info_em(symbol=code)
            if df is not None and len(df) > 0:
                info = dict(zip(df['item'], df['value']))
                name = info.get('股票简称', '')
                market_cap = info.get('总市值', None)
                # 解析市值字符串（如 "123.45亿" 或 "123456.78万"）
                if market_cap is not None and isinstance(market_cap, str):
                    market_cap = market_cap.replace(',', '')
                    try:
                        if '亿' in market_cap:
                            market_cap = float(market_cap.replace('亿', '')) * 1e8
                        elif '万' in market_cap:
                            market_cap = float(market_cap.replace('万', '')) * 1e4
                        else:
                            market_cap = float(market_cap)
                    except Exception:
                        market_cap = None
                return {
                    'code': code,
                    'name': name,
                    'market_cap': market_cap
                }
        except Exception as e:
            logger.debug(f"AkShare get_stock_info 获取失败 {code}: {e}")
        
        # 3. 兜底：返回空名称（不中断流程）
        logger.warning(f"未能获取股票信息: {code}")
        return {
            'code': code,
            'name': '',
            'market_cap': None
        }"""

new = """        # 2. 根据配置决定是否使用AkShare查询
        if not self.akshare_backup_available:
            logger.debug(f"AkShare backup disabled, skipping for {code} stock info")
        else:
            try:
                df = ak.stock_individual_info_em(symbol=code)
                if df is not None and len(df) > 0:
                    info = dict(zip(df['item'], df['value']))
                    name = info.get('股票简称', '')
                    market_cap = info.get('总市值', None)
                    # 解析市值字符串（如 "123.45亿" 或 "123456.78万"）
                    if market_cap is not None and isinstance(market_cap, str):
                        market_cap = market_cap.replace(',', '')
                        try:
                            if '亿' in market_cap:
                                market_cap = float(market_cap.replace('亿', '')) * 1e8
                            elif '万' in market_cap:
                                market_cap = float(market_cap.replace('万', '')) * 1e4
                            else:
                                market_cap = float(market_cap)
                        except Exception:
                            market_cap = None
                    return {
                        'code': code,
                        'name': name,
                        'market_cap': market_cap
                    }
            except Exception as e:
                logger.debug(f"AkShare get_stock_info 获取失败 {code}: {e}")
        
        # 3. 兜底：返回空名称（不中断流程）
        logger.warning(f"未能获取股票信息: {code}")
        return {
            'code': code,
            'name': '',
            'market_cap': None
        }"""

if old in content:
    content = content.replace(old, new, 1)
    with open('data_fetcher.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print('✓ Fixed get_stock_info() to check akshare_backup_available flag')
else:
    print('ERROR: old string not found')
    # Debug: show what's around line 425
    lines = content.split('\n')
    for i in range(420, min(460, len(lines))):
        print(f'{i+1}: {repr(lines[i])}')
