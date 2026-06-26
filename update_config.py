"""
同步更新 config.py 中的配置值
用法：python update_config.py [top_n=N] [key=value ...]
"""
import sys
import re
import os

def update_config(updates: dict):
    """更新 config.py 中的配置值"""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.py")
    
    with open(config_path, 'r', encoding='utf-8-sig') as f:
        cfg = f.read()
    
    for key, value in updates.items():
        # 匹配 "key": value, 或 "key": value}
        pattern = rf'(["\']({key})["\']\s*:\s*)(.+?)([,}}])'
        replacement = rf'\1{value}\3'
        new_cfg = re.sub(pattern, replacement, cfg)
        if new_cfg == cfg:
            print(f"  [WARN] 未找到配置项: {key}")
        else:
            cfg = new_cfg
            print(f"  [OK] {key} -> {value}")
    
    with open(config_path, 'w', encoding='utf-8-sig') as f:
        f.write(cfg)

if __name__ == "__main__":
    updates = {}
    for arg in sys.argv[1:]:
        if '=' in arg:
            key, value = arg.split('=', 1)
            updates[key] = value
    
    if updates:
        update_config(updates)
    else:
        print("用法：python update_config.py [top_n=10] [key=value ...]")
        print("示例：python update_config.py top_n=10")
