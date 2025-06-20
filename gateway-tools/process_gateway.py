#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
专门处理Gateway日志的脚本
输出结果到 data/output 目录
"""

import re
from pathlib import Path
from typing import Optional, Dict
import pandas as pd
from datetime import datetime

def main():
    print("🚀 开始处理Gateway日志...")
    
    # 设置路径
    current_dir = Path(__file__).parent
    gateway_dir = current_dir.parent / "data" / "gate_way_logs"
    output_base_dir = current_dir.parent / "data" / "output"
    
    print(f"📁 输入目录: {gateway_dir}")
    print(f"📁 输出基础目录: {output_base_dir}")
    
    # 检查输入目录
    if not gateway_dir.exists():
        print(f"❌ 输入目录不存在: {gateway_dir}")
        return
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_base_dir / f"processed_gateway_{timestamp}"
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"📁 输出目录: {output_dir}")
    
    # Gateway日志格式正则表达式
    gateway_pattern = re.compile(
        r'^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\s+(?P<level>[A-Z]+)\s+\d+\s+---\s+\[(?P<thread>[^\]]+)\]\s+(?P<classpath>.+?)\s+:\s+(?P<message>.+)$'
    )
    
    # 扫描.log文件
    log_files = list(gateway_dir.glob("*.log"))
    print(f"📊 找到 {len(log_files)} 个日志文件")
    
    if not log_files:
        print("❌ 未找到任何.log文件")
        return
    
    # 处理每个文件
    total_logs = 0
    for log_file in log_files[:3]:  # 先处理前3个文件测试
        print(f"📄 处理文件: {log_file.name}")
        
        parsed_logs = []
        try:
            with log_file.open("r", encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f):
                    if i >= 1000:  # 每个文件只处理前1000行用于测试
                        break
                    
                    match = gateway_pattern.match(line.strip())
                    if match:
                        log_data = match.groupdict()
                        # 去掉毫秒部分
                        log_data['timestamp'] = log_data['timestamp'].split('.')[0]
                        log_data['source_file'] = log_file.name
                        parsed_logs.append(log_data)
            
            if parsed_logs:
                # 保存结果
                df = pd.DataFrame(parsed_logs)
                output_file = output_dir / f"{log_file.stem}_cleaned.csv"
                df.to_csv(output_file, index=False, encoding="utf-8-sig")
                total_logs += len(parsed_logs)
                print(f"  ✅ 提取 {len(parsed_logs)} 条日志记录")
            else:
                print(f"  ❌ 未找到匹配的日志记录")
        
        except Exception as e:
            print(f"  ❌ 处理失败: {e}")
    
    print("-" * 50)
    print(f"🎉 处理完成!")
    print(f"📊 总共提取 {total_logs} 条日志记录")
    print(f"📁 结果保存在: {output_dir}")

if __name__ == "__main__":
    main() 