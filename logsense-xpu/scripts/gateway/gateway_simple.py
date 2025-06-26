#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gateway日志处理脚本 - 接受目录路径参数
用法: python gateway_simple.py <目录路径>
"""

import sys
import re
from pathlib import Path
import pandas as pd
from datetime import datetime

def process_logs(input_dir_path):
    """处理指定目录下的所有日志文件"""
    print("开始处理Gateway日志...")
    print(f"输入目录: {input_dir_path}")
    
    # 转换为Path对象并验证
    input_dir = Path(input_dir_path)
    if not input_dir.exists():
        print(f"错误：目录不存在 - {input_dir}")
        return
    
    if not input_dir.is_dir():
        print(f"错误：路径不是目录 - {input_dir}")
        return
    
    # 设置输出目录
    output_base_dir = Path(__file__).parent.parent.parent.parent / "DATA_OUTPUT"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_base_dir / f"processed_gateway_{timestamp}"
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"输出目录: {output_dir}")

    # Gateway日志格式正则表达式
    gateway_pattern = re.compile(r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\s+(?P<level>[A-Z]+)\s+\d+\s+---\s+\[(?P<thread>[^\]]+)\]\s+(?P<classpath>.+?)\s+:\s+(?P<message>.+)$")

    # 关键词过滤
    keywords = [
        "user", "auth", "jwt", "token", "security", "gateway",
        "controller", "service", "http", "api", "request", "response",
        "error", "exception", "failed", "success", "验证", "令牌"
    ]

    log_files = list(input_dir.glob("*.log"))
    print(f"找到 {len(log_files)} 个日志文件")

    total_logs = 0
    processed_files = 0
    
    # 处理所有文件，不限制数量
    for log_file in log_files:
        print(f"处理文件: {log_file.name}")
        parsed_logs = []
        try:
            with log_file.open("r", encoding="utf-8", errors="ignore") as f:
                # 处理所有行，不限制数量
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    match = gateway_pattern.match(line)
                    if match:
                        log_data = match.groupdict()
                        
                        # 关键词过滤
                        classpath = log_data.get("classpath", "").lower()
                        message = log_data.get("message", "").lower()
                        
                        if any(keyword in classpath or keyword in message for keyword in keywords):
                            log_data["timestamp"] = log_data["timestamp"].split(".")[0]
                            log_data["source_file"] = log_file.name
                            log_data["line_number"] = line_num
                            parsed_logs.append(log_data)
            
            if parsed_logs:
                df = pd.DataFrame(parsed_logs)
                output_file = output_dir / f"{log_file.stem}_cleaned.csv"
                df.to_csv(output_file, index=False, encoding="utf-8-sig")
                total_logs += len(parsed_logs)
                processed_files += 1
                print(f"  提取 {len(parsed_logs)} 条有效日志记录")
            else:
                print(f"  未找到符合条件的日志记录")
        except Exception as e:
            print(f"  处理失败: {e}")

    print("-" * 50)
    print(f"处理完成!")
    print(f"处理了 {processed_files} 个文件，共 {total_logs} 条有效日志")
    print(f"结果保存在: {output_dir}")


def main():
    """主函数"""
    if len(sys.argv) != 2:
        print("Gateway日志处理工具")
        print("=" * 30)
        print("使用方法:")
        print(f"  python {sys.argv[0]} <目录路径>")
        print("\n示例:")
        print(f"  python {sys.argv[0]} C:\\path\\to\\gateway_logs")
        return
    
    input_dir_path = sys.argv[1]
    process_logs(input_dir_path)


if __name__ == "__main__":
    main() 