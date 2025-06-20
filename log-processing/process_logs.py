#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日志处理脚本
接受目录路径参数，遍历处理其中的所有日志文件
"""

import sys
import re
from pathlib import Path
import pandas as pd
from datetime import datetime

def process_logs(input_dir_path):
    """
    处理指定目录下的所有日志文件
    
    Args:
        input_dir_path (str): 输入目录路径
    """
    print("🚀 开始处理日志文件...")
    print(f"📁 输入目录: {input_dir_path}")
    
    # 转换为Path对象
    input_dir = Path(input_dir_path)
    
    # 检查输入目录是否存在
    if not input_dir.exists():
        print(f"❌ 错误：输入目录不存在 - {input_dir}")
        return
    
    if not input_dir.is_dir():
        print(f"❌ 错误：提供的路径不是目录 - {input_dir}")
        return
    
    # 设置输出目录
    project_root = Path(__file__).parent
    output_base_dir = project_root / "data" / "output"
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = input_dir.name
    output_dir = output_base_dir / f"processed_{dir_name}_{timestamp}"
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"📁 输出目录: {output_dir}")
    
    # 扫描所有.log文件
    log_files = list(input_dir.glob("*.log"))
    print(f"📊 找到 {len(log_files)} 个 .log 文件")
    
    if not log_files:
        print("❌ 未找到任何 .log 文件")
        return
    
    # 保存文件路径列表
    file_list_path = output_dir / "file_list.txt"
    with file_list_path.open("w", encoding="utf-8") as f:
        f.write("日志文件路径列表\n")
        f.write("=" * 50 + "\n")
        for i, log_file in enumerate(log_files, 1):
            f.write(f"{i:3d}. {log_file}\n")
    print(f"📝 文件路径列表已保存: {file_list_path}")
    
    # Gateway日志格式正则表达式
    gateway_pattern = re.compile(
        r'^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\s+'
        r'(?P<level>[A-Z]+)\s+\d+\s+---\s+\[(?P<thread>[^\]]+)\]\s+'
        r'(?P<classpath>.+?)\s+:\s+(?P<message>.+)$'
    )
    
    # 关键词过滤
    keywords = [
        "user", "account", "login", "auth", "jwt", "token", "security",
        "controller", "service", "repository", "config", "gateway",
        "http", "api", "request", "response", "filter",
        "error", "exception", "failed", "timeout", "connection",
        "started", "stopped", "shutdown", "startup", "success"
    ]
    
    # 处理每个文件
    total_logs = 0
    processed_files = 0
    processing_summary = {}
    
    print("-" * 50)
    for log_file in log_files:
        print(f"📄 处理文件: {log_file.name}")
        parsed_logs = []
        
        try:
            with log_file.open("r", encoding="utf-8", errors="ignore") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    # 尝试匹配Gateway格式
                    match = gateway_pattern.match(line)
                    if match:
                        log_data = match.groupdict()
                        
                        # 检查是否包含关键词
                        classpath_lower = log_data['classpath'].lower()
                        message_lower = log_data['message'].lower()
                        
                        if any(keyword in classpath_lower or keyword in message_lower 
                               for keyword in keywords):
                            # 统一时间戳格式（去掉毫秒）
                            log_data['timestamp'] = log_data['timestamp'].split('.')[0]
                            log_data['source_file'] = log_file.name
                            log_data['line_number'] = line_num
                            parsed_logs.append(log_data)
            
            # 保存清洗后的数据
            if parsed_logs:
                df = pd.DataFrame(parsed_logs)
                output_file = output_dir / f"{log_file.stem}_cleaned.csv"
                df.to_csv(output_file, index=False, encoding="utf-8-sig")
                
                count = len(parsed_logs)
                total_logs += count
                processed_files += 1
                processing_summary[log_file.name] = count
                
                print(f"  ✅ 提取 {count} 条有效日志记录")
            else:
                print(f"  ⚠️  未找到符合条件的日志记录")
                processing_summary[log_file.name] = 0
        
        except Exception as e:
            print(f"  ❌ 处理失败: {e}")
            processing_summary[log_file.name] = f"错误: {e}"
    
    # 生成处理报告
    report_file = output_dir / "processing_report.txt"
    with report_file.open("w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("日志处理报告\n")
        f.write("=" * 60 + "\n")
        f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"输入目录: {input_dir}\n")
        f.write(f"输出目录: {output_dir}\n")
        f.write(f"扫描文件数: {len(log_files)}\n")
        f.write(f"成功处理文件数: {processed_files}\n")
        f.write(f"有效日志总数: {total_logs}\n\n")
        
        f.write("各文件处理详情:\n")
        f.write("-" * 40 + "\n")
        for filename, count in processing_summary.items():
            f.write(f"{filename:<35} {str(count):>10}\n")
        
        f.write(f"\n过滤关键词: {', '.join(keywords[:10])}...\n")
    
    print("-" * 50)
    print(f"🎉 处理完成!")
    print(f"📊 扫描了 {len(log_files)} 个文件")
    print(f"📊 成功处理 {processed_files} 个文件")
    print(f"📊 提取 {total_logs} 条有效日志记录")
    print(f"📁 结果保存在: {output_dir}")
    print(f"📄 处理报告: {report_file}")
    print(f"📝 文件列表: {file_list_path}")


def main():
    """主函数"""
    if len(sys.argv) != 2:
        print("使用方法:")
        print(f"    python {sys.argv[0]} <目录路径>")
        print("\n示例:")
        print(f"    python {sys.argv[0]} C:\\Users\\30871\\Desktop\\AILogForSpring\\logsense-xpu\\data\\gate_way_logs")
        return
    
    input_dir_path = sys.argv[1]
    process_logs(input_dir_path)


if __name__ == "__main__":
    main() 