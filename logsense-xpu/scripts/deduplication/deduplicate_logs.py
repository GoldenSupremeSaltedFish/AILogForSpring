#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日志去重脚本
支持精确去重和模糊去重两种方案
用法: python deduplicate_logs.py <输入目录> [--mode exact|fuzzy|both]
"""

import sys
import re
import pandas as pd
from pathlib import Path
from datetime import datetime
import argparse

def normalize_message(msg):
    """
    模糊归一化消息内容
    替换变化的部分（ID、时间戳、IP等）为通配符
    """
    if pd.isna(msg):
        return ""
    
    # 替换长数字（如ID、时间戳）
    msg = re.sub(r'\b\d{4,}\b', '*', msg)
    # 替换所有数字
    msg = re.sub(r'\b\d+\b', '*', msg)
    # 替换hex或UUID
    msg = re.sub(r'[a-fA-F0-9]{8,}', '*', msg)
    # 替换IP地址
    msg = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '*', msg)
    # 替换时间戳格式
    msg = re.sub(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', '*', msg)
    # 替换UUID格式
    msg = re.sub(r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}', '*', msg)
    # 合并多空格
    msg = re.sub(r'\s+', ' ', msg.strip())
    
    return msg

def deduplicate_exact(df):
    """精确去重：基于message字段完全匹配"""
    return df.drop_duplicates(subset=["message"])

def deduplicate_fuzzy(df):
    """模糊去重：基于归一化后的message字段"""
    df["normalized_message"] = df["message"].apply(normalize_message)
    df_dedup = df.drop_duplicates(subset=["normalized_message"])
    # 删除临时列
    df_dedup = df_dedup.drop(columns=["normalized_message"])
    return df_dedup

def process_single_file(input_file, output_dir, mode="both"):
    """处理单个CSV文件"""
    print(f"处理文件: {input_file.name}")
    
    try:
        # 读取CSV文件
        df = pd.read_csv(input_file, encoding='utf-8-sig')
        original_count = len(df)
        
        results = {
            "file": input_file.name,
            "original_count": original_count
        }
        
        # 精确去重
        if mode in ["exact", "both"]:
            df_exact = deduplicate_exact(df)
            exact_count = len(df_exact)
            results["exact_dedup_count"] = exact_count
            results["exact_removed"] = original_count - exact_count
            
            # 保存精确去重结果
            exact_output = output_dir / "exact" / f"{input_file.stem}_exact_dedup.csv"
            exact_output.parent.mkdir(parents=True, exist_ok=True)
            df_exact.to_csv(exact_output, index=False, encoding='utf-8-sig')
        
        # 模糊去重
        if mode in ["fuzzy", "both"]:
            df_fuzzy = deduplicate_fuzzy(df)
            fuzzy_count = len(df_fuzzy)
            results["fuzzy_dedup_count"] = fuzzy_count
            results["fuzzy_removed"] = original_count - fuzzy_count
            
            # 保存模糊去重结果
            fuzzy_output = output_dir / "fuzzy" / f"{input_file.stem}_fuzzy_dedup.csv"
            fuzzy_output.parent.mkdir(parents=True, exist_ok=True)
            df_fuzzy.to_csv(fuzzy_output, index=False, encoding='utf-8-sig')
        
        return results
        
    except Exception as e:
        print(f"  处理失败: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='日志去重工具')
    parser.add_argument('input_dir', help='输入目录路径（包含CSV文件）')
    parser.add_argument('--mode', choices=['exact', 'fuzzy', 'both'], 
                       default='both', help='去重模式 (默认: both)')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    mode = args.mode
    
    if not input_dir.exists():
        print(f"错误：输入目录不存在 - {input_dir}")
        return
    
    # 设置输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base_dir = Path(__file__).parent.parent.parent.parent / "DATA_OUTPUT"
    output_dir = output_base_dir / f"deduplicated_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("日志去重工具")
    print("=" * 50)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"去重模式: {mode}")
    print("=" * 50)
    
    # 查找所有CSV文件
    csv_files = list(input_dir.glob("*.csv"))
    if not csv_files:
        print("错误：输入目录中未找到CSV文件")
        return
    
    print(f"找到 {len(csv_files)} 个CSV文件")
    print("-" * 50)
    
    # 处理统计
    all_results = []
    total_original = 0
    total_exact = 0
    total_fuzzy = 0
    
    # 处理每个文件
    for csv_file in csv_files:
        result = process_single_file(csv_file, output_dir, mode)
        if result:
            all_results.append(result)
            total_original += result["original_count"]
            
            if mode in ["exact", "both"]:
                total_exact += result["exact_dedup_count"]
                exact_removed = result["exact_removed"]
                exact_ratio = (exact_removed / result["original_count"]) * 100
                print(f"  精确去重: {result['original_count']} -> {result['exact_dedup_count']} "
                      f"(移除 {exact_removed}, {exact_ratio:.1f}%)")
            
            if mode in ["fuzzy", "both"]:
                total_fuzzy += result["fuzzy_dedup_count"]
                fuzzy_removed = result["fuzzy_removed"]
                fuzzy_ratio = (fuzzy_removed / result["original_count"]) * 100
                print(f"  模糊去重: {result['original_count']} -> {result['fuzzy_dedup_count']} "
                      f"(移除 {fuzzy_removed}, {fuzzy_ratio:.1f}%)")
        print()
    
    # 生成汇总报告
    print("=" * 50)
    print("汇总报告")
    print("=" * 50)
    print(f"处理文件数: {len(all_results)}")
    print(f"原始日志总数: {total_original:,}")
    
    if mode in ["exact", "both"]:
        exact_removed_total = total_original - total_exact
        exact_ratio_total = (exact_removed_total / total_original) * 100
        print(f"精确去重后: {total_exact:,} (移除 {exact_removed_total:,}, {exact_ratio_total:.1f}%)")
    
    if mode in ["fuzzy", "both"]:
        fuzzy_removed_total = total_original - total_fuzzy
        fuzzy_ratio_total = (fuzzy_removed_total / total_original) * 100
        print(f"模糊去重后: {total_fuzzy:,} (移除 {fuzzy_removed_total:,}, {fuzzy_ratio_total:.1f}%)")
    
    # 保存详细报告
    report_file = output_dir / "deduplication_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("日志去重报告\n")
        f.write("=" * 50 + "\n")
        f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"输入目录: {input_dir}\n")
        f.write(f"去重模式: {mode}\n")
        f.write(f"处理文件数: {len(all_results)}\n")
        f.write(f"原始日志总数: {total_original:,}\n")
        
        if mode in ["exact", "both"]:
            f.write(f"精确去重后: {total_exact:,} (移除 {exact_removed_total:,}, {exact_ratio_total:.1f}%)\n")
        if mode in ["fuzzy", "both"]:
            f.write(f"模糊去重后: {total_fuzzy:,} (移除 {fuzzy_removed_total:,}, {fuzzy_ratio_total:.1f}%)\n")
        
        f.write("\n详细统计:\n")
        f.write("-" * 50 + "\n")
        
        for result in all_results:
            f.write(f"\n文件: {result['file']}\n")
            f.write(f"  原始记录: {result['original_count']:,}\n")
            if mode in ["exact", "both"]:
                f.write(f"  精确去重: {result['exact_dedup_count']:,} "
                       f"(移除 {result['exact_removed']:,})\n")
            if mode in ["fuzzy", "both"]:
                f.write(f"  模糊去重: {result['fuzzy_dedup_count']:,} "
                       f"(移除 {result['fuzzy_removed']:,})\n")
    
    print(f"\n详细报告已保存至: {report_file}")
    print(f"去重结果保存在: {output_dir}")


if __name__ == "__main__":
    main() 