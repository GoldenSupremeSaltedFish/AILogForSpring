#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强数据分析脚本
"""

import pandas as pd
import numpy as np

def analyze_enhanced_data():
    """分析增强数据"""
    print("=== 数据质量分析 ===")
    
    # 加载数据
    df = pd.read_csv('data/processed_logs_quality_enhanced.csv')
    print(f"总记录数: {len(df)}")
    
    # 基本统计
    print(f"有效日志数: {len(df[df['cleaned_log'].str.len() > 10])}")
    
    # 特征统计 - 修复统计逻辑
    error_codes_count = len(df[df['error_codes'].notna() & (df['error_codes'] != '')])
    paths_count = len(df[df['paths'].notna() & (df['paths'] != '')])
    numbers_count = len(df[df['numbers'].notna() & (df['numbers'] != '')])
    classes_count = len(df[df['classes'].notna() & (df['classes'] != '')])
    
    print(f"包含错误码: {error_codes_count}")
    print(f"包含路径: {paths_count}")
    print(f"包含数字: {numbers_count}")
    print(f"包含类名: {classes_count}")
    
    print("\n=== 类别分布 ===")
    category_counts = df['category'].value_counts()
    for category, count in category_counts.items():
        percentage = (count / len(df)) * 100
        print(f"{category}: {count} 条 ({percentage:.1f}%)")
    
    print("\n=== 日志级别分布 ===")
    level_counts = df['log_level'].value_counts()
    for level, count in level_counts.items():
        percentage = (count / len(df)) * 100
        print(f"{level}: {count} 条 ({percentage:.1f}%)")
    
    print("\n=== 数据样例 ===")
    sample_cols = ['category', 'cleaned_log', 'log_level', 'error_codes', 'paths']
    print(df[sample_cols].head(3).to_string())
    
    # 特征覆盖率 - 修复计算逻辑
    error_codes_coverage = (error_codes_count / len(df)) * 100
    paths_coverage = (paths_count / len(df)) * 100
    numbers_coverage = (numbers_count / len(df)) * 100
    classes_coverage = (classes_count / len(df)) * 100
    
    print("\n=== 特征覆盖率 ===")
    print(f"错误码覆盖率: {error_codes_coverage:.1f}%")
    print(f"路径覆盖率: {paths_coverage:.1f}%")
    print(f"数字覆盖率: {numbers_coverage:.1f}%")
    print(f"类名覆盖率: {classes_coverage:.1f}%")
    
    # 详细特征分析
    print("\n=== 详细特征分析 ===")
    
    # 错误码分析
    error_codes_df = df[df['error_codes'].notna() & (df['error_codes'] != '')]
    if len(error_codes_df) > 0:
        print(f"错误码样例: {error_codes_df['error_codes'].iloc[0]}")
    
    # 路径分析
    paths_df = df[df['paths'].notna() & (df['paths'] != '')]
    if len(paths_df) > 0:
        print(f"路径样例: {paths_df['paths'].iloc[0]}")
    
    # 数字分析
    numbers_df = df[df['numbers'].notna() & (df['numbers'] != '')]
    if len(numbers_df) > 0:
        print(f"数字样例: {numbers_df['numbers'].iloc[0]}")
    
    # 类名分析
    classes_df = df[df['classes'].notna() & (df['classes'] != '')]
    if len(classes_df) > 0:
        print(f"类名样例: {classes_df['classes'].iloc[0]}")

if __name__ == "__main__":
    analyze_enhanced_data() 