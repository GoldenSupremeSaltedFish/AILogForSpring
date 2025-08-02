#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据处理工具模块
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


def detect_label_column(df: pd.DataFrame) -> Optional[str]:
    """检测标签列"""
    possible_labels = ['content_type', 'final_label', 'label', 'category']
    for col in possible_labels:
        if col in df.columns:
            return col
    return None


def detect_text_column(df: pd.DataFrame) -> Optional[str]:
    """检测文本列"""
    possible_texts = ['original_log', 'message', 'content', 'text']
    for col in possible_texts:
        if col in df.columns:
            return col
    return None


def load_and_prepare_data(data_file: str, sample_size: int = None) -> Tuple[pd.DataFrame, str, str]:
    """加载和准备数据"""
    print(f"📂 加载数据文件: {data_file}")
    
    df = pd.read_csv(data_file)
    print(f"📊 原始数据: {len(df)} 条记录")
    
    # 检测标签列和文本列
    label_column = detect_label_column(df)
    text_column = detect_text_column(df)
    
    if not label_column or not text_column:
        raise ValueError("未找到标签列或文本列")
    
    print(f"🔍 使用标签列: {label_column}")
    print(f"🔍 使用文本列: {text_column}")
    
    # 过滤数据
    df_filtered = df[df[label_column] != 'other'].copy()
    print(f"🔍 过滤后数据: {len(df_filtered)} 条记录")
    
    # 统计类别分布
    category_counts = df_filtered[label_column].value_counts()
    print("\n📈 类别分布:")
    for category, count in category_counts.items():
        percentage = (count / len(df_filtered)) * 100
        print(f"  {category}: {count} 条 ({percentage:.1f}%)")
    
    # 如果指定了采样大小，进行采样
    if sample_size:
        print(f"\n🎯 进行采样，每类最多 {sample_size} 条记录")
        sampled_data = []
        for category in df_filtered[label_column].unique():
            category_data = df_filtered[df_filtered[label_column] == category]
            if len(category_data) > sample_size:
                category_data = category_data.sample(n=sample_size, random_state=42)
            sampled_data.append(category_data)
        
        df_filtered = pd.concat(sampled_data, ignore_index=True)
        print(f"📊 采样后数据: {len(df_filtered)} 条记录")
    
    return df_filtered, label_column, text_column 