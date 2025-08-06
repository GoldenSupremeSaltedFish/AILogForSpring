#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的数据清理脚本
"""

import pandas as pd
import re
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def improved_clean_log_content(text):
    """改进的日志内容清理"""
    if pd.isna(text) or text == '':
        return ''
    
    text = str(text)
    
    # 移除GitHub元数据
    patterns = [
        r'github\.com/[^,\s]+',
        r'https://github\.com/[^,\s]+',
        r'github_issue',
        r'unknown,github_issue',
        r'[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]+',
        r'[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+,\d+',
        r'https://github\.com/[^,\s]+/issues/\d+',
        r'unknown,,,',  # 移除unknown,,,
        r'https://',     # 移除https://
        r'[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+,',  # 移除仓库名,
        r',unknown,,,',  # 移除,unknown,,,
        r',unknown,',    # 移除,unknown,
        r'unknown,',     # 移除unknown,
    ]
    
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # 移除HTML标签
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'&[a-zA-Z]+;', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    
    # 清理多余字符
    text = re.sub(r'^,+|,+$', '', text)
    text = re.sub(r'^"+|"+$', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # 移除以逗号开头的部分
    if text.startswith(','):
        text = text.lstrip(',')
    
    # 移除以unknown开头的部分
    if text.lower().startswith('unknown'):
        text = re.sub(r'^unknown[,\s]*', '', text, flags=re.IGNORECASE)
    
    return text.strip()

def is_valid_log_improved(text):
    """改进的有效日志判断"""
    if pd.isna(text) or text == '':
        return False
    
    text = str(text).lower()
    
    # 检查是否包含日志关键词
    log_keywords = [
        'error', 'warn', 'info', 'debug', 'exception', 'java', 'spring', 
        'at ', 'caused by', 'connection', 'database', 'timeout', 'memory',
        'performance', 'authentication', 'authorization', 'configuration',
        'environment', 'business', 'logic', 'operation', 'monitoring',
        'heartbeat', 'startup', 'boot', 'application'
    ]
    
    has_keyword = any(keyword in text for keyword in log_keywords)
    has_length = len(text) > 30  # 增加最小长度要求
    not_mostly_metadata = not text.count(',') > 3  # 避免主要是元数据的内容
    
    return has_keyword and has_length and not_mostly_metadata

def clean_dataset(input_path, output_path):
    """清理数据集"""
    logger.info(f"📂 加载数据: {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"📊 原始数据: {len(df)} 条记录")
    
    # 清理数据
    cleaned_data = []
    for idx, row in df.iterrows():
        original_log = row.get('original_log', '')
        cleaned_log = improved_clean_log_content(original_log)
        
        if is_valid_log_improved(cleaned_log):
            cleaned_data.append({
                'original_log': cleaned_log,
                'category': row['category'],
                'log_level': row.get('log_level', 'UNKNOWN'),
                'content_type': row.get('content_type', ''),
                'priority': row.get('priority', ''),
                'source_file': row.get('source_file', '')
            })
    
    # 创建清理后的数据框
    cleaned_df = pd.DataFrame(cleaned_data)
    logger.info(f"📊 清理后数据: {len(cleaned_df)} 条记录")
    
    # 保存清理后的数据
    cleaned_df.to_csv(output_path, index=False, encoding='utf-8')
    logger.info(f"✅ 数据已保存到: {output_path}")
    
    # 显示统计信息
    category_counts = cleaned_df['category'].value_counts()
    logger.info("📊 清理后类别分布:")
    for category, count in category_counts.items():
        logger.info(f"  {category}: {count} 条")
    
    # 显示样本
    logger.info("📝 清理后的样本:")
    for i, row in cleaned_df.head(5).iterrows():
        logger.info(f"  {i+1}. {row['category']}: {row['original_log'][:100]}...")
    
    return cleaned_df

def main():
    """主函数"""
    input_path = "data/processed_logs_cleaned.csv"
    output_path = "data/processed_logs_final_cleaned.csv"
    
    logger.info("🎯 改进的数据清理")
    cleaned_df = clean_dataset(input_path, output_path)
    
    logger.info("✅ 数据清理完成!")

if __name__ == "__main__":
    main() 