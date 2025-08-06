#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据预处理 - 清理日志数据，移除GitHub元数据
"""

import os
import pandas as pd
import numpy as np
import logging
import re
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_log_content(text):
    """清理日志内容"""
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
    
    return text

def is_valid_log(text):
    """判断是否为有效的日志内容"""
    if pd.isna(text) or text == '':
        return False
    
    text = str(text).lower()
    log_keywords = ['error', 'warn', 'info', 'debug', 'exception', 'java', 'spring', 'at ', 'caused by']
    has_keyword = any(keyword in text for keyword in log_keywords)
    has_length = len(text) > 20
    
    return has_keyword or has_length

def process_category_data(category_path, category_name):
    """处理单个类别的数据"""
    logger.info(f"📁 处理类别: {category_name}")
    
    all_data = []
    
    # 遍历目录中的所有CSV文件
    for csv_file in Path(category_path).glob("*.csv"):
        try:
            df = pd.read_csv(csv_file)
            logger.info(f"  📄 读取文件: {csv_file.name} - {len(df)} 条记录")
            
            # 清理数据
            cleaned_data = []
            for idx, row in df.iterrows():
                original_log = row.get('original_log', '')
                
                # 清理日志内容
                cleaned_log = clean_log_content(original_log)
                
                # 检查是否为有效日志
                if is_valid_log(cleaned_log):
                    cleaned_data.append({
                        'original_log': cleaned_log,
                        'category': category_name,
                        'log_level': row.get('log_level', 'UNKNOWN'),
                        'content_type': row.get('content_type', ''),
                        'priority': row.get('priority', ''),
                        'source_file': csv_file.name
                    })
            
            logger.info(f"  ✅ 清理后: {len(cleaned_data)} 条有效记录")
            all_data.extend(cleaned_data)
            
        except Exception as e:
            logger.error(f"  ❌ 处理文件失败: {csv_file.name} - {e}")
    
    return pd.DataFrame(all_data)

def main():
    """主函数"""
    data_path = r"C:\Users\30871\Desktop\AILogForSpring\DATA_OUTPUT"
    output_path = "data/processed_logs_cleaned.csv"
    
    logger.info("🎯 数据预处理 - 清理日志数据")
    logger.info(f"📁 输入路径: {data_path}")
    logger.info(f"📁 输出文件: {output_path}")
    
    # 检查输入路径
    if not os.path.exists(data_path):
        logger.error(f"❌ 输入路径不存在: {data_path}")
        return
    
    # 类别映射
    category_mapping = {
        '01_堆栈异常_stack_exception': 'stack_exception',
        '02_数据库异常_database_exception': 'database_exception',
        '03_连接问题_connection_issue': 'connection_issue',
        '04_认证授权_auth_authorization': 'auth_authorization',
        '05_配置环境_config_environment': 'config_environment',
        '06_业务逻辑_business_logic': 'business_logic',
        '07_正常操作_normal_operation': 'normal_operation',
        '08_监控心跳_monitoring_heartbeat': 'monitoring_heartbeat',
        '09_内存性能_memory_performance': 'memory_performance',
        '10_超时错误_timeout': 'timeout',
        '11_SpringBoot启动失败_spring_boot_startup_failure': 'spring_boot_startup_failure'
    }
    
    all_cleaned_data = []
    
    # 处理每个类别
    for folder_name, category in category_mapping.items():
        folder_path = os.path.join(data_path, folder_name)
        
        if not os.path.exists(folder_path):
            logger.warning(f"⚠️ 目录不存在: {folder_path}")
            continue
        
        # 处理该类别的数据
        category_df = process_category_data(folder_path, category)
        
        if len(category_df) > 0:
            all_cleaned_data.append(category_df)
            logger.info(f"✅ {category}: {len(category_df)} 条记录")
        else:
            logger.warning(f"⚠️ {category}: 没有有效记录")
    
    if not all_cleaned_data:
        logger.error("❌ 没有找到有效数据")
        return
    
    # 合并所有数据
    combined_df = pd.concat(all_cleaned_data, ignore_index=True)
    
    # 数据统计
    logger.info("📊 预处理结果统计:")
    category_counts = combined_df['category'].value_counts()
    for category, count in category_counts.items():
        logger.info(f"  {category}: {count} 条")
    
    # 保存清理后的数据
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    combined_df.to_csv(output_path, index=False, encoding='utf-8')
    
    logger.info(f"✅ 数据预处理完成! 保存到: {output_path}")
    logger.info(f"📊 总记录数: {len(combined_df)}")
    
    # 显示一些清理后的样本
    logger.info("📝 清理后的样本:")
    for i, row in combined_df.head(5).iterrows():
        logger.info(f"  {i+1}. {row['category']}: {row['original_log'][:100]}...")

if __name__ == "__main__":
    main() 