#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
准备完整数据集 - 合并所有11种类别的日志
"""

import os
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 类别映射
CATEGORY_MAPPING = {
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

def load_category_data(data_output_path: str) -> pd.DataFrame:
    """加载所有类别的数据"""
    logger.info(f"📂 开始加载数据: {data_output_path}")
    
    all_data = []
    
    for folder_name, category in CATEGORY_MAPPING.items():
        folder_path = os.path.join(data_output_path, folder_name)
        
        if not os.path.exists(folder_path):
            logger.warning(f"⚠️ 目录不存在: {folder_path}")
            continue
            
        logger.info(f"📁 处理类别: {category} ({folder_name})")
        
        # 查找CSV文件
        csv_files = list(Path(folder_path).glob("*.csv"))
        
        if not csv_files:
            logger.warning(f"⚠️ 未找到CSV文件: {folder_path}")
            continue
            
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                logger.info(f"  📄 加载文件: {csv_file.name} - {len(df)} 条记录")
                
                # 确保有必要的列
                if 'original_log' not in df.columns:
                    logger.warning(f"⚠️ 文件缺少original_log列: {csv_file.name}")
                    continue
                
                # 添加类别信息
                df['category'] = category
                all_data.append(df)
                
            except Exception as e:
                logger.error(f"❌ 读取文件失败: {csv_file.name} - {e}")
                continue
    
    if not all_data:
        logger.error("❌ 没有找到任何有效数据")
        return pd.DataFrame()
    
    # 合并所有数据
    combined_df = pd.concat(all_data, ignore_index=True)
    logger.info(f"✅ 数据合并完成 - 总计: {len(combined_df)} 条记录")
    
    return combined_df

def analyze_data_distribution(df: pd.DataFrame) -> Dict:
    """分析数据分布"""
    logger.info("📊 分析数据分布:")
    
    category_counts = df['category'].value_counts()
    total_samples = len(df)
    
    distribution = {}
    for category, count in category_counts.items():
        percentage = (count / total_samples) * 100
        distribution[category] = {
            'count': count,
            'percentage': percentage
        }
        logger.info(f"  {category}: {count} 条 ({percentage:.1f}%)")
    
    # 检查数据不平衡
    max_count = category_counts.max()
    min_count = category_counts.min()
    imbalance_ratio = max_count / min_count
    
    logger.info(f"📈 数据不平衡比例: {imbalance_ratio:.2f}:1")
    
    return {
        'total_samples': total_samples,
        'category_distribution': distribution,
        'imbalance_ratio': imbalance_ratio,
        'num_categories': len(category_counts)
    }

def clean_and_prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """清洗和准备数据"""
    logger.info("🧹 开始数据清洗")
    
    # 记录原始数据量
    original_count = len(df)
    
    # 移除空值
    df_cleaned = df.dropna(subset=['original_log', 'category'])
    
    # 移除空字符串
    df_cleaned = df_cleaned[df_cleaned['original_log'].str.strip() != '']
    
    # 移除重复数据
    df_cleaned = df_cleaned.drop_duplicates(subset=['original_log'])
    
    # 记录清洗后的数据量
    cleaned_count = len(df_cleaned)
    removed_count = original_count - cleaned_count
    
    logger.info(f"✅ 数据清洗完成:")
    logger.info(f"  原始数据: {original_count} 条")
    logger.info(f"  清洗后数据: {cleaned_count} 条")
    logger.info(f"  移除数据: {removed_count} 条")
    
    return df_cleaned

def save_processed_data(df: pd.DataFrame, output_path: str):
    """保存处理后的数据"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_csv(output_path, index=False, encoding='utf-8')
    logger.info(f"💾 数据已保存: {output_path}")
    
    # 保存数据信息
    info_path = output_path.replace('.csv', '_info.txt')
    with open(info_path, 'w', encoding='utf-8') as f:
        f.write(f"数据文件信息\n")
        f.write(f"============\n")
        f.write(f"文件路径: {output_path}\n")
        f.write(f"总记录数: {len(df)}\n")
        f.write(f"类别数量: {df['category'].nunique()}\n")
        f.write(f"列名: {list(df.columns)}\n\n")
        
        f.write(f"类别分布:\n")
        category_counts = df['category'].value_counts()
        for category, count in category_counts.items():
            percentage = (count / len(df)) * 100
            f.write(f"  {category}: {count} 条 ({percentage:.1f}%)\n")
    
    logger.info(f"📋 数据信息已保存: {info_path}")

def main():
    """主函数"""
    # 数据路径
    data_output_path = r"C:\Users\30871\Desktop\AILogForSpring\DATA_OUTPUT"
    output_path = "data/processed_logs_full.csv"
    
    logger.info("🚀 开始准备完整数据集")
    
    # 检查数据目录
    if not os.path.exists(data_output_path):
        logger.error(f"❌ 数据目录不存在: {data_output_path}")
        return
    
    # 加载所有类别数据
    df = load_category_data(data_output_path)
    
    if df.empty:
        logger.error("❌ 没有加载到任何数据")
        return
    
    # 分析数据分布
    data_info = analyze_data_distribution(df)
    
    # 清洗数据
    df_cleaned = clean_and_prepare_data(df)
    
    # 保存处理后的数据
    save_processed_data(df_cleaned, output_path)
    
    logger.info("✅ 数据准备完成!")
    logger.info(f"📁 输出文件: {output_path}")
    logger.info(f"📊 最终数据: {len(df_cleaned)} 条记录, {df_cleaned['category'].nunique()} 个类别")

if __name__ == "__main__":
    main() 