#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
准备完整数据集 - 合并所有11种类别的日志
"""

import os
import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 类别映射
CATEGORIES = {
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

def load_all_data(data_path: str):
    """加载所有类别数据"""
    logger.info(f"📂 加载数据: {data_path}")
    
    all_data = []
    
    for folder_name, category in CATEGORIES.items():
        folder_path = os.path.join(data_path, folder_name)
        
        if not os.path.exists(folder_path):
            logger.warning(f"⚠️ 目录不存在: {folder_path}")
            continue
            
        logger.info(f"📁 处理: {category}")
        
        # 查找CSV文件
        csv_files = list(Path(folder_path).glob("*.csv"))
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                df['category'] = category
                all_data.append(df)
                logger.info(f"  📄 {csv_file.name}: {len(df)} 条")
            except Exception as e:
                logger.error(f"❌ 读取失败: {csv_file.name} - {e}")
    
    if not all_data:
        logger.error("❌ 没有找到数据")
        return None
    
    combined_df = pd.concat(all_data, ignore_index=True)
    logger.info(f"✅ 合并完成: {len(combined_df)} 条记录")
    
    return combined_df

def main():
    """主函数"""
    data_path = r"C:\Users\30871\Desktop\AILogForSpring\DATA_OUTPUT"
    output_path = "data/processed_logs_full.csv"
    
    # 加载数据
    df = load_all_data(data_path)
    if df is None:
        return
    
    # 清洗数据
    df_cleaned = df.dropna(subset=['original_log', 'category'])
    df_cleaned = df_cleaned[df_cleaned['original_log'].str.strip() != '']
    df_cleaned = df_cleaned.drop_duplicates(subset=['original_log'])
    
    # 分析分布
    category_counts = df_cleaned['category'].value_counts()
    logger.info("📊 数据分布:")
    for category, count in category_counts.items():
        percentage = (count / len(df_cleaned)) * 100
        logger.info(f"  {category}: {count} 条 ({percentage:.1f}%)")
    
    # 保存数据
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_cleaned.to_csv(output_path, index=False, encoding='utf-8')
    
    logger.info(f"✅ 完成! 保存到: {output_path}")
    logger.info(f"📊 最终数据: {len(df_cleaned)} 条, {len(category_counts)} 个类别")

if __name__ == "__main__":
    main() 