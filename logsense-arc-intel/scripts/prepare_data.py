#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据准备脚本
合并所有分类好的日志数据，创建训练集
"""

import pandas as pd
import os
import logging
from pathlib import Path
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_category_data(data_dir: str) -> Dict[str, pd.DataFrame]:
    """
    加载所有分类数据
    Args:
        data_dir: 数据目录路径
    Returns:
        分类数据字典
    """
    category_data = {}
    
    # 定义分类映射
    category_mapping = {
        '01_堆栈异常_stack_exception': 'stack_exception',
        '02_数据库异常_database_exception': 'database_exception', 
        '03_连接问题_connection_issue': 'connection_issue'
    }
    
    for folder_name, category in category_mapping.items():
        folder_path = os.path.join(data_dir, folder_name)
        if os.path.exists(folder_path):
            logger.info(f"📂 加载分类: {category}")
            
            # 读取该分类下的所有CSV文件
            csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
            dfs = []
            
            for csv_file in csv_files:
                file_path = os.path.join(folder_path, csv_file)
                try:
                    df = pd.read_csv(file_path)
                    # 添加分类标签
                    df['category'] = category
                    dfs.append(df)
                    logger.info(f"   ✅ 加载文件: {csv_file} ({len(df)} 条记录)")
                except Exception as e:
                    logger.warning(f"   ⚠️ 跳过文件: {csv_file} - {e}")
            
            if dfs:
                # 合并该分类的所有数据
                category_df = pd.concat(dfs, ignore_index=True)
                category_data[category] = category_df
                logger.info(f"   📊 分类 {category} 总计: {len(category_df)} 条记录")
    
    return category_data


def create_balanced_dataset(category_data: Dict[str, pd.DataFrame], 
                          max_samples_per_category: int = 1500) -> pd.DataFrame:
    """
    创建平衡的数据集
    Args:
        category_data: 分类数据字典
        max_samples_per_category: 每个分类的最大样本数
    Returns:
        平衡的数据集
    """
    balanced_dfs = []
    
    for category, df in category_data.items():
        if len(df) > max_samples_per_category:
            # 随机采样
            df_sampled = df.sample(n=max_samples_per_category, random_state=42)
            logger.info(f"📊 {category}: 采样 {len(df_sampled)} 条 (原始 {len(df)} 条)")
        else:
            df_sampled = df
            logger.info(f"📊 {category}: 使用全部 {len(df_sampled)} 条")
        
        balanced_dfs.append(df_sampled)
    
    # 合并所有分类数据
    final_df = pd.concat(balanced_dfs, ignore_index=True)
    
    # 确保有message列
    if 'message' not in final_df.columns:
        # 尝试找到包含日志内容的列
        possible_columns = ['log', 'content', 'text', 'message', '日志', '内容']
        for col in possible_columns:
            if col in final_df.columns:
                final_df['message'] = final_df[col]
                break
        
        if 'message' not in final_df.columns:
            # 如果没有找到合适的列，使用第一列作为message
            first_col = final_df.columns[0]
            final_df['message'] = final_df[first_col].astype(str)
            logger.warning(f"⚠️ 未找到message列，使用第一列: {first_col}")
    
    return final_df


def save_processed_data(df: pd.DataFrame, output_path: str):
    """
    保存处理后的数据
    Args:
        df: 数据框
        output_path: 输出路径
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存数据
    df.to_csv(output_path, index=False, encoding='utf-8')
    logger.info(f"💾 数据已保存: {output_path}")
    
    # 打印统计信息
    logger.info("📊 数据集统计:")
    logger.info(f"   总样本数: {len(df)}")
    logger.info(f"   分类数: {df['category'].nunique()}")
    
    category_counts = df['category'].value_counts()
    for category, count in category_counts.items():
        logger.info(f"   {category}: {count} 条")


def main():
    """主函数"""
    logger.info("🚀 开始准备训练数据")
    
    # 数据目录路径
    data_dir = "../DATA_OUTPUT"
    output_path = "data/processed_logs.csv"
    
    # 检查数据目录
    if not os.path.exists(data_dir):
        logger.error(f"❌ 数据目录不存在: {data_dir}")
        return
    
    # 加载所有分类数据
    category_data = load_category_data(data_dir)
    
    if not category_data:
        logger.error("❌ 未找到任何分类数据")
        return
    
    # 创建平衡数据集
    logger.info("⚖️ 创建平衡数据集...")
    balanced_df = create_balanced_dataset(category_data, max_samples_per_category=1500)
    
    # 保存处理后的数据
    save_processed_data(balanced_df, output_path)
    
    logger.info("✅ 数据准备完成！")
    logger.info(f"🎯 可以开始训练: python scripts/train.py --model textcnn --data {output_path}")


if __name__ == "__main__":
    main() 