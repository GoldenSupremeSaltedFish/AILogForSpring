#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分阶段数据准备脚本
支持小数据集快速验证和完整数据集训练
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


def create_staged_datasets(category_data: Dict[str, pd.DataFrame], 
                          small_samples_per_category: int = 100,
                          large_samples_per_category: int = 1500) -> Dict[str, pd.DataFrame]:
    """
    创建分阶段数据集
    Args:
        category_data: 分类数据字典
        small_samples_per_category: 小数据集每类样本数
        large_samples_per_category: 大数据集每类样本数
    Returns:
        包含小数据集和完整数据集的字典
    """
    datasets = {}
    
    # 创建小数据集（快速验证用）
    logger.info("🔬 创建小数据集（快速验证）...")
    small_dfs = []
    
    for category, df in category_data.items():
        # 小数据集：每类100条
        if len(df) > small_samples_per_category:
            df_small = df.sample(n=small_samples_per_category, random_state=42)
            logger.info(f"📊 {category}: 小数据集 {len(df_small)} 条 (原始 {len(df)} 条)")
        else:
            df_small = df
            logger.info(f"📊 {category}: 小数据集使用全部 {len(df_small)} 条")
        
        small_dfs.append(df_small)
    
    small_dataset = pd.concat(small_dfs, ignore_index=True)
    datasets['small'] = small_dataset
    
    # 创建完整数据集（正式训练用）
    logger.info("🚀 创建完整数据集（正式训练）...")
    large_dfs = []
    
    for category, df in category_data.items():
        # 完整数据集：每类最多1500条
        if len(df) > large_samples_per_category:
            df_large = df.sample(n=large_samples_per_category, random_state=42)
            logger.info(f"📊 {category}: 完整数据集 {len(df_large)} 条 (原始 {len(df)} 条)")
        else:
            df_large = df
            logger.info(f"📊 {category}: 完整数据集使用全部 {len(df_large)} 条")
        
        large_dfs.append(df_large)
    
    large_dataset = pd.concat(large_dfs, ignore_index=True)
    datasets['large'] = large_dataset
    
    return datasets


def process_dataset(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    处理数据集
    Args:
        df: 原始数据框
        dataset_name: 数据集名称
    Returns:
        处理后的数据框
    """
    # 确保有message列
    if 'message' not in df.columns:
        # 尝试找到包含日志内容的列
        possible_columns = ['log', 'content', 'text', 'message', '日志', '内容']
        for col in possible_columns:
            if col in df.columns:
                df['message'] = df[col]
                break
        
        if 'message' not in df.columns:
            # 如果没有找到合适的列，使用第一列作为message
            first_col = df.columns[0]
            df['message'] = df[first_col].astype(str)
            logger.warning(f"⚠️ 未找到message列，使用第一列: {first_col}")
    
    # 数据清洗
    df['message'] = df['message'].fillna('').astype(str)
    
    # 移除空消息
    df = df[df['message'].str.strip() != '']
    
    logger.info(f"📊 {dataset_name} 数据集处理完成:")
    logger.info(f"   总样本数: {len(df)}")
    logger.info(f"   分类数: {df['category'].nunique()}")
    
    category_counts = df['category'].value_counts()
    for category, count in category_counts.items():
        logger.info(f"   {category}: {count} 条")
    
    return df


def save_datasets(datasets: Dict[str, pd.DataFrame], output_dir: str = "data"):
    """
    保存数据集
    Args:
        datasets: 数据集字典
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for dataset_name, df in datasets.items():
        output_path = os.path.join(output_dir, f"processed_logs_{dataset_name}.csv")
        df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"💾 {dataset_name} 数据集已保存: {output_path}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="分阶段数据准备工具")
    parser.add_argument("--small_samples", type=int, default=100, 
                       help="小数据集每类样本数")
    parser.add_argument("--large_samples", type=int, default=1500, 
                       help="完整数据集每类样本数")
    parser.add_argument("--data_dir", type=str, default="../DATA_OUTPUT", 
                       help="数据目录路径")
    parser.add_argument("--output_dir", type=str, default="data", 
                       help="输出目录")
    
    args = parser.parse_args()
    
    logger.info("🚀 开始分阶段数据准备")
    logger.info(f"📂 数据目录: {args.data_dir}")
    logger.info(f"📦 小数据集每类样本数: {args.small_samples}")
    logger.info(f"📦 完整数据集每类样本数: {args.large_samples}")
    
    # 检查数据目录
    if not os.path.exists(args.data_dir):
        logger.error(f"❌ 数据目录不存在: {args.data_dir}")
        return
    
    # 加载所有分类数据
    category_data = load_category_data(args.data_dir)
    
    if not category_data:
        logger.error("❌ 未找到任何分类数据")
        return
    
    # 创建分阶段数据集
    datasets = create_staged_datasets(
        category_data, 
        small_samples_per_category=args.small_samples,
        large_samples_per_category=args.large_samples
    )
    
    # 处理并保存数据集
    processed_datasets = {}
    for dataset_name, df in datasets.items():
        processed_df = process_dataset(df, dataset_name)
        processed_datasets[dataset_name] = processed_df
    
    # 保存数据集
    save_datasets(processed_datasets, args.output_dir)
    
    logger.info("✅ 分阶段数据准备完成！")
    logger.info("🎯 训练命令:")
    logger.info("   小数据集验证: python scripts/train.py --model textcnn --data data/processed_logs_small.csv --epochs 3")
    logger.info("   完整数据集训练: python scripts/train.py --model textcnn --data data/processed_logs_large.csv --epochs 10")


if __name__ == "__main__":
    main() 