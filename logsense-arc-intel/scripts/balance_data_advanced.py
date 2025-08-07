#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级数据平衡脚本 - 针对Attention机制优化
"""

import pandas as pd
import numpy as np
import logging
from collections import Counter
from sklearn.utils import resample
import re

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AdvancedDataBalancer:
    """高级数据平衡器"""
    
    def __init__(self, target_samples_per_class=1000):
        self.target_samples_per_class = target_samples_per_class
    
    def load_and_analyze_data(self, data_path: str):
        """加载并分析数据"""
        logger.info(f"📂 加载数据: {data_path}")
        df = pd.read_csv(data_path)
        logger.info(f"📊 原始数据: {len(df)} 条记录")
        
        # 数据清洗
        df_cleaned = df.dropna(subset=['original_log', 'category'])
        df_cleaned = df_cleaned[df_cleaned['original_log'].str.strip() != '']
        
        # 分析类别分布
        category_counts = df_cleaned['category'].value_counts()
        logger.info("📊 原始数据分布:")
        for category, count in category_counts.items():
            percentage = (count / len(df_cleaned)) * 100
            logger.info(f"  {category}: {count} 条 ({percentage:.1f}%)")
        
        return df_cleaned, category_counts
    
    def balance_data(self, df: pd.DataFrame, category_counts: pd.Series):
        """平衡数据"""
        logger.info(f"🎯 目标每类样本数: {self.target_samples_per_class}")
        
        balanced_dfs = []
        
        for category in category_counts.index:
            category_df = df[df['category'] == category].copy()
            current_count = len(category_df)
            
            logger.info(f"📊 处理类别: {category} (当前: {current_count} 条)")
            
            if current_count < self.target_samples_per_class:
                # 上采样
                if current_count > 0:
                    # 使用重复采样
                    oversampled = resample(
                        category_df,
                        n_samples=self.target_samples_per_class,
                        replace=True,
                        random_state=42
                    )
                    logger.info(f"  ✅ 上采样到 {self.target_samples_per_class} 条")
                    balanced_dfs.append(oversampled)
                else:
                    logger.warning(f"  ⚠️ 类别 {category} 无样本，跳过")
                    continue
            elif current_count > self.target_samples_per_class:
                # 下采样
                undersampled = resample(
                    category_df,
                    n_samples=self.target_samples_per_class,
                    replace=False,
                    random_state=42
                )
                logger.info(f"  ✅ 下采样到 {self.target_samples_per_class} 条")
                balanced_dfs.append(undersampled)
            else:
                # 保持原样
                logger.info(f"  ✅ 保持 {current_count} 条")
                balanced_dfs.append(category_df)
        
        # 合并所有类别
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        
        # 打乱数据
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        logger.info(f"📊 平衡后数据: {len(balanced_df)} 条记录")
        
        # 验证平衡结果
        final_counts = balanced_df['category'].value_counts()
        logger.info("📊 平衡后分布:")
        for category, count in final_counts.items():
            percentage = (count / len(balanced_df)) * 100
            logger.info(f"  {category}: {count} 条 ({percentage:.1f}%)")
        
        return balanced_df
    
    def create_attention_optimized_data(self, data_path: str, output_path: str):
        """创建针对Attention优化的数据"""
        logger.info("🚀 开始创建Attention优化数据")
        
        # 加载数据
        df, category_counts = self.load_and_analyze_data(data_path)
        
        # 平衡数据
        balanced_df = self.balance_data(df, category_counts)
        
        # 保存结果
        balanced_df.to_csv(output_path, index=False)
        logger.info(f"💾 保存到: {output_path}")
        
        return balanced_df


def main():
    """主函数"""
    balancer = AdvancedDataBalancer(target_samples_per_class=1000)
    
    input_path = "data/processed_logs_final_cleaned.csv"
    output_path = "data/processed_logs_balanced_attention.csv"
    
    balanced_df = balancer.create_attention_optimized_data(input_path, output_path)
    
    logger.info("✅ 数据平衡完成!")


if __name__ == "__main__":
    main() 