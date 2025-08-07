#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的数据平衡器 - 使用更智能的采样策略
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


class ImprovedDataBalancer:
    """改进的数据平衡器"""
    
    def __init__(self, target_samples_per_class=500):
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
    
    def smart_balance_data(self, df: pd.DataFrame, category_counts: pd.Series):
        """智能平衡数据"""
        logger.info(f"🎯 目标每类样本数: {self.target_samples_per_class}")
        
        balanced_dfs = []
        
        for category in category_counts.index:
            category_df = df[df['category'] == category].copy()
            current_count = len(category_df)
            
            logger.info(f"📊 处理类别: {category} (当前: {current_count} 条)")
            
            if current_count < self.target_samples_per_class:
                # 上采样 - 使用更智能的策略
                if current_count > 0:
                    # 计算需要重复的次数
                    repeat_times = self.target_samples_per_class // current_count
                    remainder = self.target_samples_per_class % current_count
                    
                    # 重复采样
                    repeated_samples = []
                    for _ in range(repeat_times):
                        repeated_samples.append(category_df)
                    
                    # 添加剩余样本
                    if remainder > 0:
                        remainder_samples = category_df.sample(n=remainder, random_state=42)
                        repeated_samples.append(remainder_samples)
                    
                    # 合并并打乱
                    oversampled = pd.concat(repeated_samples, ignore_index=True)
                    oversampled = oversampled.sample(frac=1, random_state=42).reset_index(drop=True)
                    
                    logger.info(f"  ✅ 智能上采样到 {len(oversampled)} 条")
                    balanced_dfs.append(oversampled)
                else:
                    logger.warning(f"  ⚠️ 类别 {category} 无样本，跳过")
                    continue
            elif current_count > self.target_samples_per_class:
                # 下采样 - 使用分层采样
                undersampled = category_df.sample(
                    n=self.target_samples_per_class, 
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
    
    def create_improved_data(self, data_path: str, output_path: str):
        """创建改进的平衡数据"""
        logger.info("🚀 开始创建改进的平衡数据")
        
        # 加载数据
        df, category_counts = self.load_and_analyze_data(data_path)
        
        # 智能平衡数据
        balanced_df = self.smart_balance_data(df, category_counts)
        
        # 保存结果
        balanced_df.to_csv(output_path, index=False)
        logger.info(f"💾 保存到: {output_path}")
        
        return balanced_df


def main():
    """主函数"""
    balancer = ImprovedDataBalancer(target_samples_per_class=500)
    
    input_path = "data/processed_logs_final_cleaned.csv"
    output_path = "data/processed_logs_improved_balanced.csv"
    
    balanced_df = balancer.create_improved_data(input_path, output_path)
    
    logger.info("✅ 改进数据平衡完成!")


if __name__ == "__main__":
    main() 