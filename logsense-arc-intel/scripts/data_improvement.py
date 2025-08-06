#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据改进策略 - 数据增强和过采样
"""

import os
import pandas as pd
import numpy as np
import logging
from collections import Counter
from typing import List, Dict, Tuple
import random
import re

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataAugmenter:
    """数据增强器"""
    
    def __init__(self):
        # 同义词词典
        self.synonyms = {
            'error': ['exception', 'failure', 'issue', 'problem'],
            'failed': ['failed', 'error', 'exception', 'problem'],
            'exception': ['error', 'failure', 'issue', 'problem'],
            'connection': ['link', 'connect', 'network', 'socket'],
            'database': ['db', 'datastore', 'repository', 'storage'],
            'timeout': ['timeout', 'expired', 'timed_out', 'overdue'],
            'memory': ['ram', 'storage', 'cache', 'buffer'],
            'performance': ['speed', 'efficiency', 'throughput', 'latency'],
            'authentication': ['auth', 'login', 'verify', 'validate'],
            'authorization': ['permission', 'access', 'rights', 'privilege'],
            'configuration': ['config', 'setting', 'parameter', 'option'],
            'environment': ['env', 'surrounding', 'context', 'condition'],
            'business': ['commercial', 'enterprise', 'corporate', 'trade'],
            'logic': ['reasoning', 'algorithm', 'process', 'method'],
            'operation': ['action', 'task', 'process', 'function'],
            'monitoring': ['watch', 'observe', 'track', 'supervise'],
            'heartbeat': ['pulse', 'signal', 'alive', 'status'],
            'startup': ['boot', 'launch', 'initialize', 'begin'],
            'spring': ['spring', 'framework', 'boot', 'application']
        }
        
        # 插入词列表
        self.insert_words = [
            'error', 'exception', 'failed', 'timeout', 'connection',
            'database', 'memory', 'performance', 'authentication',
            'authorization', 'configuration', 'environment', 'business',
            'logic', 'operation', 'monitoring', 'heartbeat', 'startup'
        ]
        
        # 删除概率
        self.delete_prob = 0.1
        # 插入概率
        self.insert_prob = 0.15
        # 替换概率
        self.replace_prob = 0.2
    
    def augment_text(self, text: str, num_augmentations: int = 3) -> List[str]:
        """增强单个文本"""
        augmented_texts = []
        
        for _ in range(num_augmentations):
            augmented = text
            
            # 随机删除单词
            if random.random() < self.delete_prob:
                words = augmented.split()
                if len(words) > 3:
                    delete_idx = random.randint(0, len(words) - 1)
                    words.pop(delete_idx)
                    augmented = ' '.join(words)
            
            # 随机插入单词
            if random.random() < self.insert_prob:
                words = augmented.split()
                if len(words) > 0:
                    insert_word = random.choice(self.insert_words)
                    insert_idx = random.randint(0, len(words))
                    words.insert(insert_idx, insert_word)
                    augmented = ' '.join(words)
            
            # 随机替换单词
            if random.random() < self.replace_prob:
                words = augmented.split()
                for i, word in enumerate(words):
                    if word.lower() in self.synonyms and random.random() < 0.3:
                        synonyms = self.synonyms[word.lower()]
                        words[i] = random.choice(synonyms)
                augmented = ' '.join(words)
            
            if augmented != text:
                augmented_texts.append(augmented)
        
        return augmented_texts
    
    def augment_category(self, df: pd.DataFrame, category: str, target_count: int = 300) -> pd.DataFrame:
        """增强特定类别的数据"""
        category_df = df[df['category'] == category].copy()
        current_count = len(category_df)
        
        if current_count >= target_count:
            logger.info(f"  {category}: 已有 {current_count} 条，无需增强")
            return category_df
        
        needed_count = target_count - current_count
        logger.info(f"  {category}: 当前 {current_count} 条，需要增加到 {target_count} 条")
        
        augmented_data = []
        
        # 对现有样本进行增强
        for idx, row in category_df.iterrows():
            text = row['original_log']
            augmented_texts = self.augment_text(text, num_augmentations=2)
            
            for aug_text in augmented_texts:
                if len(augmented_data) < needed_count:
                    new_row = row.copy()
                    new_row['original_log'] = aug_text
                    new_row['augmented'] = True
                    augmented_data.append(new_row)
        
        # 如果还不够，继续增强
        while len(augmented_data) < needed_count and len(category_df) > 0:
            sample_row = category_df.sample(n=1).iloc[0]
            text = sample_row['original_log']
            augmented_texts = self.augment_text(text, num_augmentations=1)
            
            for aug_text in augmented_texts:
                if len(augmented_data) < needed_count:
                    new_row = sample_row.copy()
                    new_row['original_log'] = aug_text
                    new_row['augmented'] = True
                    augmented_data.append(new_row)
        
        # 合并原始数据和增强数据
        augmented_df = pd.DataFrame(augmented_data)
        result_df = pd.concat([category_df, augmented_df], ignore_index=True)
        
        logger.info(f"  {category}: 增强后 {len(result_df)} 条")
        return result_df


class DataBalancer:
    """数据平衡器"""
    
    def __init__(self, target_samples_per_class: int = 300):
        self.target_samples_per_class = target_samples_per_class
    
    def balance_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """平衡数据"""
        logger.info("🔄 开始数据平衡")
        
        # 分析当前分布
        category_counts = df['category'].value_counts()
        logger.info("📊 原始数据分布:")
        for category, count in category_counts.items():
            logger.info(f"  {category}: {count} 条")
        
        # 确定目标数量
        max_count = category_counts.max()
        target_count = min(self.target_samples_per_class, max_count)
        logger.info(f"🎯 目标每类样本数: {target_count}")
        
        balanced_dfs = []
        
        for category in df['category'].unique():
            category_df = df[df['category'] == category].copy()
            current_count = len(category_df)
            
            if current_count >= target_count:
                # 如果样本足够，随机采样
                balanced_df = category_df.sample(n=target_count, random_state=42)
                logger.info(f"  {category}: 采样 {len(balanced_df)} 条")
            else:
                # 如果样本不足，过采样
                balanced_df = self._oversample_category(category_df, target_count)
                logger.info(f"  {category}: 过采样到 {len(balanced_df)} 条")
            
            balanced_dfs.append(balanced_df)
        
        # 合并所有类别
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        
        # 分析平衡后的分布
        balanced_counts = balanced_df['category'].value_counts()
        logger.info("📊 平衡后数据分布:")
        for category, count in balanced_counts.items():
            logger.info(f"  {category}: {count} 条")
        
        return balanced_df
    
    def _oversample_category(self, category_df: pd.DataFrame, target_count: int) -> pd.DataFrame:
        """过采样特定类别"""
        current_count = len(category_df)
        
        if current_count == 0:
            return category_df
        
        # 计算需要重复的次数
        repeat_times = target_count // current_count
        remainder = target_count % current_count
        
        # 重复数据
        repeated_df = pd.concat([category_df] * repeat_times, ignore_index=True)
        
        # 添加剩余样本
        if remainder > 0:
            extra_samples = category_df.sample(n=remainder, random_state=42)
            repeated_df = pd.concat([repeated_df, extra_samples], ignore_index=True)
        
        return repeated_df


def improve_data(input_path: str, output_path: str, target_samples: int = 300):
    """改进数据"""
    logger.info("🚀 开始数据改进")
    
    # 加载数据
    df = pd.read_csv(input_path)
    logger.info(f"📂 加载数据: {len(df)} 条记录")
    
    # 数据清洗
    df_cleaned = df.dropna(subset=['original_log', 'category'])
    df_cleaned = df_cleaned[df_cleaned['original_log'].str.strip() != '']
    
    # 过滤掉样本太少的类别
    category_counts = df_cleaned['category'].value_counts()
    min_samples = 10
    valid_categories = category_counts[category_counts >= min_samples].index.tolist()
    df_filtered = df_cleaned[df_cleaned['category'].isin(valid_categories)]
    
    logger.info(f"📊 过滤后数据: {len(df_filtered)} 条记录")
    logger.info(f"📊 有效类别: {len(valid_categories)} 个")
    
    # 数据增强
    logger.info("🔧 开始数据增强")
    augmenter = DataAugmenter()
    enhanced_dfs = []
    
    for category in valid_categories:
        category_df = df_filtered[df_filtered['category'] == category].copy()
        enhanced_df = augmenter.augment_category(category_df, category, target_samples)
        enhanced_dfs.append(enhanced_df)
    
    # 合并增强后的数据
    enhanced_df = pd.concat(enhanced_dfs, ignore_index=True)
    logger.info(f"📊 增强后数据: {len(enhanced_df)} 条记录")
    
    # 数据平衡
    logger.info("⚖️ 开始数据平衡")
    balancer = DataBalancer(target_samples_per_class=target_samples)
    balanced_df = balancer.balance_data(enhanced_df)
    
    # 保存改进后的数据
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    balanced_df.to_csv(output_path, index=False, encoding='utf-8')
    
    # 分析最终分布
    final_counts = balanced_df['category'].value_counts()
    logger.info("📊 最终数据分布:")
    for category, count in final_counts.items():
        logger.info(f"  {category}: {count} 条")
    
    # 计算不平衡比例
    max_count = final_counts.max()
    min_count = final_counts.min()
    imbalance_ratio = max_count / min_count
    
    logger.info(f"📈 最终不平衡比例: {imbalance_ratio:.2f}:1")
    logger.info(f"✅ 数据改进完成! 保存到: {output_path}")
    
    return balanced_df


def main():
    """主函数"""
    input_path = "data/processed_logs_full.csv"
    output_path = "data/processed_logs_improved.csv"
    target_samples = 300
    
    logger.info("🎯 数据改进策略")
    logger.info(f"📁 输入文件: {input_path}")
    logger.info(f"📁 输出文件: {output_path}")
    logger.info(f"🎯 目标每类样本数: {target_samples}")
    
    # 检查输入文件
    if not os.path.exists(input_path):
        logger.error(f"❌ 输入文件不存在: {input_path}")
        return
    
    # 执行数据改进
    improved_df = improve_data(input_path, output_path, target_samples)
    
    logger.info("✅ 数据改进完成!")


if __name__ == "__main__":
    main() 