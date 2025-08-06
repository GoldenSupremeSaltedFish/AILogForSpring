#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级数据改进策略
"""

import os
import pandas as pd
import numpy as np
import logging
from collections import Counter
import random
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedDataAugmenter:
    def __init__(self):
        self.synonyms = {
            'error': ['exception', 'failure', 'issue', 'problem', 'bug'],
            'failed': ['failed', 'error', 'exception', 'problem', 'broken'],
            'exception': ['error', 'failure', 'issue', 'problem', 'crash'],
            'connection': ['link', 'connect', 'network', 'socket', 'channel'],
            'database': ['db', 'datastore', 'repository', 'storage', 'table'],
            'timeout': ['timeout', 'expired', 'timed_out', 'overdue', 'deadline'],
            'memory': ['ram', 'storage', 'cache', 'buffer', 'heap'],
            'performance': ['speed', 'efficiency', 'throughput', 'latency', 'response'],
            'authentication': ['auth', 'login', 'verify', 'validate', 'check'],
            'authorization': ['permission', 'access', 'rights', 'privilege', 'role'],
            'configuration': ['config', 'setting', 'parameter', 'option', 'property'],
            'environment': ['env', 'surrounding', 'context', 'condition', 'setup'],
            'business': ['commercial', 'enterprise', 'corporate', 'trade', 'service'],
            'logic': ['reasoning', 'algorithm', 'process', 'method', 'function'],
            'operation': ['action', 'task', 'process', 'function', 'work'],
            'monitoring': ['watch', 'observe', 'track', 'supervise', 'check'],
            'heartbeat': ['pulse', 'signal', 'alive', 'status', 'ping'],
            'startup': ['boot', 'launch', 'initialize', 'begin', 'start'],
            'spring': ['spring', 'framework', 'boot', 'application', 'context']
        }
        
        self.insert_words = [
            'error', 'exception', 'failed', 'timeout', 'connection',
            'database', 'memory', 'performance', 'authentication',
            'authorization', 'configuration', 'environment', 'business',
            'logic', 'operation', 'monitoring', 'heartbeat', 'startup'
        ]
        
        self.delete_prob = 0.15
        self.insert_prob = 0.25
        self.replace_prob = 0.3
        self.swap_prob = 0.1
    
    def augment_text_advanced(self, text: str, num_augmentations: int = 5) -> list:
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
                    if word.lower() in self.synonyms and random.random() < 0.4:
                        synonyms = self.synonyms[word.lower()]
                        words[i] = random.choice(synonyms)
                augmented = ' '.join(words)
            
            # 随机交换相邻单词
            if random.random() < self.swap_prob:
                words = augmented.split()
                if len(words) > 1:
                    swap_idx = random.randint(0, len(words) - 2)
                    words[swap_idx], words[swap_idx + 1] = words[swap_idx + 1], words[swap_idx]
                    augmented = ' '.join(words)
            
            if augmented != text:
                augmented_texts.append(augmented)
        
        return augmented_texts
    
    def augment_category_advanced(self, df: pd.DataFrame, category: str, target_count: int = 500) -> pd.DataFrame:
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
            augmented_texts = self.augment_text_advanced(text, num_augmentations=3)
            
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
            augmented_texts = self.augment_text_advanced(text, num_augmentations=2)
            
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

def improve_data_advanced(input_path: str, output_path: str, target_samples: int = 500):
    logger.info("🚀 开始高级数据改进")
    
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
    
    # 高级数据增强
    logger.info("🔧 开始高级数据增强")
    augmenter = AdvancedDataAugmenter()
    enhanced_dfs = []
    
    for category in valid_categories:
        category_df = df_filtered[df_filtered['category'] == category].copy()
        enhanced_df = augmenter.augment_category_advanced(category_df, category, target_samples)
        enhanced_dfs.append(enhanced_df)
    
    # 合并增强后的数据
    enhanced_df = pd.concat(enhanced_dfs, ignore_index=True)
    logger.info(f"📊 增强后数据: {len(enhanced_df)} 条记录")
    
    # 保存改进后的数据
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    enhanced_df.to_csv(output_path, index=False, encoding='utf-8')
    
    # 分析最终分布
    final_counts = enhanced_df['category'].value_counts()
    logger.info("📊 最终数据分布:")
    for category, count in final_counts.items():
        logger.info(f"  {category}: {count} 条")
    
    # 计算不平衡比例
    max_count = final_counts.max()
    min_count = final_counts.min()
    imbalance_ratio = max_count / min_count
    
    logger.info(f"📈 最终不平衡比例: {imbalance_ratio:.2f}:1")
    logger.info(f"✅ 高级数据改进完成! 保存到: {output_path}")
    
    return enhanced_df

def main():
    input_path = "data/processed_logs_full.csv"
    output_path = "data/processed_logs_advanced.csv"
    target_samples = 500
    
    logger.info("🎯 高级数据改进策略")
    logger.info(f"📁 输入文件: {input_path}")
    logger.info(f"📁 输出文件: {output_path}")
    logger.info(f"🎯 目标每类样本数: {target_samples}")
    
    # 检查输入文件
    if not os.path.exists(input_path):
        logger.error(f"❌ 输入文件不存在: {input_path}")
        return
    
    # 执行高级数据改进
    improved_df = improve_data_advanced(input_path, output_path, target_samples)
    
    logger.info("✅ 高级数据改进完成!")

if __name__ == "__main__":
    main() 