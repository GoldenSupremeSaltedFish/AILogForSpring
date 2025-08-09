#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级数据质量增强器 - 包含智能数据平衡和精细特征工程
"""

import pandas as pd
import numpy as np
import logging
import re
from collections import Counter
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AdvancedFeatureExtractor:
    """高级特征提取器"""
    
    def __init__(self):
        # 错误码模式 - 更精确的匹配
        self.error_patterns = [
            r'error\s*[:\s]*([A-Z0-9_]{3,10})',  # ERROR: CODE123
            r'err\s*[:\s]*([A-Z0-9_]{3,10})',    # ERR: CODE123
            r'([A-Z]{2,10}\d{3,6})',             # 通用错误码
            r'([A-Z]{2,10}Exception)',           # Java异常
            r'([A-Z][a-z]+Exception)',           # 异常类名
            r'([A-Z][a-z]+Error)',               # 错误类名
        ]
        
        # 路径模式 - 更精确的匹配
        self.path_patterns = [
            r'([A-Za-z]:\\[^\s,]+)',             # Windows路径
            r'(/[^\s,]+)',                        # Unix路径
            r'([A-Za-z0-9_/.-]+\.(java|py|js|ts|go|cpp|c|h))',  # 文件路径
            r'([a-zA-Z0-9_.]+\.(java|py|js|ts))', # 类文件
        ]
        
        # 数字模式 - 更精确的匹配
        self.number_patterns = [
            r'(\d+\.\d+)',                        # 浮点数
            r'(\d{3,6})',                         # 错误代码
            r'(\d+)',                             # 整数
        ]
        
        # 类名模式 - 更精确的匹配
        self.class_patterns = [
            r'([A-Z][a-zA-Z0-9_]*\.java)',       # Java类文件
            r'([A-Z][a-zA-Z0-9_]*\.py)',         # Python文件
            r'([A-Z][a-zA-Z0-9_]*\.js)',         # JavaScript文件
            r'([A-Z][a-zA-Z0-9_]*\.ts)',         # TypeScript文件
        ]
        
        # 方法名模式
        self.method_patterns = [
            r'([a-z][a-zA-Z0-9_]*\()',           # 方法调用
            r'([a-z][a-zA-Z0-9_]*\s*:)',         # 方法定义
        ]
        
        # 时间戳模式
        self.timestamp_patterns = [
            r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})',
            r'(\d{2}:\d{2}:\d{2})',
            r'(\d{4}/\d{2}/\d{2})',
        ]
    
    def extract_advanced_features(self, text: str) -> dict:
        """提取高级特征"""
        features = {}
        
        # 提取错误码
        error_codes = []
        for pattern in self.error_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                for match in matches:
                    if isinstance(match, tuple):
                        error_codes.extend([m for m in match if m and len(m) > 2])
                    else:
                        if match and len(match) > 2:
                            error_codes.append(match)
        features['error_codes'] = ' '.join(set(error_codes)) if error_codes else ''
        
        # 提取路径
        paths = []
        for pattern in self.path_patterns:
            matches = re.findall(pattern, text)
            if matches:
                for match in matches:
                    if isinstance(match, tuple):
                        paths.extend([m for m in match if m and len(m) > 3])
                    else:
                        if match and len(match) > 3:
                            paths.append(match)
        features['paths'] = ' '.join(set(paths)) if paths else ''
        
        # 提取数字
        numbers = []
        for pattern in self.number_patterns:
            matches = re.findall(pattern, text)
            if matches:
                for match in matches:
                    if isinstance(match, tuple):
                        numbers.extend([m for m in match if m])
                    else:
                        numbers.append(match)
        features['numbers'] = ' '.join(set(numbers)) if numbers else ''
        
        # 提取类名
        classes = []
        for pattern in self.class_patterns:
            matches = re.findall(pattern, text)
            if matches:
                for match in matches:
                    if isinstance(match, tuple):
                        classes.extend([m for m in match if m])
                    else:
                        classes.append(match)
        features['classes'] = ' '.join(set(classes)) if classes else ''
        
        # 提取方法名
        methods = []
        for pattern in self.method_patterns:
            matches = re.findall(pattern, text)
            if matches:
                for match in matches:
                    if isinstance(match, tuple):
                        methods.extend([m for m in match if m])
                    else:
                        methods.append(match)
        features['methods'] = ' '.join(set(methods)) if methods else ''
        
        # 提取时间戳
        timestamps = []
        for pattern in self.timestamp_patterns:
            matches = re.findall(pattern, text)
            if matches:
                for match in matches:
                    if isinstance(match, tuple):
                        timestamps.extend([m for m in match if m])
                    else:
                        timestamps.append(match)
        features['timestamps'] = ' '.join(set(timestamps)) if timestamps else ''
        
        return features


class SmartDataBalancer:
    """智能数据平衡器"""
    
    def __init__(self, target_samples_per_class=500, min_samples_per_class=10):
        self.target_samples_per_class = target_samples_per_class
        self.min_samples_per_class = min_samples_per_class
    
    def balance_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """智能平衡数据"""
        logger.info(f"🎯 开始智能数据平衡，目标每类样本数: {self.target_samples_per_class}")
        
        balanced_dfs = []
        category_counts = df['category'].value_counts()
        
        for category in category_counts.index:
            category_df = df[df['category'] == category].copy()
            current_count = len(category_df)
            
            logger.info(f"📊 处理类别: {category} (当前: {current_count} 条)")
            
            if current_count < self.min_samples_per_class:
                logger.warning(f"  ⚠️ 类别 {category} 样本数过少 ({current_count} < {self.min_samples_per_class})，跳过")
                continue
            
            if current_count < self.target_samples_per_class:
                # 智能上采样
                balanced_df = self._smart_oversample(category_df, self.target_samples_per_class)
                logger.info(f"  ✅ 智能上采样到 {len(balanced_df)} 条")
            elif current_count > self.target_samples_per_class:
                # 智能下采样
                balanced_df = self._smart_undersample(category_df, self.target_samples_per_class)
                logger.info(f"  ✅ 智能下采样到 {len(balanced_df)} 条")
            else:
                # 保持原样
                balanced_df = category_df
                logger.info(f"  ✅ 保持 {current_count} 条")
            
            balanced_dfs.append(balanced_df)
        
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
    
    def _smart_oversample(self, df: pd.DataFrame, target_count: int) -> pd.DataFrame:
        """智能上采样"""
        if len(df) == 0:
            return df
        
        # 计算需要重复的次数
        repeat_times = target_count // len(df)
        remainder = target_count % len(df)
        
        # 重复采样
        repeated_samples = []
        for _ in range(repeat_times):
            repeated_samples.append(df)
        
        # 添加剩余样本
        if remainder > 0:
            remainder_samples = df.sample(n=remainder, random_state=42)
            repeated_samples.append(remainder_samples)
        
        # 合并并打乱
        oversampled = pd.concat(repeated_samples, ignore_index=True)
        oversampled = oversampled.sample(frac=1, random_state=42).reset_index(drop=True)
        
        return oversampled
    
    def _smart_undersample(self, df: pd.DataFrame, target_count: int) -> pd.DataFrame:
        """智能下采样"""
        if len(df) <= target_count:
            return df
        
        # 分层采样
        undersampled = df.sample(n=target_count, random_state=42)
        return undersampled


class AdvancedDataQualityEnhancer:
    """高级数据质量增强器"""
    
    def __init__(self):
        self.extractor = AdvancedFeatureExtractor()
        self.balancer = SmartDataBalancer()
    
    def enhance_data_quality(self, df: pd.DataFrame, balance_data: bool = True) -> pd.DataFrame:
        """增强数据质量"""
        logger.info("🚀 开始高级数据质量增强")
        logger.info(f"📊 原始数据: {len(df)} 条记录")
        
        # 1. 数据清洗
        logger.info("🧹 步骤1: 数据清洗")
        df_cleaned = self._clean_data(df)
        
        # 2. 高级特征工程
        logger.info("🔧 步骤2: 高级特征工程")
        df_enhanced = self._extract_advanced_features(df_cleaned)
        
        # 3. 数据平衡
        if balance_data:
            logger.info("⚖️ 步骤3: 智能数据平衡")
            df_balanced = self.balancer.balance_data(df_enhanced)
        else:
            df_balanced = df_enhanced
        
        # 4. 质量评估
        logger.info("📈 步骤4: 质量评估")
        self._assess_quality(df_balanced)
        
        return df_balanced
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """清洗数据"""
        # 移除空值
        df_cleaned = df.dropna(subset=['original_log', 'category'])
        
        # 清洗日志内容
        df_cleaned['cleaned_log'] = df_cleaned['original_log'].apply(self._clean_log_content)
        
        # 提取日志级别
        df_cleaned['log_level'] = df_cleaned['cleaned_log'].apply(self._extract_log_level)
        
        # 过滤无效日志
        df_cleaned = df_cleaned[df_cleaned['cleaned_log'].apply(self._is_valid_log)]
        
        logger.info(f"✅ 清洗后数据: {len(df_cleaned)} 条记录")
        
        return df_cleaned
    
    def _clean_log_content(self, text: str) -> str:
        """清洗日志内容"""
        if pd.isna(text) or text == '':
            return ''
        
        text = str(text)
        
        # 移除元数据
        metadata_patterns = [
            r'github\.com/[^\s,]+',
            r'https://github\.com/[^\s,]+',
            r'github_issue',
            r'unknown,github_issue',
            r'[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+,\d+',
            r'https://github\.com/[^\s,]+/issues/\d+',
            r'[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]+',
            r'[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}',
            r'<[^>]+>',
            r'&[a-zA-Z]+;',
            r'\[.*?\]',
            r'^,+|,+$',
            r'^"+|"+$',
            r'unknown,,,',
            r',unknown,,,',
            r',unknown,',
            r'unknown,',
            r'https://',
        ]
        
        for pattern in metadata_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # 标准化
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # 移除以逗号开头的部分
        if text.startswith(','):
            text = text.lstrip(',')
        
        # 移除以unknown开头的部分
        if text.lower().startswith('unknown'):
            text = re.sub(r'^unknown[,\s]*', '', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def _extract_log_level(self, text: str) -> str:
        """提取日志级别"""
        log_levels = ['DEBUG', 'INFO', 'WARN', 'WARNING', 'ERROR', 'FATAL', 'TRACE']
        text_upper = text.upper()
        for level in log_levels:
            if level in text_upper:
                return level
        return 'UNKNOWN'
    
    def _is_valid_log(self, text: str) -> bool:
        """判断是否为有效日志"""
        if not text or len(text.strip()) < 10:
            return False
        
        log_indicators = ['error', 'warn', 'info', 'debug', 'exception', 'failed', 'success']
        text_lower = text.lower()
        
        return any(indicator in text_lower for indicator in log_indicators)
    
    def _extract_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """提取高级特征"""
        logger.info("🔍 提取高级结构化特征...")
        
        # 提取特征
        features_list = []
        for idx, row in df.iterrows():
            features = self.extractor.extract_advanced_features(row['cleaned_log'])
            features_list.append(features)
        
        # 转换为DataFrame
        features_df = pd.DataFrame(features_list)
        
        # 合并特征
        df_enhanced = pd.concat([df.reset_index(drop=True), features_df], axis=1)
        
        # 统计特征提取结果
        logger.info("📊 高级特征提取统计:")
        for col in features_df.columns:
            non_empty = (features_df[col] != '').sum()
            percentage = (non_empty / len(features_df)) * 100
            logger.info(f"  {col}: {non_empty} 条 ({percentage:.1f}%)")
        
        return df_enhanced
    
    def _assess_quality(self, df: pd.DataFrame) -> None:
        """评估数据质量"""
        logger.info("📈 数据质量评估:")
        
        # 计算质量指标
        total_logs = len(df)
        valid_logs = len(df[df['cleaned_log'].str.len() > 10])
        logs_with_features = len(df[df['error_codes'] != ''])
        logs_with_paths = len(df[df['paths'] != ''])
        
        logger.info(f"  📊 总日志数: {total_logs}")
        logger.info(f"  ✅ 有效日志数: {valid_logs} ({valid_logs/total_logs*100:.1f}%)")
        logger.info(f"  🔍 包含错误码: {logs_with_features} ({logs_with_features/total_logs*100:.1f}%)")
        logger.info(f"  📁 包含路径: {logs_with_paths} ({logs_with_paths/total_logs*100:.1f}%)")
        
        # 类别分布
        category_counts = df['category'].value_counts()
        logger.info("📊 类别分布:")
        for category, count in category_counts.items():
            percentage = (count / total_logs) * 100
            logger.info(f"  {category}: {count} 条 ({percentage:.1f}%)")
    
    def create_advanced_enhanced_dataset(self, input_path: str, output_path: str, balance_data: bool = True) -> pd.DataFrame:
        """创建高级增强数据集"""
        logger.info(f"📂 加载数据: {input_path}")
        df = pd.read_csv(input_path)
        
        # 增强数据质量
        df_enhanced = self.enhance_data_quality(df, balance_data)
        
        # 保存结果
        df_enhanced.to_csv(output_path, index=False)
        logger.info(f"💾 保存高级增强数据: {output_path}")
        
        return df_enhanced


def main():
    """主函数"""
    enhancer = AdvancedDataQualityEnhancer()
    
    input_path = "data/processed_logs_final_cleaned.csv"
    output_path = "data/processed_logs_advanced_enhanced.csv"
    
    # 创建高级增强数据集
    df_enhanced = enhancer.create_advanced_enhanced_dataset(input_path, output_path, balance_data=True)
    
    logger.info("✅ 高级数据质量增强完成!")
    
    # 显示增强后的数据样例
    logger.info("📋 高级增强数据样例:")
    sample_cols = ['category', 'cleaned_log', 'log_level', 'error_codes', 'paths', 'methods']
    print(df_enhanced[sample_cols].head(3).to_string())


if __name__ == "__main__":
    main() 