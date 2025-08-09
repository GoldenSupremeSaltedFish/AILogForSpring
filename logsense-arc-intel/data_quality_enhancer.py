#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据质量增强器 - 专注于日志数据清洗和特征工程
"""

import pandas as pd
import numpy as np
import logging
import re
import json
from collections import Counter
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import os

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LogFeatureExtractor:
    """日志特征提取器"""
    
    def __init__(self):
        # 错误码模式
        self.error_patterns = [
            r'error\s*[:\s]*([A-Z0-9_]+)',  # ERROR: CODE123
            r'err\s*[:\s]*([A-Z0-9_]+)',    # ERR: CODE123
            r'([A-Z]{2,10}\d{3,6})',        # 通用错误码
            r'([A-Z]{2,10}Exception)',      # Java异常
            r'([A-Z][a-z]+Exception)',      # 异常类名
        ]
        
        # 路径模式
        self.path_patterns = [
            r'([A-Za-z]:\\[^\s]+)',         # Windows路径
            r'(/[^\s]+)',                    # Unix路径
            r'([A-Za-z0-9_/.-]+\.(java|py|js|ts|go|cpp|c|h))',  # 文件路径
        ]
        
        # 时间戳模式
        self.timestamp_patterns = [
            r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})',
            r'(\d{2}:\d{2}:\d{2})',
            r'(\d{4}/\d{2}/\d{2})',
        ]
        
        # 数字模式
        self.number_patterns = [
            r'(\d+\.\d+)',                   # 浮点数
            r'(\d+)',                        # 整数
        ]
        
        # 类名模式
        self.class_patterns = [
            r'([A-Z][a-zA-Z0-9_]*\.java)',  # Java类文件
            r'([A-Z][a-zA-Z0-9_]*\.py)',    # Python文件
            r'([A-Z][a-zA-Z0-9_]*\.js)',    # JavaScript文件
            r'([A-Z][a-zA-Z0-9_]*\.ts)',    # TypeScript文件
        ]
        
        # 方法名模式
        self.method_patterns = [
            r'([a-z][a-zA-Z0-9_]*\()',      # 方法调用
            r'([a-z][a-zA-Z0-9_]*\s*:)',    # 方法定义
        ]
    
    def extract_structural_features(self, text: str) -> Dict[str, str]:
        """提取结构化特征"""
        features = {}
        
        # 提取错误码
        error_codes = []
        for pattern in self.error_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            error_codes.extend(matches)
        features['error_codes'] = ' '.join(set(error_codes)) if error_codes else ''
        
        # 提取路径
        paths = []
        for pattern in self.path_patterns:
            matches = re.findall(pattern, text)
            paths.extend(matches)
        features['paths'] = ' '.join(set(paths)) if paths else ''
        
        # 提取时间戳
        timestamps = []
        for pattern in self.timestamp_patterns:
            matches = re.findall(pattern, text)
            timestamps.extend(matches)
        features['timestamps'] = ' '.join(set(timestamps)) if timestamps else ''
        
        # 提取数字
        numbers = []
        for pattern in self.number_patterns:
            matches = re.findall(pattern, text)
            numbers.extend(matches)
        features['numbers'] = ' '.join(set(numbers)) if numbers else ''
        
        # 提取类名
        classes = []
        for pattern in self.class_patterns:
            matches = re.findall(pattern, text)
            classes.extend(matches)
        features['classes'] = ' '.join(set(classes)) if classes else ''
        
        # 提取方法名
        methods = []
        for pattern in self.method_patterns:
            matches = re.findall(pattern, text)
            methods.extend(matches)
        features['methods'] = ' '.join(set(methods)) if methods else ''
        
        return features


class AdvancedLogCleaner:
    """高级日志清洗器"""
    
    def __init__(self):
        # 需要移除的元数据模式
        self.metadata_patterns = [
            # GitHub相关
            r'github\.com/[^\s,]+',
            r'https://github\.com/[^\s,]+',
            r'github_issue',
            r'unknown,github_issue',
            r'[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+,\d+',
            r'https://github\.com/[^\s,]+/issues/\d+',
            
            # 时间戳
            r'[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]+',
            r'[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}',
            
            # HTML标签
            r'<[^>]+>',
            r'&[a-zA-Z]+;',
            r'\[.*?\]',
            
            # 多余字符
            r'^,+|,+$',
            r'^"+|"+$',
            r'unknown,,,',
            r',unknown,,,',
            r',unknown,',
            r'unknown,',
            r'https://',
        ]
        
        # 需要标准化的模式
        self.normalization_patterns = [
            (r'\s+', ' '),  # 多个空格变单个
            (r'^\s+|\s+$', ''),  # 首尾空格
        ]
        
        # 日志级别模式
        self.log_levels = ['DEBUG', 'INFO', 'WARN', 'WARNING', 'ERROR', 'FATAL', 'TRACE']
    
    def clean_log_content(self, text: str) -> str:
        """清洗日志内容"""
        if pd.isna(text) or text == '':
            return ''
        
        text = str(text)
        
        # 移除元数据
        for pattern in self.metadata_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # 标准化
        for pattern, replacement in self.normalization_patterns:
            text = re.sub(pattern, replacement, text)
        
        # 移除以逗号开头的部分
        if text.startswith(','):
            text = text.lstrip(',')
        
        # 移除以unknown开头的部分
        if text.lower().startswith('unknown'):
            text = re.sub(r'^unknown[,\s]*', '', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def extract_log_level(self, text: str) -> str:
        """提取日志级别"""
        text_upper = text.upper()
        for level in self.log_levels:
            if level in text_upper:
                return level
        return 'UNKNOWN'
    
    def is_valid_log(self, text: str) -> bool:
        """判断是否为有效日志"""
        if not text or len(text.strip()) < 10:
            return False
        
        # 检查是否包含日志特征
        log_indicators = ['error', 'warn', 'info', 'debug', 'exception', 'failed', 'success']
        text_lower = text.lower()
        
        return any(indicator in text_lower for indicator in log_indicators)


class LogStructureAnalyzer:
    """日志结构分析器"""
    
    def __init__(self):
        self.structure_patterns = {
            'java_stack': r'at\s+([a-zA-Z0-9_.]+)\.([a-zA-Z0-9_]+)\([^)]*\)',
            'python_traceback': r'File\s+"([^"]+)"\s*,\s*line\s+(\d+)',
            'json_log': r'\{[^{}]*"[^"]*"\s*:\s*[^,{}]*\}',
            'key_value': r'([a-zA-Z0-9_]+)\s*[=:]\s*([^\s,]+)',
        }
    
    def analyze_log_structure(self, text: str) -> Dict[str, any]:
        """分析日志结构"""
        structure_info = {
            'has_java_stack': False,
            'has_python_traceback': False,
            'has_json': False,
            'has_key_value': False,
            'structure_type': 'unknown'
        }
        
        # 检查Java堆栈
        if re.search(self.structure_patterns['java_stack'], text):
            structure_info['has_java_stack'] = True
            structure_info['structure_type'] = 'java_stack'
        
        # 检查Python回溯
        elif re.search(self.structure_patterns['python_traceback'], text):
            structure_info['has_python_traceback'] = True
            structure_info['structure_type'] = 'python_traceback'
        
        # 检查JSON格式
        elif re.search(self.structure_patterns['json_log'], text):
            structure_info['has_json'] = True
            structure_info['structure_type'] = 'json'
        
        # 检查键值对
        elif re.search(self.structure_patterns['key_value'], text):
            structure_info['has_key_value'] = True
            structure_info['structure_type'] = 'key_value'
        
        return structure_info


class DataQualityEnhancer:
    """数据质量增强器"""
    
    def __init__(self):
        self.cleaner = AdvancedLogCleaner()
        self.extractor = LogFeatureExtractor()
        self.analyzer = LogStructureAnalyzer()
    
    def enhance_data_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """增强数据质量"""
        logger.info("🚀 开始数据质量增强")
        logger.info(f"📊 原始数据: {len(df)} 条记录")
        
        # 1. 数据清洗
        logger.info("🧹 步骤1: 数据清洗")
        df_cleaned = self._clean_data(df)
        
        # 2. 特征工程
        logger.info("🔧 步骤2: 特征工程")
        df_enhanced = self._extract_features(df_cleaned)
        
        # 3. 结构分析
        logger.info("📋 步骤3: 结构分析")
        df_final = self._analyze_structure(df_enhanced)
        
        # 4. 质量评估
        logger.info("📈 步骤4: 质量评估")
        self._assess_quality(df_final)
        
        return df_final
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """清洗数据"""
        # 移除空值
        df_cleaned = df.dropna(subset=['original_log', 'category'])
        
        # 清洗日志内容
        df_cleaned['cleaned_log'] = df_cleaned['original_log'].apply(self.cleaner.clean_log_content)
        
        # 提取日志级别
        df_cleaned['log_level'] = df_cleaned['cleaned_log'].apply(self.cleaner.extract_log_level)
        
        # 过滤无效日志
        df_cleaned = df_cleaned[df_cleaned['cleaned_log'].apply(self.cleaner.is_valid_log)]
        
        logger.info(f"✅ 清洗后数据: {len(df_cleaned)} 条记录")
        
        # 分析日志级别分布
        level_counts = df_cleaned['log_level'].value_counts()
        logger.info("📊 日志级别分布:")
        for level, count in level_counts.items():
            percentage = (count / len(df_cleaned)) * 100
            logger.info(f"  {level}: {count} 条 ({percentage:.1f}%)")
        
        return df_cleaned
    
    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """提取特征"""
        logger.info("🔍 提取结构化特征...")
        
        # 提取特征
        features_list = []
        for idx, row in df.iterrows():
            features = self.extractor.extract_structural_features(row['cleaned_log'])
            features_list.append(features)
        
        # 转换为DataFrame
        features_df = pd.DataFrame(features_list)
        
        # 合并特征
        df_enhanced = pd.concat([df.reset_index(drop=True), features_df], axis=1)
        
        # 统计特征提取结果
        logger.info("📊 特征提取统计:")
        for col in features_df.columns:
            non_empty = (features_df[col] != '').sum()
            percentage = (non_empty / len(features_df)) * 100
            logger.info(f"  {col}: {non_empty} 条 ({percentage:.1f}%)")
        
        return df_enhanced
    
    def _analyze_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """分析日志结构"""
        logger.info("📋 分析日志结构...")
        
        # 分析结构
        structure_list = []
        for idx, row in df.iterrows():
            structure = self.analyzer.analyze_log_structure(row['cleaned_log'])
            structure_list.append(structure)
        
        # 转换为DataFrame
        structure_df = pd.DataFrame(structure_list)
        
        # 合并结构信息
        df_final = pd.concat([df.reset_index(drop=True), structure_df], axis=1)
        
        # 统计结构类型
        structure_counts = df_final['structure_type'].value_counts()
        logger.info("📊 日志结构类型分布:")
        for struct_type, count in structure_counts.items():
            percentage = (count / len(df_final)) * 100
            logger.info(f"  {struct_type}: {count} 条 ({percentage:.1f}%)")
        
        return df_final
    
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
    
    def create_enhanced_dataset(self, input_path: str, output_path: str) -> pd.DataFrame:
        """创建增强数据集"""
        logger.info(f"📂 加载数据: {input_path}")
        df = pd.read_csv(input_path)
        
        # 增强数据质量
        df_enhanced = self.enhance_data_quality(df)
        
        # 保存结果
        df_enhanced.to_csv(output_path, index=False)
        logger.info(f"💾 保存增强数据: {output_path}")
        
        return df_enhanced


def main():
    """主函数"""
    enhancer = DataQualityEnhancer()
    
    input_path = "data/processed_logs_final_cleaned.csv"
    output_path = "data/processed_logs_quality_enhanced.csv"
    
    # 创建增强数据集
    df_enhanced = enhancer.create_enhanced_dataset(input_path, output_path)
    
    logger.info("✅ 数据质量增强完成!")
    
    # 显示增强后的数据样例
    logger.info("📋 增强数据样例:")
    sample_cols = ['category', 'cleaned_log', 'log_level', 'error_codes', 'paths', 'structure_type']
    print(df_enhanced[sample_cols].head(3).to_string())


if __name__ == "__main__":
    main() 