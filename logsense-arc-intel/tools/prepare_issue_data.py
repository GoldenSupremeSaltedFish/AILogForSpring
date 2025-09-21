# -*- coding: utf-8 -*-
"""
Issue日志数据准备脚本
将清洗后的issue日志数据转换为训练格式，包括数据混淆
"""

import pandas as pd
import numpy as np
import re
import random
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IssueDataPreparer:
    """Issue日志数据准备器"""
    
    def __init__(self):
        # 定义数据混淆规则
        self.augmentation_patterns = {
            'stack_exception': [
                # 替换常见的异常类名
                (r'NullPointerException', ['NullPointerException', 'IllegalArgumentException', 'RuntimeException']),
                (r'SQLException', ['SQLException', 'DataAccessException', 'DatabaseException']),
                (r'ConnectionException', ['ConnectionException', 'ConnectException', 'SocketException']),
                # 替换常见的包名
                (r'org\.springframework\.', ['org.springframework.', 'org.hibernate.', 'org.apache.']),
                (r'java\.lang\.', ['java.lang.', 'java.util.', 'java.io.']),
                # 替换常见的错误信息
                (r'Connection refused', ['Connection refused', 'Connection timeout', 'Connection failed']),
                (r'Table.*not found', ['Table not found', 'Column not found', 'Schema not found']),
            ],
            'startup_failure': [
                (r'Failed to start', ['Failed to start', 'Unable to start', 'Cannot start']),
                (r'Port.*in use', ['Port already in use', 'Address already in use', 'Port is busy']),
                (r'BeanCreationException', ['BeanCreationException', 'ContextLoadException', 'ApplicationContextException']),
            ],
            'auth_error': [
                (r'Authentication failed', ['Authentication failed', 'Login failed', 'Auth failed']),
                (r'Access denied', ['Access denied', 'Permission denied', 'Forbidden']),
                (r'Invalid token', ['Invalid token', 'Token expired', 'Token invalid']),
            ],
            'db_error': [
                (r'SQLException', ['SQLException', 'DatabaseException', 'DataAccessException']),
                (r'Connection.*failed', ['Connection failed', 'Connection refused', 'Connection timeout']),
                (r'Duplicate entry', ['Duplicate entry', 'Constraint violation', 'Unique constraint']),
            ],
            'connection_issue': [
                (r'Connection.*timeout', ['Connection timeout', 'Connection refused', 'Connection failed']),
                (r'Network.*unreachable', ['Network unreachable', 'Host unreachable', 'No route to host']),
            ],
            'timeout': [
                (r'Request.*timeout', ['Request timeout', 'Response timeout', 'Operation timeout']),
                (r'Read.*timeout', ['Read timeout', 'Write timeout', 'Socket timeout']),
            ],
            'performance': [
                (r'OutOfMemoryError', ['OutOfMemoryError', 'MemoryError', 'Heap space']),
                (r'GC.*overhead', ['GC overhead', 'Garbage collection', 'Memory pressure']),
            ],
            'config': [
                (r'Configuration.*error', ['Configuration error', 'Config error', 'Property error']),
                (r'Property.*not found', ['Property not found', 'Environment variable not found', 'Config not found']),
            ],
            'business': [
                (r'Business.*error', ['Business error', 'Logic error', 'Service error']),
                (r'Validation.*failed', ['Validation failed', 'Invalid input', 'Data validation failed']),
            ],
            'normal': [
                (r'INFO.*Started', ['INFO: Started', 'INFO: Running', 'INFO: Application started']),
                (r'DEBUG.*', ['DEBUG: ', 'TRACE: ', 'INFO: ']),
            ]
        }
    
    def load_cleaned_data(self, data_path: str) -> pd.DataFrame:
        """加载清洗后的数据"""
        logger.info(f"📂 加载清洗数据: {data_path}")
        
        try:
            df = pd.read_csv(data_path)
            logger.info(f"✅ 成功加载 {len(df)} 条记录")
            return df
        except Exception as e:
            logger.error(f"❌ 加载数据失败: {e}")
            return pd.DataFrame()
    
    def augment_data(self, df: pd.DataFrame, augmentation_factor: float = 0.3) -> pd.DataFrame:
        """数据增强/混淆"""
        logger.info("🔄 开始数据增强...")
        
        # 检查列名并统一
        if 'label' in df.columns:
            label_col = 'label'
        elif 'auto_label' in df.columns:
            label_col = 'auto_label'
        else:
            logger.error("❌ 未找到标签列")
            return df
            
        if 'text' in df.columns:
            text_col = 'text'
        elif 'cleaned_message' in df.columns:
            text_col = 'cleaned_message'
        else:
            logger.error("❌ 未找到文本列")
            return df
        
        augmented_data = []
        
        for _, row in df.iterrows():
            # 添加原始数据
            augmented_data.append(row.to_dict())
            
            # 根据类别进行数据增强
            label = row.get(label_col, 'unknown')
            message = row.get(text_col, '')
            
            if label in self.augmentation_patterns and random.random() < augmentation_factor:
                # 对部分数据进行增强
                augmented_message = self._augment_message(message, label)
                if augmented_message != message:
                    augmented_row = row.copy()
                    augmented_row[text_col] = augmented_message
                    augmented_row['is_augmented'] = True
                    augmented_data.append(augmented_row)
        
        augmented_df = pd.DataFrame(augmented_data)
        logger.info(f"📊 数据增强完成: {len(df)} -> {len(augmented_df)} 条记录")
        
        return augmented_df
    
    def _augment_message(self, message: str, label: str) -> str:
        """对单条消息进行增强"""
        if label not in self.augmentation_patterns:
            return message
        
        augmented_message = message
        
        for pattern, replacements in self.augmentation_patterns[label]:
            if re.search(pattern, augmented_message, re.IGNORECASE):
                # 随机选择一个替换项
                replacement = random.choice(replacements)
                augmented_message = re.sub(pattern, replacement, augmented_message, flags=re.IGNORECASE)
                break  # 只替换第一个匹配的模式
        
        return augmented_message
    
    def balance_data(self, df: pd.DataFrame, max_per_class: int = 1000) -> pd.DataFrame:
        """平衡各类别数据量"""
        logger.info(f"⚖️ 开始数据平衡 (每类最多 {max_per_class} 条)...")
        
        # 检查列名并统一
        if 'label' in df.columns:
            label_col = 'label'
        elif 'auto_label' in df.columns:
            label_col = 'auto_label'
        else:
            logger.error("❌ 未找到标签列")
            return df
        
        balanced_dfs = []
        for category in df[label_col].unique():
            category_df = df[df[label_col] == category]
            if len(category_df) > max_per_class:
                # 随机采样
                category_df = category_df.sample(n=max_per_class, random_state=42)
                logger.info(f"  {category}: 采样 {len(category_df)} 条 (原始 {len(df[df[label_col] == category])} 条)")
            else:
                logger.info(f"  {category}: 使用全部 {len(category_df)} 条")
            
            balanced_dfs.append(category_df)
        
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        logger.info(f"📊 数据平衡完成: {len(balanced_df)} 条记录")
        
        return balanced_df
    
    def prepare_training_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """准备训练数据格式"""
        logger.info("📋 准备训练数据格式...")
        
        # 检查列名并统一
        if 'text' in df.columns and 'label' in df.columns:
            # 已经是标准格式
            training_df = df[['text', 'label']].copy()
        elif 'cleaned_message' in df.columns and 'auto_label' in df.columns:
            # 需要转换格式
            training_df = df[['cleaned_message', 'auto_label']].copy()
            training_df.columns = ['text', 'label']
        else:
            logger.error("❌ 未找到必要的文本和标签列")
            return df
        
        # 移除空文本
        training_df = training_df[training_df['text'].str.len() > 0]
        
        # 添加数据来源标识
        training_df['source'] = 'issue_logs'
        training_df['timestamp'] = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"✅ 训练数据准备完成: {len(training_df)} 条记录")
        
        return training_df
    
    def extract_structured_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """提取结构化特征"""
        logger.info("🔍 提取结构化特征...")
        
        # 检查文本列名
        if 'text' in df.columns:
            text_col = 'text'
        elif 'cleaned_message' in df.columns:
            text_col = 'cleaned_message'
        else:
            logger.error("❌ 未找到文本列")
            return df
        
        def extract_features(text):
            features = {}
            
            # 文本长度特征
            features['text_length'] = len(text)
            features['word_count'] = len(text.split())
            
            # 错误类型特征
            features['has_exception'] = 1 if re.search(r'Exception|Error', text, re.IGNORECASE) else 0
            features['has_stack_trace'] = 1 if re.search(r'at |Caused by:', text, re.IGNORECASE) else 0
            features['has_sql'] = 1 if re.search(r'SQL|Database|Table|Column', text, re.IGNORECASE) else 0
            features['has_connection'] = 1 if re.search(r'Connection|Socket|Network', text, re.IGNORECASE) else 0
            features['has_auth'] = 1 if re.search(r'Authentication|Authorization|Token|Login', text, re.IGNORECASE) else 0
            features['has_timeout'] = 1 if re.search(r'Timeout|timeout', text, re.IGNORECASE) else 0
            features['has_memory'] = 1 if re.search(r'Memory|OutOfMemory|GC', text, re.IGNORECASE) else 0
            features['has_config'] = 1 if re.search(r'Configuration|Property|Config', text, re.IGNORECASE) else 0
            
            # 日志级别特征
            features['has_error'] = 1 if re.search(r'ERROR|FATAL', text, re.IGNORECASE) else 0
            features['has_warn'] = 1 if re.search(r'WARN|WARNING', text, re.IGNORECASE) else 0
            features['has_info'] = 1 if re.search(r'INFO', text, re.IGNORECASE) else 0
            features['has_debug'] = 1 if re.search(r'DEBUG|TRACE', text, re.IGNORECASE) else 0
            
            # 特殊字符特征
            features['has_colon'] = 1 if ':' in text else 0
            features['has_bracket'] = 1 if re.search(r'[\(\)\[\]\{\}]', text) else 0
            features['has_dot'] = 1 if '.' in text else 0
            features['has_underscore'] = 1 if '_' in text else 0
            
            return features
        
        # 提取特征
        feature_dfs = []
        for _, row in df.iterrows():
            features = extract_features(row[text_col])
            feature_df = pd.DataFrame([features])
            feature_dfs.append(feature_df)
        
        features_df = pd.concat(feature_dfs, ignore_index=True)
        
        # 合并特征到原始数据
        result_df = pd.concat([df, features_df], axis=1)
        
        logger.info(f"✅ 结构化特征提取完成: {len(features_df.columns)} 个特征")
        
        return result_df
    
    def save_processed_data(self, df: pd.DataFrame, output_path: str):
        """保存处理后的数据"""
        logger.info(f"💾 保存处理后的数据到: {output_path}")
        
        # 创建输出目录
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存数据
        df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"✅ 数据保存成功: {len(df)} 条记录")
        
        # 显示数据统计
        if 'label' in df.columns:
            label_counts = df['label'].value_counts()
            logger.info("📈 类别分布:")
            for label, count in label_counts.items():
                percentage = (count / len(df)) * 100
                logger.info(f"  {label}: {count} 条 ({percentage:.1f}%)")

def main():
    """主函数"""
    logger.info("🚀 Issue日志数据准备开始...")
    
    # 初始化数据准备器
    preparer = IssueDataPreparer()
    
    # 加载清洗后的数据
    input_file = "../DATA_OUTPUT/issue_logs_training_20250812_001907.csv"
    df = preparer.load_cleaned_data(input_file)
    
    if df.empty:
        logger.error("❌ 没有有效数据，退出")
        return
    
    # 数据增强
    df_augmented = preparer.augment_data(df, augmentation_factor=0.3)
    
    # 数据平衡
    df_balanced = preparer.balance_data(df_augmented, max_per_class=1000)
    
    # 提取结构化特征
    df_with_features = preparer.extract_structured_features(df_balanced)
    
    # 准备训练数据
    training_df = preparer.prepare_training_data(df_with_features)
    
    # 保存处理后的数据
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"data/issue_logs_processed_{timestamp}.csv"
    preparer.save_processed_data(training_df, output_file)
    
    # 同时保存一个标准命名的文件
    standard_output = "data/processed_logs_issue_enhanced.csv"
    preparer.save_processed_data(training_df, standard_output)
    
    logger.info("🎉 Issue日志数据准备完成！")
    logger.info(f"📁 输出文件:")
    logger.info(f"  - {output_file}")
    logger.info(f"  - {standard_output}")

if __name__ == "__main__":
    main()
