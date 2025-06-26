"""
日志预处理模块
负责日志数据的清洗、规范化和特征提取
"""
import pandas as pd
import numpy as np
import re
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from utils import setup_logging, load_data, save_data, clean_text, extract_log_features

class LogPreprocessor:
    """日志预处理器"""
    
    def __init__(self, log_level: str = "INFO"):
        self.logger = setup_logging(log_level)
        self.log_patterns = {
            'timestamp': r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}',
            'ip_address': r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
            'url': r'https?://[^\s]+',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'error_code': r'[Ee]rror\s*[:\-]?\s*(\d+)',
            'session_id': r'session[_\-]?id[:\s]*([a-zA-Z0-9]+)',
            'user_id': r'user[_\-]?id[:\s]*([a-zA-Z0-9]+)'
        }
    
    def load_logs(self, file_path: str) -> pd.DataFrame:
        """加载日志数据"""
        try:
            df = load_data(file_path)
            self.logger.info(f"成功加载 {len(df)} 条日志记录")
            return df
        except Exception as e:
            self.logger.error(f"加载日志失败: {e}")
            raise
    
    def clean_log_message(self, message: str) -> str:
        """清洗单条日志消息"""
        if pd.isna(message):
            return ""
            
        # 基础文本清洗
        cleaned = clean_text(message)
        
        # 替换敏感信息
        cleaned = self._mask_sensitive_info(cleaned)
        
        # 标准化常见错误信息
        cleaned = self._standardize_error_messages(cleaned)
        
        return cleaned
    
    def _mask_sensitive_info(self, text: str) -> str:
        """遮蔽敏感信息"""
        # 遮蔽IP地址
        text = re.sub(self.log_patterns['ip_address'], '[IP_MASKED]', text)
        
        # 遮蔽邮箱
        text = re.sub(self.log_patterns['email'], '[EMAIL_MASKED]', text)
        
        # 遮蔽用户ID（保留格式）
        text = re.sub(r'user[_\-]?id[:\s]*([a-zA-Z0-9]+)', 'user_id:[USER_ID]', text, flags=re.IGNORECASE)
        
        # 遮蔽会话ID
        text = re.sub(r'session[_\-]?id[:\s]*([a-zA-Z0-9]+)', 'session_id:[SESSION_ID]', text, flags=re.IGNORECASE)
        
        return text
    
    def _standardize_error_messages(self, text: str) -> str:
        """标准化错误消息"""
        # 标准化数据库错误
        text = re.sub(r'connection\s+(timeout|failed|refused)', 'connection_error', text, flags=re.IGNORECASE)
        
        # 标准化内存错误
        text = re.sub(r'out\s+of\s+memory|memory\s+error', 'memory_error', text, flags=re.IGNORECASE)
        
        # 标准化网络错误
        text = re.sub(r'network\s+(timeout|error|unreachable)', 'network_error', text, flags=re.IGNORECASE)
        
        return text
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """提取日志特征"""
        self.logger.info("开始提取日志特征...")
        
        # 清洗消息内容
        df['cleaned_message'] = df['message'].apply(self.clean_log_message)
        
        # 基础特征
        feature_dicts = df['cleaned_message'].apply(extract_log_features)
        feature_df = pd.DataFrame(list(feature_dicts))
        
        # 日志级别编码
        level_mapping = {
            'DEBUG': 0, 'INFO': 1, 'WARN': 2, 
            'WARNING': 2, 'ERROR': 3, 'FATAL': 4, 'CRITICAL': 4
        }
        df['level_encoded'] = df['level'].map(level_mapping).fillna(1)
        
        # 时间特征（如果有时间戳）
        if 'timestamp' in df.columns:
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        
        # 来源编码
        if 'source' in df.columns:
            source_counts = df['source'].value_counts()
            df['source_frequency'] = df['source'].map(source_counts)
        
        # 合并特征
        result_df = pd.concat([df, feature_df], axis=1)
        
        self.logger.info(f"特征提取完成，共 {len(result_df.columns)} 个特征")
        return result_df
    
    def categorize_logs(self, df: pd.DataFrame) -> pd.DataFrame:
        """基于规则的日志分类"""
        def get_category(row):
            message = row['cleaned_message'].lower()
            level = row['level'].upper()
            source = row.get('source', '').lower()
            
            # 基于关键词的分类规则
            if any(keyword in message for keyword in ['login', 'logout', 'auth', 'authentication']):
                return 'auth'
            elif any(keyword in message for keyword in ['database', 'db', 'sql', 'connection']):
                return 'database'
            elif any(keyword in message for keyword in ['memory', 'cpu', 'disk', 'system']):
                return 'system'
            elif any(keyword in message for keyword in ['api', 'request', 'response', 'endpoint']):
                return 'api'
            elif any(keyword in message for keyword in ['cache', 'redis', 'memcached']):
                return 'cache'
            elif any(keyword in message for keyword in ['payment', 'order', 'transaction']):
                return 'payment'
            elif level in ['ERROR', 'FATAL', 'CRITICAL']:
                return 'error'
            elif level in ['WARN', 'WARNING']:
                return 'warning'
            else:
                return 'info'
        
        df['predicted_category'] = df.apply(get_category, axis=1)
        return df
    
    def process_logs(self, input_file: str, output_file: str = None) -> pd.DataFrame:
        """完整的日志处理流程"""
        self.logger.info("开始日志预处理流程...")
        
        # 加载数据
        df = self.load_logs(input_file)
        
        # 提取特征
        df = self.extract_features(df)
        
        # 基础分类
        df = self.categorize_logs(df)
        
        # 保存处理后的数据
        if output_file:
            save_data(df, output_file)
            self.logger.info(f"处理后的数据已保存到: {output_file}")
        
        self.logger.info("日志预处理完成!")
        return df
    
    def get_statistics(self, df: pd.DataFrame) -> Dict:
        """获取数据统计信息"""
        stats = {
            'total_logs': len(df),
            'log_levels': df['level'].value_counts().to_dict(),
            'sources': df.get('source', pd.Series()).value_counts().to_dict(),
            'categories': df.get('predicted_category', pd.Series()).value_counts().to_dict(),
            'avg_message_length': df['cleaned_message'].str.len().mean(),
            'missing_values': df.isnull().sum().to_dict()
        }
        return stats

if __name__ == "__main__":
    # 示例用法
    preprocessor = LogPreprocessor()
    
    # 处理示例数据
    processed_df = preprocessor.process_logs(
        input_file="data/logs.csv",
        output_file="data/processed_logs.csv"
    )
    
    # 显示统计信息
    stats = preprocessor.get_statistics(processed_df)
    print("数据统计信息:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print(f"\n处理后的数据预览:")
    print(processed_df.head()) 