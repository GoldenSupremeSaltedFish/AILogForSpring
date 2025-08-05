#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日志数据预处理器
"""

import re
import pandas as pd
from typing import List, Dict, Any


class LogPreprocessor:
    """日志数据预处理器"""
    
    def __init__(self):
        # 常见日志模式
        self.log_patterns = {
            'timestamp': r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}',
            'ip_address': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
            'url': r'https?://[^\s]+',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'uuid': r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
            'hex': r'0x[0-9a-fA-F]+',
            'numbers': r'\b\d+\b'
        }
    
    def clean_text(self, text: str) -> str:
        """
        清理文本
        Args:
            text: 原始文本
        Returns:
            清理后的文本
        """
        if pd.isna(text):
            return ""
        
        text = str(text)
        
        # 移除特殊字符，保留中英文数字和常用标点
        text = re.sub(r'[^\w\s\u4e00-\u9fff.,!?;:-]', ' ', text)
        
        # 规范化空白字符
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_features(self, message: str) -> Dict[str, Any]:
        """
        从日志消息中提取特征
        Args:
            message: 日志消息
        Returns:
            特征字典
        """
        features = {
            'length': len(message),
            'word_count': len(message.split()),
            'has_numbers': bool(re.search(r'\d', message)),
            'has_ip': bool(re.search(self.log_patterns['ip_address'], message)),
            'has_url': bool(re.search(self.log_patterns['url'], message)),
            'has_email': bool(re.search(self.log_patterns['email'], message)),
            'has_uuid': bool(re.search(self.log_patterns['uuid'], message)),
            'has_hex': bool(re.search(self.log_patterns['hex'], message)),
            'has_error_keywords': bool(re.search(r'(error|failed|exception|timeout)', message.lower())),
            'has_success_keywords': bool(re.search(r'(success|complete|ok|done)', message.lower())),
            'has_warning_keywords': bool(re.search(r'(warning|warn|caution)', message.lower())),
            'has_info_keywords': bool(re.search(r'(info|information|log)', message.lower()))
        }
        
        return features
    
    def normalize_text(self, text: str) -> str:
        """
        标准化文本
        Args:
            text: 原始文本
        Returns:
            标准化后的文本
        """
        text = self.clean_text(text)
        
        # 替换常见模式为占位符
        text = re.sub(self.log_patterns['timestamp'], '[TIMESTAMP]', text)
        text = re.sub(self.log_patterns['ip_address'], '[IP]', text)
        text = re.sub(self.log_patterns['url'], '[URL]', text)
        text = re.sub(self.log_patterns['email'], '[EMAIL]', text)
        text = re.sub(self.log_patterns['uuid'], '[UUID]', text)
        text = re.sub(self.log_patterns['hex'], '[HEX]', text)
        
        return text
    
    def process_batch(self, texts: List[str], normalize: bool = True) -> List[str]:
        """
        批量处理文本
        Args:
            texts: 文本列表
            normalize: 是否标准化
        Returns:
            处理后的文本列表
        """
        processed_texts = []
        
        for text in texts:
            if normalize:
                processed_text = self.normalize_text(text)
            else:
                processed_text = self.clean_text(text)
            
            processed_texts.append(processed_text)
        
        return processed_texts
    
    def get_statistics(self, texts: List[str]) -> Dict[str, Any]:
        """
        获取文本统计信息
        Args:
            texts: 文本列表
        Returns:
            统计信息字典
        """
        if not texts:
            return {}
        
        lengths = [len(text) for text in texts]
        word_counts = [len(text.split()) for text in texts]
        
        stats = {
            'total_texts': len(texts),
            'avg_length': sum(lengths) / len(lengths),
            'max_length': max(lengths),
            'min_length': min(lengths),
            'avg_word_count': sum(word_counts) / len(word_counts),
            'max_word_count': max(word_counts),
            'min_word_count': min(word_counts)
        }
        
        return stats 