#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日志数据集实现
"""

import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any


class LogDataset(Dataset):
    """日志数据集"""
    
    def __init__(self, texts: List[str], labels: List[int], 
                 max_length: int = 128, vocab_size: int = 10000):
        """
        初始化数据集
        Args:
            texts: 日志文本列表
            labels: 标签列表
            max_length: 最大序列长度
            vocab_size: 词汇表大小
        """
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        self.vocab_size = vocab_size
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        Args:
            idx: 样本索引
        Returns:
            包含input_ids和labels的字典
        """
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # 简单的分词处理
        tokens = text.split()[:self.max_length]
        token_ids = [hash(token) % self.vocab_size for token in tokens]
        
        # 补齐到固定长度
        if len(token_ids) < self.max_length:
            token_ids += [0] * (self.max_length - len(token_ids))
        else:
            token_ids = token_ids[:self.max_length]
        
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }
    
    def get_sample(self, idx: int) -> Dict[str, Any]:
        """
        获取样本信息（用于调试）
        Args:
            idx: 样本索引
        Returns:
            样本信息字典
        """
        if idx >= len(self):
            raise IndexError(f"索引 {idx} 超出范围")
        
        return {
            'text': self.texts[idx],
            'label': self.labels[idx],
            'text_length': len(self.texts[idx].split()),
            'token_ids': self.__getitem__(idx)['input_ids'].tolist()
        } 