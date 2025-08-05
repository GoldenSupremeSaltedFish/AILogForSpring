#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据加载器工厂类
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from typing import Tuple, Dict, Any
from .dataset import LogDataset


class DataLoaderFactory:
    """数据加载器工厂类"""
    
    @staticmethod
    def load_csv_data(data_path: str, text_column: str = 'message', 
                     label_column: str = 'category') -> Tuple[list, list, Dict[int, str]]:
        """
        从CSV文件加载数据
        Args:
            data_path: 数据文件路径
            text_column: 文本列名
            label_column: 标签列名
        Returns:
            texts: 文本列表
            labels: 标签列表
            label_encoder: 标签编码器
        """
        df = pd.read_csv(data_path)
        texts = df[text_column].fillna('').tolist()
        labels = df[label_column].tolist()
        
        # 标签编码
        unique_labels = sorted(list(set(labels)))
        label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
        encoded_labels = [label_to_id[label] for label in labels]
        
        label_encoder = {idx: label for label, idx in label_to_id.items()}
        
        return texts, encoded_labels, label_encoder
    
    @staticmethod
    def split_data(texts: list, labels: list, test_size: float = 0.2, 
                   val_size: float = 0.2, random_state: int = 42) -> Tuple[list, list, list, list, list, list]:
        """
        分割数据集
        Args:
            texts: 文本列表
            labels: 标签列表
            test_size: 测试集比例
            val_size: 验证集比例
            random_state: 随机种子
        Returns:
            训练、验证、测试集的文本和标签
        """
        # 首先分割出测试集
        train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
            texts, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
        
        # 从剩余数据中分割出验证集
        val_size_adjusted = val_size / (1 - test_size)
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_val_texts, train_val_labels, test_size=val_size_adjusted, 
            random_state=random_state, stratify=train_val_labels
        )
        
        return train_texts, val_texts, test_texts, train_labels, val_labels, test_labels
    
    @staticmethod
    def create_data_loaders(train_texts: list, val_texts: list, test_texts: list,
                           train_labels: list, val_labels: list, test_labels: list,
                           batch_size: int = 32, max_length: int = 128, 
                           vocab_size: int = 10000) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        创建数据加载器
        Args:
            train_texts: 训练文本
            val_texts: 验证文本
            test_texts: 测试文本
            train_labels: 训练标签
            val_labels: 验证标签
            test_labels: 测试标签
            batch_size: 批次大小
            max_length: 最大序列长度
            vocab_size: 词汇表大小
        Returns:
            训练、验证、测试数据加载器
        """
        # 创建数据集
        train_dataset = LogDataset(train_texts, train_labels, max_length, vocab_size)
        val_dataset = LogDataset(val_texts, val_labels, max_length, vocab_size)
        test_dataset = LogDataset(test_texts, test_labels, max_length, vocab_size)
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    @staticmethod
    def get_dataset_info(train_loader: DataLoader, val_loader: DataLoader, 
                        test_loader: DataLoader = None) -> Dict[str, Any]:
        """
        获取数据集信息
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            test_loader: 测试数据加载器
        Returns:
            数据集信息字典
        """
        info = {
            'train_samples': len(train_loader.dataset),
            'val_samples': len(val_loader.dataset),
            'train_batches': len(train_loader),
            'val_batches': len(val_loader),
            'batch_size': train_loader.batch_size
        }
        
        if test_loader:
            info.update({
                'test_samples': len(test_loader.dataset),
                'test_batches': len(test_loader)
            })
        
        return info 