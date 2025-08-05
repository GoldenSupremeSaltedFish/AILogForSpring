#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型工厂类
用于创建不同类型的模型
"""

from typing import Dict, Any
from .textcnn import TextCNN
from .fasttext import FastTextModel


class ModelFactory:
    """模型工厂类"""
    
    @staticmethod
    def create_model(model_type: str, **kwargs) -> Any:
        """
        创建模型实例
        Args:
            model_type: 模型类型 ('textcnn', 'fasttext')
            **kwargs: 模型参数
        Returns:
            模型实例
        """
        if model_type.lower() == 'textcnn':
            return TextCNN(**kwargs)
        elif model_type.lower() == 'fasttext':
            return FastTextModel(**kwargs)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
    
    @staticmethod
    def get_default_config(model_type: str) -> Dict[str, Any]:
        """
        获取默认配置
        Args:
            model_type: 模型类型
        Returns:
            默认配置字典
        """
        if model_type.lower() == 'textcnn':
            return {
                'vocab_size': 10000,
                'embed_dim': 128,
                'num_classes': 10,
                'filter_sizes': [3, 4, 5],
                'num_filters': 128,
                'dropout': 0.5
            }
        elif model_type.lower() == 'fasttext':
            return {
                'vocab_size': 10000,
                'embed_dim': 128,
                'num_classes': 10,
                'dropout': 0.3
            }
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
    
    @staticmethod
    def get_supported_models() -> list:
        """获取支持的模型类型列表"""
        return ['textcnn', 'fasttext'] 