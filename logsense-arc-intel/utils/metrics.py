#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
指标计算模块
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """指标计算器"""
    
    @staticmethod
    def calculate_metrics(y_true, y_pred, labels=None):
        """计算各种指标"""
        # 转换为numpy数组
        if torch.is_tensor(y_true):
            y_true = y_true.cpu().numpy()
        if torch.is_tensor(y_pred):
            y_pred = y_pred.cpu().numpy()
        
        # 计算准确率
        accuracy = accuracy_score(y_true, y_pred)
        
        # 计算精确率、召回率、F1分数
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm
        }
        
        return metrics
    
    @staticmethod
    def print_metrics(metrics: Dict[str, Any]):
        """打印指标"""
        logger.info("📊 模型性能指标:")
        logger.info(f"   准确率: {metrics['accuracy']:.4f}")
        logger.info(f"   精确率: {metrics['precision']:.4f}")
        logger.info(f"   召回率: {metrics['recall']:.4f}")
        logger.info(f"   F1分数: {metrics['f1']:.4f}")
        
        # 打印混淆矩阵
        logger.info("   混淆矩阵:")
        cm = metrics['confusion_matrix']
        for i, row in enumerate(cm):
            logger.info(f"     {row}")
    
    @staticmethod
    def calculate_class_metrics(y_true, y_pred, class_names=None):
        """计算每个类别的指标"""
        if torch.is_tensor(y_true):
            y_true = y_true.cpu().numpy()
        if torch.is_tensor(y_pred):
            y_pred = y_pred.cpu().numpy()
        
        # 计算每个类别的精确率、召回率、F1分数
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        class_metrics = {}
        for i in range(len(precision)):
            class_name = class_names[i] if class_names else f"Class_{i}"
            class_metrics[class_name] = {
                'precision': precision[i],
                'recall': recall[i],
                'f1': f1[i],
                'support': support[i]
            }
        
        return class_metrics 