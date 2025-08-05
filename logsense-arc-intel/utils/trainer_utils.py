#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练工具模块
"""

import torch
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class TrainerUtils:
    """训练工具类"""
    
    @staticmethod
    def calculate_accuracy(outputs, labels):
        """计算准确率"""
        _, predicted = torch.max(outputs.data, 1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()
        return 100 * correct / total
    
    @staticmethod
    def save_checkpoint(model, optimizer, epoch, loss, save_path):
        """保存检查点"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, save_path)
        logger.info(f"检查点已保存: {save_path}")
    
    @staticmethod
    def load_checkpoint(model, optimizer, checkpoint_path):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        logger.info(f"检查点已加载: {checkpoint_path}")
        return epoch, loss 