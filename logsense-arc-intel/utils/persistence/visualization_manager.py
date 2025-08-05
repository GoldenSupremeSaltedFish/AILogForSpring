#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化管理器 - 处理图表和可视化的生成
"""

import os
import logging
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class VisualizationManager:
    """可视化管理器"""
    
    def __init__(self, session_dir: str, timestamp: str):
        self.session_dir = session_dir
        self.timestamp = timestamp
        self.plots_dir = os.path.join(session_dir, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)
    
    def save_training_plots(self, history: Dict[str, List[float]]) -> str:
        """保存训练图表"""
        # 创建损失曲线图
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(history.get('train_loss', []), label='训练损失')
        plt.plot(history.get('val_loss', []), label='验证损失')
        plt.title('损失曲线')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.plot(history.get('train_acc', []), label='训练准确率')
        plt.plot(history.get('val_acc', []), label='验证准确率')
        plt.title('准确率曲线')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        plt.plot(history.get('val_f1', []), label='验证F1分数')
        plt.title('F1分数曲线')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        plt.plot(history.get('learning_rate', []), label='学习率')
        plt.title('学习率变化')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(self.plots_dir, f"training_plots_{self.timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 训练图表已保存: {plot_path}")
        return plot_path
    
    def save_confusion_matrix(self, y_true: List[int], y_pred: List[int], 
                            class_names: List[str] = None) -> str:
        """保存混淆矩阵"""
        from sklearn.metrics import confusion_matrix
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        
        # 绘制混淆矩阵
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('混淆矩阵')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        
        cm_path = os.path.join(self.plots_dir, f"confusion_matrix_{self.timestamp}.png")
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 混淆矩阵已保存: {cm_path}")
        return cm_path 