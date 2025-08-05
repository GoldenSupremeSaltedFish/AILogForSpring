#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intel Arc GPU 训练器 - 重构版本
"""

import torch
import torch.nn as nn
import logging
import os
from datetime import datetime
from typing import Dict, Any

# 导入模块
from arc_models import ModelFactory
from arc_data import DataLoaderFactory
from arc_utils import ArcGPUDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArcTrainerRefactored:
    """重构后的训练器"""
    
    def __init__(self, model_type: str = "textcnn"):
        self.model_type = model_type
        self.device = ArcGPUDetector.get_device()
        self.model = None
        self.label_encoder = None
        
        logger.info(f"🎯 初始化训练器 - 模型类型: {model_type}")
        logger.info(f"🖥️  计算设备: {self.device}")
    
    def load_data(self, data_path: str):
        """加载数据"""
        logger.info(f"📂 加载数据: {data_path}")
        
        # 加载和分割数据
        texts, labels, label_encoder = DataLoaderFactory.load_csv_data(data_path)
        train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = \
            DataLoaderFactory.split_data(texts, labels)
        
        # 创建数据加载器
        train_loader, val_loader, test_loader = DataLoaderFactory.create_data_loaders(
            train_texts, val_texts, test_texts,
            train_labels, val_labels, test_labels
        )
        
        self.label_encoder = label_encoder
        logger.info(f"📊 数据加载完成 - 类别数: {len(label_encoder)}")
        
        return train_loader, val_loader, test_loader
    
    def create_model(self, num_classes: int):
        """创建模型"""
        logger.info(f"🏗️ 创建模型: {self.model_type}")
        
        model_config = ModelFactory.get_default_config(self.model_type)
        model_config['num_classes'] = num_classes
        
        self.model = ModelFactory.create_model(self.model_type, **model_config)
        self.model.to(self.device)
        
        param_count = self.model.count_parameters()
        logger.info(f"✅ 模型创建成功 - 参数数量: {param_count:,}")
    
    def train_epoch(self, train_loader, criterion, optimizer):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(input_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        return total_loss / len(train_loader), 100 * correct / total
    
    def validate_epoch(self, val_loader, criterion):
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return total_loss / len(val_loader), 100 * correct / total
    
    def train(self, data_path: str, epochs: int = 10):
        """完整训练流程"""
        logger.info("🚀 开始训练")
        
        # 加载数据
        train_loader, val_loader, test_loader = self.load_data(data_path)
        
        # 创建模型
        num_classes = len(self.label_encoder)
        self.create_model(num_classes)
        
        # 设置训练组件
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        best_acc = 0
        
        for epoch in range(epochs):
            logger.info(f"📈 Epoch {epoch+1}/{epochs}")
            
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            
            # 验证
            val_loss, val_acc = self.validate_epoch(val_loader, criterion)
            
            logger.info(f"  训练: 损失={train_loss:.4f}, 准确率={train_acc:.2f}%")
            logger.info(f"  验证: 损失={val_loss:.4f}, 准确率={val_acc:.2f}%")
            
            # 保存最佳模型
            if val_acc > best_acc:
                best_acc = val_acc
                self.save_model("best")
                logger.info(f"💾 保存最佳模型 (准确率: {val_acc:.2f}%)")
        
        # 保存最终模型
        self.save_model("final")
        
        # 测试集评估
        test_loss, test_acc = self.validate_epoch(test_loader, criterion)
        logger.info(f"🧪 测试集: 损失={test_loss:.4f}, 准确率={test_acc:.2f}%")
        
        logger.info(f"✅ 训练完成 - 最佳准确率: {best_acc:.2f}%")
    
    def save_model(self, suffix: str = ""):
        """保存模型"""
        os.makedirs("results/models", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_path = f"results/models/arc_gpu_model_{self.model_type}_{suffix}_{timestamp}.pth"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': self.model.get_config(),
            'label_encoder': self.label_encoder
        }, model_path)
        
        logger.info(f"💾 模型已保存: {model_path}")


def main():
    """主函数"""
    logger.info("🎯 Intel Arc GPU 训练器 - 重构版本")
    
    if not ArcGPUDetector.check_arc_gpu():
        logger.warning("⚠️ 未检测到Intel Arc GPU，将使用CPU训练")
    
    trainer = ArcTrainerRefactored(model_type="textcnn")
    trainer.train("DATA_OUTPUT/processed_logs.csv", epochs=10)


if __name__ == "__main__":
    main() 