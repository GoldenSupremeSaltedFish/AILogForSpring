#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intel Arc GPU 优化训练脚本
针对内存使用进行优化，避免OOM
"""

import argparse
import torch
import torch.nn as nn
import logging
import os
import gc
from datetime import datetime
from typing import Dict, Any

# 导入自定义模块
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import ModelFactory
from data import DataLoaderFactory, LogPreprocessor
from utils import ArcGPUDetector

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OptimizedArcTrainer:
    """优化的Intel Arc GPU 训练器"""
    
    def __init__(self, model_type: str = "textcnn", batch_size: int = 16):
        self.model_type = model_type
        self.device = ArcGPUDetector.get_device()
        self.model = None
        self.label_encoder = None
        self.batch_size = batch_size
        
        logger.info(f"🎯 初始化优化训练器 - 模型类型: {model_type}")
        logger.info(f"🖥️  计算设备: {self.device}")
        logger.info(f"📦 批次大小: {batch_size}")
    
    def load_data(self, data_path: str):
        """加载数据"""
        logger.info(f"📂 加载数据: {data_path}")
        
        # 加载和分割数据
        texts, labels, label_encoder = DataLoaderFactory.load_csv_data(data_path)
        train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = \
            DataLoaderFactory.split_data(texts, labels)
        
        # 创建数据加载器（使用较小的批次大小）
        train_loader, val_loader, test_loader = DataLoaderFactory.create_data_loaders(
            train_texts, val_texts, test_texts,
            train_labels, val_labels, test_labels,
            batch_size=self.batch_size
        )
        
        self.label_encoder = label_encoder
        logger.info(f"📊 数据加载完成 - 类别数: {len(label_encoder)}")
        
        return train_loader, val_loader, test_loader
    
    def create_model(self, num_classes: int):
        """创建模型"""
        logger.info(f"🏗️ 创建模型: {self.model_type}")
        
        # 使用较小的模型配置以避免OOM
        model_config = ModelFactory.get_default_config(self.model_type)
        model_config['num_classes'] = num_classes
        
        # 针对Intel Arc GPU优化配置
        if self.model_type == "textcnn":
            model_config.update({
                'embed_dim': 64,  # 减小嵌入维度
                'num_filters': 64,  # 减小卷积核数量
                'dropout': 0.3
            })
        elif self.model_type == "fasttext":
            model_config.update({
                'embed_dim': 64,  # 减小嵌入维度
                'dropout': 0.2
            })
        
        self.model = ModelFactory.create_model(self.model_type, **model_config)
        self.model.to(self.device)
        
        param_count = self.model.count_parameters()
        logger.info(f"✅ 模型创建成功 - 参数数量: {param_count:,}")
        logger.info(f"📋 模型配置: {self.model.get_config()}")
    
    def train_epoch(self, train_loader, criterion, optimizer):
        """训练一个epoch（内存优化版本）"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # 清理GPU内存
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                if hasattr(torch, 'xpu') and torch.xpu.is_available():
                    torch.xpu.empty_cache()
            
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(input_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # 梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 清理中间变量
            del outputs, loss
            if batch_idx % 50 == 0:
                logger.info(f"   批次 {batch_idx}/{len(train_loader)} - 损失: {total_loss/(batch_idx+1):.4f}")
        
        return total_loss / len(train_loader), 100 * correct / total
    
    def validate_epoch(self, val_loader, criterion):
        """验证一个epoch（内存优化版本）"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                # 清理GPU内存
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    if hasattr(torch, 'xpu') and torch.xpu.is_available():
                        torch.xpu.empty_cache()
                
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # 清理中间变量
                del outputs, loss
        
        return total_loss / len(val_loader), 100 * correct / total
    
    def train(self, data_path: str, epochs: int = 10, save_dir: str = "results/models"):
        """完整训练流程（内存优化版本）"""
        logger.info("🚀 开始优化训练")
        
        # 检查GPU状态
        ArcGPUDetector.print_gpu_status()
        
        # 加载数据
        train_loader, val_loader, test_loader = self.load_data(data_path)
        
        # 创建模型
        num_classes = len(self.label_encoder)
        self.create_model(num_classes)
        
        # 设置训练组件
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        
        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=2, verbose=True
        )
        
        best_acc = 0
        patience_counter = 0
        max_patience = 5
        
        for epoch in range(epochs):
            logger.info(f"📈 Epoch {epoch+1}/{epochs}")
            
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            
            # 验证
            val_loss, val_acc = self.validate_epoch(val_loader, criterion)
            
            # 更新学习率
            scheduler.step(val_acc)
            
            logger.info(f"  训练: 损失={train_loss:.4f}, 准确率={train_acc:.2f}%")
            logger.info(f"  验证: 损失={val_loss:.4f}, 准确率={val_acc:.2f}%")
            
            # 保存最佳模型
            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
                self.save_model(save_dir, "best")
                logger.info(f"💾 保存最佳模型 (准确率: {val_acc:.2f}%)")
            else:
                patience_counter += 1
                logger.info(f"⏳ 耐心计数: {patience_counter}/{max_patience}")
            
            # 早停机制
            if patience_counter >= max_patience:
                logger.info(f"🛑 早停触发，最佳准确率: {best_acc:.2f}%")
                break
            
            # 强制垃圾回收
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, 'xpu') and torch.xpu.is_available():
                torch.xpu.empty_cache()
        
        # 保存最终模型
        self.save_model(save_dir, "final")
        
        # 测试集评估
        test_loss, test_acc = self.validate_epoch(test_loader, criterion)
        logger.info(f"🧪 测试集: 损失={test_loss:.4f}, 准确率={test_acc:.2f}%")
        
        logger.info(f"✅ 训练完成 - 最佳准确率: {best_acc:.2f}%")
    
    def save_model(self, save_dir: str, suffix: str = ""):
        """保存模型"""
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_path = os.path.join(save_dir, f"arc_gpu_optimized_{self.model_type}_{suffix}_{timestamp}.pth")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': self.model.get_config(),
            'label_encoder': self.label_encoder,
            'training_config': {
                'model_type': self.model_type,
                'batch_size': self.batch_size,
                'device': str(self.device)
            }
        }, model_path)
        
        logger.info(f"💾 模型已保存: {model_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Intel Arc GPU 优化训练器")
    parser.add_argument("--model", type=str, default="textcnn", 
                       choices=["textcnn", "fasttext"], help="模型类型")
    parser.add_argument("--data", type=str, required=True, help="数据文件路径")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument("--save_dir", type=str, default="results/models", help="模型保存目录")
    
    args = parser.parse_args()
    
    logger.info("🎯 Intel Arc GPU 优化训练器")
    logger.info(f"📦 批次大小: {args.batch_size}")
    
    if not ArcGPUDetector.check_arc_gpu():
        logger.warning("⚠️ 未检测到Intel Arc GPU，将使用CPU训练")
    
    trainer = OptimizedArcTrainer(model_type=args.model, batch_size=args.batch_size)
    trainer.train(args.data, args.epochs, args.save_dir)


if __name__ == "__main__":
    main() 