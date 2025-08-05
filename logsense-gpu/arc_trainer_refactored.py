#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intel Arc GPU 训练器 - 重构版本
使用模块化设计，提高代码可读性和可维护性
"""

import torch
import torch.nn as nn
import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional

# 导入自定义模块
from arc_models import ModelFactory
from arc_data import DataLoaderFactory, LogPreprocessor
from arc_utils import ArcGPUDetector, TrainerUtils, ModelSaver, MetricsCalculator

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ArcTrainer:
    """Intel Arc GPU 训练器 - 重构版本"""
    
    def __init__(self, model_type: str = "textcnn", config: Optional[Dict[str, Any]] = None):
        """
        初始化训练器
        Args:
            model_type: 模型类型 ('textcnn', 'fasttext')
            config: 训练配置
        """
        self.model_type = model_type
        self.config = config or self._get_default_config()
        self.device = ArcGPUDetector.get_device()
        
        # 初始化组件
        self.model = None
        self.label_encoder = None
        self.trainer_utils = TrainerUtils()
        self.model_saver = ModelSaver()
        self.metrics_calculator = MetricsCalculator()
        self.preprocessor = LogPreprocessor()
        
        logger.info(f"🎯 初始化训练器 - 模型类型: {model_type}")
        logger.info(f"🖥️  计算设备: {self.device}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'model_config': ModelFactory.get_default_config(self.model_type),
            'training_config': {
                'batch_size': 32,
                'epochs': 10,
                'learning_rate': 0.001,
                'max_length': 128,
                'vocab_size': 10000,
                'test_size': 0.2,
                'val_size': 0.2,
                'random_state': 42
            },
            'data_config': {
                'text_column': 'message',
                'label_column': 'category',
                'normalize_text': True
            }
        }
    
    def load_and_preprocess_data(self, data_path: str) -> tuple:
        """
        加载和预处理数据
        Args:
            data_path: 数据文件路径
        Returns:
            训练、验证、测试数据加载器和标签编码器
        """
        logger.info(f"📂 加载数据: {data_path}")
        
        # 加载原始数据
        texts, labels, label_encoder = DataLoaderFactory.load_csv_data(
            data_path, 
            self.config['data_config']['text_column'],
            self.config['data_config']['label_column']
        )
        
        # 预处理文本
        if self.config['data_config']['normalize_text']:
            texts = self.preprocessor.process_batch(texts, normalize=True)
            logger.info("✅ 文本预处理完成")
        
        # 分割数据
        train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = \
            DataLoaderFactory.split_data(
                texts, labels,
                test_size=self.config['training_config']['test_size'],
                val_size=self.config['training_config']['val_size'],
                random_state=self.config['training_config']['random_state']
            )
        
        # 创建数据加载器
        train_loader, val_loader, test_loader = DataLoaderFactory.create_data_loaders(
            train_texts, val_texts, test_texts,
            train_labels, val_labels, test_labels,
            batch_size=self.config['training_config']['batch_size'],
            max_length=self.config['training_config']['max_length'],
            vocab_size=self.config['training_config']['vocab_size']
        )
        
        # 保存标签编码器
        self.label_encoder = label_encoder
        
        # 打印数据集信息
        dataset_info = DataLoaderFactory.get_dataset_info(train_loader, val_loader, test_loader)
        logger.info(f"📊 数据集信息: {dataset_info}")
        
        return train_loader, val_loader, test_loader
    
    def create_model(self, num_classes: int):
        """创建模型"""
        logger.info(f"🏗️ 创建模型: {self.model_type}")
        
        # 更新模型配置
        model_config = self.config['model_config'].copy()
        model_config['num_classes'] = num_classes
        
        # 创建模型
        self.model = ModelFactory.create_model(self.model_type, **model_config)
        self.model.to(self.device)
        
        # 打印模型信息
        param_count = self.model.count_parameters()
        logger.info(f"✅ 模型创建成功 - 参数数量: {param_count:,}")
        logger.info(f"📋 模型配置: {self.model.get_config()}")
    
    def train(self, data_path: str, save_dir: str = "results/models"):
        """
        完整训练流程
        Args:
            data_path: 数据文件路径
            save_dir: 模型保存目录
        """
        logger.info("🚀 开始训练流程")
        
        # 检查GPU状态
        ArcGPUDetector.print_gpu_status()
        
        # 加载数据
        train_loader, val_loader, test_loader = self.load_and_preprocess_data(data_path)
        
        # 创建模型
        num_classes = len(self.label_encoder)
        self.create_model(num_classes)
        
        # 设置训练组件
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.config['training_config']['learning_rate']
        )
        
        # 训练循环
        best_val_acc = 0
        training_history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        
        for epoch in range(self.config['training_config']['epochs']):
            logger.info(f"📈 Epoch {epoch+1}/{self.config['training_config']['epochs']}")
            
            # 训练
            train_loss, train_acc = self.trainer_utils.train_epoch(
                self.model, train_loader, criterion, optimizer, self.device
            )
            
            # 验证
            val_loss, val_acc = self.trainer_utils.validate_epoch(
                self.model, val_loader, criterion, self.device
            )
            
            # 记录历史
            training_history['train_loss'].append(train_loss)
            training_history['train_acc'].append(train_acc)
            training_history['val_loss'].append(val_loss)
            training_history['val_acc'].append(val_acc)
            
            # 打印进度
            logger.info(f"  训练: 损失={train_loss:.4f}, 准确率={train_acc:.2f}%")
            logger.info(f"  验证: 损失={val_loss:.4f}, 准确率={val_acc:.2f}%")
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.model_saver.save_model(
                    self.model, self.label_encoder, save_dir, "best",
                    model_type=self.model_type, epoch=epoch+1, val_acc=val_acc
                )
                logger.info(f"💾 保存最佳模型 (验证准确率: {val_acc:.2f}%)")
        
        # 保存最终模型
        self.model_saver.save_model(
            self.model, self.label_encoder, save_dir, "final",
            model_type=self.model_type, epoch=self.config['training_config']['epochs']
        )
        
        # 绘制训练曲线
        self.metrics_calculator.plot_training_curves(training_history, save_dir)
        
        # 测试集评估
        test_loss, test_acc = self.trainer_utils.validate_epoch(
            self.model, test_loader, criterion, self.device
        )
        logger.info(f"🧪 测试集评估: 损失={test_loss:.4f}, 准确率={test_acc:.2f}%")
        
        logger.info(f"✅ 训练完成 - 最佳验证准确率: {best_val_acc:.2f}%")
        
        return {
            'best_val_acc': best_val_acc,
            'test_acc': test_acc,
            'training_history': training_history
        }


def main():
    """主函数"""
    logger.info("🎯 Intel Arc GPU 训练器 - 重构版本")
    
    # 检查GPU
    if not ArcGPUDetector.check_arc_gpu():
        logger.warning("⚠️ 未检测到Intel Arc GPU，将使用CPU训练")
    
    # 创建训练器
    trainer = ArcTrainer(model_type="textcnn")
    
    # 训练参数
    data_path = "DATA_OUTPUT/processed_logs.csv"  # 根据实际数据路径调整
    save_dir = "results/models"
    
    # 开始训练
    results = trainer.train(data_path, save_dir)
    
    # 打印最终结果
    logger.info("📊 训练结果总结:")
    logger.info(f"   最佳验证准确率: {results['best_val_acc']:.2f}%")
    logger.info(f"   测试集准确率: {results['test_acc']:.2f}%")


if __name__ == "__main__":
    main() 