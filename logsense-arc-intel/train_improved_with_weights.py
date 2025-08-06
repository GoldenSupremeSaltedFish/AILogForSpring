#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的训练脚本 - 包含类别权重和更强的正则化
"""

import argparse
import torch
import torch.nn as nn
import logging
import os
from datetime import datetime
from typing import Dict, Any, List
import numpy as np
from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

# 导入自定义模块
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import ModelFactory
from utils import create_persistence_manager

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ImprovedDataset(torch.utils.data.Dataset):
    """改进的数据集"""
    
    def __init__(self, texts: list, labels: list, max_length: int = 128, vocab_size: int = 8000):
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.vocab = self._build_vocab()
        logger.info(f"📚 词汇表大小: {len(self.vocab)}")
    
    def _build_vocab(self):
        """构建词汇表"""
        word_counts = Counter()
        for text in self.texts:
            words = text.lower().split()
            word_counts.update(words)
        
        vocab = {'<PAD>': 0, '<UNK>': 1}
        for word, count in word_counts.most_common(self.vocab_size - 2):
            if count >= 2:  # 至少出现2次
                vocab[word] = len(vocab)
        return vocab
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx]).lower()
        label = self.labels[idx]
        
        words = text.split()[:self.max_length]
        token_ids = []
        
        for word in words:
            if word in self.vocab:
                token_ids.append(self.vocab[word])
            else:
                token_ids.append(self.vocab['<UNK>'])
        
        if len(token_ids) < self.max_length:
            token_ids += [self.vocab['<PAD>']] * (self.max_length - len(token_ids))
        else:
            token_ids = token_ids[:self.max_length]
        
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class ImprovedTrainer:
    """改进的训练器"""
    
    def __init__(self, model_type: str = "textcnn"):
        self.model_type = model_type
        if torch.xpu.is_available():
            self.device = torch.device("xpu:0")
            logger.info("✅ 使用Intel XPU GPU加速")
        else:
            self.device = torch.device("cpu")
            logger.warning("⚠️ 使用CPU训练")
        
        self.model = None
        self.label_encoder = None
        self.class_weights = None
        
        # 初始化持久化管理器
        self.persistence = create_persistence_manager()
        
        # 训练历史记录
        self.training_history = {
            'train_loss': [], 'val_loss': [], 'train_acc': [], 
            'val_acc': [], 'val_f1': [], 'learning_rate': []
        }
        
        logger.info(f"🎯 初始化改进训练器 - 模型类型: {model_type}")
        logger.info(f"🖥️  计算设备: {self.device}")
    
    def load_data(self, data_path: str):
        """加载数据"""
        logger.info(f"📂 加载数据: {data_path}")
        
        df = pd.read_csv(data_path)
        logger.info(f"📊 原始数据: {len(df)} 条记录")
        
        # 数据清洗
        df_cleaned = df.dropna(subset=['original_log', 'category'])
        df_cleaned = df_cleaned[df_cleaned['original_log'].str.strip() != '']
        
        # 检查数据平衡性
        category_counts = df_cleaned['category'].value_counts()
        logger.info("📊 数据分布检查:")
        for category, count in category_counts.items():
            logger.info(f"  {category}: {count} 条")
        
        # 过滤掉样本太少的类别（少于10条记录）
        min_samples = 10
        valid_categories = category_counts[category_counts >= min_samples].index.tolist()
        df_balanced = df_cleaned[df_cleaned['category'].isin(valid_categories)]
        
        logger.info(f"📊 平衡后数据: {len(df_balanced)} 条记录")
        logger.info(f"📊 有效类别数: {len(valid_categories)}")
        logger.info(f"📊 有效类别: {valid_categories}")
        
        texts = df_balanced['original_log'].fillna('').tolist()
        labels = df_balanced['category'].tolist()
        
        # 分析数据分布
        data_info = self._analyze_data_distribution(labels)
        self.persistence.save_data_info(data_info)
        self.persistence.copy_data_files(data_path)
        
        # 标签编码
        unique_labels = sorted(list(set(labels)))
        label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
        encoded_labels = [label_to_id[label] for label in labels]
        
        self.label_encoder = {idx: label for label, idx in label_to_id.items()}
        
        # 计算类别权重
        self.class_weights = self._calculate_class_weights(encoded_labels)
        
        logger.info(f"✅ 数据加载完成 - 类别数: {len(self.label_encoder)}")
        
        return texts, encoded_labels
    
    def _analyze_data_distribution(self, labels):
        """分析数据分布"""
        label_counts = Counter(labels)
        total = len(labels)
        
        distribution = {}
        for label, count in label_counts.items():
            percentage = (count / total) * 100
            distribution[label] = {'count': count, 'percentage': percentage}
            logger.info(f"   {label}: {count} 条 ({percentage:.1f}%)")
        
        max_count = max(label_counts.values())
        min_count = min(label_counts.values())
        imbalance_ratio = max_count / min_count
        
        logger.info(f"📈 数据不平衡比例: {imbalance_ratio:.2f}:1")
        
        return {
            'total_samples': total,
            'label_distribution': distribution,
            'imbalance_ratio': imbalance_ratio,
            'num_classes': len(label_counts)
        }
    
    def _calculate_class_weights(self, labels: List[int]) -> torch.Tensor:
        """计算类别权重"""
        label_counts = Counter(labels)
        total_samples = len(labels)
        num_classes = len(label_counts)
        
        weights = torch.ones(num_classes)
        for label, count in label_counts.items():
            # 使用逆频率权重
            weight = total_samples / (num_classes * count)
            weights[label] = weight
        
        logger.info("📊 类别权重:")
        for i, weight in enumerate(weights):
            logger.info(f"  类别 {i} ({self.label_encoder[i]}): {weight:.3f}")
        
        return weights.to(self.device)
    
    def create_model(self, num_classes: int):
        """创建模型"""
        logger.info(f"🏗️ 创建模型: {self.model_type}")
        
        model_config = {
            'vocab_size': 8000,
            'embed_dim': 128,
            'num_classes': num_classes,
            'filter_sizes': [3, 4, 5],
            'num_filters': 128,
            'dropout': 0.6  # 增加Dropout
        }
        
        self.model = ModelFactory.create_model(self.model_type, **model_config)
        self.model.to(self.device)
        
        param_count = self.model.count_parameters()
        logger.info(f"✅ 模型创建成功 - 参数数量: {param_count:,}")
        
        return model_config
    
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
            
            # 更强的梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
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
        all_predictions = []
        all_labels = []
        
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
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        return total_loss / len(val_loader), 100 * correct / total, f1, all_predictions, all_labels
    
    def train(self, data_path: str, epochs: int = 30, patience: int = 8, batch_size: int = 32):
        """训练流程"""
        logger.info("🚀 开始改进训练")
        
        # 加载数据
        texts, labels = self.load_data(data_path)
        
        # 使用Stratified Split
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, stratify=labels, random_state=42
        )
        
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            train_texts, train_labels, test_size=0.2, stratify=train_labels, random_state=42
        )
        
        logger.info(f"📊 数据分割完成:")
        logger.info(f"   训练集: {len(train_texts)} 条")
        logger.info(f"   验证集: {len(val_texts)} 条")
        logger.info(f"   测试集: {len(test_texts)} 条")
        
        # 创建数据集
        train_dataset = ImprovedDataset(train_texts, train_labels)
        val_dataset = ImprovedDataset(val_texts, val_labels)
        test_dataset = ImprovedDataset(test_texts, test_labels)
        
        # 创建数据加载器
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # 创建模型
        num_classes = len(self.label_encoder)
        model_config = self.create_model(num_classes)
        
        # 保存训练配置
        training_config = {
            'model_type': self.model_type,
            'epochs': epochs,
            'patience': patience,
            'batch_size': batch_size,
            'learning_rate': 0.0005,  # 降低学习率
            'weight_decay': 0.05,     # 增加L2正则化
            'model_config': model_config,
            'device': str(self.device),
            'data_path': data_path,
            'num_classes': num_classes,
            'use_class_weights': True
        }
        self.persistence.save_training_config(training_config)
        
        # 训练组件 - 使用类别权重
        criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.0005, weight_decay=0.05)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.3)
        
        best_acc = 0
        best_f1 = 0
        patience_counter = 0
        best_predictions = None
        best_labels = None
        
        for epoch in range(epochs):
            logger.info(f"📈 Epoch {epoch+1}/{epochs}")
            
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            
            # 验证
            val_loss, val_acc, val_f1, val_predictions, val_labels = self.validate_epoch(val_loader, criterion)
            
            # 学习率调度
            scheduler.step(val_f1)
            current_lr = optimizer.param_groups[0]['lr']
            
            # 记录历史
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_acc'].append(val_acc)
            self.training_history['val_f1'].append(val_f1)
            self.training_history['learning_rate'].append(current_lr)
            
            logger.info(f"  训练: 损失={train_loss:.4f}, 准确率={train_acc:.2f}%")
            logger.info(f"  验证: 损失={val_loss:.4f}, 准确率={val_acc:.2f}%, F1={val_f1:.4f}")
            
            # 早停检查
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_acc = val_acc
                best_predictions = val_predictions
                best_labels = val_labels
                patience_counter = 0
                
                self.persistence.save_model(self.model, "best", model_config)
                logger.info(f" 保存最佳模型 (F1: {val_f1:.4f}, 准确率: {val_acc:.2f}%)")
            else:
                patience_counter += 1
                logger.info(f"⏳ 早停计数: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                logger.info(f" 早停触发 - {patience} 轮未改善")
                break
        
        # 保存最终模型
        self.persistence.save_model(self.model, "final", model_config)
        
        # 测试集评估
        test_loss, test_acc, test_f1, test_predictions, test_labels = self.validate_epoch(test_loader, criterion)
        logger.info(f" 测试集: 损失={test_loss:.4f}, 准确率={test_acc:.2f}%, F1={test_f1:.4f}")
        
        # 保存所有结果
        self._save_all_results(best_acc, best_f1, test_acc, test_f1, epoch + 1, 
                             best_predictions, best_labels)
        
        logger.info(f"✅ 改进训练完成 - 最佳F1: {best_f1:.4f}, 最佳准确率: {best_acc:.2f}%")
        logger.info(f"📁 所有结果已保存到: {self.persistence.session_dir}")
    
    def _save_all_results(self, best_acc, best_f1, test_acc, test_f1, final_epoch,
                         best_predictions, best_labels):
        """保存所有结果"""
        # 保存训练历史
        self.persistence.save_training_history(self.training_history)
        
        # 保存训练指标
        metrics = {
            'best_val_acc': best_acc,
            'best_val_f1': best_f1,
            'test_acc': test_acc,
            'test_f1': test_f1,
            'final_epoch': final_epoch,
            'total_epochs': len(self.training_history['train_loss'])
        }
        self.persistence.save_training_metrics(metrics)
        
        # 保存训练图表
        self.persistence.save_plots(self.training_history)
        
        # 保存混淆矩阵
        actual_class_names = [self.label_encoder[i] for i in range(len(self.label_encoder))]
        logger.info(f"📊 实际类别数量: {len(actual_class_names)}")
        logger.info(f"📊 实际类别: {actual_class_names}")
        
        self.persistence.save_confusion_matrix(best_labels, best_predictions, actual_class_names)
        
        # 保存分类报告
        self.persistence.save_classification_report(best_labels, best_predictions, actual_class_names)
        
        # 保存会话总结
        summary = {
            'model_type': self.model_type,
            'best_val_acc': best_acc,
            'best_f1': best_f1,
            'test_acc': test_acc,
            'test_f1': test_f1,
            'total_epochs': len(self.training_history['train_loss']),
            'final_epoch': final_epoch,
            'num_classes': len(self.label_encoder),
            'actual_classes': actual_class_names,
            'device': str(self.device),
            'use_class_weights': True
        }
        self.persistence.save_session_summary(summary)
        
        # 创建README
        training_info = {
            'model_type': self.model_type,
            'epochs': len(self.training_history['train_loss']),
            'batch_size': 32,
            'learning_rate': 0.0005,
            'best_val_acc': best_acc,
            'best_f1': best_f1,
            'test_acc': test_acc,
            'test_f1': test_f1,
            'num_classes': len(self.label_encoder),
            'use_class_weights': True
        }
        self.persistence.create_readme(training_info)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="改进的Intel Arc GPU 训练器")
    parser.add_argument("--model", type=str, default="textcnn",
                       choices=["textcnn", "fasttext"], help="模型类型")
    parser.add_argument("--data", type=str, default="data/processed_logs_improved.csv", 
                       help="数据文件路径")
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数")
    parser.add_argument("--patience", type=int, default=8, help="早停耐心值")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    
    args = parser.parse_args()
    
    logger.info("🎯 改进的Intel Arc GPU 训练器")
    
    trainer = ImprovedTrainer(model_type=args.model)
    trainer.train(args.data, args.epochs, args.patience, args.batch_size)


if __name__ == "__main__":
    main() 