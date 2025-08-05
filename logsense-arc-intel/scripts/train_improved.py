#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的Intel Arc GPU 训练脚本 - 解决过拟合问题
"""

import argparse
import torch
import torch.nn as nn
import logging
import os
from datetime import datetime
from typing import Dict, Any
import numpy as np
from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, classification_report

# 导入自定义模块
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import ModelFactory
from data import LogPreprocessor

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ImprovedLogDataset(torch.utils.data.Dataset):
    """改进的日志数据集 - 解决Tokenizer问题"""
    
    def __init__(self, texts: list, labels: list, max_length: int = 128, vocab_size: int = 5000):
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        self.vocab_size = vocab_size
        
        # 构建词汇表
        self.vocab = self._build_vocab()
        logger.info(f" 词汇表大小: {len(self.vocab)}")
    
    def _build_vocab(self):
        """构建词汇表"""
        word_counts = Counter()
        
        # 统计所有词汇
        for text in self.texts:
            words = text.lower().split()
            word_counts.update(words)
        
        # 选择最常见的词汇
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
        
        # 改进的分词处理
        words = text.split()[:self.max_length]
        token_ids = []
        
        for word in words:
            if word in self.vocab:
                token_ids.append(self.vocab[word])
            else:
                token_ids.append(self.vocab['<UNK>'])
        
        # 补齐到固定长度
        if len(token_ids) < self.max_length:
            token_ids += [self.vocab['<PAD>']] * (self.max_length - len(token_ids))
        else:
            token_ids = token_ids[:self.max_length]
        
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class ImprovedArcTrainer:
    """改进的Intel Arc GPU 训练器"""

    def __init__(self, model_type: str = "textcnn"):
        self.model_type = model_type
        # 优先使用XPU，如果没有则使用CPU
        if torch.xpu.is_available():
            self.device = torch.device("xpu:0")
            logger.info("✅ 使用Intel XPU GPU加速")
        else:
            self.device = torch.device("cpu")
            logger.warning("⚠️ 使用CPU训练")
        
        self.model = None
        self.label_encoder = None
        self.vocab = None

        logger.info(f" 初始化改进训练器 - 模型类型: {model_type}")
        logger.info(f"🖥️  计算设备: {self.device}")

    def load_and_clean_data(self, data_path: str):
        """加载和清洗数据"""
        logger.info(f"📂 加载数据: {data_path}")

        # 加载数据
        df = pd.read_csv(data_path)
        logger.info(f"📊 原始数据: {len(df)} 条记录")
        
        # 数据质量检查
        logger.info("🔍 数据质量检查:")
        logger.info(f"   空值数量: {df.isnull().sum().sum()}")
        logger.info(f"   重复样本: {df.duplicated().sum()}")
        
        # 检查列名
        logger.info(f"   列名: {list(df.columns)}")
        
        # 找到正确的文本列和标签列
        text_column = None
        label_column = None
        
        # 尝试找到文本列
        possible_text_cols = ['original_log', 'message', 'content', 'text']
        for col in possible_text_cols:
            if col in df.columns:
                text_column = col
                break
        
        if not text_column:
            logger.error("❌ 未找到合适的文本列")
            return None, None, None
        
        # 标签列
        label_column = 'category'
        
        logger.info(f"🔍 使用文本列: {text_column}")
        logger.info(f"🔍 使用标签列: {label_column}")
        
        # 数据清洗
        df_cleaned = df.dropna(subset=[text_column, label_column])
        df_cleaned = df_cleaned[df_cleaned[label_column] != 'other']
        
        logger.info(f" 清洗后数据: {len(df_cleaned)} 条记录")
        
        # 分析数据分布
        texts = df_cleaned[text_column].fillna('').tolist()
        labels = df_cleaned[label_column].tolist()
        
        self._analyze_data_distribution(labels)
        
        # 标签编码
        unique_labels = sorted(list(set(labels)))
        label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
        encoded_labels = [label_to_id[label] for label in labels]
        
        self.label_encoder = {idx: label for label, idx in label_to_id.items()}
        
        logger.info(f"✅ 数据加载完成 - 类别数: {len(self.label_encoder)}")
        
        return texts, encoded_labels, self.label_encoder

    def _analyze_data_distribution(self, labels):
        """分析数据分布"""
        label_counts = Counter(labels)
        total = len(labels)
        
        logger.info("📊 数据分布分析:")
        for label, count in label_counts.items():
            percentage = (count / total) * 100
            logger.info(f"   {label}: {count} 条 ({percentage:.1f}%)")
        
        # 检查数据不平衡
        max_count = max(label_counts.values())
        min_count = min(label_counts.values())
        imbalance_ratio = max_count / min_count
        
        if imbalance_ratio > 2:
            logger.warning(f"⚠️ 数据严重不平衡 (比例: {imbalance_ratio:.1f}:1)")
        else:
            logger.info("✅ 数据分布相对均衡")

    def create_improved_model(self, num_classes: int):
        """创建改进的模型 - 减少复杂度"""
        logger.info(f"🏗️ 创建改进模型: {self.model_type}")

        # 使用更小的模型配置
        model_config = {
            'vocab_size': 5000,  # 减少词汇表大小
            'embed_dim': 64,      # 减少嵌入维度
            'num_classes': num_classes,
            'filter_sizes': [3, 4, 5],
            'num_filters': 64,    # 减少卷积核数量
            'dropout': 0.7        # 增加Dropout
        }

        self.model = ModelFactory.create_model(self.model_type, **model_config)
        self.model.to(self.device)

        param_count = self.model.count_parameters()
        logger.info(f"✅ 改进模型创建成功 - 参数数量: {param_count:,}")
        logger.info(f"📊 参数/样本比: {param_count / 3288:.1f}:1")

    def train_epoch(self, train_loader, criterion, optimizer, scheduler=None):
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
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            if scheduler:
                scheduler.step()

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

        # 计算F1分数
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        return total_loss / len(val_loader), 100 * correct / total, f1

    def train_with_early_stopping(self, data_path: str, epochs: int = 20, 
                                 patience: int = 5, save_dir: str = "results/models_improved"):
        """带早停的训练流程"""
        logger.info("🚀 开始改进训练")

        # 检查GPU状态
        if torch.xpu.is_available():
            logger.info(f"🖥️ 使用Intel XPU GPU: {torch.xpu.get_device_name(0)}")
            logger.info(f"💾 GPU内存: {torch.xpu.get_device_properties(0).total_memory / (1024**3):.1f} GB")
        else:
            logger.warning("⚠️ 未检测到Intel XPU GPU，使用CPU训练")

        # 加载和清洗数据
        texts, labels, label_encoder = self.load_and_clean_data(data_path)
        if texts is None:
            logger.error("❌ 数据加载失败")
            return

        # 使用Stratified Split
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, stratify=labels, random_state=42
        )
        
        # 进一步分割验证集
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            train_texts, train_labels, test_size=0.2, stratify=train_labels, random_state=42
        )

        logger.info(f"📊 数据分割完成:")
        logger.info(f"   训练集: {len(train_texts)} 条")
        logger.info(f"   验证集: {len(val_texts)} 条")
        logger.info(f"   测试集: {len(test_texts)} 条")

        # 创建改进的数据集
        train_dataset = ImprovedLogDataset(train_texts, train_labels)
        val_dataset = ImprovedLogDataset(val_texts, val_labels)
        test_dataset = ImprovedLogDataset(test_texts, test_labels)

        # 创建数据加载器
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)  # 减小批次大小
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

        # 创建改进的模型
        num_classes = len(self.label_encoder)
        self.create_improved_model(num_classes)

        # 设置训练组件 - 添加L2正则化
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)  # 添加L2正则化
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

        best_acc = 0
        best_f1 = 0
        patience_counter = 0

        for epoch in range(epochs):
            logger.info(f"📈 Epoch {epoch+1}/{epochs}")

            # 训练
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)

            # 验证
            val_loss, val_acc, val_f1 = self.validate_epoch(val_loader, criterion)

            # 学习率调度
            scheduler.step(val_f1)

            logger.info(f"  训练: 损失={train_loss:.4f}, 准确率={train_acc:.2f}%")
            logger.info(f"  验证: 损失={val_loss:.4f}, 准确率={val_acc:.2f}%, F1={val_f1:.4f}")
            logger.info(f"  学习率: {optimizer.param_groups[0]['lr']:.6f}")

            # 早停检查
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_acc = val_acc
                patience_counter = 0
                self.save_model(save_dir, "best")
                logger.info(f" 保存最佳模型 (F1: {val_f1:.4f}, 准确率: {val_acc:.2f}%)")
            else:
                patience_counter += 1
                logger.info(f"⏳ 早停计数: {patience_counter}/{patience}")

            # 早停
            if patience_counter >= patience:
                logger.info(f" 早停触发 - {patience} 轮未改善")
                break

        # 保存最终模型
        self.save_model(save_dir, "final")

        # 测试集评估
        test_loss, test_acc, test_f1 = self.validate_epoch(test_loader, criterion)
        logger.info(f" 测试集: 损失={test_loss:.4f}, 准确率={test_acc:.2f}%, F1={test_f1:.4f}")

        logger.info(f"✅ 改进训练完成 - 最佳F1: {best_f1:.4f}, 最佳准确率: {best_acc:.2f}%")

    def save_model(self, save_dir: str, suffix: str = ""):
        """保存模型"""
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        model_path = os.path.join(save_dir, f"improved_arc_gpu_model_{self.model_type}_{suffix}_{timestamp}.pth")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': self.model.get_config(),
            'label_encoder': self.label_encoder,
            'vocab': getattr(self.model, 'vocab', None)
        }, model_path)

        logger.info(f" 模型已保存: {model_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="改进的Intel Arc GPU 训练器")
    parser.add_argument("--model", type=str, default="textcnn",
                       choices=["textcnn", "fasttext"], help="模型类型")
    parser.add_argument("--data", type=str, required=True, help="数据文件路径")
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数")
    parser.add_argument("--patience", type=int, default=5, help="早停耐心值")
    parser.add_argument("--save_dir", type=str, default="results/models_improved", help="模型保存目录")

    args = parser.parse_args()

    logger.info("🎯 改进的Intel Arc GPU 训练器")

    trainer = ImprovedArcTrainer(model_type=args.model)
    trainer.train_with_early_stopping(args.data, args.epochs, args.patience, args.save_dir)


if __name__ == "__main__":
    main()