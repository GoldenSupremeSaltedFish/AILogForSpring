#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多层Attention对比训练脚本
"""

import argparse
import torch
import torch.nn as nn
import logging
import os
from datetime import datetime
import numpy as np
from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

# 导入自定义模块
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.textcnn import TextCNN
from models.textcnn_with_attention import TextCNNWithAttention
from models.textcnn_multi_attention import TextCNNMultiAttention

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Dataset(torch.utils.data.Dataset):
    """数据集"""
    
    def __init__(self, texts: list, labels: list, max_length: int = 128, vocab_size: int = 8000):
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.vocab = self._build_vocab()
        logger.info(f" 词汇表大小: {len(self.vocab)}")
    
    def _build_vocab(self):
        """构建词汇表"""
        word_counts = Counter()
        for text in self.texts:
            words = text.lower().split()
            word_counts.update(words)
        
        vocab = {'<PAD>': 0, '<UNK>': 1}
        for word, count in word_counts.most_common(self.vocab_size - 2):
            if count >= 2:
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


class MultiAttentionComparisonTrainer:
    """多层Attention对比训练器"""

    def __init__(self):
        if torch.xpu.is_available():
            self.device = torch.device("xpu:0")
            logger.info("✅ 使用Intel XPU GPU加速")
        else:
            self.device = torch.device("cpu")
            logger.warning("⚠️ 使用CPU训练")
    
    def load_data(self, data_path: str):
        """加载数据"""
        logger.info(f"📂 加载数据: {data_path}")
        df = pd.read_csv(data_path)
        logger.info(f"📊 原始数据: {len(df)} 条记录")
        
        # 数据清洗
        df_cleaned = df.dropna(subset=['original_log', 'category'])
        df_cleaned = df_cleaned[df_cleaned['original_log'].str.strip() != '']
        
        # 过滤样本太少的类别
        category_counts = df_cleaned['category'].value_counts()
        min_samples = 10
        valid_categories = category_counts[category_counts >= min_samples].index.tolist()
        df_filtered = df_cleaned[df_cleaned['category'].isin(valid_categories)]
        
        logger.info(f"📊 过滤后数据: {len(df_filtered)} 条记录")
        logger.info(f" 有效类别: {len(valid_categories)} 个")
        
        # 标签编码
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(df_filtered['category'])
        
        texts = df_filtered['original_log'].tolist()
        
        # 分析数据分布
        self._analyze_data_distribution(labels, label_encoder.classes_)
        
        return texts, labels, label_encoder
    
    def _analyze_data_distribution(self, labels, class_names):
        """分析数据分布"""
        unique, counts = np.unique(labels, return_counts=True)
        total = len(labels)
        
        logger.info("📊 数据分布:")
        for i, (label, count) in enumerate(zip(unique, counts)):
            percentage = (count / total) * 100
            logger.info(f"  {class_names[label]}: {count} 条 ({percentage:.1f}%)")
        
        # 计算不平衡比例
        max_count = counts.max()
        min_count = counts.min()
        imbalance_ratio = max_count / min_count
        logger.info(f"📈 数据不平衡比例: {imbalance_ratio:.2f}:1")
    
    def train_model(self, model, train_loader, val_loader, epochs=10, model_name="Model"):
        """训练模型"""
        logger.info(f"🚀 开始训练 {model_name}")
        
        model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3)
        
        best_f1 = 0
        best_acc = 0
        patience = 5
        patience_counter = 0
        
        for epoch in range(epochs):
            # 训练
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                outputs = model(input_ids)
                loss = criterion(outputs, labels)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            train_acc = 100 * train_correct / train_total
            
            # 验证
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            all_predictions = []
            all_labels = []
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    outputs = model(input_ids)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    
                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            val_acc = 100 * val_correct / val_total
            val_f1 = f1_score(all_labels, all_predictions, average='weighted')
            
            logger.info(f"📈 Epoch {epoch+1}/{epochs}")
            logger.info(f"   训练: 损失={train_loss/len(train_loader):.4f}, 准确率={train_acc:.2f}%")
            logger.info(f"   验证: 损失={val_loss/len(val_loader):.4f}, 准确率={val_acc:.2f}%, F1={val_f1:.4f}")
            
            # 早停
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
            
            scheduler.step(val_f1)
            
            if patience_counter >= patience:
                logger.info(f"⏳ 早停触发 - {patience} 轮未改善")
                break
        
        return best_acc, best_f1
    
    def compare_models(self, data_path: str, epochs: int = 10):
        """对比模型"""
        logger.info("🎯 多层Attention对比训练")
        
        # 加载数据
        texts, labels, label_encoder = self.load_data(data_path)
        
        # 数据分割
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, stratify=labels, random_state=42
        )
        
        logger.info(f"📊 数据分割完成:")
        logger.info(f"   训练集: {len(train_texts)} 条")
        logger.info(f"   验证集: {len(val_texts)} 条")
        
        # 创建数据集
        train_dataset = Dataset(train_texts, train_labels)
        val_dataset = Dataset(val_texts, val_labels)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        num_classes = len(label_encoder.classes_)
        logger.info(f"📊 类别数: {num_classes}")
        
        # 模型配置
        model_config = {
            'vocab_size': 8000,
            'embed_dim': 128,
            'num_classes': num_classes,
            'filter_sizes': [3, 4, 5],
            'num_filters': 128,
            'dropout': 0.5
        }
        
        results = {}
        
        # 训练原始TextCNN
        logger.info("🔍 训练原始TextCNN (无Attention)")
        original_model = TextCNN(**model_config)
        original_acc, original_f1 = self.train_model(
            original_model, train_loader, val_loader, epochs, "原始TextCNN"
        )
        results['original'] = {'acc': original_acc, 'f1': original_f1}
        
        # 训练单层Attention TextCNN
        logger.info("🔍 训练TextCNN with Single Attention")
        single_attention_model = TextCNNWithAttention(**model_config, attention_dim=128)
        single_acc, single_f1 = self.train_model(
            single_attention_model, train_loader, val_loader, epochs, "TextCNN with Single Attention"
        )
        results['single_attention'] = {'acc': single_acc, 'f1': single_f1}
        
        # 训练多层Attention TextCNN
        logger.info("🔍 训练TextCNN with Multi-Layer Attention")
        multi_attention_model = TextCNNMultiAttention(
            **model_config, 
            attention_layers=2, 
            num_heads=4
        )
        multi_acc, multi_f1 = self.train_model(
            multi_attention_model, train_loader, val_loader, epochs, "TextCNN with Multi-Layer Attention"
        )
        results['multi_attention'] = {'acc': multi_acc, 'f1': multi_f1}
        
        # 对比结果
        logger.info("📊 对比结果:")
        logger.info(f"  原始TextCNN: 准确率={original_acc:.2f}%, F1={original_f1:.4f}")
        logger.info(f"  单层Attention: 准确率={single_acc:.2f}%, F1={single_f1:.4f}")
        logger.info(f"  多层Attention: 准确率={multi_acc:.2f}%, F1={multi_f1:.4f}")
        
        # 计算改进幅度
        single_acc_improvement = single_acc - original_acc
        single_f1_improvement = single_f1 - original_f1
        multi_acc_improvement = multi_acc - original_acc
        multi_f1_improvement = multi_f1 - original_f1
        
        logger.info(f"📈 改进幅度:")
        logger.info(f"  单层Attention vs 原始:")
        logger.info(f"    准确率提升: {single_acc_improvement:.2f}%")
        logger.info(f"    F1分数提升: {single_f1_improvement:.4f}")
        logger.info(f"  多层Attention vs 原始:")
        logger.info(f"    准确率提升: {multi_acc_improvement:.2f}%")
        logger.info(f"    F1分数提升: {multi_f1_improvement:.4f}")
        logger.info(f"  多层Attention vs 单层Attention:")
        logger.info(f"    准确率提升: {multi_acc - single_acc:.2f}%")
        logger.info(f"    F1分数提升: {multi_f1 - single_f1:.4f}")
        
        # 判断最佳模型
        best_model = max(results.items(), key=lambda x: x[1]['f1'])
        logger.info(f"🏆 最佳模型: {best_model[0]} (F1: {best_model[1]['f1']:.4f})")
        
        if multi_f1 > single_f1 and multi_f1 > original_f1:
            logger.info("✅ 多层Attention机制显著提升了模型性能!")
        elif single_f1 > original_f1:
            logger.info("✅ 单层Attention机制有效提升了模型性能!")
        else:
            logger.info("⚠️ Attention机制未显著提升性能")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='多层Attention对比训练')
    parser.add_argument('--data', type=str, default='data/processed_logs_final_cleaned.csv',
                       help='数据文件路径')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    
    args = parser.parse_args()
    
    trainer = MultiAttentionComparisonTrainer()
    trainer.compare_models(args.data, args.epochs)


if __name__ == "__main__":
    main() 