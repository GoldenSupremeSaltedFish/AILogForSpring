#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于平衡数据的质量训练脚本
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
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 导入自定义模块
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.textcnn import TextCNN
from utils.persistence_manager import TrainingPersistenceManager

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BalancedQualityDataset(torch.utils.data.Dataset):
    """基于平衡数据质量的数据集"""
    
    def __init__(self, texts: list, labels: list, features: dict = None, max_length: int = 128, vocab_size: int = 8000):
        self.texts = texts
        self.labels = labels
        self.features = features
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


class BalancedQualityTrainer:
    """基于平衡数据的质量训练器"""
    
    def __init__(self):
        if torch.xpu.is_available():
            self.device = torch.device("xpu:0")
            logger.info("✅ 使用Intel XPU GPU加速")
        else:
            self.device = torch.device("cpu")
            logger.warning("⚠️ 使用CPU训练")
    
    def load_balanced_data(self, data_path: str):
        """加载平衡数据"""
        logger.info(f"📂 加载平衡数据: {data_path}")
        df = pd.read_csv(data_path)
        logger.info(f"📊 平衡数据: {len(df)} 条记录")
        
        # 检查必要的列
        required_cols = ['category', 'cleaned_log', 'log_level', 'error_codes', 'paths']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"缺少必要的列: {missing_cols}")
        
        # 数据清洗
        df_cleaned = df.dropna(subset=['cleaned_log', 'category'])
        df_cleaned = df_cleaned[df_cleaned['cleaned_log'].str.strip() != '']
        
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
        
        texts = df_filtered['cleaned_log'].tolist()
        
        # 分析数据分布
        self._analyze_data_distribution(labels, label_encoder.classes_)
        
        # 分析特征分布
        self._analyze_feature_distribution(df_filtered)
        
        return texts, labels, label_encoder, df_filtered
    
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
    
    def _analyze_feature_distribution(self, df):
        """分析特征分布"""
        logger.info("🔍 特征分布分析:")
        
        # 日志级别分布
        level_counts = df['log_level'].value_counts()
        logger.info("📊 日志级别分布:")
        for level, count in level_counts.items():
            percentage = (count / len(df)) * 100
            logger.info(f"  {level}: {count} 条 ({percentage:.1f}%)")
        
        # 错误码分布
        error_codes_count = (df['error_codes'] != '').sum()
        error_codes_percentage = (error_codes_count / len(df)) * 100
        logger.info(f"🔍 包含错误码: {error_codes_count} 条 ({error_codes_percentage:.1f}%)")
        
        # 路径分布
        paths_count = (df['paths'] != '').sum()
        paths_percentage = (paths_count / len(df)) * 100
        logger.info(f"📁 包含路径: {paths_count} 条 ({paths_percentage:.1f}%)")
    
    def train_model(self, model, train_loader, val_loader, epochs=15, model_name="BalancedQualityModel"):
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
        
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        val_f1s = []
        
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
            train_losses.append(train_loss / len(train_loader))
            train_accs.append(train_acc)
            
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
            
            val_losses.append(val_loss / len(val_loader))
            val_accs.append(val_acc)
            val_f1s.append(val_f1)
            
            logger.info(f"📈 Epoch {epoch+1}/{epochs}")
            logger.info(f"   训练: 损失={train_losses[-1]:.4f}, 准确率={train_acc:.2f}%")
            logger.info(f"   验证: 损失={val_losses[-1]:.4f}, 准确率={val_acc:.2f}%, F1={val_f1:.4f}")
            
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
        
        # 返回训练历史
        history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs,
            'val_f1s': val_f1s
        }
        
        return best_acc, best_f1, history
    
    def train_balanced_quality(self, data_path: str, epochs: int = 15):
        """基于平衡数据的质量训练"""
        logger.info("🎯 基于平衡数据的质量训练")
        
        # 加载平衡数据
        texts, labels, label_encoder, df_balanced = self.load_balanced_data(data_path)
        
        # 数据分割
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, stratify=labels, random_state=42
        )
        
        logger.info(f"📊 数据分割完成:")
        logger.info(f"   训练集: {len(train_texts)} 条")
        logger.info(f"   验证集: {len(val_texts)} 条")
        
        # 创建数据集
        train_dataset = BalancedQualityDataset(train_texts, train_labels)
        val_dataset = BalancedQualityDataset(val_texts, val_labels)
        
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
        
        # 训练模型
        logger.info("🔍 训练基于平衡数据的TextCNN")
        model = TextCNN(**model_config)
        best_acc, best_f1, history = self.train_model(
            model, train_loader, val_loader, epochs, "BalancedQualityTextCNN"
        )
        
        logger.info("📊 训练结果:")
        logger.info(f"  最佳准确率: {best_acc:.2f}%")
        logger.info(f"  最佳F1分数: {best_f1:.4f}")
        
        # 保存结果
        self._save_results(model, label_encoder, history, best_acc, best_f1, df_balanced)
        
        return model, label_encoder, history
    
    def _save_results(self, model, label_encoder, history, best_acc, best_f1, df_balanced):
        """保存训练结果"""
        # 创建时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存模型
        model_path = f"results/models/balanced_quality_model_{timestamp}.pth"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': model.get_config(),
            'label_encoder': label_encoder,
            'best_acc': best_acc,
            'best_f1': best_f1,
            'timestamp': timestamp
        }, model_path)
        
        logger.info(f"💾 模型已保存: {model_path}")
        
        # 保存训练历史
        history_path = f"results/history/balanced_quality_history_{timestamp}.json"
        os.makedirs(os.path.dirname(history_path), exist_ok=True)
        
        import json
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"💾 训练历史已保存: {history_path}")
        
        # 生成训练曲线
        self._plot_training_curves(history, timestamp)
    
    def _plot_training_curves(self, history, timestamp):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 损失曲线
        axes[0, 0].plot(history['train_losses'], label='训练损失')
        axes[0, 0].plot(history['val_losses'], label='验证损失')
        axes[0, 0].set_title('损失曲线')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        # 准确率曲线
        axes[0, 1].plot(history['train_accs'], label='训练准确率')
        axes[0, 1].plot(history['val_accs'], label='验证准确率')
        axes[0, 1].set_title('准确率曲线')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        
        # F1分数曲线
        axes[1, 0].plot(history['val_f1s'], label='验证F1分数')
        axes[1, 0].set_title('F1分数曲线')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        
        # 训练进度
        epochs = range(1, len(history['train_losses']) + 1)
        axes[1, 1].plot(epochs, history['train_losses'], 'b-', label='训练损失')
        axes[1, 1].plot(epochs, history['val_losses'], 'r-', label='验证损失')
        axes[1, 1].set_title('训练进度')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        # 保存图片
        plot_path = f"results/plots/balanced_quality_training_{timestamp}.png"
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 训练曲线已保存: {plot_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='基于平衡数据的质量训练')
    parser.add_argument('--data', type=str, default='data/processed_logs_advanced_enhanced.csv',
                       help='平衡数据文件路径')
    parser.add_argument('--epochs', type=int, default=15, help='训练轮数')
    
    args = parser.parse_args()
    
    trainer = BalancedQualityTrainer()
    model, label_encoder, history = trainer.train_balanced_quality(args.data, args.epochs)
    
    logger.info("✅ 基于平衡数据的质量训练完成!")


if __name__ == "__main__":
    main() 