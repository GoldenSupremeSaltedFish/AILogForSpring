#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化文本验证脚本
只验证TextCNN部分，不包含结构化特征
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import sys
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 定义StructuredFeatureExtractor类（用于加载模型）
class StructuredFeatureExtractor:
    def __init__(self, max_tfidf_features=1000):
        self.max_tfidf_features = max_tfidf_features
        self.label_encoders = {}
        self.tfidf_vectorizer = None
        self.scaler = None
        self.feature_names = []

class SimpleTextCNN(nn.Module):
    """简化的TextCNN模型"""
    def __init__(self, vocab_size, embed_dim, num_filters, filter_sizes, num_classes, dropout=0.5):
        super(SimpleTextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (size, embed_dim)) for size in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
    
    def forward(self, x):
        x = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        x = x.unsqueeze(1)  # [batch_size, 1, seq_len, embed_dim]
        
        conv_outputs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(x))  # [batch_size, num_filters, seq_len-k+1, 1]
            conv_out = conv_out.squeeze(3)  # [batch_size, num_filters, seq_len-k+1]
            pooled = torch.max_pool1d(conv_out, conv_out.size(2))  # [batch_size, num_filters, 1]
            conv_outputs.append(pooled.squeeze(2))  # [batch_size, num_filters]
        
        x = torch.cat(conv_outputs, dim=1)  # [batch_size, len(filter_sizes) * num_filters]
        x = self.dropout(x)
        x = self.fc(x)
        return x

def text_to_sequence(text, vocab, max_len=100):
    """将文本转换为序列"""
    words = text.lower().split()
    sequence = []
    for word in words[:max_len]:
        sequence.append(vocab.get(word, vocab.get('<UNK>', 1)))
    
    # 填充到最大长度
    while len(sequence) < max_len:
        sequence.append(vocab.get('<PAD>', 0))
    
    return sequence

class SimpleTextValidator:
    """简化文本验证器"""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
        self.model = None
        self.vocab = None
        self.label_encoder = None
        
    def load_model(self):
        """加载模型"""
        print(f"🔍 加载模型: {self.model_path}")
        
        # 添加安全的全局变量
        torch.serialization.add_safe_globals([
            'sklearn.preprocessing._label.LabelEncoder',
            'sklearn.preprocessing._data.StandardScaler', 
            'sklearn.feature_extraction.text.TfidfVectorizer'
        ])
        
        # 加载检查点
        checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
        
        # 提取组件
        self.vocab = checkpoint['vocab']
        self.label_encoder = checkpoint['label_encoder']
        
        print(f"✅ 模型加载成功!")
        print(f"   词汇表大小: {len(self.vocab)}")
        print(f"   类别数: {len(self.label_encoder.classes_)}")
        print(f"   类别: {list(self.label_encoder.classes_)}")
        
        # 创建简化的TextCNN模型
        vocab_size = len(self.vocab)
        num_classes = len(self.label_encoder.classes_)
        
        self.model = SimpleTextCNN(
            vocab_size=vocab_size,
            embed_dim=128,
            num_filters=128,
            filter_sizes=[3, 4, 5],
            num_classes=num_classes,
            dropout=0.5
        )
        
        # 只加载文本编码器的权重
        text_encoder_state_dict = {}
        for key, value in checkpoint['model_state_dict'].items():
            if key.startswith('text_encoder.'):
                # 移除'text_encoder.'前缀
                new_key = key.replace('text_encoder.', '')
                text_encoder_state_dict[new_key] = value
        
        self.model.load_state_dict(text_encoder_state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ 文本模型权重加载成功!")
        
    def load_validation_data(self, data_path):
        """加载验证数据"""
        print(f"📊 加载验证数据: {data_path}")
        
        if not os.path.exists(data_path):
            print(f"❌ 数据文件不存在: {data_path}")
            return None
        
        df = pd.read_csv(data_path)
        print(f"✅ 数据加载成功! 样本数: {len(df)}")
        print(f"   原始类别分布:\n{df['category'].value_counts()}")
        
        # 过滤掉训练时没有的类别
        valid_categories = set(self.label_encoder.classes_)
        df_filtered = df[df['category'].isin(valid_categories)].copy()
        
        print(f"✅ 过滤后数据: 样本数: {len(df_filtered)}")
        print(f"   过滤后类别分布:\n{df_filtered['category'].value_counts()}")
        
        return df_filtered
    
    def prepare_features(self, df):
        """准备特征"""
        print("🔧 准备特征...")
        
        # 文本序列化
        text_sequences = []
        for text in df['original_log']:
            sequence = text_to_sequence(text, self.vocab)
            text_sequences.append(sequence)
        
        text_tensor = torch.tensor(text_sequences, dtype=torch.long)
        
        # 标签
        labels = self.label_encoder.transform(df['category'])
        label_tensor = torch.tensor(labels, dtype=torch.long)
        
        print(f"✅ 特征准备完成!")
        print(f"   文本特征形状: {text_tensor.shape}")
        print(f"   标签形状: {label_tensor.shape}")
        
        return text_tensor, label_tensor
    
    def validate_model(self, text_tensor, label_tensor):
        """验证模型"""
        print("🔍 开始模型验证...")
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            # 分批处理
            batch_size = 32
            num_samples = len(text_tensor)
            
            for i in range(0, num_samples, batch_size):
                end_idx = min(i + batch_size, num_samples)
                
                batch_text = text_tensor[i:end_idx].to(self.device)
                batch_labels = label_tensor[i:end_idx]
                
                outputs = self.model(batch_text)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch_labels.numpy())
        
        # 计算指标
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        print(f"\n📊 验证结果:")
        print(f"   总体准确率: {accuracy:.4f}")
        print(f"   总体F1分数: {f1:.4f}")
        
        # 分类报告
        print(f"\n📋 详细分类报告:")
        report = classification_report(all_labels, all_predictions, 
                                     target_names=self.label_encoder.classes_,
                                     digits=4)
        print(report)
        
        # 混淆矩阵
        cm = confusion_matrix(all_labels, all_predictions)
        self.plot_confusion_matrix(cm, self.label_encoder.classes_)
        
        return accuracy, f1, all_predictions, all_labels
    
    def plot_confusion_matrix(self, cm, class_names):
        """绘制混淆矩阵"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('混淆矩阵')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # 保存图片
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = f"results/plots/simple_text_validation_confusion_matrix_{timestamp}.png"
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 混淆矩阵已保存: {plot_path}")
    
    def run_validation(self, data_path):
        """运行完整验证"""
        print("=" * 60)
        print("🚀 简化文本模型验证")
        print("=" * 60)
        
        # 加载模型
        self.load_model()
        
        # 加载数据
        df = self.load_validation_data(data_path)
        if df is None:
            return
        
        # 准备特征
        text_tensor, label_tensor = self.prepare_features(df)
        
        # 验证模型
        accuracy, f1, predictions, labels = self.validate_model(text_tensor, label_tensor)
        
        # 保存结果
        self.save_results(accuracy, f1, predictions, labels, df)
        
        print("\n" + "=" * 60)
        print("✅ 验证完成!")
        print("=" * 60)
    
    def save_results(self, accuracy, f1, predictions, labels, df):
        """保存验证结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建结果目录
        results_dir = f"results/validation_results_{timestamp}"
        os.makedirs(results_dir, exist_ok=True)
        
        # 保存详细结果
        results_df = df.copy()
        results_df['true_label'] = labels
        results_df['predicted_label'] = predictions
        results_df['correct'] = (labels == predictions)
        
        # 转换标签名称
        results_df['true_category'] = self.label_encoder.inverse_transform(labels)
        results_df['predicted_category'] = self.label_encoder.inverse_transform(predictions)
        
        # 保存到CSV
        results_path = os.path.join(results_dir, "validation_results.csv")
        results_df.to_csv(results_path, index=False)
        
        # 保存摘要
        summary = {
            'timestamp': timestamp,
            'model_path': self.model_path,
            'total_samples': len(df),
            'accuracy': accuracy,
            'f1_score': f1,
            'class_distribution': df['category'].value_counts().to_dict(),
            'per_class_accuracy': {}
        }
        
        # 计算每个类别的准确率
        for i, class_name in enumerate(self.label_encoder.classes_):
            class_mask = (labels == i)
            if np.sum(class_mask) > 0:
                class_accuracy = (predictions[class_mask] == labels[class_mask]).mean()
                summary['per_class_accuracy'][class_name] = class_accuracy
        
        # 保存摘要
        import json
        summary_path = os.path.join(results_dir, "validation_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"📁 验证结果已保存到: {results_dir}")
        print(f"   - 详细结果: validation_results.csv")
        print(f"   - 摘要报告: validation_summary.json")

def main():
    """主函数"""
    # 模型文件路径
    model_path = "results/models/feature_enhanced_model_20250812_004934.pth"
    
    # 验证数据路径
    data_path = "data/processed_logs_full.csv"
    
    # 创建验证器
    validator = SimpleTextValidator(model_path)
    
    # 运行验证
    validator.run_validation(data_path)

if __name__ == "__main__":
    main()
