#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双通道特征融合模型 - 结合结构化特征和深度语义特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import logging
import re
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import json
from collections import Counter

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StructuredFeatureExtractor:
    """结构化特征提取器"""
    
    def __init__(self, max_tfidf_features=1000):
        self.max_tfidf_features = max_tfidf_features
        self.label_encoders = {}
        self.tfidf_vectorizer = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def extract_structured_features(self, df):
        """提取结构化特征"""
        logger.info("🔍 提取结构化特征...")
        
        features = {}
        
        # 1. 日志级别特征
        logger.info("📊 处理日志级别特征")
        if 'log_level' in df.columns:
            features['log_level'] = self._encode_categorical(df['log_level'], 'log_level')
        
        # 2. 错误码特征
        logger.info("🔍 处理错误码特征")
        if 'error_codes' in df.columns:
            features['has_error_code'] = (df['error_codes'] != '').astype(int)
            features['error_code_count'] = df['error_codes'].str.count(' ').fillna(0) + (df['error_codes'] != '').astype(int)
        
        # 3. 路径特征
        logger.info("📁 处理路径特征")
        if 'paths' in df.columns:
            features['has_path'] = (df['paths'] != '').astype(int)
            features['path_count'] = df['paths'].str.count(' ').fillna(0) + (df['paths'] != '').astype(int)
            # 修复反斜杠转义问题
            features['path_depth'] = df['paths'].str.count('/').fillna(0) + df['paths'].str.count(r'\\').fillna(0)
        
        # 4. 数字特征
        logger.info("🔢 处理数字特征")
        if 'numbers' in df.columns:
            features['has_numbers'] = (df['numbers'] != '').astype(int)
            features['number_count'] = df['numbers'].str.count(' ').fillna(0) + (df['numbers'] != '').astype(int)
        
        # 5. 类名特征
        logger.info("🏷️ 处理类名特征")
        if 'classes' in df.columns:
            features['has_classes'] = (df['classes'] != '').astype(int)
            features['class_count'] = df['classes'].str.count(' ').fillna(0) + (df['classes'] != '').astype(int)
        
        # 6. 方法名特征
        logger.info("⚙️ 处理方法名特征")
        if 'methods' in df.columns:
            features['has_methods'] = (df['methods'] != '').astype(int)
            features['method_count'] = df['methods'].str.count(' ').fillna(0) + (df['methods'] != '').astype(int)
        
        # 7. 时间戳特征
        logger.info("⏰ 处理时间戳特征")
        if 'timestamps' in df.columns:
            features['has_timestamps'] = (df['timestamps'] != '').astype(int)
        
        # 8. 日志长度特征
        logger.info("📏 处理日志长度特征")
        if 'cleaned_log' in df.columns:
            features['log_length'] = df['cleaned_log'].str.len().fillna(0)
            features['word_count'] = df['cleaned_log'].str.split().str.len().fillna(0)
        
        # 9. 特殊字符特征
        logger.info("🔤 处理特殊字符特征")
        if 'cleaned_log' in df.columns:
            features['special_char_count'] = df['cleaned_log'].str.count(r'[^a-zA-Z0-9\s]').fillna(0)
            features['uppercase_count'] = df['cleaned_log'].str.count(r'[A-Z]').fillna(0)
            features['digit_count'] = df['cleaned_log'].str.count(r'\d').fillna(0)
        
        # 10. TF-IDF特征（简化版本）
        logger.info("📝 处理TF-IDF特征")
        if 'cleaned_log' in df.columns:
            tfidf_features = self._extract_tfidf_features(df['cleaned_log'])
            features.update(tfidf_features)
        
        # 合并所有特征
        feature_df = pd.DataFrame(features)
        logger.info(f"📊 结构化特征维度: {feature_df.shape}")
        
        # 标准化数值特征
        numeric_features = feature_df.select_dtypes(include=[np.number]).columns
        if len(numeric_features) > 0:
            feature_df[numeric_features] = self.scaler.fit_transform(feature_df[numeric_features])
        
        return feature_df
    
    def _encode_categorical(self, series, name):
        """编码分类特征"""
        if name not in self.label_encoders:
            self.label_encoders[name] = LabelEncoder()
            return self.label_encoders[name].fit_transform(series.astype(str))
        else:
            return self.label_encoders[name].transform(series.astype(str))
    
    def _extract_tfidf_features(self, texts, max_features=None):
        """提取TF-IDF特征"""
        if max_features is None:
            max_features = self.max_tfidf_features
        
        if self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        else:
            tfidf_matrix = self.tfidf_vectorizer.transform(texts)
        
        # 转换为DataFrame
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
        )
        
        logger.info(f"📝 TF-IDF特征维度: {tfidf_df.shape}")
        return tfidf_df


class TextCNN(nn.Module):
    """TextCNN模型"""
    
    def __init__(self, vocab_size, embed_dim, num_classes, filter_sizes=[3, 4, 5], num_filters=128, dropout=0.5):
        super(TextCNN, self).__init__()
        self.embed_dim = embed_dim
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 卷积层
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embed_dim)) for k in filter_sizes
        ])
        
        # Dropout和分类层
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        
    def forward(self, x):
        # x: [batch_size, seq_len]
        embedded = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        embedded = embedded.unsqueeze(1)  # [batch_size, 1, seq_len, embed_dim]
        
        # 卷积
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(embedded))  # [batch_size, num_filters, seq_len-k+1, 1]
            conv_out = conv_out.squeeze(3)  # [batch_size, num_filters, seq_len-k+1]
            pooled = F.max_pool1d(conv_out, conv_out.size(2))  # [batch_size, num_filters, 1]
            conv_outputs.append(pooled.squeeze(2))  # [batch_size, num_filters]
        
        # 拼接
        concatenated = torch.cat(conv_outputs, dim=1)  # [batch_size, len(filter_sizes) * num_filters]
        
        # Dropout和分类
        dropped = self.dropout(concatenated)
        output = self.fc(dropped)
        
        return output
    
    def get_output_dim(self):
        return len(self.filter_sizes) * self.num_filters


class StructuredFeatureMLP(nn.Module):
    """结构化特征MLP"""
    
    def __init__(self, input_dim, hidden_dims=[256, 128], dropout=0.3):
        super(StructuredFeatureMLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]
    
    def forward(self, x):
        return self.mlp(x)


class DualChannelLogClassifier(nn.Module):
    """双通道日志分类器"""
    
    def __init__(self, text_encoder, struct_input_dim, num_classes, fusion_dim=256):
        super(DualChannelLogClassifier, self).__init__()
        
        self.text_encoder = text_encoder
        self.struct_mlp = StructuredFeatureMLP(struct_input_dim)
        
        # 融合层
        total_features = text_encoder.get_output_dim() + self.struct_mlp.output_dim
        self.fusion_layer = nn.Sequential(
            nn.Linear(total_features, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, num_classes)
        )
        
        logger.info(f"🔗 双通道模型结构:")
        logger.info(f"  文本编码器输出维度: {text_encoder.get_output_dim()}")
        logger.info(f"  结构化特征输出维度: {self.struct_mlp.output_dim}")
        logger.info(f"  融合层输入维度: {total_features}")
        logger.info(f"  融合层输出维度: {fusion_dim}")
    
    def forward(self, text_inputs, struct_inputs):
        # 文本特征提取 - 修改为只返回特征，不进行分类
        embedded = self.text_encoder.embedding(text_inputs)  # [batch_size, seq_len, embed_dim]
        embedded = embedded.unsqueeze(1)  # [batch_size, 1, seq_len, embed_dim]
        
        # 卷积
        conv_outputs = []
        for conv in self.text_encoder.convs:
            conv_out = F.relu(conv(embedded))  # [batch_size, num_filters, seq_len-k+1, 1]
            conv_out = conv_out.squeeze(3)  # [batch_size, num_filters, seq_len-k+1]
            pooled = F.max_pool1d(conv_out, conv_out.size(2))  # [batch_size, num_filters, 1]
            conv_outputs.append(pooled.squeeze(2))  # [batch_size, num_filters]
        
        # 拼接
        text_features = torch.cat(conv_outputs, dim=1)  # [batch_size, len(filter_sizes) * num_filters]
        text_features = self.text_encoder.dropout(text_features)
        
        # 结构化特征提取
        struct_features = self.struct_mlp(struct_inputs)  # [batch, struct_dim]
        
        # 特征融合
        combined_features = torch.cat([text_features, struct_features], dim=1)
        
        # 分类
        output = self.fusion_layer(combined_features)
        
        return output


class FeatureEnhancedDataset(torch.utils.data.Dataset):
    """特征增强数据集"""
    
    def __init__(self, texts, struct_features, labels, vocab, max_length=128):
        self.texts = texts
        self.struct_features = struct_features
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx]).lower()
        struct_feature = self.struct_features[idx]
        label = self.labels[idx]
        
        # 文本编码
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
            'struct_features': torch.tensor(struct_feature, dtype=torch.float),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class FeatureEnhancedTrainer:
    """特征增强训练器"""
    
    def __init__(self):
        if torch.xpu.is_available():
            self.device = torch.device("xpu:0")
            logger.info("✅ 使用Intel XPU GPU加速")
        else:
            self.device = torch.device("cpu")
            logger.warning("⚠️ 使用CPU训练")
    
    def prepare_data(self, data_path):
        """准备数据"""
        logger.info(f"📂 加载数据: {data_path}")
        df = pd.read_csv(data_path)
        
        # 数据清洗
        df_cleaned = df.dropna(subset=['cleaned_log', 'category'])
        df_cleaned = df_cleaned[df_cleaned['cleaned_log'].str.strip() != '']
        
        # 提取结构化特征
        feature_extractor = StructuredFeatureExtractor()
        struct_features = feature_extractor.extract_structured_features(df_cleaned)
        
        # 标签编码
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(df_cleaned['category'])
        
        # 构建词汇表
        texts = df_cleaned['cleaned_log'].tolist()
        vocab = self._build_vocab(texts)
        
        logger.info(f"📊 数据准备完成:")
        logger.info(f"  总样本数: {len(df_cleaned)}")
        logger.info(f"  类别数: {len(label_encoder.classes_)}")
        logger.info(f"  结构化特征维度: {struct_features.shape[1]}")
        logger.info(f"  词汇表大小: {len(vocab)}")
        
        return texts, struct_features.values, labels, label_encoder, vocab, feature_extractor
    
    def _build_vocab(self, texts, vocab_size=8000):
        """构建词汇表"""
        word_counts = Counter()
        for text in texts:
            words = text.lower().split()
            word_counts.update(words)
        
        vocab = {'<PAD>': 0, '<UNK>': 1}
        for word, count in word_counts.most_common(vocab_size - 2):
            if count >= 2:
                vocab[word] = len(vocab)
        
        return vocab
    
    def train_model(self, data_path, epochs=15):
        """训练特征增强模型"""
        logger.info("🚀 开始特征增强模型训练")
        
        # 准备数据
        texts, struct_features, labels, label_encoder, vocab, feature_extractor = self.prepare_data(data_path)
        
        # 数据分割
        train_texts, val_texts, train_struct, val_struct, train_labels, val_labels = train_test_split(
            texts, struct_features, labels, test_size=0.2, stratify=labels, random_state=42
        )
        
        # 创建数据集
        train_dataset = FeatureEnhancedDataset(train_texts, train_struct, train_labels, vocab)
        val_dataset = FeatureEnhancedDataset(val_texts, val_struct, val_labels, vocab)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # 模型配置
        num_classes = len(label_encoder.classes_)
        struct_input_dim = struct_features.shape[1]
        
        text_encoder = TextCNN(
            vocab_size=len(vocab),
            embed_dim=128,
            num_classes=num_classes,
            filter_sizes=[3, 4, 5],
            num_filters=128,
            dropout=0.5
        )
        
        model = DualChannelLogClassifier(text_encoder, struct_input_dim, num_classes)
        model.to(self.device)
        
        # 训练配置
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3)
        
        # 训练循环
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
                struct_features = batch['struct_features'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                outputs = model(input_ids, struct_features)
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
                    struct_features = batch['struct_features'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    outputs = model(input_ids, struct_features)
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
        
        # 保存结果
        self._save_results(model, label_encoder, feature_extractor, best_acc, best_f1, {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs,
            'val_f1s': val_f1s
        })
        
        return model, label_encoder, feature_extractor
    
    def _save_results(self, model, label_encoder, feature_extractor, best_acc, best_f1, history):
        """保存训练结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存模型
        model_path = f"results/models/feature_enhanced_model_{timestamp}.pth"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'label_encoder': label_encoder,
            'feature_extractor': feature_extractor,
            'best_acc': best_acc,
            'best_f1': best_f1,
            'timestamp': timestamp
        }, model_path)
        
        logger.info(f"💾 模型已保存: {model_path}")
        
        # 保存训练历史
        history_path = f"results/history/feature_enhanced_history_{timestamp}.json"
        os.makedirs(os.path.dirname(history_path), exist_ok=True)
        
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
        plot_path = f"results/plots/feature_enhanced_training_{timestamp}.png"
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 训练曲线已保存: {plot_path}")


def main():
    """主函数"""
    trainer = FeatureEnhancedTrainer()
    
    data_path = "data/processed_logs_advanced_enhanced.csv"
    model, label_encoder, feature_extractor = trainer.train_model(data_path, epochs=15)
    
    logger.info("✅ 特征增强模型训练完成!")


if __name__ == "__main__":
    main() 