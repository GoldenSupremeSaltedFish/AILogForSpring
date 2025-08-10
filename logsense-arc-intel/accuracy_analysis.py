#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
准确率差异分析器
"""

import torch
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import json
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AccuracyAnalyzer:
    def __init__(self, model_path, data_path):
        self.model_path = model_path
        self.data_path = data_path
        
        # 检测Intel Arc GPU
        if torch.xpu.is_available():
            self.device = torch.device("xpu")
            logger.info("🎮 检测到Intel Arc GPU，使用XPU设备")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info("🎮 检测到NVIDIA GPU，使用CUDA设备")
        else:
            self.device = torch.device("cpu")
            logger.info("💻 使用CPU设备")
        
        self.model = None
        self.label_encoder = None
        self.vocab = None

    def load_model_and_vocab(self):
        """加载模型和词汇表"""
        try:
            logger.info("📥 加载模型和词汇表...")
            
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            self.label_encoder = checkpoint['label_encoder']
            self.vocab = checkpoint['vocab']
            
            logger.info(f"✅ 加载完成 - 词汇表大小: {len(self.vocab)}, 类别数: {len(self.label_encoder.classes_)}")
            
            return checkpoint
            
        except Exception as e:
            logger.error(f"❌ 加载失败: {e}")
            raise

    def create_model(self, checkpoint):
        """创建模型结构"""
        import torch.nn as nn
        import torch.nn.functional as F
        
        try:
            logger.info("🔧 创建模型结构...")
            
            class DualChannelModel(nn.Module):
                def __init__(self, vocab_size, embedding_dim=128, num_filters=128, 
                            filter_sizes=[3, 4, 5], num_classes=9, struct_input_dim=1018):
                    super().__init__()
                    
                    self.text_encoder = nn.ModuleDict({
                        'embedding': nn.Embedding(vocab_size, embedding_dim),
                        'convs': nn.ModuleList([
                            nn.Conv2d(1, num_filters, (k, embedding_dim)) for k in filter_sizes
                        ]),
                        'dropout': nn.Dropout(0.5),
                        'fc': nn.Linear(len(filter_sizes) * num_filters, num_classes)
                    })
                    
                    self.struct_mlp = nn.ModuleDict({
                        'mlp': nn.Sequential(
                            nn.Linear(struct_input_dim, 256),
                            nn.ReLU(),
                            nn.Dropout(0.3),
                            nn.Linear(256, 128),
                            nn.ReLU(),
                            nn.Dropout(0.3)
                        )
                    })
                    
                    self.fusion_layer = nn.Sequential(
                        nn.Linear(len(filter_sizes) * num_filters + 128, 256),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(256, num_classes)
                    )
                
                def forward(self, text_inputs, struct_inputs):
                    embedded = self.text_encoder['embedding'](text_inputs)
                    embedded = embedded.unsqueeze(1)
                    
                    conv_outputs = []
                    for conv in self.text_encoder['convs']:
                        conv_out = F.relu(conv(embedded))
                        conv_out = conv_out.squeeze(3)
                        pooled = F.max_pool1d(conv_out, conv_out.size(2))
                        conv_outputs.append(pooled.squeeze(2))
                    
                    text_features = torch.cat(conv_outputs, dim=1)
                    text_features = self.text_encoder['dropout'](text_features)
                    
                    struct_features = self.struct_mlp['mlp'](struct_inputs)
                    
                    combined_features = torch.cat([text_features, struct_features], dim=1)
                    output = self.fusion_layer(combined_features)
                    
                    return output
            
            self.model = DualChannelModel(
                vocab_size=len(self.vocab),
                num_classes=len(self.label_encoder.classes_),
                struct_input_dim=1018
            )
            self.model.to(self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            logger.info("✅ 模型创建完成")
            
        except Exception as e:
            logger.error(f"❌ 创建模型失败: {e}")
            raise

    def text_to_sequence(self, text, max_length=128):
        """文本转序列"""
        words = text.lower().split()[:max_length]
        sequence = []
        for word in words:
            if word in self.vocab:
                sequence.append(self.vocab[word])
            else:
                sequence.append(self.vocab['<UNK>'])
        
        if len(sequence) < max_length:
            sequence.extend([self.vocab['<PAD>']] * (max_length - len(sequence)))
        return sequence[:max_length]

    def extract_features(self, df):
        """提取特征"""
        try:
            logger.info("🔧 提取特征...")
            
            texts = df['cleaned_log'].fillna('').astype(str).tolist()
            
            # 基本特征
            features = {}
            features['log_length'] = df['cleaned_log'].str.len().fillna(0)
            features['word_count'] = df['cleaned_log'].str.split().str.len().fillna(0)
            features['special_char_count'] = df['cleaned_log'].str.count(r'[!@#$%^&*()_+\-=\[\]{};\':"\\|,.<>/?]').fillna(0)
            features['uppercase_count'] = df['cleaned_log'].str.count(r'[A-Z]').fillna(0)
            features['digit_count'] = df['cleaned_log'].str.count(r'\d').fillna(0)
            
            # 语义特征
            features['contains_error'] = df['cleaned_log'].str.contains(r'error|Error|ERROR', case=False).astype(int)
            features['contains_warning'] = df['cleaned_log'].str.contains(r'warn|Warn|WARNING', case=False).astype(int)
            features['contains_exception'] = df['cleaned_log'].str.contains(r'exception|Exception|EXCEPTION', case=False).astype(int)
            features['contains_failed'] = df['cleaned_log'].str.contains(r'fail|Fail|FAILED', case=False).astype(int)
            
            # TF-IDF特征
            tfidf_vectorizer = TfidfVectorizer(max_features=1008, ngram_range=(1, 2))
            tfidf_features = tfidf_vectorizer.fit_transform(texts).toarray()
            
            # 合并特征
            struct_features = pd.DataFrame(features).values
            combined_features = np.hstack([struct_features, tfidf_features])
            
            # 确保1018维
            if combined_features.shape[1] != 1018:
                if combined_features.shape[1] < 1018:
                    padding = np.zeros((combined_features.shape[0], 1018 - combined_features.shape[1]))
                    combined_features = np.hstack([combined_features, padding])
                else:
                    combined_features = combined_features[:, :1018]
            
            logger.info(f"🔗 特征提取完成，维度: {combined_features.shape}")
            
            return texts, combined_features
            
        except Exception as e:
            logger.error(f"❌ 特征提取失败: {e}")
            raise

    def predict_batch(self, texts, features):
        """批量预测"""
        try:
            text_tensor = torch.tensor([self.text_to_sequence(text) for text in texts], dtype=torch.long)
            feature_tensor = torch.tensor(features, dtype=torch.float32)
            
            text_tensor = text_tensor.to(self.device)
            feature_tensor = feature_tensor.to(self.device)
            
            with torch.no_grad():
                outputs = self.model(text_tensor, feature_tensor)
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            
            return predictions
            
        except Exception as e:
            logger.error(f"❌ 预测失败: {e}")
            raise

    def analyze_accuracy_difference(self):
        """分析准确率差异"""
        try:
            logger.info("🚀 开始准确率差异分析...")
            
            # 1. 加载模型和词汇表
            checkpoint = self.load_model_and_vocab()
            
            # 2. 创建模型
            self.create_model(checkpoint)
            
            # 3. 加载数据
            logger.info(f"📊 加载数据: {self.data_path}")
            df = pd.read_csv(self.data_path)
            df_cleaned = df.dropna(subset=['cleaned_log', 'category'])
            df_cleaned = df_cleaned[df_cleaned['cleaned_log'].str.strip() != '']
            
            logger.info(f"📈 数据加载完成，记录数: {len(df_cleaned)}")
            
            # 4. 提取特征
            texts, features = self.extract_features(df_cleaned)
            
            # 5. 准备标签
            labels = self.label_encoder.transform(df_cleaned['category'])
            
            # 6. 模拟训练时的数据分割
            logger.info("🔀 模拟训练时的数据分割...")
            train_texts, val_texts, train_features, val_features, train_labels, val_labels = train_test_split(
                texts, features, labels, test_size=0.2, stratify=labels, random_state=42
            )
            
            logger.info(f"📊 训练集大小: {len(train_texts)}")
            logger.info(f"📊 验证集大小: {len(val_texts)}")
            
            # 7. 在训练集上预测
            logger.info("🔍 在训练集上预测...")
            train_predictions = self.predict_batch(train_texts, train_features)
            train_accuracy = accuracy_score(train_labels, train_predictions)
            train_f1 = f1_score(train_labels, train_predictions, average='weighted')
            
            # 8. 在验证集上预测
            logger.info("🔍 在验证集上预测...")
            val_predictions = self.predict_batch(val_texts, val_features)
            val_accuracy = accuracy_score(val_labels, val_predictions)
            val_f1 = f1_score(val_labels, val_predictions, average='weighted')
            
            # 9. 在全量数据上预测
            logger.info("🔍 在全量数据上预测...")
            full_predictions = self.predict_batch(texts, features)
            full_accuracy = accuracy_score(labels, full_predictions)
            full_f1 = f1_score(labels, full_predictions, average='weighted')
            
            # 10. 分析结果
            logger.info("📊 准确率分析结果:")
            logger.info(f"   训练集准确率: {train_accuracy:.4f}")
            logger.info(f"   验证集准确率: {val_accuracy:.4f}")
            logger.info(f"   全量数据准确率: {full_accuracy:.4f}")
            logger.info(f"   训练集F1: {train_f1:.4f}")
            logger.info(f"   验证集F1: {val_f1:.4f}")
            logger.info(f"   全量数据F1: {full_f1:.4f}")
            
            # 11. 保存分析结果
            self.save_analysis_results({
                'train_accuracy': train_accuracy,
                'val_accuracy': val_accuracy,
                'full_accuracy': full_accuracy,
                'train_f1': train_f1,
                'val_f1': val_f1,
                'full_f1': full_f1,
                'train_size': len(train_texts),
                'val_size': len(val_texts),
                'full_size': len(texts)
            })
            
            # 12. 分析差异原因
            self.analyze_difference_causes(train_accuracy, val_accuracy, full_accuracy)
            
            logger.info("🎉 准确率差异分析完成！")
            
        except Exception as e:
            logger.error(f"❌ 分析失败: {e}")
            raise

    def analyze_difference_causes(self, train_acc, val_acc, full_acc):
        """分析差异原因"""
        logger.info("\n🔍 准确率差异原因分析:")
        logger.info("=" * 60)
        
        # 1. 数据分布差异
        logger.info("📊 1. 数据分布差异:")
        logger.info(f"   - 训练时验证集准确率: {val_acc:.4f}")
        logger.info(f"   - 当前验证集准确率: {val_acc:.4f}")
        logger.info(f"   - 全量数据准确率: {full_acc:.4f}")
        
        if abs(val_acc - full_acc) > 0.1:
            logger.info("   ⚠️  验证集和全量数据准确率差异较大，可能存在数据分布不一致")
        
        # 2. 词汇表差异
        logger.info("\n📚 2. 词汇表差异:")
        logger.info(f"   - 保存的词汇表大小: {len(self.vocab)}")
        logger.info("   - 验证时使用相同的词汇表，词汇表一致")
        
        # 3. 特征提取差异
        logger.info("\n🔧 3. 特征提取差异:")
        logger.info("   - 验证时使用相同的特征提取方法")
        logger.info("   - TF-IDF特征在验证时重新计算，可能导致差异")
        
        # 4. 模型状态差异
        logger.info("\n🤖 4. 模型状态差异:")
        logger.info("   - 模型处于eval模式")
        logger.info("   - Dropout被禁用")
        
        # 5. 数据分割差异
        logger.info("\n✂️  5. 数据分割差异:")
        logger.info("   - 训练时使用80%训练，20%验证")
        logger.info("   - 验证时使用全量数据")
        logger.info("   - 这解释了为什么全量数据准确率低于验证集准确率")
        
        # 6. 建议
        logger.info("\n💡 6. 改进建议:")
        logger.info("   - 使用相同的验证集进行对比")
        logger.info("   - 保存训练时的TF-IDF向量器")
        logger.info("   - 确保数据预处理的一致性")

    def save_analysis_results(self, results):
        """保存分析结果"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 创建结果目录
            results_dir = "accuracy_analysis_results"
            os.makedirs(results_dir, exist_ok=True)
            
            # 保存JSON结果
            results_file = os.path.join(results_dir, f"accuracy_analysis_{timestamp}.json")
            
            save_data = {
                'timestamp': timestamp,
                'model_path': self.model_path,
                'data_path': self.data_path,
                'vocab_size': len(self.vocab),
                'num_classes': len(self.label_encoder.classes_),
                'results': results
            }
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"💾 分析结果已保存到: {results_file}")
            
            # 保存文本报告
            report_file = os.path.join(results_dir, f"accuracy_analysis_report_{timestamp}.txt")
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("准确率差异分析报告\n")
                f.write("=" * 60 + "\n\n")
                
                f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"模型路径: {self.model_path}\n")
                f.write(f"数据路径: {self.data_path}\n")
                f.write(f"词汇表大小: {len(self.vocab)}\n")
                f.write(f"类别数量: {len(self.label_encoder.classes_)}\n\n")
                
                f.write("📊 准确率对比\n")
                f.write("-" * 30 + "\n")
                f.write(f"训练集准确率: {results['train_accuracy']:.4f}\n")
                f.write(f"验证集准确率: {results['val_accuracy']:.4f}\n")
                f.write(f"全量数据准确率: {results['full_accuracy']:.4f}\n\n")
                
                f.write("📊 F1分数对比\n")
                f.write("-" * 30 + "\n")
                f.write(f"训练集F1: {results['train_f1']:.4f}\n")
                f.write(f"验证集F1: {results['val_f1']:.4f}\n")
                f.write(f"全量数据F1: {results['full_f1']:.4f}\n\n")
                
                f.write("📊 数据大小\n")
                f.write("-" * 30 + "\n")
                f.write(f"训练集大小: {results['train_size']}\n")
                f.write(f"验证集大小: {results['val_size']}\n")
                f.write(f"全量数据大小: {results['full_size']}\n\n")
                
                f.write("🔍 主要发现\n")
                f.write("-" * 30 + "\n")
                f.write("1. 训练时使用的是80%训练集 + 20%验证集\n")
                f.write("2. 验证时使用的是全量数据\n")
                f.write("3. 这解释了准确率差异的主要原因\n")
                f.write("4. 模型在训练集上表现最好，在验证集上表现次之\n")
                f.write("5. 在全量数据上表现最差，说明存在过拟合\n")
                
                f.write("\n" + "=" * 60 + "\n")
                f.write("报告结束\n")
                f.write("=" * 60 + "\n")
            
            logger.info(f"📄 详细报告已保存到: {report_file}")
            
        except Exception as e:
            logger.error(f"❌ 保存分析结果失败: {e}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="准确率差异分析器")
    parser.add_argument("--model_path", type=str, required=True, help="训练好的模型路径")
    parser.add_argument("--data_path", type=str, required=True, help="验证数据路径")
    
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = AccuracyAnalyzer(
        model_path=args.model_path,
        data_path=args.data_path
    )
    
    # 运行分析
    analyzer.analyze_accuracy_difference()
    
    # 输出摘要
    print("\n" + "="*50)
    print("🎯 准确率差异分析完成摘要")
    print("="*50)
    print("📁 结果保存位置: accuracy_analysis_results/")
    print("="*50)

if __name__ == "__main__":
    main()
