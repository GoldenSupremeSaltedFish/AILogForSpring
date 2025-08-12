#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最终模型运行器 - 使用训练时的原始数据重建词汇表
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import logging
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime
import os
import json
import warnings
from collections import Counter
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 定义训练时使用的类
class StructuredFeatureExtractor:
    """结构化特征提取器"""
    
    def __init__(self, max_tfidf_features=1000):
        self.max_tfidf_features = max_tfidf_features
        self.label_encoders = {}
        self.tfidf_vectorizer = None
        self.scaler = StandardScaler()
        self.feature_names = []

class FinalModelRunner:
    """最终模型运行器"""

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
        
        # 创建结果目录
        self.results_dir = "final_validation_results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 模型组件
        self.model = None
        self.label_encoder = None
        self.vocab = None
        
        logger.info(f"🚀 初始化最终模型运行器，使用设备: {self.device}")

    def build_vocab_from_data(self, texts, vocab_size=8000):
        """从数据构建词汇表 - 与训练时相同的方法"""
        logger.info("🔤 构建词汇表...")
        word_counts = Counter()
        for text in texts:
            words = text.lower().split()
            word_counts.update(words)
        
        vocab = {'<PAD>': 0, '<UNK>': 1}
        for word, count in word_counts.most_common(vocab_size - 2):
            if count >= 2:
                vocab[word] = len(vocab)
        
        logger.info(f"📚 词汇表构建完成，大小: {len(vocab)}")
        return vocab

    def text_to_sequence(self, text, vocab, max_length=128):
        """将文本转换为序列 - 使用词汇表"""
        words = text.lower().split()[:max_length]
        sequence = []
        for word in words:
            if word in vocab:
                sequence.append(vocab[word])
            else:
                sequence.append(vocab['<UNK>'])
        
        if len(sequence) < max_length:
            sequence.extend([vocab['<PAD>']] * (max_length - len(sequence)))
        return sequence[:max_length]

    def create_model(self):
        """创建模型结构"""
        try:
            logger.info("🔧 创建模型结构...")
            
            # 创建双通道模型 - 匹配训练时的结构
            class DualChannelModel(nn.Module):
                def __init__(self, vocab_size, embedding_dim=128, num_filters=128, 
                            filter_sizes=[3, 4, 5], num_classes=9, struct_input_dim=1018):
                    super().__init__()
                    
                    # 文本编码器 (TextCNN)
                    self.text_encoder = nn.ModuleDict({
                        'embedding': nn.Embedding(vocab_size, embedding_dim),
                        'convs': nn.ModuleList([
                            nn.Conv2d(1, num_filters, (k, embedding_dim)) for k in filter_sizes
                        ]),
                        'dropout': nn.Dropout(0.5),
                        'fc': nn.Linear(len(filter_sizes) * num_filters, num_classes)
                    })
                    
                    # 结构化特征MLP
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
                    
                    # 融合层
                    self.fusion_layer = nn.Sequential(
                        nn.Linear(len(filter_sizes) * num_filters + 128, 256),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(256, num_classes)
                    )
                
                def forward(self, text_inputs, struct_inputs):
                    # 文本特征提取
                    embedded = self.text_encoder['embedding'](text_inputs)
                    embedded = embedded.unsqueeze(1)
                    
                    # 卷积特征提取
                    conv_outputs = []
                    for conv in self.text_encoder['convs']:
                        conv_out = F.relu(conv(embedded))
                        conv_out = conv_out.squeeze(3)
                        pooled = F.max_pool1d(conv_out, conv_out.size(2))
                        conv_outputs.append(pooled.squeeze(2))
                    
                    # 拼接卷积输出
                    text_features = torch.cat(conv_outputs, dim=1)
                    text_features = self.text_encoder['dropout'](text_features)
                    
                    # 结构化特征处理
                    struct_features = self.struct_mlp['mlp'](struct_inputs)
                    
                    # 特征融合
                    combined_features = torch.cat([text_features, struct_features], dim=1)
                    output = self.fusion_layer(combined_features)
                    
                    return output
            
            # 创建模型实例
            self.model = DualChannelModel(
                vocab_size=len(self.vocab),
                num_classes=len(self.label_encoder.classes_),
                struct_input_dim=1018
            )
            self.model.to(self.device)
            
            logger.info("✅ 模型结构创建完成")
            
            # 加载模型权重
            if hasattr(self, 'checkpoint') and 'model_state_dict' in self.checkpoint:
                self.model.load_state_dict(self.checkpoint['model_state_dict'])
                logger.info("✅ 模型权重加载完成")
            else:
                logger.warning("⚠️ 未找到模型权重，使用随机初始化的模型")
            
            self.model.eval()
            
        except Exception as e:
            logger.error(f"❌ 创建模型失败: {e}")
            raise

    def load_model_weights(self):
        """加载模型权重"""
        try:
            logger.info("📥 加载模型权重...")
            
            # 尝试加载checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            # 提取组件
            self.label_encoder = checkpoint['label_encoder']
            
            # 加载训练时保存的词汇表
            if 'vocab' in checkpoint:
                self.vocab = checkpoint['vocab']
                logger.info(f"📚 加载训练时保存的词汇表，大小: {len(self.vocab)}")
            else:
                logger.warning("⚠️ 未找到训练时保存的词汇表，将重新构建")
                self.vocab = None
            
            # 保存checkpoint供后续使用
            self.checkpoint = checkpoint
            
            logger.info("✅ 模型权重加载完成")
            
        except Exception as e:
            logger.error(f"❌ 加载模型权重失败: {e}")
            raise

    def extract_features(self, df):
        """提取特征"""
        try:
            logger.info("🔧 开始特征提取...")
            
            # 文本特征
            texts = df['cleaned_log'].fillna('').astype(str).tolist()
            
            # 基本结构化特征
            features = {}
            features['log_length'] = df['cleaned_log'].str.len().fillna(0)
            features['word_count'] = df['cleaned_log'].str.split().str.len().fillna(0)
            features['special_char_count'] = df['cleaned_log'].str.count(r'[!@#$%^&*()_+\-=\[\]{};\':"\\|,.<>/?]').fillna(0)
            features['uppercase_count'] = df['cleaned_log'].str.count(r'[A-Z]').fillna(0)
            features['digit_count'] = df['cleaned_log'].str.count(r'\d').fillna(0)
            
            # 添加更多结构化特征
            features['contains_error'] = df['cleaned_log'].str.contains(r'error|Error|ERROR', case=False).astype(int)
            features['contains_warning'] = df['cleaned_log'].str.contains(r'warn|Warn|WARNING', case=False).astype(int)
            features['contains_exception'] = df['cleaned_log'].str.contains(r'exception|Exception|EXCEPTION', case=False).astype(int)
            features['contains_failed'] = df['cleaned_log'].str.contains(r'fail|Fail|FAILED', case=False).astype(int)
            
            # TF-IDF特征 - 调整到合适的大小以匹配1018维
            tfidf_vectorizer = TfidfVectorizer(max_features=1008, ngram_range=(1, 2))
            tfidf_features = tfidf_vectorizer.fit_transform(texts).toarray()
            
            # 合并特征
            struct_features = pd.DataFrame(features).values
            combined_features = np.hstack([struct_features, tfidf_features])
            
            # 确保输出1018维
            if combined_features.shape[1] != 1018:
                if combined_features.shape[1] < 1018:
                    # 如果维度不够，用零填充
                    padding = np.zeros((combined_features.shape[0], 1018 - combined_features.shape[1]))
                    combined_features = np.hstack([combined_features, padding])
                else:
                    # 如果维度过多，截取前1018维
                    combined_features = combined_features[:, :1018]
            
            logger.info(f"🔗 特征提取完成，总维度: {combined_features.shape}")
            
            return texts, combined_features
            
        except Exception as e:
            logger.error(f"❌ 特征提取失败: {e}")
            raise

    def load_and_prepare_data(self):
        """加载并准备数据"""
        try:
            logger.info(f"📊 加载数据: {self.data_path}")
            
            if self.data_path.endswith('.csv'):
                df = pd.read_csv(self.data_path)
            else:
                raise ValueError("请使用CSV格式的数据文件")
            
            # 数据清洗
            df_cleaned = df.dropna(subset=['cleaned_log', 'category'])
            df_cleaned = df_cleaned[df_cleaned['cleaned_log'].str.strip() != '']
            
            logger.info(f"📈 数据加载完成，总记录数: {len(df_cleaned)}")
            
            # 显示类别分布
            logger.info("🎯 类别分布:")
            category_counts = df_cleaned['category'].value_counts()
            for category, count in category_counts.items():
                logger.info(f"   {category}: {count}")
            
            # 使用已加载的词汇表，如果没有则重新构建
            texts = df_cleaned['cleaned_log'].tolist()
            if self.vocab is None:
                logger.info("🔤 重新构建词汇表...")
                self.vocab = self.build_vocab_from_data(texts)
            else:
                logger.info(f"📚 使用已加载的词汇表，大小: {len(self.vocab)}")
            
            # 提取结构化特征
            texts, struct_features = self.extract_features(df_cleaned)
            
            # 准备标签
            labels = self.label_encoder.transform(df_cleaned['category'])
            
            return df_cleaned, texts, struct_features, labels
            
        except Exception as e:
            logger.error(f"❌ 数据加载失败: {e}")
            raise

    def validate_model(self, texts, features, labels, validation_name="validation"):
        """验证模型"""
        try:
            logger.info(f"🔍 开始模型验证: {validation_name}")
            
            # 准备数据
            text_tensor = torch.tensor([self.text_to_sequence(text, self.vocab) for text in texts], dtype=torch.long)
            feature_tensor = torch.tensor(features, dtype=torch.float32)
            
            # 移动到设备
            text_tensor = text_tensor.to(self.device)
            feature_tensor = feature_tensor.to(self.device)
            
            # 预测
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(text_tensor, feature_tensor)
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            
            # 计算指标
            accuracy = accuracy_score(labels, predictions)
            f1 = f1_score(labels, predictions, average='weighted')
            
            # 分类报告
            class_report = classification_report(
                labels, predictions,
                target_names=self.label_encoder.classes_,
                output_dict=True
            )
            
            # 混淆矩阵
            conf_matrix = confusion_matrix(labels, predictions)
            
            logger.info(f"📊 验证结果 - {validation_name}:")
            logger.info(f"   准确率: {accuracy:.4f}")
            logger.info(f"   F1分数: {f1:.4f}")
            
            return {
                'validation_name': validation_name,
                'accuracy': accuracy,
                'f1_score': f1,
                'predictions': predictions,
                'true_labels': labels,
                'classification_report': class_report,
                'confusion_matrix': conf_matrix,
                'class_names': self.label_encoder.classes_
            }
            
        except Exception as e:
            logger.error(f"❌ 模型验证失败: {e}")
            raise

    def run_validation(self):
        """运行验证"""
        try:
            logger.info("🚀 开始模型验证...")
            
            # 1. 加载模型权重（获取标签编码器）
            self.load_model_weights()
            
            # 2. 加载并准备数据（构建词汇表）
            df, texts, features, labels = self.load_and_prepare_data()
            
            # 3. 创建模型结构
            self.create_model()
            
            # 4. 验证模型
            result = self.validate_model(texts, features, labels, "full_validation")
            
            # 5. 保存结果
            self.save_results(result)
            
            logger.info("🎉 验证完成！")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 验证失败: {e}")
            raise

    def save_results(self, result):
        """保存结果"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 保存验证结果
            results_file = os.path.join(self.results_dir, f"final_validation_results_{timestamp}.json")
            
            # 准备保存的数据
            save_data = {
                'timestamp': timestamp,
                'model_path': self.model_path,
                'data_path': self.data_path,
                'vocab_size': len(self.vocab),
                'num_classes': len(self.label_encoder.classes_),
                'accuracy': result['accuracy'],
                'f1_score': result['f1_score'],
                'classification_report': result['classification_report'],
                'confusion_matrix': result['confusion_matrix'].tolist(),
                'class_names': result['class_names'].tolist()
            }
            
            # 保存到JSON文件
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"💾 结果已保存到: {results_file}")
            
            # 保存详细报告
            self.save_detailed_report(result, timestamp)
            
            return results_file
            
        except Exception as e:
            logger.error(f"❌ 保存结果失败: {e}")
            raise

    def save_detailed_report(self, result, timestamp):
        """保存详细报告"""
        try:
            report_file = os.path.join(self.results_dir, f"final_detailed_report_{timestamp}.txt")
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("最终模型验证报告\n")
                f.write("=" * 80 + "\n\n")
                
                f.write(f"验证时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"模型路径: {self.model_path}\n")
                f.write(f"数据路径: {self.data_path}\n")
                f.write(f"词汇表大小: {len(self.vocab)}\n")
                f.write(f"类别数量: {len(self.label_encoder.classes_)}\n\n")
                
                # 验证结果
                f.write("📊 验证结果\n")
                f.write("-" * 40 + "\n")
                f.write(f"准确率: {result['accuracy']:.4f}\n")
                f.write(f"F1分数: {result['f1_score']:.4f}\n\n")
                
                # 分类报告
                f.write("🎯 分类报告\n")
                f.write("-" * 40 + "\n")
                for class_name in result['class_names']:
                    if class_name in result['classification_report']:
                        class_metrics = result['classification_report'][class_name]
                        f.write(f"\n类别: {class_name}\n")
                        f.write(f"  精确率: {class_metrics['precision']:.4f}\n")
                        f.write(f"  召回率: {class_metrics['recall']:.4f}\n")
                        f.write(f"  F1分数: {class_metrics['f1-score']:.4f}\n")
                        f.write(f"  支持数: {class_metrics['support']}\n")
                
                f.write("\n" + "=" * 80 + "\n")
                f.write("报告结束\n")
                f.write("=" * 80 + "\n")
            
            logger.info(f"📄 详细报告已保存到: {report_file}")
            
        except Exception as e:
            logger.error(f"❌ 保存详细报告失败: {e}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="最终模型运行器")
    parser.add_argument("--model_path", type=str, required=True, help="训练好的模型路径")
    parser.add_argument("--data_path", type=str, required=True, help="验证数据路径")
    
    args = parser.parse_args()
    
    # 创建模型运行器
    runner = FinalModelRunner(
        model_path=args.model_path,
        data_path=args.data_path
    )
    
    # 运行验证
    result = runner.run_validation()
    
    # 输出摘要
    print("\n" + "="*60)
    print("🎯 验证完成摘要")
    print("="*60)
    print(f"📊 准确率: {result['accuracy']:.4f}")
    print(f"📊 F1分数: {result['f1_score']:.4f}")
    print(f"📁 结果保存位置: {runner.results_dir}/")
    print("="*60)

if __name__ == "__main__":
    main()
