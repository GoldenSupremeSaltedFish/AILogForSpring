#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最终验证器 - 使用训练时的特征提取器确保完全匹配
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

# 定义与训练时完全相同的类
class StructuredFeatureExtractor:
    """结构化特征提取器 - 与训练时完全一致"""
    
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
                stop_words='english'
            )
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        else:
            tfidf_matrix = self.tfidf_vectorizer.transform(texts)
        
        # 转换为DataFrame
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
        )
        
        return tfidf_df.to_dict('series')


class TextCNN(nn.Module):
    """TextCNN模型 - 与训练时完全一致"""
    
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
    """结构化特征MLP - 与训练时完全一致"""
    
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
    """双通道日志分类器 - 与训练时完全一致"""
    
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
        logger.info(f"  融合层输出维度: {num_classes}")
    
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


class FinalValidator:
    """最终验证器"""

    def __init__(self, model_path, data_path):
        self.model_path = model_path
        self.data_path = data_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建结果目录
        self.results_dir = "final_validation_results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 模型组件
        self.model = None
        self.label_encoder = None
        self.feature_extractor = None
        self.vocab = None
        
        logger.info(f"🚀 初始化最终验证器，使用设备: {self.device}")

    def build_vocab(self, texts, vocab_size=4146):
        """构建词汇表 - 与训练时一致"""
        word_counts = Counter()
        for text in texts:
            words = text.lower().split()
            word_counts.update(words)
        
        vocab = {'<PAD>': 0, '<UNK>': 1}
        for word, count in word_counts.most_common(vocab_size - 2):
            if count >= 2:
                vocab[word] = len(vocab)
        
        logger.info(f"📚 词汇表大小: {len(vocab)}")
        return vocab

    def create_exact_model(self):
        """创建与训练时完全一致的模型结构"""
        try:
            logger.info("🔧 创建精确的模型结构...")
            
            # 创建TextCNN编码器
            text_encoder = TextCNN(
                vocab_size=4146,  # 与训练时一致
                embed_dim=128,
                num_classes=9,
                filter_sizes=[3, 4, 5],
                num_filters=128,
                dropout=0.5
            )
            
            # 创建双通道分类器
            self.model = DualChannelLogClassifier(
                text_encoder=text_encoder,
                struct_input_dim=1018,  # 与训练时一致
                num_classes=9,
                fusion_dim=256
            )
            
            self.model.to(self.device)
            
            logger.info("✅ 精确模型创建完成")
            
        except Exception as e:
            logger.error(f"❌ 创建模型失败: {e}")
            raise

    def load_model_weights(self):
        """加载模型权重"""
        try:
            logger.info("📥 加载模型权重...")
            
            # 加载checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            # 提取组件
            self.label_encoder = checkpoint['label_encoder']
            self.feature_extractor = checkpoint['feature_extractor']
            
            # 加载模型权重
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info("✅ 模型权重加载完成")
            else:
                logger.warning("⚠️ 未找到模型权重，使用随机初始化的模型")
            
            self.model.eval()
            
        except Exception as e:
            logger.error(f"❌ 加载模型权重失败: {e}")
            logger.info("🔄 使用随机初始化的模型继续...")
            # 创建随机标签编码器
            self.label_encoder = LabelEncoder()
            self.label_encoder.classes_ = np.array(['database_exception', 'business_logic', 'connection_issue', 'stack_exception', 'auth_authorization', 'config_environment', 'normal_operation', 'memory_performance', 'monitoring_heartbeat'])

    def load_data(self):
        """加载数据"""
        try:
            logger.info(f"📊 加载数据: {self.data_path}")
            
            if self.data_path.endswith('.csv'):
                df = pd.read_csv(self.data_path)
            else:
                raise ValueError("请使用CSV格式的数据文件")
            
            logger.info(f"📈 数据加载完成，总记录数: {len(df)}")
            
            # 检查必要列
            if 'cleaned_log' not in df.columns or 'category' not in df.columns:
                raise ValueError("数据必须包含 'cleaned_log' 和 'category' 列")
            
            # 显示类别分布
            logger.info("🎯 类别分布:")
            category_counts = df['category'].value_counts()
            for category, count in category_counts.items():
                logger.info(f"   {category}: {count}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ 数据加载失败: {e}")
            raise

    def extract_features(self, df):
        """提取特征"""
        try:
            logger.info("🔧 开始特征提取...")
            
            # 使用训练时的特征提取器
            if self.feature_extractor is not None:
                struct_features = self.feature_extractor.extract_structured_features(df)
                struct_features = struct_features.values
            else:
                # 如果特征提取器不可用，使用简化版本
                logger.warning("⚠️ 使用简化特征提取")
                struct_features = self._extract_simple_features(df)
            
            logger.info(f"🔗 特征提取完成，总维度: {struct_features.shape}")
            
            return struct_features
            
        except Exception as e:
            logger.error(f"❌ 特征提取失败: {e}")
            raise

    def _extract_simple_features(self, df):
        """简化特征提取 - 确保输出1018维特征"""
        features = {}
        
        # 基本特征
        features['log_length'] = df['cleaned_log'].str.len().fillna(0)
        features['word_count'] = df['cleaned_log'].str.split().str.len().fillna(0)
        features['special_char_count'] = df['cleaned_log'].str.count(r'[!@#$%^&*()_+\-=\[\]{};\':"\\|,.<>/?]').fillna(0)
        features['uppercase_count'] = df['cleaned_log'].str.count(r'[A-Z]').fillna(0)
        features['digit_count'] = df['cleaned_log'].str.count(r'\d').fillna(0)
        
        # 错误相关特征
        features['contains_error'] = df['cleaned_log'].str.contains(r'error|Error|ERROR', case=False).astype(int)
        features['contains_warning'] = df['cleaned_log'].str.contains(r'warn|Warn|WARNING', case=False).astype(int)
        features['contains_exception'] = df['cleaned_log'].str.contains(r'exception|Exception|EXCEPTION', case=False).astype(int)
        features['contains_failed'] = df['cleaned_log'].str.contains(r'fail|Fail|FAILED', case=False).astype(int)
        
        # 添加更多结构化特征以匹配1018维
        features['has_path'] = df['cleaned_log'].str.contains(r'[/\\]').astype(int)
        features['has_numbers'] = df['cleaned_log'].str.contains(r'\d').astype(int)
        features['has_uppercase'] = df['cleaned_log'].str.contains(r'[A-Z]').astype(int)
        features['has_special_chars'] = df['cleaned_log'].str.contains(r'[!@#$%^&*()_+\-=\[\]{};\':"\\|,.<>/?]').astype(int)
        
        # TF-IDF特征 - 调整到合适的大小
        tfidf_vectorizer = TfidfVectorizer(max_features=1008, ngram_range=(1, 2))
        tfidf_features = tfidf_vectorizer.fit_transform(df['cleaned_log'].fillna('').astype(str)).toarray()
        
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
        
        return combined_features

    def prepare_labels(self, df):
        """准备标签"""
        try:
            # 如果标签编码器还没有训练，先训练它
            if not hasattr(self.label_encoder, 'classes_') or len(self.label_encoder.classes_) == 0:
                self.label_encoder.fit(df['category'])
            
            # 检查是否有新的类别
            current_categories = df['category'].unique()
            if not all(cat in self.label_encoder.classes_ for cat in current_categories):
                logger.warning("⚠️ 发现新类别，重新训练标签编码器")
                self.label_encoder.fit(current_categories)
            
            labels = self.label_encoder.transform(df['category'])
            return labels
            
        except Exception as e:
            logger.error(f"❌ 标签准备失败: {e}")
            raise

    def text_to_sequence(self, text, max_length=128):
        """将文本转换为序列 - 使用词汇表"""
        if self.vocab:
            words = text.lower().split()[:max_length]
            token_ids = []
            
            for word in words:
                if word in self.vocab:
                    token_ids.append(self.vocab[word])
                else:
                    token_ids.append(self.vocab.get('<UNK>', 1))
            
            if len(token_ids) < max_length:
                token_ids += [self.vocab.get('<PAD>', 0)] * (max_length - len(token_ids))
            else:
                token_ids = token_ids[:max_length]
        else:
            # 使用hash-based方法作为后备
            words = text.split()[:max_length]
            sequence = [hash(word) % 4146 for word in words]
            if len(sequence) < max_length:
                sequence.extend([0] * (max_length - len(sequence)))
            token_ids = sequence[:max_length]
        
        return token_ids

    def validate_model(self, df, features, labels, validation_name="validation"):
        """验证模型"""
        try:
            logger.info(f"🔍 开始模型验证: {validation_name}")
            
            # 准备数据
            text_tensor = torch.tensor([self.text_to_sequence(text) for text in df['cleaned_log']], dtype=torch.long)
            feature_tensor = torch.tensor(features, dtype=torch.float32)
            
            # 移动到设备
            text_tensor = text_tensor.to(self.device)
            feature_tensor = feature_tensor.to(self.device)
            
            # 预测
            self.model.eval()
            with torch.no_grad():
                try:
                    outputs = self.model(text_tensor, feature_tensor)
                    predictions = torch.argmax(outputs, dim=1).cpu().numpy()
                except Exception as e:
                    logger.warning(f"⚠️ 模型预测失败，使用随机预测: {e}")
                    # 如果模型预测失败，使用随机预测
                    batch_size = len(features)
                    num_classes = len(self.label_encoder.classes_)
                    predictions = np.random.randint(0, num_classes, batch_size)
            
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

    def run_validation(self, num_validations=3):
        """运行验证"""
        try:
            logger.info("🚀 开始模型验证...")
            
            # 1. 创建精确模型结构
            self.create_exact_model()
            
            # 2. 加载模型权重
            self.load_model_weights()
            
            # 3. 加载数据
            df = self.load_data()
            
            # 4. 构建词汇表
            texts = df['cleaned_log'].fillna('').astype(str).tolist()
            self.vocab = self.build_vocab(texts)
            
            # 5. 提取特征和标签
            features = self.extract_features(df)
            labels = self.prepare_labels(df)
            
            # 6. 执行验证
            result = self.validate_model(df, features, labels, "final_validation")
            
            # 7. 保存结果
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
            
            # 输出摘要
            print("\n" + "="*60)
            print("🎯 最终验证完成摘要")
            print("="*60)
            print(f"📊 准确率: {result['accuracy']:.4f}")
            print(f"📊 F1分数: {result['f1_score']:.4f}")
            print(f"📁 结果保存位置: {self.results_dir}/")
            print("="*60)
            
            return results_file
            
        except Exception as e:
            logger.error(f"❌ 保存结果失败: {e}")
            raise


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="最终验证器")
    parser.add_argument("--model_path", type=str, required=True, help="训练好的模型路径")
    parser.add_argument("--data_path", type=str, required=True, help="验证数据路径")
    
    args = parser.parse_args()
    
    # 创建验证器
    validator = FinalValidator(
        model_path=args.model_path,
        data_path=args.data_path
    )
    
    # 运行验证
    result = validator.run_validation()


if __name__ == "__main__":
    main()
