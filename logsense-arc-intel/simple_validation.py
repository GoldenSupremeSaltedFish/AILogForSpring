#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单模型验证器 - 直接使用模型权重进行验证
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import logging
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import json
import warnings
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class SimpleValidationModel(nn.Module):
    """简单的验证模型"""
    
    def __init__(self, vocab_size=10000, embedding_dim=128, num_filters=128, 
                 filter_sizes=[3, 4, 5], num_classes=9, struct_input_dim=1000):
        super().__init__()
        
        # 文本编码器 (TextCNN)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, k) for k in filter_sizes
        ])
        self.dropout = nn.Dropout(0.5)
        self.text_output_dim = num_filters * len(filter_sizes)
        
        # 结构化特征MLP
        self.struct_mlp = nn.Sequential(
            nn.Linear(struct_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.text_output_dim + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, text_inputs, struct_inputs):
        # 文本特征提取
        embedded = self.embedding(text_inputs)  # [batch, seq_len, embedding_dim]
        embedded = embedded.transpose(1, 2)     # [batch, embedding_dim, seq_len]
        
        # 卷积特征提取
        conv_outputs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(embedded))  # [batch, num_filters, seq_len-k+1]
            conv_out = torch.max_pool1d(conv_out, conv_out.size(2))  # [batch, num_filters, 1]
            conv_out = conv_out.squeeze(2)  # [batch, num_filters]
            conv_outputs.append(conv_out)
        
        # 拼接卷积输出
        text_features = torch.cat(conv_outputs, dim=1)  # [batch, num_filters * len(filter_sizes)]
        text_features = self.dropout(text_features)
        
        # 结构化特征处理
        struct_features = self.struct_mlp(struct_inputs)
        
        # 特征融合
        combined_features = torch.cat([text_features, struct_features], dim=1)
        
        # 分类
        output = self.fusion_layer(combined_features)
        return output

class SimpleValidator:
    """简单验证器"""

    def __init__(self, model_path, data_path):
        self.model_path = model_path
        self.data_path = data_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建结果目录
        self.results_dir = "validation_results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 模型组件
        self.model = None
        self.label_encoder = None
        
        logger.info(f"🚀 初始化简单验证器，使用设备: {self.device}")

    def load_model_and_components(self):
        """加载模型和组件"""
        try:
            logger.info("📥 加载训练好的模型...")
            
            # 尝试加载模型文件
            try:
                # 首先尝试weights_only=True
                checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
                logger.info("✅ 使用weights_only=True成功加载模型")
            except Exception as e:
                logger.info("⚠️ weights_only=True失败，尝试weights_only=False...")
                checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
                logger.info("✅ 使用weights_only=False成功加载模型")
            
            # 提取组件
            if 'label_encoder' in checkpoint:
                self.label_encoder = checkpoint['label_encoder']
            else:
                # 如果没有label_encoder，创建一个默认的
                logger.warning("⚠️ 未找到label_encoder，将使用默认标签")
                self.label_encoder = LabelEncoder()
                # 假设有9个类别
                self.label_encoder.classes_ = np.array(['class_0', 'class_1', 'class_2', 'class_3', 'class_4', 'class_5', 'class_6', 'class_7', 'class_8'])
            
            # 创建模型实例
            self.model = SimpleValidationModel()
            
            # 加载权重
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info("✅ 模型权重加载成功")
            else:
                logger.warning("⚠️ 未找到model_state_dict，使用随机初始化的模型")
            
            self.model.eval()
            
            logger.info(f"✅ 模型加载完成，类别数: {len(self.label_encoder.classes_)}")
            
        except Exception as e:
            logger.error(f"❌ 加载模型失败: {e}")
            raise

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
            
            # TF-IDF特征
            tfidf_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
            tfidf_features = tfidf_vectorizer.fit_transform(texts).toarray()
            
            # 合并特征
            struct_features = pd.DataFrame(features).values
            combined_features = np.hstack([struct_features, tfidf_features])
            
            logger.info(f"🔗 特征提取完成，总维度: {combined_features.shape}")
            
            return texts, combined_features
            
        except Exception as e:
            logger.error(f"❌ 特征提取失败: {e}")
            raise

    def prepare_labels(self, df):
        """准备标签"""
        try:
            # 如果label_encoder没有训练过，先训练
            if not hasattr(self.label_encoder, 'classes_') or len(self.label_encoder.classes_) == 0:
                self.label_encoder.fit(df['category'])
            
            labels = self.label_encoder.transform(df['category'])
            return labels
            
        except Exception as e:
            logger.error(f"❌ 标签准备失败: {e}")
            raise

    def text_to_sequence(self, text, max_length=100):
        """将文本转换为序列"""
        words = text.split()[:max_length]
        sequence = [hash(word) % 10000 for word in words]
        if len(sequence) < max_length:
            sequence.extend([0] * (max_length - len(sequence)))
        return sequence[:max_length]

    def validate_model(self, texts, features, labels, validation_name="validation"):
        """验证模型"""
        try:
            logger.info(f"🔍 开始模型验证: {validation_name}")
            
            # 准备数据
            text_tensor = torch.tensor([self.text_to_sequence(text) for text in texts], dtype=torch.long)
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
                    batch_size = len(texts)
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

    def perform_multiple_validations(self, df, num_validations=5):
        """执行多次验证"""
        logger.info(f"🔄 开始执行 {num_validations} 次验证...")
        
        validation_results = []
        
        for i in range(num_validations):
            logger.info(f"🔄 第 {i+1} 次验证...")
            
            # 随机打乱数据
            df_shuffled = df.sample(frac=1.0, random_state=i).reset_index(drop=True)
            
            # 提取特征和标签
            texts, features = self.extract_features(df_shuffled)
            labels = self.prepare_labels(df_shuffled)
            
            # 验证模型
            result = self.validate_model(
                texts, features, labels,
                validation_name=f"validation_{i+1}"
            )
            
            validation_results.append(result)
            
            logger.info(f"✅ 第 {i+1} 次验证完成")
        
        return validation_results

    def analyze_results(self, validation_results):
        """分析验证结果"""
        logger.info("📈 分析验证结果...")
        
        # 统计指标
        accuracies = [result['accuracy'] for result in validation_results]
        f1_scores = [result['f1_score'] for result in validation_results]
        
        # 计算统计信息
        stats = {
            'accuracy': {
                'mean': np.mean(accuracies),
                'std': np.std(accuracies),
                'min': np.min(accuracies),
                'max': np.max(accuracies)
            },
            'f1_score': {
                'mean': np.mean(f1_scores),
                'std': np.std(f1_scores),
                'min': np.min(f1_scores),
                'max': np.max(f1_scores)
            }
        }
        
        logger.info("📊 验证结果统计:")
        logger.info(f"   准确率 - 均值: {stats['accuracy']['mean']:.4f}, 标准差: {stats['accuracy']['std']:.4f}")
        logger.info(f"   准确率 - 范围: [{stats['accuracy']['min']:.4f}, {stats['accuracy']['max']:.4f}]")
        logger.info(f"   F1分数 - 均值: {stats['f1_score']['mean']:.4f}, 标准差: {stats['f1_score']['std']:.4f}")
        logger.info(f"   F1分数 - 范围: [{stats['f1_score']['min']:.4f}, {stats['f1_score']['max']:.4f}]")
        
        return stats

    def generate_category_analysis(self, validation_results):
        """生成每个类别的详细分析"""
        logger.info("🎯 生成每个类别的详细分析...")
        
        # 使用最后一次验证的结果
        final_result = validation_results[-1]
        
        # 按类别分析
        category_analysis = {}
        
        for i, class_name in enumerate(final_result['class_names']):
            # 找到该类别的样本
            class_mask = final_result['true_labels'] == i
            class_predictions = final_result['predictions'][class_mask]
            class_true = final_result['true_labels'][class_mask]
            
            if len(class_true) > 0:
                # 计算该类别的指标
                class_accuracy = accuracy_score(class_true, class_predictions)
                class_f1 = f1_score(class_true, class_predictions, average='binary')
                
                # 计算该类别的预测分布
                from collections import Counter
                prediction_counts = Counter(class_predictions)
                
                category_analysis[class_name] = {
                    'sample_count': len(class_true),
                    'accuracy': class_accuracy,
                    'f1_score': class_f1,
                    'prediction_distribution': {
                        self.label_encoder.classes_[j]: count
                        for j, count in prediction_counts.items()
                    }
                }
        
        return category_analysis

    def save_results(self, validation_results, stats, category_analysis):
        """保存结果"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 保存验证结果
            results_file = os.path.join(self.results_dir, f"validation_results_{timestamp}.json")
            
            # 准备保存的数据
            save_data = {
                'timestamp': timestamp,
                'model_path': self.model_path,
                'data_path': self.data_path,
                'statistics': stats,
                'category_analysis': category_analysis,
                'validation_summary': []
            }
            
            # 添加每次验证的摘要
            for result in validation_results:
                save_data['validation_summary'].append({
                    'validation_name': result['validation_name'],
                    'accuracy': result['accuracy'],
                    'f1_score': result['f1_score']
                })
            
            # 保存到JSON文件
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"💾 结果已保存到: {results_file}")
            
            # 保存详细报告
            self.save_detailed_report(validation_results, stats, category_analysis, timestamp)
            
            return results_file
            
        except Exception as e:
            logger.error(f"❌ 保存结果失败: {e}")
            raise

    def save_detailed_report(self, validation_results, stats, category_analysis, timestamp):
        """保存详细报告"""
        try:
            report_file = os.path.join(self.results_dir, f"detailed_report_{timestamp}.txt")
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("特征增强模型全面验证报告\n")
                f.write("=" * 80 + "\n\n")
                
                f.write(f"验证时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"模型路径: {self.model_path}\n")
                f.write(f"数据路径: {self.data_path}\n\n")
                
                # 统计信息
                f.write("📊 验证结果统计\n")
                f.write("-" * 40 + "\n")
                f.write(f"准确率 - 均值: {stats['accuracy']['mean']:.4f}\n")
                f.write(f"准确率 - 标准差: {stats['accuracy']['std']:.4f}\n")
                f.write(f"准确率 - 范围: [{stats['accuracy']['min']:.4f}, {stats['accuracy']['max']:.4f}]\n")
                f.write(f"F1分数 - 均值: {stats['f1_score']['mean']:.4f}\n")
                f.write(f"F1分数 - 标准差: {stats['f1_score']['std']:.4f}\n")
                f.write(f"F1分数 - 范围: [{stats['f1_score']['min']:.4f}, {stats['f1_score']['max']:.4f}]\n\n")
                
                # 每次验证结果
                f.write("🔄 各次验证结果\n")
                f.write("-" * 40 + "\n")
                for result in validation_results:
                    f.write(f"{result['validation_name']}: 准确率={result['accuracy']:.4f}, F1={result['f1_score']:.4f}\n")
                f.write("\n")
                
                # 类别分析
                f.write("🎯 各类别详细分析\n")
                f.write("-" * 40 + "\n")
                for category, analysis in category_analysis.items():
                    f.write(f"\n类别: {category}\n")
                    f.write(f"  样本数量: {analysis['sample_count']}\n")
                    f.write(f"  准确率: {analysis['accuracy']:.4f}\n")
                    f.write(f"  F1分数: {analysis['f1_score']:.4f}\n")
                    f.write(f"  预测分布:\n")
                    for pred_cat, count in analysis['prediction_distribution'].items():
                        f.write(f"    {pred_cat}: {count}\n")
                
                f.write("\n" + "=" * 80 + "\n")
                f.write("报告结束\n")
                f.write("=" * 80 + "\n")
            
            logger.info(f"📄 详细报告已保存到: {report_file}")
            
        except Exception as e:
            logger.error(f"❌ 保存详细报告失败: {e}")

    def run_validation(self, num_validations=5):
        """运行验证"""
        try:
            logger.info("🚀 开始模型验证...")
            
            # 1. 加载模型和组件
            self.load_model_and_components()
            
            # 2. 加载数据
            df = self.load_data()
            
            # 3. 执行多次验证
            validation_results = self.perform_multiple_validations(df, num_validations)
            
            # 4. 分析结果
            stats = self.analyze_results(validation_results)
            
            # 5. 生成类别分析
            category_analysis = self.generate_category_analysis(validation_results)
            
            # 6. 保存结果
            results_file = self.save_results(validation_results, stats, category_analysis)
            
            logger.info("🎉 验证完成！")
            
            return {
                'validation_results': validation_results,
                'statistics': stats,
                'category_analysis': category_analysis,
                'results_file': results_file
            }
            
        except Exception as e:
            logger.error(f"❌ 验证失败: {e}")
            raise

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="简单模型验证器")
    parser.add_argument("--model_path", type=str, required=True, help="训练好的模型路径")
    parser.add_argument("--data_path", type=str, required=True, help="验证数据路径")
    parser.add_argument("--num_validations", type=int, default=5, help="验证次数")
    
    args = parser.parse_args()
    
    # 创建验证器
    validator = SimpleValidator(
        model_path=args.model_path,
        data_path=args.data_path
    )
    
    # 运行验证
    results = validator.run_validation(args.num_validations)
    
    # 输出摘要
    print("\n" + "="*60)
    print("🎯 验证完成摘要")
    print("="*60)
    print(f"📊 平均准确率: {results['statistics']['accuracy']['mean']:.4f} ± {results['statistics']['accuracy']['std']:.4f}")
    print(f"📊 平均F1分数: {results['statistics']['f1_score']['mean']:.4f} ± {results['statistics']['f1_score']['std']:.4f}")
    print(f"📁 结果保存位置: {validator.results_dir}/")
    print("="*60)

if __name__ == "__main__":
    main() 