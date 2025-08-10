#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全面模型验证器
对训练好的特征增强模型进行多次验证，使用全部数据集，并分类别输出准确性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import logging
import json
import os
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ComprehensiveModelValidator:
    """全面模型验证器"""
    
    def __init__(self, model_path, data_path, results_dir="validation_results"):
        self.model_path = model_path
        self.data_path = data_path
        self.results_dir = results_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建结果目录
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 初始化组件
        self.model = None
        self.label_encoder = None
        self.struct_extractor = None
        self.tfidf_vectorizer = None
        self.scaler = None
        
        logger.info(f"🚀 初始化验证器，使用设备: {self.device}")
    
    def load_model_and_components(self):
        """加载模型和相关组件"""
        try:
            # 加载模型
            logger.info("📥 加载训练好的模型...")
            self.model = torch.load(self.model_path, map_location=self.device)
            self.model.eval()
            
            # 加载标签编码器
            label_encoder_path = self.model_path.replace('.pth', '_label_encoder.pkl')
            if os.path.exists(label_encoder_path):
                import pickle
                with open(label_encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
                logger.info(f"✅ 标签编码器已加载，类别数: {len(self.label_encoder.classes_)}")
            
            # 加载结构化特征提取器
            struct_path = self.model_path.replace('.pth', '_struct_extractor.pkl')
            if os.path.exists(struct_path):
                import pickle
                with open(struct_path, 'rb') as f:
                    self.struct_extractor = pickle.load(f)
                logger.info("✅ 结构化特征提取器已加载")
            
            # 加载TF-IDF向量器
            tfidf_path = self.model_path.replace('.pth', '_tfidf_vectorizer.pkl')
            if os.path.exists(tfidf_path):
                import pickle
                with open(tfidf_path, 'rb') as f:
                    self.tfidf_vectorizer = pickle.load(f)
                logger.info("✅ TF-IDF向量器已加载")
            
            # 加载标准化器
            scaler_path = self.model_path.replace('.pth', '_scaler.pkl')
            if os.path.exists(scaler_path):
                import pickle
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info("✅ 标准化器已加载")
            
            logger.info("🎯 所有组件加载完成")
            
        except Exception as e:
            logger.error(f"❌ 加载模型失败: {e}")
            raise
    
    def load_and_prepare_data(self):
        """加载和准备验证数据"""
        try:
            logger.info(f"📊 加载数据: {self.data_path}")
            
            # 加载数据
            if self.data_path.endswith('.csv'):
                df = pd.read_csv(self.data_path)
            elif self.data_path.endswith('.json'):
                df = pd.read_json(self.data_path)
            else:
                raise ValueError("不支持的数据格式，请使用CSV或JSON文件")
            
            logger.info(f"📈 数据加载完成，总记录数: {len(df)}")
            logger.info(f"📋 数据列: {list(df.columns)}")
            
            # 检查必要列
            required_columns = ['cleaned_log', 'category']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"缺少必要列: {missing_columns}")
            
            # 数据基本信息
            logger.info(f"🎯 类别分布:")
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
            
            # 结构化特征
            if self.struct_extractor:
                struct_features = self.struct_extractor.extract_structured_features(df)
                logger.info(f"📊 结构化特征维度: {struct_features.shape}")
            else:
                # 如果没有预训练的结构化特征提取器，创建基本特征
                struct_features = self.create_basic_features(df)
                logger.info(f"📊 基本特征维度: {struct_features.shape}")
            
            # TF-IDF特征
            if self.tfidf_vectorizer:
                tfidf_features = self.tfidf_vectorizer.transform(texts).toarray()
                logger.info(f"📝 TF-IDF特征维度: {tfidf_features.shape}")
            else:
                # 创建基本的TF-IDF特征
                tfidf_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
                tfidf_features = tfidf_vectorizer.fit_transform(texts).toarray()
                logger.info(f"📝 基本TF-IDF特征维度: {tfidf_features.shape}")
            
            # 合并特征
            if struct_features is not None:
                combined_features = np.hstack([struct_features, tfidf_features])
            else:
                combined_features = tfidf_features
            
            logger.info(f"🔗 合并后特征维度: {combined_features.shape}")
            
            return texts, combined_features
            
        except Exception as e:
            logger.error(f"❌ 特征提取失败: {e}")
            raise
    
    def create_basic_features(self, df):
        """创建基本特征"""
        features = {}
        
        # 日志长度特征
        features['log_length'] = df['cleaned_log'].str.len().fillna(0)
        features['word_count'] = df['cleaned_log'].str.split().str.len().fillna(0)
        
        # 特殊字符计数
        features['special_char_count'] = df['cleaned_log'].str.count(r'[!@#$%^&*()_+\-=\[\]{};\':"\\|,.<>/?]').fillna(0)
        features['uppercase_count'] = df['cleaned_log'].str.count(r'[A-Z]').fillna(0)
        features['digit_count'] = df['cleaned_log'].str.count(r'\d').fillna(0)
        
        # 转换为DataFrame
        feature_df = pd.DataFrame(features)
        
        # 标准化
        if self.scaler:
            feature_array = self.scaler.transform(feature_df)
        else:
            feature_array = feature_df.values
        
        return feature_array
    
    def prepare_labels(self, df):
        """准备标签"""
        try:
            if self.label_encoder:
                labels = self.label_encoder.transform(df['category'])
                logger.info(f"🏷️ 标签编码完成，类别数: {len(self.label_encoder.classes_)}")
            else:
                # 如果没有预训练的标签编码器，创建新的
                self.label_encoder = LabelEncoder()
                labels = self.label_encoder.fit_transform(df['category'])
                logger.info(f"🏷️ 新标签编码器创建完成，类别数: {len(self.label_encoder.classes_)}")
            
            return labels
            
        except Exception as e:
            logger.error(f"❌ 标签准备失败: {e}")
            raise
    
    def validate_model(self, texts, features, labels, validation_name="full_dataset"):
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
                outputs = self.model(text_tensor, feature_tensor)
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()
                probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
            
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
                'probabilities': probabilities,
                'true_labels': labels,
                'classification_report': class_report,
                'confusion_matrix': conf_matrix,
                'class_names': self.label_encoder.classes_
            }
            
        except Exception as e:
            logger.error(f"❌ 模型验证失败: {e}")
            raise
    
    def text_to_sequence(self, text, max_length=100):
        """将文本转换为序列"""
        # 简单的分词和索引化
        words = text.split()[:max_length]
        # 这里使用简单的哈希方法，实际应用中应该使用预训练的词汇表
        sequence = [hash(word) % 10000 for word in words]
        # 填充到固定长度
        if len(sequence) < max_length:
            sequence.extend([0] * (max_length - len(sequence)))
        return sequence[:max_length]
    
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
    
    def analyze_validation_results(self, validation_results):
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
    
    def generate_per_category_analysis(self, validation_results):
        """生成每个类别的详细分析"""
        logger.info("🎯 生成每个类别的详细分析...")
        
        # 使用最后一次验证的结果进行详细分析
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
            # 创建详细报告
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
    
    def create_visualizations(self, validation_results, stats, category_analysis):
        """创建可视化图表"""
        try:
            logger.info("📊 创建可视化图表...")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 1. 验证结果趋势图
            plt.figure(figsize=(12, 8))
            
            # 准确率趋势
            plt.subplot(2, 2, 1)
            accuracies = [result['accuracy'] for result in validation_results]
            plt.plot(range(1, len(accuracies) + 1), accuracies, 'bo-', linewidth=2, markersize=8)
            plt.axhline(y=stats['accuracy']['mean'], color='r', linestyle='--', label=f'均值: {stats["accuracy"]["mean"]:.4f}')
            plt.fill_between(range(1, len(accuracies) + 1), 
                           [stats['accuracy']['mean'] - stats['accuracy']['std']] * len(accuracies),
                           [stats['accuracy']['mean'] + stats['accuracy']['std']] * len(accuracies),
                           alpha=0.3, color='r', label=f'±1标准差')
            plt.xlabel('验证次数')
            plt.ylabel('准确率')
            plt.title('验证准确率趋势')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # F1分数趋势
            plt.subplot(2, 2, 2)
            f1_scores = [result['f1_score'] for result in validation_results]
            plt.plot(range(1, len(f1_scores) + 1), f1_scores, 'go-', linewidth=2, markersize=8)
            plt.axhline(y=stats['f1_score']['mean'], color='r', linestyle='--', label=f'均值: {stats["f1_score"]["mean"]:.4f}')
            plt.fill_between(range(1, len(f1_scores) + 1), 
                           [stats['f1_score']['mean'] - stats['f1_score']['std']] * len(f1_scores),
                           [stats['f1_score']['mean'] + stats['f1_score']['std']] * len(f1_scores),
                           alpha=0.3, color='r', label=f'±1标准差')
            plt.xlabel('验证次数')
            plt.ylabel('F1分数')
            plt.title('验证F1分数趋势')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 类别准确率分布
            plt.subplot(2, 2, 3)
            categories = list(category_analysis.keys())
            category_accuracies = [category_analysis[cat]['accuracy'] for cat in categories]
            plt.bar(categories, category_accuracies, color='skyblue', alpha=0.7)
            plt.xlabel('类别')
            plt.ylabel('准确率')
            plt.title('各类别准确率')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, alpha=0.3)
            
            # 类别样本数量分布
            plt.subplot(2, 2, 4)
            category_counts = [category_analysis[cat]['sample_count'] for cat in categories]
            plt.bar(categories, category_counts, color='lightgreen', alpha=0.7)
            plt.xlabel('类别')
            plt.ylabel('样本数量')
            plt.title('各类别样本数量')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图表
            plot_file = os.path.join(self.results_dir, f"validation_plots_{timestamp}.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"📊 可视化图表已保存到: {plot_file}")
            
            # 2. 混淆矩阵热力图
            final_result = validation_results[-1]
            plt.figure(figsize=(10, 8))
            sns.heatmap(final_result['confusion_matrix'], 
                       annot=True, fmt='d', cmap='Blues',
                       xticklabels=final_result['class_names'],
                       yticklabels=final_result['class_names'])
            plt.title('混淆矩阵热力图')
            plt.xlabel('预测类别')
            plt.ylabel('真实类别')
            
            # 保存混淆矩阵
            cm_file = os.path.join(self.results_dir, f"confusion_matrix_{timestamp}.png")
            plt.savefig(cm_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"🔥 混淆矩阵已保存到: {cm_file}")
            
        except Exception as e:
            logger.error(f"❌ 创建可视化图表失败: {e}")
    
    def run_comprehensive_validation(self, num_validations=5):
        """运行全面验证"""
        try:
            logger.info("🚀 开始全面模型验证...")
            
            # 1. 加载模型和组件
            self.load_model_and_components()
            
            # 2. 加载数据
            df = self.load_and_prepare_data()
            
            # 3. 执行多次验证
            validation_results = self.perform_multiple_validations(df, num_validations)
            
            # 4. 分析结果
            stats = self.analyze_validation_results(validation_results)
            
            # 5. 生成类别分析
            category_analysis = self.generate_per_category_analysis(validation_results)
            
            # 6. 保存结果
            results_file = self.save_results(validation_results, stats, category_analysis)
            
            # 7. 创建可视化
            self.create_visualizations(validation_results, stats, category_analysis)
            
            logger.info("🎉 全面验证完成！")
            
            return {
                'validation_results': validation_results,
                'statistics': stats,
                'category_analysis': category_analysis,
                'results_file': results_file
            }
            
        except Exception as e:
            logger.error(f"❌ 全面验证失败: {e}")
            raise

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="全面模型验证器")
    parser.add_argument("--model_path", type=str, required=True, help="训练好的模型路径")
    parser.add_argument("--data_path", type=str, required=True, help="验证数据路径")
    parser.add_argument("--num_validations", type=int, default=5, help="验证次数")
    parser.add_argument("--results_dir", type=str, default="validation_results", help="结果保存目录")
    
    args = parser.parse_args()
    
    # 创建验证器
    validator = ComprehensiveModelValidator(
        model_path=args.model_path,
        data_path=args.data_path,
        results_dir=args.results_dir
    )
    
    # 运行验证
    results = validator.run_comprehensive_validation(args.num_validations)
    
    # 输出摘要
    print("\n" + "="*60)
    print("🎯 验证完成摘要")
    print("="*60)
    print(f"📊 平均准确率: {results['statistics']['accuracy']['mean']:.4f} ± {results['statistics']['accuracy']['std']:.4f}")
    print(f"📊 平均F1分数: {results['statistics']['f1_score']['mean']:.4f} ± {results['statistics']['f1_score']['std']:.4f}")
    print(f"📁 结果保存位置: {results['results_dir']}")
    print("="*60)

if __name__ == "__main__":
    main() 