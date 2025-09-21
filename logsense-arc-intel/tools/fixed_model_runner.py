# -*- coding: utf-8 -*-
"""
修复版模型验证脚本
使用训练时保存的词汇表，避免词汇表大小不匹配问题
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
import logging
import argparse
import os
from datetime import datetime
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pickle

# 导入自定义模块
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import ModelFactory
from data import LogPreprocessor
from utils import ArcGPUDetector

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FixedModelRunner:
    """修复版模型验证器"""
    
    def __init__(self, model_path: str, data_path: str):
        self.model_path = model_path
        self.data_path = data_path
        self.device = None
        self.model = None
        self.vocab = None
        self.label_encoder = None
        self.checkpoint = None
        
        # 检测设备
        self.detect_device()
        
    def detect_device(self):
        """检测计算设备"""
        if torch.xpu.is_available():
            self.device = torch.device("xpu:0")
            logger.info("🎮 检测到Intel Arc GPU，使用XPU设备")
        else:
            self.device = torch.device("cpu")
            logger.warning("⚠️ 使用CPU设备")
        
        logger.info(f"🚀 初始化修复版模型验证器，使用设备: {self.device}")
    
    def load_model_weights(self):
        """加载模型权重和词汇表"""
        try:
            logger.info("📥 加载模型权重...")
            
            # 加载checkpoint
            self.checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            # 加载词汇表
            if 'vocab' in self.checkpoint:
                self.vocab = self.checkpoint['vocab']
                logger.info(f"📚 词汇表加载完成，大小: {len(self.vocab)}")
            else:
                logger.error("❌ 未找到词汇表")
                raise ValueError("模型文件中未包含词汇表")
            
            # 加载标签编码器
            if 'label_encoder' in self.checkpoint:
                self.label_encoder = self.checkpoint['label_encoder']
                logger.info(f"🏷️ 标签编码器加载完成，类别数: {len(self.label_encoder.classes_)}")
            else:
                logger.error("❌ 未找到标签编码器")
                raise ValueError("模型文件中未包含标签编码器")
            
            logger.info("✅ 模型权重加载完成")
            
        except Exception as e:
            logger.error(f"❌ 模型权重加载失败: {e}")
            raise
    
    def text_to_sequence(self, text: str, vocab: dict, max_length: int = 100) -> list:
        """将文本转换为序列"""
        words = text.lower().split()
        sequence = []
        
        for word in words[:max_length]:
            if word in vocab:
                sequence.append(vocab[word])
            else:
                sequence.append(vocab.get('<UNK>', 0))
        
        # 填充到固定长度
        while len(sequence) < max_length:
            sequence.append(vocab.get('<PAD>', 0))
        
        return sequence[:max_length]
    
    def extract_structured_features(self, df: pd.DataFrame) -> np.ndarray:
        """提取结构化特征"""
        logger.info("🔧 开始特征提取...")
        
        # 使用训练时的特征提取器
        preprocessor = LogPreprocessor()
        features = preprocessor.extract_structured_features(df['cleaned_log'].tolist())
        
        logger.info(f"🔗 特征提取完成，总维度: {features.shape}")
        return features
    
    def create_model(self):
        """创建模型结构"""
        try:
            logger.info("🔧 创建模型结构...")
            
            # 获取模型配置
            num_classes = len(self.label_encoder.classes_)
            vocab_size = len(self.vocab)
            
            # 创建模型
            model_config = {
                'vocab_size': vocab_size,
                'embedding_dim': 128,
                'num_filters': 128,
                'filter_sizes': [3, 4, 5],
                'num_classes': num_classes,
                'dropout': 0.5
            }
            
            self.model = ModelFactory.create_model('textcnn', **model_config)
            self.model.to(self.device)
            
            # 加载权重
            self.model.load_state_dict(self.checkpoint['model_state_dict'])
            
            logger.info("✅ 模型结构创建完成")
            
        except Exception as e:
            logger.error(f"❌ 创建模型失败: {e}")
            raise
    
    def load_and_prepare_data(self):
        """加载并准备数据"""
        try:
            logger.info(f"📊 加载数据: {self.data_path}")
            
            # 加载数据
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
            
            # 提取结构化特征
            features = self.extract_structured_features(df_cleaned)
            
            # 准备文本序列
            texts = [self.text_to_sequence(text, self.vocab) for text in df_cleaned['cleaned_log']]
            
            # 准备标签
            labels = self.label_encoder.transform(df_cleaned['category'])
            
            return df_cleaned, texts, features, labels
            
        except Exception as e:
            logger.error(f"❌ 数据加载失败: {e}")
            raise
    
    def validate_model(self, texts, features, labels, validation_name="validation"):
        """验证模型"""
        try:
            logger.info(f"🔍 开始模型验证: {validation_name}")
            
            # 准备数据
            text_tensor = torch.tensor(texts, dtype=torch.long)
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
    
    def save_results(self, results: dict):
        """保存验证结果"""
        try:
            # 创建结果目录
            results_dir = Path("final_validation_results")
            results_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 保存JSON结果
            json_path = results_dir / f"fixed_validation_results_{timestamp}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'timestamp': timestamp,
                    'model_path': self.model_path,
                    'data_path': self.data_path,
                    'vocab_size': len(self.vocab),
                    'num_classes': len(self.label_encoder.classes_),
                    'accuracy': results['accuracy'],
                    'f1_score': results['f1_score'],
                    'classification_report': results['classification_report'],
                    'confusion_matrix': results['confusion_matrix'].tolist(),
                    'class_names': results['class_names'].tolist()
                }, f, ensure_ascii=False, indent=2)
            
            # 保存详细报告
            report_path = results_dir / f"fixed_detailed_report_{timestamp}.txt"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("修复版模型验证报告\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"验证时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"模型路径: {self.model_path}\n")
                f.write(f"数据路径: {self.data_path}\n")
                f.write(f"词汇表大小: {len(self.vocab)}\n")
                f.write(f"类别数量: {len(self.label_encoder.classes_)}\n\n")
                
                f.write("📊 验证结果\n")
                f.write("-" * 40 + "\n")
                f.write(f"准确率: {results['accuracy']:.4f}\n")
                f.write(f"F1分数: {results['f1_score']:.4f}\n\n")
                
                f.write("🎯 分类报告\n")
                f.write("-" * 40 + "\n")
                for class_name in results['class_names']:
                    if class_name in results['classification_report']:
                        report = results['classification_report'][class_name]
                        f.write(f"\n类别: {class_name}\n")
                        f.write(f"  精确率: {report['precision']:.4f}\n")
                        f.write(f"  召回率: {report['recall']:.4f}\n")
                        f.write(f"  F1分数: {report['f1-score']:.4f}\n")
                        f.write(f"  支持数: {report['support']:.0f}\n")
            
            logger.info(f"💾 结果已保存到: {json_path}")
            logger.info(f"📄 详细报告已保存到: {report_path}")
            
        except Exception as e:
            logger.error(f"❌ 保存结果失败: {e}")
    
    def run_validation(self):
        """运行验证"""
        try:
            logger.info("🚀 开始修复版模型验证...")
            
            # 1. 加载模型权重（获取词汇表和标签编码器）
            self.load_model_weights()
            
            # 2. 加载并准备数据
            df, texts, features, labels = self.load_and_prepare_data()
            
            # 3. 创建模型结构
            self.create_model()
            
            # 4. 验证模型
            results = self.validate_model(texts, features, labels, "fixed_validation")
            
            # 5. 保存结果
            self.save_results(results)
            
            logger.info("🎉 修复版验证完成！")
            
            # 打印摘要
            print("\n" + "=" * 60)
            print("🎯 修复版验证完成摘要")
            print("=" * 60)
            print(f"📊 准确率: {results['accuracy']:.4f}")
            print(f"📊 F1分数: {results['f1_score']:.4f}")
            print(f"📁 结果保存位置: final_validation_results/")
            print("=" * 60)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ 修复版验证失败: {e}")
            raise

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='修复版模型验证脚本')
    parser.add_argument('--model_path', required=True, help='模型文件路径')
    parser.add_argument('--data_path', required=True, help='数据文件路径')
    
    args = parser.parse_args()
    
    try:
        # 创建验证器
        runner = FixedModelRunner(args.model_path, args.data_path)
        
        # 运行验证
        results = runner.run_validation()
        
    except Exception as e:
        logger.error(f"❌ 验证失败: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
