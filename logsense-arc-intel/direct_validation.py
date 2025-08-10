#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接验证训练好的特征增强模型
"""

import sys
import os
import torch
import pandas as pd
import numpy as np
import logging
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """主函数"""
    try:
        logger.info("🚀 开始直接验证训练好的模型...")
        
        # 导入训练脚本中的类
        from feature_enhanced_model import DualChannelLogClassifier, StructuredFeatureExtractor
        
        # 模型路径
        model_path = "results/models/feature_enhanced_model_20250809_103658.pth"
        data_path = "data/processed_logs_advanced_enhanced.csv"
        
        logger.info(f"📥 加载模型: {model_path}")
        
        # 加载模型
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # 提取组件
        label_encoder = checkpoint['label_encoder']
        feature_extractor = checkpoint['feature_extractor']
        model_state_dict = checkpoint['model_state_dict']
        
        logger.info(f"✅ 模型组件加载完成，类别数: {len(label_encoder.classes_)}")
        
        # 重新创建模型
        model = DualChannelLogClassifier(
            vocab_size=10000,
            embedding_dim=128,
            num_filters=128,
            filter_sizes=[3, 4, 5],
            num_classes=len(label_encoder.classes_),
            struct_input_dim=1018
        )
        
        # 加载权重
        model.load_state_dict(model_state_dict)
        model.eval()
        
        logger.info("🔧 模型重建完成")
        
        # 加载数据
        logger.info(f"📊 加载数据: {data_path}")
        df = pd.read_csv(data_path)
        logger.info(f"📈 数据加载完成，总记录数: {len(df)}")
        
        # 显示类别分布
        logger.info("🎯 类别分布:")
        category_counts = df['category'].value_counts()
        for category, count in category_counts.items():
            logger.info(f"   {category}: {count}")
        
        # 执行验证
        logger.info("🔍 开始模型验证...")
        
        # 提取特征
        texts = df['cleaned_log'].fillna('').astype(str).tolist()
        features = feature_extractor.extract_features(df)
        labels = label_encoder.transform(df['category'])
        
        # 准备输入
        text_tensor = torch.tensor([feature_extractor.text_to_sequence(text) for text in texts], dtype=torch.long)
        feature_tensor = torch.tensor(features, dtype=torch.float32)
        
        # 预测
        with torch.no_grad():
            outputs = model(text_tensor, feature_tensor)
            predictions = torch.argmax(outputs, dim=1).numpy()
        
        # 计算指标
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='weighted')
        
        # 分类报告
        class_report = classification_report(
            labels, predictions,
            target_names=label_encoder.classes_,
            output_dict=True
        )
        
        # 混淆矩阵
        conf_matrix = confusion_matrix(labels, predictions)
        
        # 输出结果
        logger.info("📊 验证结果:")
        logger.info(f"   准确率: {accuracy:.4f}")
        logger.info(f"   F1分数: {f1:.4f}")
        
        # 按类别分析
        logger.info("\n🎯 各类别详细分析:")
        for i, class_name in enumerate(label_encoder.classes_):
            class_mask = labels == i
            if np.sum(class_mask) > 0:
                class_accuracy = accuracy_score(labels[class_mask], predictions[class_mask])
                class_f1 = f1_score(labels[class_mask], predictions[class_mask], average='binary')
                logger.info(f"   {class_name}: 准确率={class_accuracy:.4f}, F1={class_f1:.4f}, 样本数={np.sum(class_mask)}")
        
        # 保存结果
        results_dir = "direct_validation_results"
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(results_dir, f"direct_validation_{timestamp}.json")
        
        # 准备保存的数据
        save_data = {
            'timestamp': timestamp,
            'model_path': model_path,
            'data_path': data_path,
            'overall_accuracy': accuracy,
            'overall_f1_score': f1,
            'category_analysis': {}
        }
        
        # 添加每个类别的分析
        for i, class_name in enumerate(label_encoder.classes_):
            class_mask = labels == i
            if np.sum(class_mask) > 0:
                class_accuracy = accuracy_score(labels[class_mask], predictions[class_mask])
                class_f1 = f1_score(labels[class_mask], predictions[class_mask], average='binary')
                
                save_data['category_analysis'][class_name] = {
                    'sample_count': int(np.sum(class_mask)),
                    'accuracy': float(class_accuracy),
                    'f1_score': float(class_f1)
                }
        
        # 保存到JSON文件
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 结果已保存到: {results_file}")
        
        # 输出摘要
        print("\n" + "="*60)
        print("🎯 直接验证完成摘要")
        print("="*60)
        print(f"📊 整体准确率: {accuracy:.4f}")
        print(f"📊 整体F1分数: {f1:.4f}")
        print(f"📁 结果保存位置: {results_dir}/")
        print("="*60)
        
    except Exception as e:
        logger.error(f"❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main() 