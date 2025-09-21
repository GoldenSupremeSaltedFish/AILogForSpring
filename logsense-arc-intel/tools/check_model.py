#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查模型文件是否正确保存了权重和词汇表
"""

import torch
import os
import sys
import re
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# 定义与训练时完全一致的StructuredFeatureExtractor类
class StructuredFeatureExtractor:
    """结构化特征提取器"""
    
    def __init__(self, max_tfidf_features=1000):
        self.max_tfidf_features = max_tfidf_features
        self.label_encoders = {}
        self.tfidf_vectorizer = None
        self.scaler = None
        self.feature_names = []
    
    def extract_features(self, df):
        """提取结构化特征"""
        features = {}
        
        # 日志级别
        features['log_level'] = df['log_level'].fillna('unknown')
        
        # 是否包含堆栈跟踪
        features['contains_stack'] = df['original_log'].str.contains(r'at\s+\w+\.\w+\(', regex=True).astype(int)
        
        # 异常类型
        features['exception_type'] = df['original_log'].str.extract(r'(\w+Exception|\w+Error)')[0].fillna('none')
        
        # 文件路径
        features['file_path'] = df['original_log'].str.extract(r'at\s+([\w\.]+)\.\w+\([^)]*\)')[0].fillna('unknown')
        
        # 函数名
        features['function'] = df['original_log'].str.extract(r'at\s+[\w\.]+\.(\w+)\([^)]*\)')[0].fillna('unknown')
        
        # 行号
        features['line_number'] = df['original_log'].str.extract(r'\(([^)]+\.java:\d+)\)')[0].fillna('unknown')
        
        # 数字特征
        features['number_count'] = df['original_log'].str.count(r'\d+')
        features['special_char_count'] = df['original_log'].str.count(r'[^\w\s]')
        features['log_length'] = df['original_log'].str.len()
        features['word_count'] = df['original_log'].str.split().str.len()
        
        # 时间戳特征
        features['has_timestamp'] = df['original_log'].str.contains(r'\d{4}-\d{2}-\d{2}|\d{2}:\d{2}:\d{2}').astype(int)
        
        # 路径深度
        paths = df['original_log'].str.extract(r'([/\w\.]+\.java)')[0].fillna('')
        features['path_depth'] = paths.str.count('/').fillna(0) + paths.str.count(r'\\').fillna(0)
        
        return pd.DataFrame(features)

def check_model_file(model_path):
    """检查模型文件"""
    print(f"🔍 检查模型文件: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return False
    
    try:
        # 添加安全的全局变量
        torch.serialization.add_safe_globals([
            'sklearn.preprocessing._label.LabelEncoder',
            'sklearn.preprocessing._data.StandardScaler', 
            'sklearn.feature_extraction.text.TfidfVectorizer'
        ])
        
        # 加载模型文件
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        print("✅ 模型文件加载成功!")
        print(f"📦 检查点包含的键: {list(checkpoint.keys())}")
        
        # 检查各个组件
        checks = {
            'model_state_dict': 'model_state_dict' in checkpoint,
            'label_encoder': 'label_encoder' in checkpoint,
            'feature_extractor': 'feature_extractor' in checkpoint,
            'vocab': 'vocab' in checkpoint,
            'best_acc': 'best_acc' in checkpoint,
            'best_f1': 'best_f1' in checkpoint,
            'timestamp': 'timestamp' in checkpoint
        }
        
        print("\n📊 组件检查结果:")
        for component, exists in checks.items():
            status = "✅" if exists else "❌"
            print(f"   {status} {component}")
        
        # 详细检查词汇表
        if 'vocab' in checkpoint:
            vocab = checkpoint['vocab']
            print(f"\n📚 词汇表信息:")
            print(f"   大小: {len(vocab)}")
            print(f"   包含PAD: {'<PAD>' in vocab}")
            print(f"   包含UNK: {'<UNK>' in vocab}")
            print(f"   前5个词: {list(vocab.items())[:5]}")
        else:
            print("\n❌ 词汇表未找到!")
        
        # 检查模型权重
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"\n🤖 模型权重信息:")
            print(f"   权重键数量: {len(state_dict)}")
            print(f"   权重键: {list(state_dict.keys())}")
        else:
            print("\n❌ 模型权重未找到!")
        
        # 检查标签编码器
        if 'label_encoder' in checkpoint:
            label_encoder = checkpoint['label_encoder']
            print(f"\n🏷️ 标签编码器信息:")
            print(f"   类别数: {len(label_encoder.classes_)}")
            print(f"   类别: {list(label_encoder.classes_)}")
        else:
            print("\n❌ 标签编码器未找到!")
        
        # 检查性能指标
        if 'best_acc' in checkpoint and 'best_f1' in checkpoint:
            print(f"\n📈 性能指标:")
            print(f"   最佳准确率: {checkpoint['best_acc']:.4f}")
            print(f"   最佳F1分数: {checkpoint['best_f1']:.4f}")
        
        if 'timestamp' in checkpoint:
            print(f"\n⏰ 保存时间: {checkpoint['timestamp']}")
        
        return True
        
    except Exception as e:
        print(f"❌ 检查失败: {e}")
        return False

def main():
    """主函数"""
    # 检查最新的feature_enhanced模型文件
    models_dir = "results/models"
    
    if not os.path.exists(models_dir):
        print(f"❌ 模型目录不存在: {models_dir}")
        return
    
    # 获取所有feature_enhanced模型文件
    model_files = [f for f in os.listdir(models_dir) if 'feature_enhanced' in f and f.endswith('.pth')]
    
    if not model_files:
        print(f"❌ 模型目录中没有找到feature_enhanced模型文件: {models_dir}")
        return
    
    # 按时间排序，检查最新的
    model_files.sort()
    latest_model = os.path.join(models_dir, model_files[-1])
    
    print("=" * 60)
    print("🔍 模型文件检查工具")
    print("=" * 60)
    
    success = check_model_file(latest_model)
    
    if success:
        print("\n" + "=" * 60)
        print("✅ 模型文件检查完成!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ 模型文件检查失败!")
        print("=" * 60)

if __name__ == "__main__":
    main()
