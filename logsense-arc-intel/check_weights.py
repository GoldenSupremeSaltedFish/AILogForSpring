#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查模型权重维度
"""

import torch
import os

# 定义StructuredFeatureExtractor类
class StructuredFeatureExtractor:
    def __init__(self, max_tfidf_features=1000):
        self.max_tfidf_features = max_tfidf_features
        self.label_encoders = {}
        self.tfidf_vectorizer = None
        self.scaler = None
        self.feature_names = []

def main():
    model_path = "results/models/feature_enhanced_model_20250812_004934.pth"
    
    # 添加安全的全局变量
    torch.serialization.add_safe_globals([
        'sklearn.preprocessing._label.LabelEncoder',
        'sklearn.preprocessing._data.StandardScaler', 
        'sklearn.feature_extraction.text.TfidfVectorizer'
    ])
    
    # 加载检查点
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # 检查权重维度
    state_dict = checkpoint['model_state_dict']
    
    print("🔍 模型权重维度分析:")
    print("=" * 50)
    
    for key, tensor in state_dict.items():
        print(f"{key}: {tensor.shape}")
    
    # 特别关注结构化特征维度
    if 'struct_mlp.mlp.0.weight' in state_dict:
        struct_input_dim = state_dict['struct_mlp.mlp.0.weight'].shape[1]
        print(f"\n📊 结构化特征输入维度: {struct_input_dim}")

if __name__ == "__main__":
    main()
