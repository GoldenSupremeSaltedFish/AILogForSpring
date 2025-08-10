#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提取模型信息脚本 - 不加载完整模型类
"""

import torch
import pickle
import sys
import os

def extract_model_info(model_path):
    """提取模型信息"""
    try:
        print(f"🔍 分析模型文件: {model_path}")
        
        # 定义必要的类
        class StructuredFeatureExtractor:
            def __init__(self, max_tfidf_features=1000):
                self.max_tfidf_features = max_tfidf_features
                self.label_encoders = {}
                self.tfidf_vectorizer = None
                self.scaler = None
                self.feature_names = []
        
        # 设置到全局命名空间
        import __main__
        __main__.StructuredFeatureExtractor = StructuredFeatureExtractor
        
        # 尝试直接读取文件内容
        with open(model_path, 'rb') as f:
            # 跳过PyTorch的头部信息
            f.seek(0)
            
            # 尝试使用pickle加载，但不执行代码
            try:
                # 添加安全全局变量
                import torch.serialization
                torch.serialization.add_safe_globals([
                    'sklearn.preprocessing._label.LabelEncoder',
                    'sklearn.preprocessing._data.StandardScaler',
                    'sklearn.feature_extraction.text.TfidfVectorizer'
                ])
                
                # 使用weights_only=False来加载完整模型
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                print("✅ 成功加载模型权重")
                
                if isinstance(checkpoint, dict):
                    print(f"📊 Checkpoint keys: {list(checkpoint.keys())}")
                    
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                        print(f"🔧 模型状态字典包含 {len(state_dict)} 个参数:")
                        
                        # 分析关键参数
                        for key, tensor in state_dict.items():
                            print(f"  {key}: {tensor.shape}")
                            
                            # 特别关注嵌入层和MLP层
                            if 'embedding' in key and 'weight' in key:
                                print(f"    -> 词汇表大小: {tensor.shape[0]}")
                            elif 'mlp.0.weight' in key:
                                print(f"    -> 结构化特征输入维度: {tensor.shape[1]}")
                            elif 'fusion_layer.0.weight' in key:
                                print(f"    -> 融合层输入维度: {tensor.shape[1]}")
                            elif 'fusion_layer.3.weight' in key:
                                print(f"    -> 融合层输出维度: {tensor.shape[0]}")
                        
                        # 提取关键信息
                        vocab_size = None
                        struct_input_dim = None
                        num_classes = None
                        
                        for key, tensor in state_dict.items():
                            if 'text_encoder.embedding.weight' in key:
                                vocab_size = tensor.shape[0]
                            elif 'struct_mlp.mlp.0.weight' in key:
                                struct_input_dim = tensor.shape[1]
                            elif 'fusion_layer.3.weight' in key:
                                num_classes = tensor.shape[0]
                        
                        print(f"\n📋 提取的关键信息:")
                        print(f"  词汇表大小: {vocab_size}")
                        print(f"  结构化特征输入维度: {struct_input_dim}")
                        print(f"  类别数量: {num_classes}")
                        
                        # 提取词汇表
                        vocab = checkpoint.get('vocab', {})
                        label_encoder = checkpoint.get('label_encoder', None)
                        
                        return {
                            'vocab_size': vocab_size,
                            'struct_input_dim': struct_input_dim,
                            'num_classes': num_classes,
                            'state_dict': state_dict,
                            'vocab': vocab,
                            'label_encoder': label_encoder
                        }
                    else:
                        print("❌ 未找到model_state_dict")
                        return None
                else:
                    print(f"❌ Checkpoint不是字典格式: {type(checkpoint)}")
                    return None
                    
            except Exception as e:
                print(f"❌ 加载失败: {e}")
                return None
                
    except Exception as e:
        print(f"❌ 文件读取失败: {e}")
        return None

def main():
    """主函数"""
    model_path = "results/models/feature_enhanced_model_20250809_103658.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return
    
    info = extract_model_info(model_path)
    
    if info:
        print(f"\n🎯 模型信息提取完成")
        print(f"   词汇表大小: {info['vocab_size']}")
        print(f"   结构化特征输入维度: {info['struct_input_dim']}")
        print(f"   类别数量: {info['num_classes']}")
    else:
        print("❌ 模型信息提取失败")

if __name__ == "__main__":
    main()
