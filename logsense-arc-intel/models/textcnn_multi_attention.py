#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TextCNN with Multi-Layer Attention 模型实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

# 导入多层Attention模块
from .multi_attention import MultiLayerAttention


class TextCNNMultiAttention(nn.Module):
    """带多层注意力机制的TextCNN模型"""
    
    def __init__(self, vocab_size: int = 10000, embed_dim: int = 128, 
                 num_classes: int = 10, filter_sizes: List[int] = None,
                 num_filters: int = 128, dropout: float = 0.5,
                 attention_layers: int = 2, num_heads: int = 4):
        super(TextCNNMultiAttention, self).__init__()
        
        if filter_sizes is None:
            filter_sizes = [3, 4, 5]
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=k) 
            for k in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        
        # 多层注意力模块
        self.multi_attention = MultiLayerAttention(
            embed_dim=len(filter_sizes) * num_filters,
            num_layers=attention_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # 全连接层
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        
        # 模型配置
        self.config = {
            'vocab_size': vocab_size,
            'embed_dim': embed_dim,
            'num_classes': num_classes,
            'filter_sizes': filter_sizes,
            'num_filters': num_filters,
            'dropout': dropout,
            'attention_layers': attention_layers,
            'num_heads': num_heads
        }
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入张量 [batch_size, seq_len]
        Returns:
            输出张量 [batch_size, num_classes]
        """
        # 嵌入层
        embedded = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        embedded = embedded.permute(0, 2, 1)  # [batch_size, embed_dim, seq_len]
        
        # 卷积层
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(embedded))  # [batch_size, num_filters, seq_len-k+1]
            pooled = F.max_pool1d(conv_out, conv_out.size(2))  # [batch_size, num_filters, 1]
            conv_outputs.append(pooled.squeeze(2))
        
        # 拼接所有卷积输出
        concatenated = torch.cat(conv_outputs, dim=1)  # [batch_size, len(filter_sizes) * num_filters]
        
        # 应用多层注意力机制
        # 将特征重塑为序列形式用于注意力
        features_seq = concatenated.unsqueeze(1)  # [batch_size, 1, feature_dim]
        
        # 应用多层注意力
        attended_features = self.multi_attention(features_seq)
        attended_features = attended_features.squeeze(1)  # [batch_size, feature_dim]
        
        # Dropout
        dropped = self.dropout(attended_features)
        
        # 全连接层
        output = self.fc(dropped)
        
        return output
    
    def get_config(self):
        """获取模型配置"""
        return self.config.copy()
    
    def count_parameters(self):
        """计算模型参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# 为了兼容性，保留原始TextCNN
class TextCNN(nn.Module):
    """原始TextCNN模型（无注意力机制）"""
    
    def __init__(self, vocab_size: int = 10000, embed_dim: int = 128, 
                 num_classes: int = 10, filter_sizes: List[int] = None,
                 num_filters: int = 128, dropout: float = 0.5):
        super(TextCNN, self).__init__()
        
        if filter_sizes is None:
            filter_sizes = [3, 4, 5]
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=k) 
            for k in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        
        # 模型配置
        self.config = {
            'vocab_size': vocab_size,
            'embed_dim': embed_dim,
            'num_classes': num_classes,
            'filter_sizes': filter_sizes,
            'num_filters': num_filters,
            'dropout': dropout
        }
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入张量 [batch_size, seq_len]
        Returns:
            输出张量 [batch_size, num_classes]
        """
        # 嵌入层
        embedded = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        embedded = embedded.permute(0, 2, 1)  # [batch_size, embed_dim, seq_len]
        
        # 卷积层
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(embedded))  # [batch_size, num_filters, seq_len-k+1]
            pooled = F.max_pool1d(conv_out, conv_out.size(2))  # [batch_size, num_filters, 1]
            conv_outputs.append(pooled.squeeze(2))
        
        # 拼接所有卷积输出
        concatenated = torch.cat(conv_outputs, dim=1)  # [batch_size, len(filter_sizes) * num_filters]
        dropped = self.dropout(concatenated)
        output = self.fc(dropped)
        
        return output
    
    def get_config(self):
        """获取模型配置"""
        return self.config.copy()
    
    def count_parameters(self):
        """计算模型参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad) 