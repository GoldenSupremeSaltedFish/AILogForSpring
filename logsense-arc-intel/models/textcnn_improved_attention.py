#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的TextCNN with Attention - 在卷积层后直接应用注意力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from .multi_attention import MultiHeadAttention


class TextCNNImprovedAttention(nn.Module):
    """改进的TextCNN with Attention - 在卷积特征上直接应用注意力"""
    
    def __init__(self, vocab_size: int = 10000, embed_dim: int = 128, 
                 num_classes: int = 10, filter_sizes: List[int] = None,
                 num_filters: int = 128, dropout: float = 0.5,
                 num_heads: int = 4):
        super(TextCNNImprovedAttention, self).__init__()
        
        if filter_sizes is None:
            filter_sizes = [3, 4, 5]
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=k) 
            for k in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        
        # 在卷积特征上应用注意力
        self.attention = MultiHeadAttention(
            embed_dim=num_filters,
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
        
        # 卷积层 + 注意力
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(embedded))  # [batch_size, num_filters, seq_len-k+1]
            
            # 将卷积输出转换为序列形式用于注意力
            # [batch_size, num_filters, seq_len-k+1] -> [batch_size, seq_len-k+1, num_filters]
            conv_out = conv_out.permute(0, 2, 1)
            
            # 应用注意力
            attended = self.attention(conv_out)  # [batch_size, seq_len-k+1, num_filters]
            
            # 全局平均池化
            pooled = torch.mean(attended, dim=1)  # [batch_size, num_filters]
            conv_outputs.append(pooled)
        
        # 拼接所有卷积输出
        concatenated = torch.cat(conv_outputs, dim=1)  # [batch_size, len(filter_sizes) * num_filters]
        
        # Dropout
        dropped = self.dropout(concatenated)
        
        # 全连接层
        output = self.fc(dropped)
        
        return output
    
    def get_config(self):
        """获取模型配置"""
        return self.config.copy()
    
    def count_parameters(self):
        """计算模型参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad) 