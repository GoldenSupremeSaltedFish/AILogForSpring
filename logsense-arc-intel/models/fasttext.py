#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastText 模型实现
轻量级的文本分类模型
"""

import torch
import torch.nn as nn


class FastTextModel(nn.Module):
    """FastText模型 - 轻量级文本分类"""
    
    def __init__(self, vocab_size: int = 10000, embed_dim: int = 128, 
                 num_classes: int = 10, dropout: float = 0.3):
        super(FastTextModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        
        # 模型配置
        self.config = {
            'vocab_size': vocab_size,
            'embed_dim': embed_dim,
            'num_classes': num_classes,
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
        
        # 平均池化
        pooled = torch.mean(embedded, dim=1)  # [batch_size, embed_dim]
        
        # Dropout
        dropped = self.dropout(pooled)
        
        # 全连接层
        output = self.fc(dropped)
        
        return output
    
    def get_config(self):
        """获取模型配置"""
        return self.config.copy()
    
    def count_parameters(self):
        """计算模型参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad) 