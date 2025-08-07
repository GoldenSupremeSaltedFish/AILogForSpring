#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attention层实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    """注意力层"""
    
    def __init__(self, hidden_dim: int, attention_dim: int = 128):
        super(AttentionLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        
        # 注意力权重计算
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入张量 [batch_size, seq_len, hidden_dim]
        Returns:
            attended: 注意力加权后的特征 [batch_size, hidden_dim]
            attention_weights: 注意力权重 [batch_size, seq_len]
        """
        # 计算注意力权重
        attention_weights = self.attention(x)  # [batch_size, seq_len, 1]
        attention_weights = F.softmax(attention_weights, dim=1)  # [batch_size, seq_len, 1]
        
        # 应用注意力权重
        attended = torch.sum(x * attention_weights, dim=1)  # [batch_size, hidden_dim]
        
        return attended, attention_weights.squeeze(-1) 