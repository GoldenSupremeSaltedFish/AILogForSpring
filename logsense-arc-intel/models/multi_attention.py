#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多层Attention模块实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.3):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入张量 [batch_size, seq_len, embed_dim]
        Returns:
            输出张量 [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, embed_dim = x.size()
        
        # 线性变换
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        # 输出投影
        output = self.out_proj(attn_output)
        
        return output


class AttentionBlock(nn.Module):
    """注意力块（包含残差连接和层归一化）"""
    
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.3):
        super(AttentionBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim)
        )
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入张量 [batch_size, seq_len, embed_dim]
        Returns:
            输出张量 [batch_size, seq_len, embed_dim]
        """
        # 第一个子层：多头注意力 + 残差连接
        attn_output = self.attention(x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 第二个子层：前馈网络 + 残差连接
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        
        return x


class MultiLayerAttention(nn.Module):
    """多层注意力模块"""
    
    def __init__(self, embed_dim: int, num_layers: int = 2, num_heads: int = 4, dropout: float = 0.3):
        super(MultiLayerAttention, self).__init__()
        self.layers = nn.ModuleList([
            AttentionBlock(embed_dim, num_heads, dropout) 
            for _ in range(num_layers)
        ])
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入张量 [batch_size, seq_len, embed_dim]
        Returns:
            输出张量 [batch_size, seq_len, embed_dim]
        """
        for layer in self.layers:
            x = layer(x)
        return x 