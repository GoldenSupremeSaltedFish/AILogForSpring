#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intel Arc GPU 深度模型训练器
支持TextCNN、FastText等轻量级模型在Arc GPU上训练
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ArcGPUDetector:
    """Intel Arc GPU 检测器"""
    
    @staticmethod
    def check_arc_gpu():
        """检查Intel Arc GPU是否可用"""
        try:
            import intel_extension_for_pytorch as ipex
            if torch.xpu.is_available():
                device_count = torch.xpu.device_count()
                device_name = torch.xpu.get_device_name(0)
                logger.info(f"✅ 检测到Intel GPU: {device_name}")
                logger.info(f"   GPU数量: {device_count}")
                return True
            else:
                logger.warning("⚠️ 未检测到Intel XPU设备")
                return False
        except ImportError:
            logger.error("❌ Intel Extension for PyTorch未安装")
            return False
        except Exception as e:
            logger.error(f"❌ GPU检测失败: {e}")
            return False
    
    @staticmethod
    def get_device():
        """获取最佳计算设备"""
        if ArcGPUDetector.check_arc_gpu():
            return torch.device("xpu:0")
        else:
            return torch.device("cpu")

class TextCNN(nn.Module):
    """TextCNN模型 - 适合日志分类"""
    
    def __init__(self, vocab_size: int, embed_dim: int, num_classes: int, 
                 filter_sizes: List[int] = [3, 4, 5], num_filters: int = 128):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=k) 
            for k in filter_sizes
        ])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
    
    def forward(self, x):
        embedded = self.embedding(x)
        embedded = embedded.permute(0, 2, 1)
        
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(embedded))
            pooled = F.max_pool1d(conv_out, conv_out.size(2))
            conv_outputs.append(pooled.squeeze(2))
        
        concatenated = torch.cat(conv_outputs, dim=1)
        dropped = self.dropout(concatenated)
        output = self.fc(dropped)
        return output

class LogDataset(Dataset):
    """日志数据集"""
    
    def __init__(self, texts: List[str], labels: List[int], max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # 简单的分词处理
        tokens = text.split()[:self.max_length]
        token_ids = [hash(token) % 10000 for token in tokens]
        if len(token_ids) < self.max_length:
            token_ids += [0] * (self.max_length - len(token_ids))
        else:
            token_ids = token_ids[:self.max_length]
        
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class ArcGPUTrainer:
    """Intel Arc GPU 训练器"""
    
    def __init__(self, model_type: str = "textcnn", device: Optional[torch.device] = None):
        self.model_type = model_type
        self.device = device or ArcGPUDetector.get_device()
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.label_encoder = None
        self.vocab_size = 10000
        self.embed_dim = 128
        self.max_length = 128
        
        logger.info(f"🎯 初始化训练器 - 模型类型: {model_type}")
        logger.info(f"🖥️  计算设备: {self.device}")
    
    def create_model(self, num_classes: int):
        """创建模型"""
        self.model = TextCNN(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            num_classes=num_classes
        )
        
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        logger.info(f"✅ 模型创建成功 - 参数数量: {sum(p.numel() for p in self.model.parameters())}")
    
    def load_data(self, data_path: str) -> Tuple[DataLoader, DataLoader, int]:
        """加载数据"""
        logger.info(f"📂 加载数据: {data_path}")
        
        df = pd.read_csv(data_path)
        texts = df['message'].fillna('').tolist()
        labels = df['category'].tolist()
        
        # 标签编码
        unique_labels = sorted(list(set(labels)))
        label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
        encoded_labels = [label_to_id[label] for label in labels]
        
        self.label_encoder = {idx: label for label, idx in label_to_id.items()}
        
        # 分割数据
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
        )
        
        # 创建数据集
        train_dataset = LogDataset(train_texts, train_labels, self.max_length)
        val_dataset = LogDataset(val_texts, val_labels, self.max_length)
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        logger.info(f"📊 数据加载完成 - 训练样本: {len(train_texts)}, 验证样本: {len(val_texts)}")
        logger.info(f"🏷️  类别数量: {len(unique_labels)}")
        
        return train_loader, val_loader, len(unique_labels)
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc="训练中")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(input_ids)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
        
        return total_loss / len(train_loader), 100 * correct / total
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """验证模型"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return total_loss / len(val_loader), 100 * correct / total
    
    def train(self, data_path: str, epochs: int = 10, save_dir: str = "results/models"):
        """完整训练流程"""
        logger.info("🚀 开始训练流程")
        
        # 加载数据
        train_loader, val_loader, num_classes = self.load_data(data_path)
        
        # 创建模型
        self.create_model(num_classes)
        
        # 训练循环
        best_val_acc = 0
        
        for epoch in range(epochs):
            logger.info(f"📈 Epoch {epoch+1}/{epochs}")
            
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_acc = self.validate(val_loader)
            
            logger.info(f"  训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%")
            logger.info(f"  验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%")
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model(save_dir, "best")
                logger.info(f"💾 保存最佳模型 (验证准确率: {val_acc:.2f}%)")
        
        # 保存最终模型
        self.save_model(save_dir, "final")
        
        logger.info(f"✅ 训练完成 - 最佳验证准确率: {best_val_acc:.2f}%")
    
    def save_model(self, save_dir: str, suffix: str = ""):
        """保存模型"""
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存PyTorch模型
        model_path = os.path.join(save_dir, f"arc_gpu_model_{self.model_type}_{suffix}_{timestamp}.pth")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'model_type': self.model_type,
                'vocab_size': self.vocab_size,
                'embed_dim': self.embed_dim,
                'max_length': self.max_length
            },
            'label_encoder': self.label_encoder
        }, model_path)
        
        # 保存ONNX模型
        self.save_onnx_model(save_dir, timestamp)
        
        logger.info(f"💾 模型已保存: {model_path}")
    
    def save_onnx_model(self, save_dir: str, timestamp: str):
        """保存ONNX模型"""
        try:
            dummy_input = torch.randint(0, self.vocab_size, (1, self.max_length)).to(self.device)
            
            onnx_path = os.path.join(save_dir, f"arc_gpu_model_{self.model_type}_onnx_{timestamp}.onnx")
            torch.onnx.export(
                self.model,
                dummy_input,
                onnx_path,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": [0, 1], "output": [0]},
                opset_version=11
            )
            logger.info(f"💾 ONNX模型已保存: {onnx_path}")
        except Exception as e:
            logger.warning(f"⚠️ ONNX导出失败: {e}")

def main():
    """主函数"""
    logger.info("🎯 Intel Arc GPU 深度模型训练器")
    
    # 检查GPU
    if not ArcGPUDetector.check_arc_gpu():
        logger.warning("⚠️ 未检测到Intel Arc GPU，将使用CPU训练")
    
    # 创建训练器
    trainer = ArcGPUTrainer(model_type="textcnn")
    
    # 训练参数
    data_path = "DATA_OUTPUT/processed_logs.csv"  # 根据实际数据路径调整
    epochs = 10
    save_dir = "results/models"
    
    # 开始训练
    trainer.train(data_path, epochs, save_dir)

if __name__ == "__main__":
    main() 