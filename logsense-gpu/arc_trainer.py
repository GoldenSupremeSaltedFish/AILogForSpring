#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intel Arc GPU 训练器 - 简化版本
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArcGPUDetector:
    @staticmethod
    def check_arc_gpu():
        try:
            import intel_extension_for_pytorch as ipex
            if torch.xpu.is_available():
                device_name = torch.xpu.get_device_name(0)
                logger.info(f"✅ 检测到Intel GPU: {device_name}")
                return True
            else:
                logger.warning("⚠️ 未检测到Intel XPU设备")
                return False
        except ImportError:
            logger.error("❌ Intel Extension for PyTorch未安装")
            return False
    
    @staticmethod
    def get_device():
        if ArcGPUDetector.check_arc_gpu():
            return torch.device("xpu:0")
        else:
            return torch.device("cpu")

class TextCNN(nn.Module):
    def __init__(self, vocab_size=10000, embed_dim=128, num_classes=10):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv1 = nn.Conv1d(embed_dim, 128, kernel_size=3)
        self.conv2 = nn.Conv1d(embed_dim, 128, kernel_size=4)
        self.conv3 = nn.Conv1d(embed_dim, 128, kernel_size=5)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(384, num_classes)
    
    def forward(self, x):
        embedded = self.embedding(x).permute(0, 2, 1)
        conv1 = F.relu(self.conv1(embedded))
        conv2 = F.relu(self.conv2(embedded))
        conv3 = F.relu(self.conv3(embedded))
        
        pooled1 = F.max_pool1d(conv1, conv1.size(2)).squeeze(2)
        pooled2 = F.max_pool1d(conv2, conv2.size(2)).squeeze(2)
        pooled3 = F.max_pool1d(conv3, conv3.size(2)).squeeze(2)
        
        concatenated = torch.cat([pooled1, pooled2, pooled3], dim=1)
        dropped = self.dropout(concatenated)
        return self.fc(dropped)

class LogDataset(Dataset):
    def __init__(self, texts, labels, max_length=128):
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
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

class ArcTrainer:
    def __init__(self):
        self.device = ArcGPUDetector.get_device()
        self.model = None
        self.label_encoder = None
        logger.info(f"🖥️ 计算设备: {self.device}")
    
    def load_data(self, data_path):
        logger.info(f"📂 加载数据: {data_path}")
        
        df = pd.read_csv(data_path)
        texts = df['message'].fillna('').tolist()
        labels = df['category'].tolist()
        
        unique_labels = sorted(list(set(labels)))
        label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
        encoded_labels = [label_to_id[label] for label in labels]
        
        self.label_encoder = {idx: label for label, idx in label_to_id.items()}
        
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
        )
        
        train_dataset = LogDataset(train_texts, train_labels)
        val_dataset = LogDataset(val_texts, val_labels)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        logger.info(f"📊 数据加载完成 - 训练: {len(train_texts)}, 验证: {len(val_texts)}")
        return train_loader, val_loader, len(unique_labels)
    
    def train(self, data_path, epochs=10):
        logger.info("🚀 开始训练")
        
        train_loader, val_loader, num_classes = self.load_data(data_path)
        
        self.model = TextCNN(num_classes=num_classes).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        best_acc = 0
        
        for epoch in range(epochs):
            # 训练
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(input_ids)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            # 验证
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    outputs = self.model(input_ids)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total
            
            logger.info(f"Epoch {epoch+1}/{epochs}")
            logger.info(f"  训练: 损失={train_loss/len(train_loader):.4f}, 准确率={train_acc:.2f}%")
            logger.info(f"  验证: 损失={val_loss/len(val_loader):.4f}, 准确率={val_acc:.2f}%")
            
            if val_acc > best_acc:
                best_acc = val_acc
                self.save_model()
                logger.info(f"💾 保存最佳模型 (准确率: {val_acc:.2f}%)")
        
        logger.info(f"✅ 训练完成 - 最佳准确率: {best_acc:.2f}%")
    
    def save_model(self):
        os.makedirs("results/models", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_path = f"results/models/arc_gpu_model_{timestamp}.pth"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'label_encoder': self.label_encoder
        }, model_path)
        
        logger.info(f"💾 模型已保存: {model_path}")

def main():
    logger.info("🎯 Intel Arc GPU 训练器")
    
    if not ArcGPUDetector.check_arc_gpu():
        logger.warning("⚠️ 未检测到Intel Arc GPU，将使用CPU训练")
    
    trainer = ArcTrainer()
    trainer.train("DATA_OUTPUT/processed_logs.csv", epochs=10)

if __name__ == "__main__":
    main() 