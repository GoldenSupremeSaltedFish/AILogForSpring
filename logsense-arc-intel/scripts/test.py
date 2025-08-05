#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intel Arc GPU 测试脚本
"""

import argparse
import torch
import logging
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import ModelFactory
from utils import ArcGPUDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_model_creation():
    """测试模型创建"""
    logger.info("🧪 测试模型创建...")
    
    # 测试TextCNN
    try:
        textcnn = ModelFactory.create_model("textcnn", num_classes=5)
        logger.info("✅ TextCNN 创建成功")
        logger.info(f"   参数数量: {textcnn.count_parameters():,}")
    except Exception as e:
        logger.error(f"❌ TextCNN 创建失败: {e}")
    
    # 测试FastText
    try:
        fasttext = ModelFactory.create_model("fasttext", num_classes=5)
        logger.info("✅ FastText 创建成功")
        logger.info(f"   参数数量: {fasttext.count_parameters():,}")
    except Exception as e:
        logger.error(f"❌ FastText 创建失败: {e}")


def test_gpu_inference():
    """测试GPU推理"""
    logger.info("🧪 测试GPU推理...")
    
    device = ArcGPUDetector.get_device()
    logger.info(f"   使用设备: {device}")
    
    # 创建模型
    model = ModelFactory.create_model("textcnn", num_classes=5)
    model.to(device)
    
    # 创建测试输入
    batch_size = 4
    seq_len = 128
    test_input = torch.randint(0, 10000, (batch_size, seq_len)).to(device)
    
    try:
        with torch.no_grad():
            output = model(test_input)
            logger.info("✅ GPU推理成功")
            logger.info(f"   输入形状: {test_input.shape}")
            logger.info(f"   输出形状: {output.shape}")
    except Exception as e:
        logger.error(f"❌ GPU推理失败: {e}")


def test_data_loading():
    """测试数据加载"""
    logger.info("🧪 测试数据加载...")
    
    try:
        from data import DataLoaderFactory
        
        # 创建模拟数据
        import pandas as pd
        test_data = {
            'message': [
                'Error: Connection timeout',
                'Info: User login successful',
                'Warning: Disk space low',
                'Error: Database connection failed',
                'Info: Backup completed'
            ],
            'category': ['error', 'info', 'warning', 'error', 'info']
        }
        
        df = pd.DataFrame(test_data)
        df.to_csv('test_data.csv', index=False)
        
        # 测试数据加载
        texts, labels, label_encoder = DataLoaderFactory.load_csv_data('test_data.csv')
        logger.info("✅ 数据加载成功")
        logger.info(f"   文本数量: {len(texts)}")
        logger.info(f"   标签数量: {len(labels)}")
        logger.info(f"   类别数: {len(label_encoder)}")
        
        # 清理测试文件
        os.remove('test_data.csv')
        
    except Exception as e:
        logger.error(f"❌ 数据加载失败: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Intel Arc GPU 测试工具")
    parser.add_argument("--test", type=str, default="all", 
                       choices=["all", "model", "gpu", "data"], help="测试类型")
    
    args = parser.parse_args()
    
    logger.info("🎯 Intel Arc GPU 测试工具")
    logger.info("=" * 50)
    
    if args.test in ["all", "model"]:
        test_model_creation()
        print()
    
    if args.test in ["all", "gpu"]:
        test_gpu_inference()
        print()
    
    if args.test in ["all", "data"]:
        test_data_loading()
        print()
    
    logger.info("✅ 测试完成")


if __name__ == "__main__":
    main() 