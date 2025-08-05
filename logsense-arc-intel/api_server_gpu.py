#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intel Arc GPU 加速的API服务器
"""

import torch
import torch.nn as nn
import logging
import json
import os
import sys
from datetime import datetime
from typing import Dict, Any, List

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import ModelFactory
from data import LogPreprocessor

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LogPredictor:
    """日志预测器"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.label_encoder = None
        self.preprocessor = LogPreprocessor()
        
        # 设置设备
        if torch.xpu.is_available():
            self.device = torch.device("xpu:0")
            logger.info("✅ 使用Intel XPU GPU加速推理")
        else:
            self.device = torch.device("cpu")
            logger.warning("⚠️ 使用CPU推理")
        
        self.load_model()
    
    def load_model(self):
        """加载模型"""
        logger.info(f"📂 加载模型: {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # 获取模型配置
        model_config = checkpoint.get('model_config', {})
        self.label_encoder = checkpoint.get('label_encoder', {})
        
        # 创建模型
        self.model = ModelFactory.create_model("textcnn", **model_config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"✅ 模型加载成功 - 类别数: {len(self.label_encoder)}")
        logger.info(f"🖥️  推理设备: {self.device}")
    
    def preprocess_text(self, text: str) -> torch.Tensor:
        """预处理文本"""
        # 清理和标准化文本
        processed_text = self.preprocessor.normalize_text(text)
        
        # 简单分词
        tokens = processed_text.split()[:128]  # 最大长度128
        token_ids = [hash(token) % 10000 for token in tokens]  # 词汇表大小10000
        
        # 补齐到固定长度
        if len(token_ids) < 128:
            token_ids += [0] * (128 - len(token_ids))
        
        return torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(self.device)
    
    def predict(self, text: str) -> Dict[str, Any]:
        """预测日志类别"""
        try:
            # 预处理
            input_tensor = self.preprocess_text(text)
            
            # 推理
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(outputs, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            # 获取类别名称
            class_name = self.label_encoder.get(predicted_class, f"class_{predicted_class}")
            
            # 获取所有类别的概率
            all_probabilities = {}
            for class_id, class_name in self.label_encoder.items():
                all_probabilities[class_name] = probabilities[0][class_id].item()
            
            result = {
                'predicted_class': class_name,
                'confidence': confidence,
                'all_probabilities': all_probabilities,
                'input_text': text,
                'processed_text': self.preprocessor.normalize_text(text)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"预测失败: {e}")
            return {
                'error': str(e),
                'input_text': text
            }
    
    def batch_predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """批量预测"""
        results = []
        for text in texts:
            result = self.predict(text)
            results.append(result)
        return results


class GPULogAPI:
    """GPU加速的日志API服务器"""
    
    def __init__(self, model_path: str):
        self.predictor = LogPredictor(model_path)
        logger.info("🚀 GPU加速日志API服务器已启动")
    
    def predict_single(self, text: str) -> Dict[str, Any]:
        """单条日志预测"""
        return self.predictor.predict(text)
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """批量日志预测"""
        return self.predictor.batch_predict(texts)
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'model_path': self.predictor.model_path,
            'device': str(self.predictor.device),
            'num_classes': len(self.predictor.label_encoder),
            'classes': list(self.predictor.label_encoder.values()),
            'model_parameters': sum(p.numel() for p in self.predictor.model.parameters())
        }


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="GPU加速日志分类API")
    parser.add_argument("--model", type=str, 
                       default="results/models_gpu/arc_gpu_model_textcnn_best_20250805_003813.pth",
                       help="模型文件路径")
    parser.add_argument("--test", action="store_true", help="运行测试")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        logger.error(f"❌ 模型文件不存在: {args.model}")
        return
    
    # 创建API实例
    api = GPULogAPI(args.model)
    
    if args.test:
        # 运行测试
        logger.info("🧪 运行测试...")
        
        test_texts = [
            "Error: Connection timeout to database server",
            "Info: User login successful from IP 192.168.1.100",
            "Warning: Disk space is running low on /var/log",
            "Exception: java.lang.NullPointerException at com.example.Main.main",
            "Database connection failed: timeout after 30 seconds"
        ]
        
        for text in test_texts:
            result = api.predict_single(text)
            logger.info(f"📝 输入: {text}")
            logger.info(f"🎯 预测: {result['predicted_class']} (置信度: {result['confidence']:.3f})")
            logger.info("---")
    
    # 显示模型信息
    model_info = api.get_model_info()
    logger.info("📊 模型信息:")
    logger.info(f"   设备: {model_info['device']}")
    logger.info(f"   类别数: {model_info['num_classes']}")
    logger.info(f"   类别: {model_info['classes']}")
    logger.info(f"   参数数量: {model_info['model_parameters']:,}")


if __name__ == "__main__":
    main() 