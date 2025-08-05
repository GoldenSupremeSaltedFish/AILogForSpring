#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型保存器模块
"""

import torch
import os
import logging
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ModelSaver:
    """模型保存器"""
    
    @staticmethod
    def save_model(model, save_dir: str, model_name: str, **kwargs):
        """保存模型"""
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_path = os.path.join(save_dir, f"{model_name}_{timestamp}.pth")
        
        save_dict = {
            'model_state_dict': model.state_dict(),
            'model_config': model.get_config() if hasattr(model, 'get_config') else {},
            **kwargs
        }
        
        torch.save(save_dict, model_path)
        logger.info(f"模型已保存: {model_path}")
        return model_path
    
    @staticmethod
    def load_model(model_class, model_path: str, **kwargs):
        """加载模型"""
        checkpoint = torch.load(model_path)
        
        # 创建模型实例
        model_config = checkpoint.get('model_config', {})
        model = model_class(**model_config)
        
        # 加载模型权重
        model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"模型已加载: {model_path}")
        return model, checkpoint 