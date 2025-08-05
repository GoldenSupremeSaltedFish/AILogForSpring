#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型管理器 - 处理模型的保存和加载
"""

import os
import torch
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ModelManager:
    """模型管理器"""
    
    def __init__(self, session_dir: str, timestamp: str):
        self.session_dir = session_dir
        self.timestamp = timestamp
        self.model_dir = os.path.join(session_dir, "models")
        os.makedirs(self.model_dir, exist_ok=True)
    
    def save_model(self, model, model_name: str, model_config: Dict[str, Any] = None) -> str:
        """保存模型文件"""
        model_path = os.path.join(self.model_dir, f"{model_name}_{self.timestamp}.pth")
        
        # 保存模型
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': model_config or model.get_config(),
            'timestamp': self.timestamp,
            'session_dir': self.session_dir
        }, model_path)
        
        logger.info(f"💾 模型已保存: {model_path}")
        return model_path
    
    def load_model(self, model_path: str) -> Dict[str, Any]:
        """加载模型文件"""
        checkpoint = torch.load(model_path, map_location='cpu')
        logger.info(f"💾 模型已加载: {model_path}")
        return checkpoint 