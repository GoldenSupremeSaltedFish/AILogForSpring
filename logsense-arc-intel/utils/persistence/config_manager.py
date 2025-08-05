#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置管理器 - 处理训练配置的保存和加载
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, session_dir: str, timestamp: str):
        self.session_dir = session_dir
        self.timestamp = timestamp
        self.config_dir = os.path.join(session_dir, "configs")
        os.makedirs(self.config_dir, exist_ok=True)
    
    def save_training_config(self, config: Dict[str, Any]) -> str:
        """保存训练配置"""
        config_file = os.path.join(self.config_dir, "training_config.json")
        
        # 添加时间戳信息
        config['timestamp'] = self.timestamp
        config['session_dir'] = self.session_dir
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📋 训练配置已保存: {config_file}")
        return config_file
    
    def load_training_config(self, config_file: str) -> Dict[str, Any]:
        """加载训练配置"""
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        logger.info(f"📋 训练配置已加载: {config_file}")
        return config 