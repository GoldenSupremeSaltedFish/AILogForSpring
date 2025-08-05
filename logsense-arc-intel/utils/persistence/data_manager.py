#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据管理器 - 处理数据文件和信息的保存
"""

import os
import json
import shutil
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class DataManager:
    """数据管理器"""
    
    def __init__(self, session_dir: str, timestamp: str):
        self.session_dir = session_dir
        self.timestamp = timestamp
        self.data_dir = os.path.join(session_dir, "data")
        os.makedirs(self.data_dir, exist_ok=True)
    
    def save_data_info(self, data_info: Dict[str, Any]) -> str:
        """保存数据信息"""
        data_file = os.path.join(self.data_dir, "data_info.json")
        
        # 添加时间戳信息
        data_info['timestamp'] = self.timestamp
        data_info['session_dir'] = self.session_dir
        
        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump(data_info, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📂 数据信息已保存: {data_file}")
        return data_file
    
    def copy_data_files(self, data_path: str) -> str:
        """复制数据文件"""
        data_filename = os.path.basename(data_path)
        data_copy_path = os.path.join(self.data_dir, data_filename)
        
        shutil.copy2(data_path, data_copy_path)
        logger.info(f"📁 数据文件已复制: {data_copy_path}")
        return data_copy_path 