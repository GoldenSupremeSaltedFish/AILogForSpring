"""
公共工具函数模块
用于日志分类项目的通用功能
"""
import os
import re
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """设置日志记录器"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def load_data(file_path: str) -> pd.DataFrame:
    """加载CSV数据文件"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件不存在: {file_path}")
    return pd.read_csv(file_path)

def save_data(df: pd.DataFrame, file_path: str) -> None:
    """保存数据到CSV文件"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)

def clean_text(text: str) -> str:
    """基础文本清洗函数"""
    if pd.isna(text):
        return ""
    # 移除特殊字符，保留中英文数字和常用标点
    text = re.sub(r'[^\w\s\u4e00-\u9fff.,!?;:-]', ' ', str(text))
    # 规范化空白字符
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_log_features(message: str) -> Dict[str, any]:
    """从日志消息中提取特征"""
    features = {
        'length': len(message),
        'word_count': len(message.split()),
        'has_numbers': bool(re.search(r'\d', message)),
        'has_ip': bool(re.search(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', message)),
        'has_url': bool(re.search(r'https?://', message)),
        'has_error_keywords': bool(re.search(r'(error|failed|exception|timeout)', message.lower())),
        'has_success_keywords': bool(re.search(r'(success|complete|ok|done)', message.lower()))
    }
    return features

def check_xpu_availability() -> bool:
    """检查Intel XPU是否可用"""
    try:
        import torch
        return torch.xpu.is_available()
    except ImportError:
        return False

def get_device() -> str:
    """获取可用的计算设备"""
    try:
        import torch
        if torch.xpu.is_available():
            return 'xpu'
        elif torch.cuda.is_available():
            return 'cuda'
        else:
            return 'cpu' 
    except ImportError:
        return 'cpu' 