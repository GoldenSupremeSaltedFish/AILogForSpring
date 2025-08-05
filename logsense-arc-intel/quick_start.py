#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intel Arc GPU 快速启动脚本
"""

import os
import sys
import logging
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from utils import ArcGPUDetector
from models import ModelFactory

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_environment():
    """检查环境"""
    logger.info("🔍 检查环境...")
    
    # 检查Python版本
    python_version = sys.version_info
    logger.info(f"Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # 检查PyTorch
    try:
        import torch
        logger.info(f"PyTorch版本: {torch.__version__}")
    except ImportError:
        logger.error("❌ PyTorch未安装")
        return False
    
    # 检查Intel Extension for PyTorch
    try:
        import intel_extension_for_pytorch as ipex
        logger.info(f"Intel Extension for PyTorch版本: {ipex.__version__}")
    except ImportError:
        logger.warning("⚠️ Intel Extension for PyTorch未安装")
        logger.info("💡 建议安装: pip install intel-extension-for-pytorch")
    
    return True


def check_gpu():
    """检查GPU"""
    logger.info("🖥️ 检查GPU...")
    
    if ArcGPUDetector.check_arc_gpu():
        logger.info("✅ Intel Arc GPU 可用")
        return True
    else:
        logger.warning("⚠️ Intel Arc GPU 不可用，将使用CPU")
        return False


def test_models():
    """测试模型"""
    logger.info("🧪 测试模型...")
    
    try:
        # 测试TextCNN
        textcnn = ModelFactory.create_model("textcnn", num_classes=5)
        logger.info("✅ TextCNN 创建成功")
        
        # 测试FastText
        fasttext = ModelFactory.create_model("fasttext", num_classes=5)
        logger.info("✅ FastText 创建成功")
        
        return True
    except Exception as e:
        logger.error(f"❌ 模型测试失败: {e}")
        return False


def main():
    """主函数"""
    logger.info("🚀 Intel Arc GPU 快速启动")
    logger.info("=" * 50)
    
    # 检查环境
    if not check_environment():
        logger.error("❌ 环境检查失败")
        return
    
    # 检查GPU
    gpu_available = check_gpu()
    
    # 测试模型
    if not test_models():
        logger.error("❌ 模型测试失败")
        return
    
    logger.info("✅ 所有检查通过！")
    logger.info("=" * 50)
    
    if gpu_available:
        logger.info("🎯 可以开始训练:")
        logger.info("   python scripts/train.py --model textcnn --data DATA_OUTPUT/processed_logs.csv")
    else:
        logger.info("💡 建议安装Intel Extension for PyTorch以获得GPU加速")
        logger.info("   pip install intel-extension-for-pytorch")
    
    logger.info("📖 更多信息请查看 README.md")


if __name__ == "__main__":
    main() 