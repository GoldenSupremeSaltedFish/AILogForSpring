#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intel Arc GPU 训练启动脚本
"""

import os
import sys
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_data_exists():
    """检查数据是否存在"""
    data_path = "data/processed_logs.csv"
    if os.path.exists(data_path):
        logger.info(f"✅ 找到数据文件: {data_path}")
        return True
    else:
        logger.warning(f"⚠️ 数据文件不存在: {data_path}")
        return False


def prepare_data():
    """准备数据"""
    logger.info("📂 准备训练数据...")
    try:
        result = subprocess.run([sys.executable, "scripts/prepare_data.py"], 
                              capture_output=True, text=True, check=True)
        logger.info("✅ 数据准备完成")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ 数据准备失败: {e}")
        logger.error(f"错误输出: {e.stderr}")
        return False


def start_training():
    """开始训练"""
    logger.info("🚀 开始训练...")
    
    # 训练参数
    model_type = "textcnn"  # 或 "fasttext"
    data_path = "data/processed_logs.csv"
    epochs = 10
    batch_size = 16  # 较小的批次大小避免OOM
    
    cmd = [
        sys.executable, "scripts/train.py",
        "--model", model_type,
        "--data", data_path,
        "--epochs", str(epochs)
    ]
    
    try:
        logger.info(f"执行命令: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True)
        logger.info("✅ 训练完成")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ 训练失败: {e}")
        return False


def main():
    """主函数"""
    logger.info("🎯 Intel Arc GPU 训练启动器")
    logger.info("=" * 50)
    
    # 检查数据
    if not check_data_exists():
        logger.info("📂 准备数据...")
        if not prepare_data():
            logger.error("❌ 数据准备失败，退出")
            return
    
    # 开始训练
    if start_training():
        logger.info("🎉 训练成功完成！")
    else:
        logger.error("❌ 训练失败")


if __name__ == "__main__":
    main() 