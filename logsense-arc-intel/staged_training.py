#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分阶段训练脚本
先使用小数据集快速验证，再使用完整数据集进行训练
"""

import os
import sys
import subprocess
import logging
import time
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_environment():
    """检查环境"""
    logger.info("🔍 检查训练环境...")
    try:
        result = subprocess.run([sys.executable, "quick_start.py"], 
                              capture_output=True, text=True, check=True)
        logger.info("✅ 环境检查通过")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ 环境检查失败: {e}")
        logger.error(f"错误输出: {e.stderr}")
        return False


def prepare_staged_data():
    """准备分阶段数据"""
    logger.info("📂 准备分阶段训练数据...")
    try:
        result = subprocess.run([sys.executable, "scripts/prepare_data_staged.py"], 
                              capture_output=True, text=True, check=True)
        logger.info("✅ 分阶段数据准备完成")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ 数据准备失败: {e}")
        logger.error(f"错误输出: {e.stderr}")
        return False


def run_small_dataset_training():
    """运行小数据集训练（快速验证）"""
    logger.info("🔬 开始小数据集训练（快速验证）...")
    
    small_data_path = "data/processed_logs_small.csv"
    if not os.path.exists(small_data_path):
        logger.error(f"❌ 小数据集文件不存在: {small_data_path}")
        return False
    
    # 小数据集训练参数
    cmd = [
        sys.executable, "scripts/train.py",
        "--model", "textcnn",
        "--data", small_data_path,
        "--epochs", "3",  # 较少的epoch用于快速验证
        "--save_dir", "results/models_small"
    ]
    
    try:
        logger.info(f"执行命令: {' '.join(cmd)}")
        start_time = time.time()
        result = subprocess.run(cmd, check=True)
        end_time = time.time()
        
        training_time = end_time - start_time
        logger.info(f"✅ 小数据集训练完成！耗时: {training_time:.1f} 秒")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ 小数据集训练失败: {e}")
        return False


def run_large_dataset_training():
    """运行完整数据集训练"""
    logger.info("🚀 开始完整数据集训练...")
    
    large_data_path = "data/processed_logs_large.csv"
    if not os.path.exists(large_data_path):
        logger.error(f"❌ 完整数据集文件不存在: {large_data_path}")
        return False
    
    # 完整数据集训练参数
    cmd = [
        sys.executable, "scripts/train.py",
        "--model", "textcnn",
        "--data", large_data_path,
        "--epochs", "10",  # 完整的epoch用于正式训练
        "--save_dir", "results/models_large"
    ]
    
    try:
        logger.info(f"执行命令: {' '.join(cmd)}")
        start_time = time.time()
        result = subprocess.run(cmd, check=True)
        end_time = time.time()
        
        training_time = end_time - start_time
        logger.info(f"✅ 完整数据集训练完成！耗时: {training_time:.1f} 秒")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ 完整数据集训练失败: {e}")
        return False


def compare_results():
    """比较训练结果"""
    logger.info("📊 比较训练结果...")
    
    small_model_dir = "results/models_small"
    large_model_dir = "results/models_large"
    
    if os.path.exists(small_model_dir) and os.path.exists(large_model_dir):
        logger.info("✅ 两个阶段的模型都已保存")
        logger.info(f"   小数据集模型: {small_model_dir}")
        logger.info(f"   完整数据集模型: {large_model_dir}")
    else:
        logger.warning("⚠️ 部分模型文件缺失")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="分阶段训练工具")
    parser.add_argument("--skip_small", action="store_true", 
                       help="跳过小数据集训练")
    parser.add_argument("--skip_large", action="store_true", 
                       help="跳过完整数据集训练")
    parser.add_argument("--skip_data_prep", action="store_true", 
                       help="跳过数据准备")
    
    args = parser.parse_args()
    
    logger.info("🎯 Intel Arc GPU 分阶段训练")
    logger.info("=" * 60)
    logger.info("📋 训练计划:")
    logger.info("   阶段1: 小数据集快速验证 (3 epochs)")
    logger.info("   阶段2: 完整数据集正式训练 (10 epochs)")
    logger.info("=" * 60)
    
    # 检查环境
    if not check_environment():
        logger.error("❌ 环境检查失败，退出")
        return
    
    # 准备数据
    if not args.skip_data_prep:
        if not prepare_staged_data():
            logger.error("❌ 数据准备失败，退出")
            return
    else:
        logger.info("⏭️ 跳过数据准备")
    
    # 阶段1: 小数据集训练
    if not args.skip_small:
        logger.info("\n" + "="*60)
        logger.info("🔬 阶段1: 小数据集快速验证")
        logger.info("="*60)
        
        if not run_small_dataset_training():
            logger.error("❌ 小数据集训练失败")
            return
    else:
        logger.info("⏭️ 跳过小数据集训练")
    
    # 阶段2: 完整数据集训练
    if not args.skip_large:
        logger.info("\n" + "="*60)
        logger.info("🚀 阶段2: 完整数据集正式训练")
        logger.info("="*60)
        
        if not run_large_dataset_training():
            logger.error("❌ 完整数据集训练失败")
            return
    else:
        logger.info("⏭️ 跳过完整数据集训练")
    
    # 比较结果
    compare_results()
    
    logger.info("\n" + "="*60)
    logger.info("🎉 分阶段训练完成！")
    logger.info("📁 模型保存在:")
    logger.info("   results/models_small/  (小数据集模型)")
    logger.info("   results/models_large/  (完整数据集模型)")
    logger.info("="*60)


if __name__ == "__main__":
    main() 