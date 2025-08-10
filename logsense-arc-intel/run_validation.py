#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行模型验证的脚本
"""

import os
import sys

def main():
    """主函数"""
    # 检查模型文件
    model_path = "results/models/feature_enhanced_model_20250809_103658.pth"
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return
    
    # 检查数据文件
    data_path = "data/processed_logs_advanced_enhanced.csv"
    if not os.path.exists(data_path):
        print(f"❌ 数据文件不存在: {data_path}")
        return
    
    print("🚀 开始运行模型验证...")
    print(f"📁 模型路径: {model_path}")
    print(f"📊 数据路径: {data_path}")
    
    # 运行验证
    cmd = f'python complete_validator.py --model_path "{model_path}" --data_path "{data_path}" --num_validations 5'
    print(f"🔧 执行命令: {cmd}")
    
    # 执行验证
    os.system(cmd)

if __name__ == "__main__":
    main() 