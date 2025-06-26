#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LogSense-XPU 项目主入口
日志分类和异常检测项目
"""

import sys
from pathlib import Path

# 添加核心模块路径
sys.path.append(str(Path(__file__).parent / "core"))
sys.path.append(str(Path(__file__).parent / "scripts"))

def main():
    """主函数"""
    print("LogSense-XPU 日志分类项目")
    print("=" * 40)
    print("可用功能:")
    print("1. 数据预处理 (core/preprocessing.py)")
    print("2. 向量化 (core/embed.py)")
    print("3. 模型训练 (core/train.py)")
    print("4. 预测 (core/predict.py)")
    print("5. Gateway日志处理 (scripts/gateway/)")
    print("6. 日志去重 (scripts/deduplication/)")
    print()
    print("请查看 docs/README.md 获取详细使用说明")

if __name__ == "__main__":
    main()
