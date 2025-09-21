#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AILogForSpring - Intel Arc GPU 日志分类系统
主入口文件

使用方法:
    python main.py --mode train          # 训练模式
    python main.py --mode validate       # 验证模式
    python main.py --mode prepare        # 数据准备模式
    python main.py --mode check          # 模型检查模式
"""

import argparse
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    parser = argparse.ArgumentParser(
        description="AILogForSpring - Intel Arc GPU 日志分类系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main.py --mode train                    # 训练模型
  python main.py --mode validate --model_path results/models/best_model.pth  # 验证模型
  python main.py --mode prepare                  # 准备数据
  python main.py --mode check --model_path results/models/best_model.pth     # 检查模型
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['train', 'validate', 'prepare', 'check'],
        required=True,
        help='运行模式: train(训练), validate(验证), prepare(数据准备), check(模型检查)'
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        help='模型文件路径 (用于验证和检查模式)'
    )
    
    parser.add_argument(
        '--data_path',
        type=str,
        default='data/processed_logs_advanced_enhanced.csv',
        help='数据文件路径 (默认: data/processed_logs_advanced_enhanced.csv)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results',
        help='输出目录 (默认: results)'
    )
    
    args = parser.parse_args()
    
    print("🚀 AILogForSpring - Intel Arc GPU 日志分类系统")
    print("=" * 50)
    
    try:
        if args.mode == 'train':
            print("🎯 启动训练模式...")
            from feature_enhanced_model import main as train_main
            train_main()
            
        elif args.mode == 'validate':
            if not args.model_path:
                print("❌ 验证模式需要指定 --model_path 参数")
                sys.exit(1)
            print(f"🔍 启动验证模式... 模型: {args.model_path}")
            from final_model_runner import main as validate_main
            validate_main()
            
        elif args.mode == 'prepare':
            print("📊 启动数据准备模式...")
            from prepare_full_data import main as prepare_main
            prepare_main()
            
        elif args.mode == 'check':
            if not args.model_path:
                print("❌ 检查模式需要指定 --model_path 参数")
                sys.exit(1)
            print(f"🔧 启动模型检查模式... 模型: {args.model_path}")
            from tools.check_model import main as check_main
            check_main()
            
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请确保所有依赖已正确安装")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 运行错误: {e}")
        sys.exit(1)
    
    print("✅ 任务完成!")

if __name__ == "__main__":
    main()
