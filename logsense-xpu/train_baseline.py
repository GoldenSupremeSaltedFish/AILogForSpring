#!/usr/bin/env python3
"""
Baseline模型训练脚本
用于训练TF-IDF + LightGBM和FastText模型
"""
import os
import sys
import argparse
from datetime import datetime
import json

# 添加core目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

from baseline_model import BaselineLogClassifier

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练Baseline日志分类模型')
    parser.add_argument('--data_dir', type=str, default='../../DATA_OUTPUT',
                       help='数据目录路径')
    parser.add_argument('--model_type', type=str, default='lightgbm',
                       choices=['lightgbm', 'fasttext'],
                       help='模型类型: lightgbm 或 fasttext')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='测试集比例')
    parser.add_argument('--random_state', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--use_xpu', action='store_true',
                       help='是否使用Intel XPU')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='结果输出目录')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=== Baseline模型训练 ===")
    print(f"数据目录: {args.data_dir}")
    print(f"模型类型: {args.model_type}")
    print(f"测试集比例: {args.test_size}")
    print(f"使用XPU: {args.use_xpu}")
    print(f"输出目录: {args.output_dir}")
    print()
    
    try:
        # 创建分类器
        classifier = BaselineLogClassifier(
            model_type=args.model_type,
            use_xpu=args.use_xpu
        )
        
        # 训练模型
        print("开始训练模型...")
        results = classifier.train(
            data_dir=args.data_dir,
            test_size=args.test_size,
            random_state=args.random_state
        )
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(args.output_dir, f"baseline_results_{timestamp}.json")
        
        # 转换numpy数组为列表以便JSON序列化
        serializable_results = {
            'model_type': results['model_type'],
            'categories': results['categories'],
            'train_metrics': {
                'accuracy': float(results['train_metrics']['accuracy']),
                'precision': float(results['train_metrics']['precision']),
                'recall': float(results['train_metrics']['recall']),
                'f1_score': float(results['train_metrics']['f1_score'])
            },
            'val_metrics': {
                'accuracy': float(results['val_metrics']['accuracy']),
                'precision': float(results['val_metrics']['precision']),
                'recall': float(results['val_metrics']['recall']),
                'f1_score': float(results['val_metrics']['f1_score'])
            },
            'test_metrics': {
                'accuracy': float(results['test_metrics']['accuracy']),
                'precision': float(results['test_metrics']['precision']),
                'recall': float(results['test_metrics']['recall']),
                'f1_score': float(results['test_metrics']['f1_score'])
            },
            'training_config': {
                'data_dir': args.data_dir,
                'test_size': args.test_size,
                'random_state': args.random_state,
                'use_xpu': args.use_xpu
            },
            'timestamp': timestamp
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        # 绘制混淆矩阵
        confusion_matrix_file = os.path.join(args.output_dir, f"confusion_matrix_{timestamp}.png")
        classifier.plot_confusion_matrix(
            results['test_metrics']['confusion_matrix'],
            results['categories'],
            confusion_matrix_file
        )
        
        # 打印结果
        print("\n=== 训练完成 ===")
        print(f"模型类型: {results['model_type']}")
        print(f"类别数量: {len(results['categories'])}")
        print(f"训练集准确率: {results['train_metrics']['accuracy']:.4f}")
        print(f"验证集准确率: {results['val_metrics']['accuracy']:.4f}")
        print(f"测试集准确率: {results['test_metrics']['accuracy']:.4f}")
        print(f"测试集F1分数: {results['test_metrics']['f1_score']:.4f}")
        print(f"结果已保存到: {results_file}")
        print(f"混淆矩阵已保存到: {confusion_matrix_file}")
        
    except Exception as e:
        print(f"训练失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 