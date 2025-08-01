#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小样本验证 + Base 模型训练
功能：
1. 小样本验证（3-5个类别，每类500条样本）
2. 构建baseline模型（TF-IDF + LightGBM/GradientBoosting）
3. 支持多平台计算（CPU/GPU）
4. 模型评估和可视化

使用方法:
python baseline_model.py                           # 使用默认配置训练
python baseline_model.py --sample-size 300        # 设置每类样本数
python baseline_model.py --model-type lightgbm    # 选择模型类型
python baseline_model.py --gpu                    # 启用GPU加速
"""

import pandas as pd
import numpy as np
import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# 机器学习库
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    from sklearn.pipeline import Pipeline
    import matplotlib.pyplot as plt
    import seaborn as sns
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️  scikit-learn未安装")

# LightGBM
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️  lightgbm未安装")

# GPU支持
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    if GPU_AVAILABLE:
        print(f"✅ GPU可用: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  GPU不可用，使用CPU")
except ImportError:
    GPU_AVAILABLE = False
    print("⚠️  PyTorch未安装")


class BaselineModelTrainer:
    """Baseline模型训练器"""
    
    def __init__(self, sample_size: int = 500, model_type: str = 'gradient_boosting'):
        self.sample_size = sample_size
        self.model_type = model_type
        self.vectorizer = None
        self.model = None
        self.classes = None
        
        # 支持的类别（选择3-5个主要类别）
        self.target_classes = [
            'stack_exception',      # 堆栈异常
            'connection_issue',     # 连接问题
            'database_exception',   # 数据库异常
            'auth_authorization',   # 认证授权
            'memory_performance'    # 内存性能
        ]
        
        print(f"🎯 目标类别: {self.target_classes}")
        print(f"📊 每类样本数: {self.sample_size}")
        print(f"🤖 模型类型: {model_type}")
    
    def load_and_sample_data(self, data_file: str) -> pd.DataFrame:
        """加载数据并进行小样本采样"""
        try:
            print(f"📂 加载数据文件: {data_file}")
            df = pd.read_csv(data_file)
            print(f"📊 原始数据: {len(df)} 条记录")
            
            # 检查必要的列
            if 'text' not in df.columns or 'label' not in df.columns:
                print("❌ 数据文件缺少必要的列: text, label")
                return pd.DataFrame()
            
            # 过滤目标类别
            df_filtered = df[df['label'].isin(self.target_classes)].copy()
            print(f"🔍 过滤后数据: {len(df_filtered)} 条记录")
            
            # 显示类别分布
            class_counts = df_filtered['label'].value_counts()
            print("\n📈 类别分布:")
            for class_name, count in class_counts.items():
                print(f"  {class_name}: {count} 条")
            
            # 小样本采样
            sampled_data = []
            for class_name in self.target_classes:
                class_data = df_filtered[df_filtered['label'] == class_name]
                if len(class_data) >= self.sample_size:
                    # 随机采样
                    sampled = class_data.sample(n=self.sample_size, random_state=42)
                else:
                    # 如果数据不足，使用所有可用数据
                    sampled = class_data
                    print(f"⚠️  {class_name} 类别数据不足，使用全部 {len(sampled)} 条")
                
                sampled_data.append(sampled)
            
            # 合并采样数据
            df_sampled = pd.concat(sampled_data, ignore_index=True)
            print(f"\n⚖️ 采样后数据: {len(df_sampled)} 条记录")
            
            # 最终类别分布
            final_counts = df_sampled['label'].value_counts()
            print("\n📈 最终类别分布:")
            for class_name, count in final_counts.items():
                percentage = (count / len(df_sampled)) * 100
                print(f"  {class_name}: {count} 条 ({percentage:.1f}%)")
            
            return df_sampled
            
        except Exception as e:
            print(f"❌ 加载数据失败: {e}")
            return pd.DataFrame()
    
    def create_model(self):
        """创建模型"""
        if self.model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
            # LightGBM模型
            self.model = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                verbose=-1
            )
            print("🤖 使用LightGBM模型")
            
        else:
            # GradientBoosting模型
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
            print("🤖 使用GradientBoosting模型")
        
        # TF-IDF向量化器
        self.vectorizer = TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        print("📝 使用TF-IDF向量化器")
    
    def train_model(self, X_train: pd.Series, y_train: pd.Series, 
                   X_test: pd.Series, y_test: pd.Series) -> Dict:
        """训练模型"""
        print("\n🚀 开始训练模型...")
        
        # 创建模型
        self.create_model()
        
        # 向量化训练数据
        print("📝 向量化训练数据...")
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        print(f"📊 特征维度: {X_train_vec.shape[1]}")
        
        # 训练模型
        print("🏋️ 训练模型...")
        start_time = datetime.now()
        self.model.fit(X_train_vec, y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        print(f"⏱️ 训练时间: {training_time:.2f} 秒")
        
        # 预测
        print("🔮 进行预测...")
        y_pred = self.model.predict(X_test_vec)
        y_pred_proba = self.model.predict_proba(X_test_vec)
        
        # 评估结果
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        print(f"\n📊 模型评估结果:")
        print(f"准确率: {accuracy:.4f}")
        print(f"宏平均F1: {report['macro avg']['f1-score']:.4f}")
        print(f"加权平均F1: {report['weighted avg']['f1-score']:.4f}")
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'training_time': training_time,
            'feature_dim': X_train_vec.shape[1]
        }
    
    def save_model(self, model_dir: Path):
        """保存模型"""
        try:
            model_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 保存模型
            model_file = model_dir / f"baseline_model_{self.model_type}_{timestamp}.pkl"
            import pickle
            with open(model_file, 'wb') as f:
                pickle.dump({
                    'vectorizer': self.vectorizer,
                    'model': self.model,
                    'classes': self.target_classes,
                    'model_type': self.model_type
                }, f)
            
            print(f"💾 模型已保存到: {model_file}")
            
            # 保存配置
            config_file = model_dir / f"model_config_{timestamp}.json"
            import json
            config = {
                'model_type': self.model_type,
                'sample_size': self.sample_size,
                'target_classes': self.target_classes,
                'feature_dim': self.vectorizer.get_feature_names_out().shape[0],
                'training_time': datetime.now().isoformat()
            }
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            print(f"📋 配置已保存到: {config_file}")
            
        except Exception as e:
            print(f"❌ 保存模型失败: {e}")
    
    def plot_results(self, y_test: pd.Series, y_pred: np.ndarray, results_dir: Path):
        """绘制结果图表"""
        try:
            results_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 混淆矩阵
            plt.figure(figsize=(10, 8))
            cm = confusion_matrix(y_test, y_pred, labels=self.target_classes)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=self.target_classes, 
                       yticklabels=self.target_classes)
            plt.title('混淆矩阵')
            plt.xlabel('预测类别')
            plt.ylabel('真实类别')
            plt.tight_layout()
            
            cm_file = results_dir / f"confusion_matrix_{timestamp}.png"
            plt.savefig(cm_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"📊 混淆矩阵已保存到: {cm_file}")
            
            # 类别准确率条形图
            report = classification_report(y_test, y_pred, output_dict=True)
            f1_scores = [report[cls]['f1-score'] for cls in self.target_classes]
            
            plt.figure(figsize=(12, 6))
            bars = plt.bar(self.target_classes, f1_scores, color='skyblue')
            plt.title('各类别F1分数')
            plt.xlabel('类别')
            plt.ylabel('F1分数')
            plt.xticks(rotation=45)
            plt.ylim(0, 1)
            
            # 添加数值标签
            for bar, score in zip(bars, f1_scores):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            f1_file = results_dir / f"f1_scores_{timestamp}.png"
            plt.savefig(f1_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"📈 F1分数图已保存到: {f1_file}")
            
        except Exception as e:
            print(f"❌ 绘制图表失败: {e}")
    
    def run_experiment(self, data_file: str, output_dir: Path = None):
        """运行完整实验"""
        if output_dir is None:
            output_dir = Path("logsense-gpu/results")
        
        print("🧪 开始小样本验证实验")
        print("=" * 60)
        
        # 加载和采样数据
        df = self.load_and_sample_data(data_file)
        if df.empty:
            return False
        
        # 数据分割
        X = df['text']
        y = df['label']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n📊 训练集: {len(X_train)} 条记录")
        print(f"📊 测试集: {len(X_test)} 条记录")
        
        # 训练模型
        results = self.train_model(X_train, y_train, X_test, y_test)
        
        # 保存模型
        model_dir = output_dir / "models"
        self.save_model(model_dir)
        
        # 绘制结果
        results_dir = output_dir / "plots"
        self.plot_results(y_test, results['y_pred'], results_dir)
        
        # 保存结果
        results_file = output_dir / f"experiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        import json
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'accuracy': results['accuracy'],
                'training_time': results['training_time'],
                'feature_dim': results['feature_dim'],
                'sample_size': self.sample_size,
                'model_type': self.model_type,
                'target_classes': self.target_classes
            }, f, indent=2, ensure_ascii=False)
        
        print(f"📋 实验结果已保存到: {results_file}")
        
        return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="小样本验证 + Base模型训练")
    parser.add_argument("--data-file", default="DATA_OUTPUT/training_dataset_20250802_013437.csv",
                       help="训练数据文件路径")
    parser.add_argument("--sample-size", type=int, default=500,
                       help="每类样本数 (默认: 500)")
    parser.add_argument("--model-type", choices=['gradient_boosting', 'lightgbm'], 
                       default='gradient_boosting', help="模型类型")
    parser.add_argument("--output-dir", help="输出目录")
    parser.add_argument("--gpu", action="store_true", help="启用GPU加速")
    
    args = parser.parse_args()
    
    # 检查依赖
    if not SKLEARN_AVAILABLE:
        print("❌ 请安装scikit-learn: pip install scikit-learn")
        sys.exit(1)
    
    if args.model_type == 'lightgbm' and not LIGHTGBM_AVAILABLE:
        print("❌ 请安装lightgbm: pip install lightgbm")
        sys.exit(1)
    
    # 设置输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("logsense-gpu/results")
    
    # 创建训练器
    trainer = BaselineModelTrainer(
        sample_size=args.sample_size,
        model_type=args.model_type
    )
    
    # 运行实验
    success = trainer.run_experiment(args.data_file, output_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 