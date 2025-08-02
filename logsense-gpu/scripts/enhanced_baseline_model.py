#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版基线模型训练脚本
功能：使用更大的数据集进行训练，输出企业级评估指标
"""

import pandas as pd
import numpy as np
import argparse
import sys
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 机器学习相关导入
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    precision_score, recall_score, f1_score
)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# 可选导入
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class EnhancedBaselineModel:
    """增强版基线模型训练器"""
    
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.label_encoder = LabelEncoder()
        self.results = {}
        
        # 企业级目标指标
        self.enterprise_targets = {
            'accuracy': 0.90,
            'macro_f1': 0.88,
            'weighted_f1': 0.90,
            'recall': 0.85,
            'precision': 0.90
        }
    
    def load_and_prepare_data(self, data_file: str, sample_size: int = None):
        """加载和准备数据"""
        print(f"📂 加载数据文件: {data_file}")
        
        df = pd.read_csv(data_file)
        print(f"📊 原始数据: {len(df)} 条记录")
        
        # 检测标签列和文本列
        label_column = self._detect_label_column(df)
        text_column = self._detect_text_column(df)
        
        if not label_column or not text_column:
            raise ValueError("未找到标签列或文本列")
        
        print(f"🔍 使用标签列: {label_column}")
        print(f"🔍 使用文本列: {text_column}")
        
        # 过滤数据
        df_filtered = df[df[label_column] != 'other'].copy()
        print(f"🔍 过滤后数据: {len(df_filtered)} 条记录")
        
        # 统计类别分布
        category_counts = df_filtered[label_column].value_counts()
        print("\n📈 类别分布:")
        for category, count in category_counts.items():
            percentage = (count / len(df_filtered)) * 100
            print(f"  {category}: {count} 条 ({percentage:.1f}%)")
        
        # 如果指定了采样大小，进行采样
        if sample_size:
            print(f"\n🎯 进行采样，每类最多 {sample_size} 条记录")
            sampled_data = []
            for category in df_filtered[label_column].unique():
                category_data = df_filtered[df_filtered[label_column] == category]
                if len(category_data) > sample_size:
                    category_data = category_data.sample(n=sample_size, random_state=42)
                sampled_data.append(category_data)
            
            df_filtered = pd.concat(sampled_data, ignore_index=True)
            print(f"📊 采样后数据: {len(df_filtered)} 条记录")
        
        return df_filtered, label_column, text_column
    
    def _detect_label_column(self, df: pd.DataFrame) -> str:
        """检测标签列"""
        possible_labels = ['content_type', 'final_label', 'label', 'category']
        for col in possible_labels:
            if col in df.columns:
                return col
        return None
    
    def _detect_text_column(self, df: pd.DataFrame) -> str:
        """检测文本列"""
        possible_texts = ['original_log', 'message', 'content', 'text']
        for col in possible_texts:
            if col in df.columns:
                return col
        return None
    
    def create_model(self, model_type: str = 'gradient_boosting'):
        """创建模型"""
        if model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
            self.model = lgb.LGBMClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=8,
                num_leaves=31,
                random_state=42,
                verbose=-1
            )
            print("🤖 使用LightGBM模型")
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            print("🤖 使用RandomForest模型")
        else:
            self.model = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=8,
                subsample=0.8,
                random_state=42
            )
            print("🤖 使用GradientBoosting模型")
    
    def train_model(self, X_train, y_train, X_test, y_test):
        """训练模型"""
        print("\n🚀 开始训练模型...")
        
        # 特征工程
        print("📝 向量化训练数据...")
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.95,
            stop_words='english'
        )
        
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
        y_pred = self.model.predict(X_test_vec)
        y_pred_proba = self.model.predict_proba(X_test_vec)
        
        return y_pred, y_pred_proba, X_test_vec
    
    def evaluate_model(self, y_test, y_pred, y_pred_proba):
        """评估模型性能"""
        print("\n📊 模型评估结果:")
        print("=" * 60)
        
        # 计算各项指标
        accuracy = accuracy_score(y_test, y_pred)
        precision_macro = precision_score(y_test, y_pred, average='macro')
        recall_macro = recall_score(y_test, y_pred, average='macro')
        f1_macro = f1_score(y_test, y_pred, average='macro')
        
        precision_weighted = precision_score(y_test, y_pred, average='weighted')
        recall_weighted = recall_score(y_test, y_pred, average='weighted')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        # 计算每个类别的指标
        precision_per_class = precision_score(y_test, y_pred, average=None)
        recall_per_class = recall_score(y_test, y_pred, average=None)
        f1_per_class = f1_score(y_test, y_pred, average=None)
        
        # 存储结果
        self.results = {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        # 输出企业级评估表格
        self._print_enterprise_metrics()
        
        # 输出详细分类报告
        print("\n📋 详细分类报告:")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))
        
        return self.results
    
    def _print_enterprise_metrics(self):
        """打印企业级评估指标"""
        print("\n📊 企业级评估指标")
        print("=" * 80)
        print("| 指标                 | 当前值           | 企业级目标         | 状态    |")
        print("| ------------------ | ------------ | ----------------- | ------ |")
        
        metrics = [
            ('Accuracy', self.results['accuracy'], self.enterprise_targets['accuracy']),
            ('Macro F1', self.results['f1_macro'], self.enterprise_targets['macro_f1']),
            ('Weighted F1', self.results['f1_weighted'], self.enterprise_targets['weighted_f1']),
            ('Macro Recall', self.results['recall_macro'], self.enterprise_targets['recall']),
            ('Macro Precision', self.results['precision_macro'], self.enterprise_targets['precision'])
        ]
        
        for metric_name, current_value, target_value in metrics:
            status = "✅ 优秀" if current_value >= target_value else "⚠️ 需改进"
            print(f"| {metric_name:<20} | {current_value:.4f}         | {target_value:.2f}            | {status:<6} |")
        
        print("=" * 80)
        
        # 计算总体达标情况
        achieved_count = sum(1 for _, current, target in metrics if current >= target)
        total_count = len(metrics)
        print(f"\n🎯 总体达标情况: {achieved_count}/{total_count} 项指标达到企业级标准")
        
        if achieved_count == total_count:
            print("🎉 恭喜！所有指标都达到了企业级标准！")
        elif achieved_count >= total_count * 0.8:
            print("👍 表现良好，大部分指标达到企业级标准")
        else:
            print("💡 建议进一步优化模型参数或增加训练数据")
    
    def save_results(self, output_dir: str = "logsense-gpu/results"):
        """保存训练结果"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存模型
        models_dir = output_path / "models"
        models_dir.mkdir(exist_ok=True)
        
        import joblib
        model_file = models_dir / f"enhanced_model_{timestamp}.joblib"
        vectorizer_file = models_dir / f"vectorizer_{timestamp}.joblib"
        
        joblib.dump(self.model, model_file)
        joblib.dump(self.vectorizer, vectorizer_file)
        joblib.dump(self.label_encoder, models_dir / f"label_encoder_{timestamp}.joblib")
        
        print(f"💾 模型已保存到: {model_file}")
        
        # 保存结果
        results_file = output_path / f"enhanced_results_{timestamp}.json"
        import json
        
        # 转换numpy数组为列表以便JSON序列化
        results_for_json = {}
        for key, value in self.results.items():
            if isinstance(value, np.ndarray):
                results_for_json[key] = value.tolist()
            else:
                results_for_json[key] = value
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_for_json, f, indent=2, ensure_ascii=False)
        
        print(f"📊 结果已保存到: {results_file}")
        
        # 生成可视化图表
        self._generate_plots(output_path, timestamp)
    
    def _generate_plots(self, output_path: Path, timestamp: str):
        """生成可视化图表"""
        plots_dir = output_path / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # 混淆矩阵
        plt.figure(figsize=(12, 10))
        cm = confusion_matrix(self.results['y_test'], self.results['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title('混淆矩阵 - 增强版模型')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.tight_layout()
        plt.savefig(plots_dir / f"confusion_matrix_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # F1分数对比
        plt.figure(figsize=(12, 8))
        classes = self.label_encoder.classes_
        f1_scores = self.results['f1_per_class']
        
        bars = plt.bar(range(len(classes)), f1_scores, color='skyblue', alpha=0.7)
        plt.xlabel('类别')
        plt.ylabel('F1分数')
        plt.title('各类别F1分数 - 增强版模型')
        plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
        plt.ylim(0, 1)
        
        # 添加数值标签
        for i, (bar, score) in enumerate(zip(bars, f1_scores)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(plots_dir / f"f1_scores_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 图表已保存到: {plots_dir}")
    
    def run_training(self, data_file: str, sample_size: int = None, 
                    model_type: str = 'gradient_boosting', test_size: float = 0.2):
        """运行完整训练流程"""
        print("🧪 开始增强版模型训练")
        print("=" * 60)
        
        # 加载数据
        df, label_column, text_column = self.load_and_prepare_data(data_file, sample_size)
        
        # 准备特征和标签
        X = df[text_column].fillna('')
        y = df[label_column]
        
        # 编码标签
        y_encoded = self.label_encoder.fit_transform(y)
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        print(f"\n📊 训练集: {len(X_train)} 条记录")
        print(f"📊 测试集: {len(X_test)} 条记录")
        
        # 创建和训练模型
        self.create_model(model_type)
        y_pred, y_pred_proba, X_test_vec = self.train_model(X_train, y_train, X_test, y_test)
        
        # 评估模型
        results = self.evaluate_model(y_test, y_pred, y_pred_proba)
        
        # 保存结果
        self.save_results()
        
        return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="增强版基线模型训练脚本")
    parser.add_argument("--data-file", 
                       default="DATA_OUTPUT/training_data/combined_dataset_20250802_131542.csv",
                       help="训练数据文件路径")
    parser.add_argument("--sample-size", type=int, default=None,
                       help="每类样本数（None表示使用全部数据）")
    parser.add_argument("--model-type", 
                       choices=['gradient_boosting', 'lightgbm', 'random_forest'],
                       default='gradient_boosting',
                       help="模型类型")
    parser.add_argument("--test-size", type=float, default=0.2,
                       help="测试集比例")
    
    args = parser.parse_args()
    
    # 检查数据文件
    if not Path(args.data_file).exists():
        print(f"❌ 数据文件不存在: {args.data_file}")
        sys.exit(1)
    
    # 创建训练器并运行
    trainer = EnhancedBaselineModel()
    results = trainer.run_training(
        data_file=args.data_file,
        sample_size=args.sample_size,
        model_type=args.model_type,
        test_size=args.test_size
    )
    
    print("\n✅ 训练完成！")


if __name__ == "__main__":
    main() 