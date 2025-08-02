#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估工具模块
"""

from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from datetime import datetime
import json

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self):
        # 企业级目标指标
        self.enterprise_targets = {
            'accuracy': 0.90,
            'macro_f1': 0.88,
            'weighted_f1': 0.90,
            'recall': 0.85,
            'precision': 0.90
        }
        self.results = {}
    
    def evaluate_model(self, y_test, y_pred, y_pred_proba, label_encoder):
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
            'y_pred_proba': y_pred_proba,
            'label_encoder': label_encoder
        }
        
        # 输出企业级评估表格
        self._print_enterprise_metrics()
        
        # 输出详细分类报告
        print("\n📋 详细分类报告:")
        print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
        
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
        
        # 保存结果
        results_file = output_path / f"enhanced_results_{timestamp}.json"
        
        # 转换numpy数组为列表以便JSON序列化
        results_for_json = {}
        for key, value in self.results.items():
            if key == 'label_encoder':
                continue  # 跳过label_encoder
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
        
        label_encoder = self.results['label_encoder']
        
        # 混淆矩阵
        plt.figure(figsize=(12, 10))
        cm = confusion_matrix(self.results['y_test'], self.results['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=label_encoder.classes_,
                   yticklabels=label_encoder.classes_)
        plt.title('混淆矩阵 - 增强版模型')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.tight_layout()
        plt.savefig(plots_dir / f"confusion_matrix_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # F1分数对比
        plt.figure(figsize=(12, 8))
        classes = label_encoder.classes_
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