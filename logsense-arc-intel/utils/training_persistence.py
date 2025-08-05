#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练结果持久化管理器
"""

import os
import json
import shutil
import logging
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logger = logging.getLogger(__name__)


class TrainingPersistenceManager:
    """训练结果持久化管理器"""
    
    def __init__(self, base_dir: str = "results/history_results"):
        self.base_dir = base_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(base_dir, f"training_session_{self.timestamp}")
        
        # 创建目录结构
        self._create_directory_structure()
        
        logger.info(f"📁 训练会话目录: {self.session_dir}")
    
    def _create_directory_structure(self):
        """创建目录结构"""
        directories = [
            self.session_dir,
            os.path.join(self.session_dir, "models"),
            os.path.join(self.session_dir, "logs"),
            os.path.join(self.session_dir, "plots"),
            os.path.join(self.session_dir, "data"),
            os.path.join(self.session_dir, "configs"),
            os.path.join(self.session_dir, "metrics")
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def save_training_config(self, config: Dict[str, Any]):
        """保存训练配置"""
        config_file = os.path.join(self.session_dir, "configs", "training_config.json")
        
        # 添加时间戳信息
        config['timestamp'] = self.timestamp
        config['session_dir'] = self.session_dir
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📋 训练配置已保存: {config_file}")
        return config_file
    
    def save_model(self, model, model_name: str, model_config: Dict[str, Any] = None):
        """保存模型文件"""
        model_dir = os.path.join(self.session_dir, "models")
        model_path = os.path.join(model_dir, f"{model_name}_{self.timestamp}.pth")
        
        # 保存模型
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': model_config or model.get_config(),
            'timestamp': self.timestamp,
            'session_dir': self.session_dir
        }, model_path)
        
        logger.info(f"💾 模型已保存: {model_path}")
        return model_path
    
    def save_training_metrics(self, metrics: Dict[str, Any]):
        """保存训练指标"""
        metrics_file = os.path.join(self.session_dir, "metrics", "training_metrics.json")
        
        # 添加时间戳信息
        metrics['timestamp'] = self.timestamp
        metrics['session_dir'] = self.session_dir
        
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📊 训练指标已保存: {metrics_file}")
        return metrics_file
    
    def save_training_history(self, history: Dict[str, List[float]]):
        """保存训练历史"""
        history_file = os.path.join(self.session_dir, "metrics", "training_history.json")
        
        # 添加时间戳信息
        history['timestamp'] = self.timestamp
        history['session_dir'] = self.session_dir
        
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📈 训练历史已保存: {history_file}")
        return history_file
    
    def save_data_info(self, data_info: Dict[str, Any]):
        """保存数据信息"""
        data_file = os.path.join(self.session_dir, "data", "data_info.json")
        
        # 添加时间戳信息
        data_info['timestamp'] = self.timestamp
        data_info['session_dir'] = self.session_dir
        
        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump(data_info, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📂 数据信息已保存: {data_file}")
        return data_file
    
    def save_plots(self, history: Dict[str, List[float]]):
        """保存训练图表"""
        plots_dir = os.path.join(self.session_dir, "plots")
        
        # 创建损失曲线图
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(history.get('train_loss', []), label='训练损失')
        plt.plot(history.get('val_loss', []), label='验证损失')
        plt.title('损失曲线')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.plot(history.get('train_acc', []), label='训练准确率')
        plt.plot(history.get('val_acc', []), label='验证准确率')
        plt.title('准确率曲线')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        plt.plot(history.get('val_f1', []), label='验证F1分数')
        plt.title('F1分数曲线')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        plt.plot(history.get('learning_rate', []), label='学习率')
        plt.title('学习率变化')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(plots_dir, f"training_plots_{self.timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 训练图表已保存: {plot_path}")
        return plot_path
    
    def save_confusion_matrix(self, y_true: List[int], y_pred: List[int], 
                            class_names: List[str] = None):
        """保存混淆矩阵"""
        from sklearn.metrics import confusion_matrix
        
        plots_dir = os.path.join(self.session_dir, "plots")
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        
        # 绘制混淆矩阵
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('混淆矩阵')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        
        cm_path = os.path.join(plots_dir, f"confusion_matrix_{self.timestamp}.png")
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 混淆矩阵已保存: {cm_path}")
        return cm_path
    
    def save_classification_report(self, y_true: List[int], y_pred: List[int], 
                                 class_names: List[str] = None):
        """保存分类报告"""
        from sklearn.metrics import classification_report
        
        reports_dir = os.path.join(self.session_dir, "metrics")
        
        # 生成分类报告
        report = classification_report(y_true, y_pred, 
                                    target_names=class_names,
                                    output_dict=True)
        
        # 保存为JSON
        report_path = os.path.join(reports_dir, f"classification_report_{self.timestamp}.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📋 分类报告已保存: {report_path}")
        return report_path
    
    def save_session_summary(self, summary: Dict[str, Any]):
        """保存会话总结"""
        summary_file = os.path.join(self.session_dir, "session_summary.json")
        
        # 添加时间戳信息
        summary['timestamp'] = self.timestamp
        summary['session_dir'] = self.session_dir
        summary['created_at'] = datetime.now().isoformat()
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📝 会话总结已保存: {summary_file}")
        return summary_file
    
    def copy_data_files(self, data_path: str):
        """复制数据文件"""
        data_dir = os.path.join(self.session_dir, "data")
        data_filename = os.path.basename(data_path)
        data_copy_path = os.path.join(data_dir, data_filename)
        
        shutil.copy2(data_path, data_copy_path)
        logger.info(f"📁 数据文件已复制: {data_copy_path}")
        return data_copy_path
    
    def create_readme(self, training_info: Dict[str, Any]):
        """创建README文件"""
        readme_path = os.path.join(self.session_dir, "README.md")
        
        readme_content = f"""# 训练会话记录

## 基本信息
- **会话时间**: {self.timestamp}
- **会话目录**: {self.session_dir}
- **创建时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 训练配置
- **模型类型**: {training_info.get('model_type', 'N/A')}
- **训练轮数**: {training_info.get('epochs', 'N/A')}
- **批次大小**: {training_info.get('batch_size', 'N/A')}
- **学习率**: {training_info.get('learning_rate', 'N/A')}

## 训练结果
- **最佳验证准确率**: {training_info.get('best_val_acc', 'N/A')}%
- **最佳F1分数**: {training_info.get('best_f1', 'N/A')}
- **测试集准确率**: {training_info.get('test_acc', 'N/A')}%
- **测试集F1分数**: {training_info.get('test_f1', 'N/A')}

## 文件结构
```
{self.session_dir}/
├── models/           # 模型文件
├── logs/            # 日志文件
├── plots/           # 图表文件
├── data/            # 数据文件
├── configs/         # 配置文件
├── metrics/         # 指标文件
└── README.md        # 本文件
```

## 使用说明
1. 模型文件位于 `models/` 目录
2. 训练图表位于 `plots/` 目录
3. 详细指标位于 `metrics/` 目录
4. 配置文件位于 `configs/` 目录
"""
        
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        logger.info(f"📖 README文件已创建: {readme_path}")
        return readme_path
    
    def get_session_info(self) -> Dict[str, Any]:
        """获取会话信息"""
        return {
            'timestamp': self.timestamp,
            'session_dir': self.session_dir,
            'created_at': datetime.now().isoformat()
        }


def create_persistence_manager(base_dir: str = "results/history_results") -> TrainingPersistenceManager:
    """创建持久化管理器"""
    return TrainingPersistenceManager(base_dir) 