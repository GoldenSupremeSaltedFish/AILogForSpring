#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
指标管理器 - 处理训练指标和历史的保存
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class MetricsManager:
    """指标管理器"""
    
    def __init__(self, session_dir: str, timestamp: str):
        self.session_dir = session_dir
        self.timestamp = timestamp
        self.metrics_dir = os.path.join(session_dir, "metrics")
        os.makedirs(self.metrics_dir, exist_ok=True)
    
    def save_training_metrics(self, metrics: Dict[str, Any]) -> str:
        """保存训练指标"""
        metrics_file = os.path.join(self.metrics_dir, "training_metrics.json")
        
        # 添加时间戳信息
        metrics['timestamp'] = self.timestamp
        metrics['session_dir'] = self.session_dir
        
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📊 训练指标已保存: {metrics_file}")
        return metrics_file
    
    def save_training_history(self, history: Dict[str, List[float]]) -> str:
        """保存训练历史"""
        history_file = os.path.join(self.metrics_dir, "training_history.json")
        
        # 添加时间戳信息
        history['timestamp'] = self.timestamp
        history['session_dir'] = self.session_dir
        
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📈 训练历史已保存: {history_file}")
        return history_file
    
    def save_classification_report(self, y_true: List[int], y_pred: List[int], 
                                 class_names: List[str] = None) -> str:
        """保存分类报告"""
        from sklearn.metrics import classification_report
        
        # 生成分类报告
        report = classification_report(y_true, y_pred, 
                                    target_names=class_names,
                                    output_dict=True)
        
        # 保存为JSON
        report_path = os.path.join(self.metrics_dir, f"classification_report_{self.timestamp}.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📋 分类报告已保存: {report_path}")
        return report_path
    
    def save_session_summary(self, summary: Dict[str, Any]) -> str:
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