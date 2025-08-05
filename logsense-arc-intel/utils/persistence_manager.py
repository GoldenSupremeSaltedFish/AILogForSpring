#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的持久化管理器 - 整合所有子管理器
"""

import os
import logging
from datetime import datetime
from typing import Dict, Any, List

from .persistence import (
    ConfigManager, ModelManager, MetricsManager,
    VisualizationManager, DataManager, DocumentManager
)

logger = logging.getLogger(__name__)


class TrainingPersistenceManager:
    """简化的训练结果持久化管理器"""
    
    def __init__(self, base_dir: str = "results/history_results"):
        self.base_dir = base_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(base_dir, f"training_session_{self.timestamp}")
        
        # 创建目录结构
        self._create_directory_structure()
        
        # 初始化子管理器
        self.config_manager = ConfigManager(self.session_dir, self.timestamp)
        self.model_manager = ModelManager(self.session_dir, self.timestamp)
        self.metrics_manager = MetricsManager(self.session_dir, self.timestamp)
        self.visualization_manager = VisualizationManager(self.session_dir, self.timestamp)
        self.data_manager = DataManager(self.session_dir, self.timestamp)
        self.document_manager = DocumentManager(self.session_dir, self.timestamp)
        
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
    
    # 配置管理
    def save_training_config(self, config: Dict[str, Any]) -> str:
        return self.config_manager.save_training_config(config)
    
    # 模型管理
    def save_model(self, model, model_name: str, model_config: Dict[str, Any] = None) -> str:
        return self.model_manager.save_model(model, model_name, model_config)
    
    # 指标管理
    def save_training_metrics(self, metrics: Dict[str, Any]) -> str:
        return self.metrics_manager.save_training_metrics(metrics)
    
    def save_training_history(self, history: Dict[str, List[float]]) -> str:
        return self.metrics_manager.save_training_history(history)
    
    def save_classification_report(self, y_true: List[int], y_pred: List[int], 
                                 class_names: List[str] = None) -> str:
        return self.metrics_manager.save_classification_report(y_true, y_pred, class_names)
    
    def save_session_summary(self, summary: Dict[str, Any]) -> str:
        return self.metrics_manager.save_session_summary(summary)
    
    # 可视化管理
    def save_plots(self, history: Dict[str, List[float]]) -> str:
        return self.visualization_manager.save_training_plots(history)
    
    def save_confusion_matrix(self, y_true: List[int], y_pred: List[int], 
                            class_names: List[str] = None) -> str:
        return self.visualization_manager.save_confusion_matrix(y_true, y_pred, class_names)
    
    # 数据管理
    def save_data_info(self, data_info: Dict[str, Any]) -> str:
        return self.data_manager.save_data_info(data_info)
    
    def copy_data_files(self, data_path: str) -> str:
        return self.data_manager.copy_data_files(data_path)
    
    # 文档管理
    def create_readme(self, training_info: Dict[str, Any]) -> str:
        return self.document_manager.create_readme(training_info)
    
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