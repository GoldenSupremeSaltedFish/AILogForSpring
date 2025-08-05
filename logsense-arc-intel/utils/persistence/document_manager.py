#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文档管理器 - 处理README和文档的生成
"""

import os
import logging
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger(__name__)


class DocumentManager:
    """文档管理器"""
    
    def __init__(self, session_dir: str, timestamp: str):
        self.session_dir = session_dir
        self.timestamp = timestamp
    
    def create_readme(self, training_info: Dict[str, Any]) -> str:
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