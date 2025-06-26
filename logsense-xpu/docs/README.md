# LogSense-XPU

基于Intel XPU加速的智能日志分类系统

## 项目概述

LogSense-XPU是一个利用Intel XPU硬件加速的日志分析和分类系统。系统能够自动处理、分析和分类各种类型的日志数据，帮助运维人员快速识别和定位问题。

## 项目结构

```
logsense-xpu/
├── data/
│   ├── logs.csv            # 原始日志数据
│   └── labels.csv          # 标签映射文件
├── preprocessing.py        # 日志预处理模块
├── embed.py                # 向量构造模块
├── train.py                # 模型训练模块
├── predict.py              # 预测接口模块
├── utils.py                # 公共工具函数
└── README.md               # 项目说明
```

## 功能模块

### 1. 数据预处理 (preprocessing.py)
- 日志清洗和标准化
- 敏感信息脱敏
- 特征提取
- 基础分类规则

### 2. 向量构造 (embed.py)
- 基于预训练模型的文本向量化
- Intel XPU加速处理
- 批量向量化支持

### 3. 模型训练 (train.py)
- 深度学习分类模型
- XPU优化训练过程
- 模型评估和保存

### 4. 预测服务 (predict.py)
- 单条/批量日志预测
- RESTful API接口
- 实时分类服务

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- Intel Extension for PyTorch
- Intel XPU驱动

## 快速开始

### 1. 环境准备
```bash
pip install torch torchvision
pip install intel-extension-for-pytorch
pip install pandas numpy scikit-learn
```

### 2. 数据预处理
```python
from preprocessing import LogPreprocessor

preprocessor = LogPreprocessor()
processed_data = preprocessor.process_logs("data/logs.csv")
```

### 3. 向量构造
```python
from embed import LogEmbedder

embedder = LogEmbedder()
embeddings = embedder.embed_logs(processed_data)
```

### 4. 模型训练
```python
from train import LogClassifier

classifier = LogClassifier()
classifier.train(embeddings, labels)
```

### 5. 预测使用
```python
from predict import LogPredictor

predictor = LogPredictor("model.pth")
result = predictor.predict_single("User login failed")
```

## 支持的日志类型

- 认证日志 (auth)
- 数据库日志 (database)
- 系统日志 (system)
- API日志 (api)
- 缓存日志 (cache)
- 支付日志 (payment)
- 错误日志 (error)
- 警告日志 (warning)
- 信息日志 (info)

## 性能优化

项目充分利用Intel XPU的并行计算能力：
- 向量化计算加速
- 批量处理优化
- 内存使用优化
- 实时推理加速

## 后续开发计划

- [ ] 完成embedding模块实现
- [ ] 实现深度学习模型训练
- [ ] 添加Web界面
- [ ] 支持更多日志格式
- [ ] 集成监控告警功能

## 许可证

MIT License 