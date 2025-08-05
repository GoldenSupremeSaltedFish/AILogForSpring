# LogSense Intel Arc GPU 版本

基于Intel Arc GPU的日志分类系统，使用深度学习方法替代传统的TF-IDF + LightGBM方案。

## 🎯 项目特点

- **Intel Arc GPU 加速**: 完全支持Intel Arc显卡训练和推理
- **模块化设计**: 清晰的代码结构，易于维护和扩展
- **多种模型支持**: TextCNN、FastText等轻量级模型
- **完整的训练流程**: 从数据预处理到模型部署的全流程

## 📁 项目结构

```
logsense-arc-intel/
├── models/                 # 模型定义
│   ├── __init__.py
│   ├── textcnn.py         # TextCNN模型
│   ├── fasttext.py        # FastText模型
│   └── model_factory.py   # 模型工厂
├── data/                   # 数据处理
│   ├── __init__.py
│   ├── dataset.py         # 数据集类
│   ├── data_loader.py     # 数据加载器
│   └── preprocessor.py    # 数据预处理器
├── utils/                  # 工具模块
│   ├── __init__.py
│   ├── gpu_detector.py    # GPU检测器
│   ├── trainer_utils.py   # 训练工具
│   ├── model_saver.py     # 模型保存器
│   └── metrics.py         # 指标计算
├── api/                    # API服务
│   ├── __init__.py
│   ├── server.py          # 主服务器
│   └── predictor.py       # 预测器
├── config/                 # 配置文件
│   ├── __init__.py
│   └── settings.py        # 全局设置
├── scripts/                # 脚本文件
│   ├── train.py           # 训练脚本
│   ├── test.py            # 测试脚本
│   └── deploy.py          # 部署脚本
├── requirements.txt        # 依赖文件
└── README.md              # 项目文档
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装Intel Extension for PyTorch
pip install intel-extension-for-pytorch

# 安装其他依赖
pip install -r requirements.txt
```

### 2. 检查GPU状态

```bash
python scripts/check_gpu.py
```

### 3. 训练模型

```bash
python scripts/train.py --model textcnn --data DATA_OUTPUT/processed_logs.csv
```

### 4. 启动API服务

```bash
python scripts/deploy.py --model results/models/best_model.pth
```

## 🔧 配置说明

### 模型配置

- **TextCNN**: 适合日志分类的卷积神经网络
- **FastText**: 轻量级的文本分类模型

### GPU配置

- 自动检测Intel Arc GPU
- 支持XPU设备加速
- 自动回退到CPU

## 📊 性能优化

- 使用Intel Extension for PyTorch
- 支持ONNX模型导出
- 优化的数据加载和预处理
- 内存使用监控

## 🛠️ 开发指南

### 添加新模型

1. 在`models/`目录下创建新模型文件
2. 实现`forward()`方法
3. 在`model_factory.py`中注册模型

### 自定义数据预处理

1. 继承`LogPreprocessor`类
2. 重写预处理方法
3. 在训练脚本中使用

## 📈 监控和日志

- 训练过程实时监控
- GPU使用率统计
- 模型性能指标
- 详细的日志记录

## 🤝 贡献指南

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 创建Pull Request

## 📄 许可证

MIT License

## 📞 联系方式

如有问题，请提交Issue或联系开发团队。 