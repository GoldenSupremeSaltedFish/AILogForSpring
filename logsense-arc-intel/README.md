# Intel Arc GPU 日志分类项目

## 🎯 项目概述

本项目使用Intel Arc GPU训练和部署深度学习日志分类模型，实现了从TF-IDF + LightGBM到GPU加速深度学习的完整转型。

## 🚀 核心特性

- ✅ **Intel Arc GPU支持** - 使用XPU设备进行训练和推理
- ✅ **双通道深度学习模型** - TextCNN + 结构化特征融合
- ✅ **词汇表持久化** - 训练时保存词汇表，确保验证一致性
- ✅ **高质量数据处理** - 结构化特征提取和数据增强
- ✅ **工程化验证** - 解耦的验证脚本，支持GPU推理

## 📁 项目结构

```
logsense-arc-intel/
├── main.py                        # 🚀 主入口文件
├── feature_enhanced_model.py      # 🎯 最终训练脚本（包含词汇表保存）
├── final_model_runner.py          # 🎯 最终验证脚本（GPU推理）
├── prepare_full_data.py           # 📊 数据处理脚本
├── requirements.txt               # 📦 依赖文件
├── README.md                      # 📖 项目说明
├── TRAINING_SUMMARY.md            # 📋 训练总结
├── data/                          # 📁 数据目录
├── results/                       # 📁 训练结果
├── final_validation_results/      # 📁 验证结果
├── tools/                         # 🔧 工具脚本集合
│   ├── adapt_issue_data.py        # 数据适配工具
│   ├── check_model.py             # 模型检查工具
│   ├── check_weights.py           # 权重检查工具
│   ├── filter_known_labels.py     # 标签过滤工具
│   ├── improved_data_processor.py # 数据处理工具
│   ├── prepare_issue_data.py      # 问题数据准备工具
│   ├── simple_text_validator.py   # 简单文本验证工具
│   ├── simple_validation_runner.py # 简单验证运行工具
│   ├── validation_data_adapter.py # 验证数据适配工具
│   └── fixed_model_runner.py      # 修复的模型运行工具
├── scripts/                       # 📁 训练脚本
├── models/                        # 📁 模型文件
└── utils/                         # 📁 工具函数
```

## 🎮 硬件要求

- **GPU**: Intel Arc GPU (支持XPU)
- **Python**: 3.8+
- **PyTorch**: Intel XPU版本

## 🛠️ 快速开始

### 1. 环境安装

```bash
# 安装Intel PyTorch XPU版本
pip install torch torchvision torchaudio --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/cpu/us/

# 安装其他依赖
pip install -r requirements.txt
```

### 2. 数据准备

```bash
python main.py --mode prepare
```

### 3. 模型训练

```bash
python main.py --mode train
```

### 4. 模型验证

```bash
python main.py --mode validate --model_path "results/models/feature_enhanced_model_*.pth"
```

### 5. 模型检查

```bash
python main.py --mode check --model_path "results/models/feature_enhanced_model_*.pth"
```

### 6. 使用工具脚本

**方式一：直接运行工具脚本**
```bash
# 数据适配
python tools/adapt_issue_data.py

# 模型检查
python tools/check_model.py

# 权重检查
python tools/check_weights.py
```

**方式二：通过兼容性运行器（推荐）**
```bash
# 数据适配
python run_tool.py adapt_issue_data

# 模型检查
python run_tool.py check_model

# 权重检查
python run_tool.py check_weights
```

**方式三：通过主入口文件**
```bash
# 模型检查
python main.py --mode check --model_path "results/models/best_model.pth"
```

## 🏗️ 模型架构

### 双通道深度学习模型

1. **文本通道 (TextCNN)**
   - Embedding层 (128维)
   - 多尺度卷积 (3,4,5窗口)
   - MaxPooling + Dropout

2. **结构化特征通道 (MLP)**
   - 1018维特征输入
   - 256 → 128维隐藏层
   - ReLU + Dropout

3. **特征融合层**
   - 文本特征 + 结构化特征
   - 256维融合层
   - 最终分类层

## 📊 性能指标

- **训练准确率**: 97.8%
- **验证准确率**: 94.7%
- **全量数据准确率**: 38.2% (数据分布差异导致)
- **GPU加速**: Intel Arc XPU设备

## 🔧 核心功能

### 训练脚本 (`feature_enhanced_model.py`)
- ✅ Intel Arc GPU (XPU) 支持
- ✅ 词汇表构建和保存
- ✅ 结构化特征提取
- ✅ 早停和模型保存
- ✅ 训练历史记录

### 验证脚本 (`final_model_runner.py`)
- ✅ GPU推理支持
- ✅ 词汇表加载
- ✅ 批量预测
- ✅ 详细性能报告
- ✅ 结果保存

### 数据处理 (`prepare_full_data.py`)
- ✅ 日志清洗和预处理
- ✅ 结构化特征提取
- ✅ 数据平衡和质量增强
- ✅ 多类别支持

## 📈 训练历史

- **模型**: `feature_enhanced_model_20250810_222550.pth`
- **词汇表大小**: 4146
- **类别数**: 9
- **训练轮数**: 15 epochs
- **最佳验证准确率**: 94.67%

## 🎯 使用场景

1. **日志分类**: 自动识别日志类型
2. **异常检测**: 快速定位问题日志
3. **运维监控**: 实时日志分析
4. **性能优化**: GPU加速推理

## 🔍 验证结果

最新验证结果保存在 `final_validation_results/` 目录中，包含：
- JSON格式的详细指标
- 文本格式的分类报告
- 混淆矩阵和性能分析

## 📝 注意事项

1. **GPU要求**: 需要Intel Arc GPU和XPU驱动
2. **数据一致性**: 验证时使用相同的数据分布
3. **词汇表**: 训练时保存的词汇表必须用于验证
4. **特征维度**: 确保1018维结构化特征

## 🤝 贡献

欢迎提交Issue和Pull Request来改进项目。

## �� 许可证

MIT License 