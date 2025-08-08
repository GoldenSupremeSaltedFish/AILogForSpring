# AILogForSpring 项目

## 项目概述
基于AI的智能日志处理和分析平台，支持多种日志格式的自动清洗、分类、标签化和机器学习分析。项目集成了传统机器学习方法和深度学习模型，提供端到端的日志分析解决方案。

## 核心功能

### 🔍 智能日志处理
- **自动标签系统**: 基于机器学习的半自动日志标签辅助器
- **数据清洗**: 智能去重、格式标准化、质量分析
- **分类系统**: 多级日志分类和异常检测
- **质量评估**: 自动评估日志数据质量和完整性

### 🤖 机器学习分析
- **Intel Arc GPU支持**: 基于Intel Arc显卡的深度学习训练
- **多模型架构**: TextCNN、FastText、注意力机制等
- **分阶段训练**: 支持大规模数据的分阶段训练策略
- **模型比较**: 多种模型性能对比和优化

### 🛠️ 实用工具
- **批处理脚本**: Windows一键运行脚本
- **API服务**: RESTful API接口服务
- **数据可视化**: 训练过程和结果可视化
- **持久化管理**: 完整的训练状态和模型管理

## 快速开始

### 1. 环境准备
```bash
# 安装基础依赖
pip install -r logsense-arc-intel/requirements.txt

# 安装Intel GPU支持（可选）
pip install intel-extension-for-pytorch
```

### 2. 日志处理流程

#### 自动标签处理
```bash
# 使用批处理脚本（推荐）
batch-scripts/run_auto_labeler.bat

# 或直接运行
python log-processing/auto_labeler.py
```

#### 数据清洗和平衡
```bash
# 清洗数据并创建训练集
batch-scripts/run_data_cleaner.bat

# 或直接运行
python log-processing/data_cleaner.py --combined
```

#### 质量分析
```bash
# 分析日志质量
batch-scripts/run_quality_analyzer.bat

# 或直接运行
python log-processing/quality_analyzer.py
```

### 3. 机器学习训练

#### Intel Arc GPU训练
```bash
cd logsense-arc-intel
python start_training.py
```

#### 分阶段训练
```bash
# 使用批处理脚本
batch-scripts/run_enhanced_training.bat

# 或直接运行
python staged_training.py
```

### 4. API服务启动
```bash
# GPU版本API
python logsense-arc-intel/api_server_gpu.py

# 或使用批处理脚本
batch-scripts/run_enhanced_training.bat
```

## 项目模块说明

### 📁 log-processing/
日志处理核心模块，包含：
- `auto_labeler.py`: 智能日志标签辅助器
- `data_cleaner.py`: 数据清洗和平衡工具
- `quality_analyzer.py`: 日志质量分析器
- `log_categorizer.py`: 日志分类器
- `log_reviewer.py`: 日志审查工具
- `enhanced_pre_classifier.py`: 增强预分类器

### 📁 logsense-arc-intel/
基于Intel Arc GPU的深度学习模块：
- 支持TextCNN、FastText等多种模型
- 完整的训练流程和模型管理
- GPU加速训练和推理
- 分阶段训练策略

### 📁 logsense-gpu/
GPU优化版本，包含：
- 基线模型训练
- 增强模型开发
- API服务接口
- 结果可视化

### 📁 logsense-xpu/
XPU优化版本，专注于：
- 数据预处理和清洗
- 基线模型实现
- 轻量级API服务

### 📁 batch-scripts/
Windows批处理脚本集合：
- 一键运行各种处理流程
- 自动化数据清洗和训练
- 简化操作流程

### 📁 gateway-tools/
Gateway日志专用处理工具：
- Gateway日志格式识别
- 专用清洗和过滤
- 批量处理支持

## 使用示例

### 完整工作流程
1. **数据准备**: 将日志文件放入`DATA_OUTPUT/`目录
2. **自动标签**: 运行`run_auto_labeler.bat`进行智能标签
3. **数据清洗**: 运行`run_data_cleaner.bat`清洗和平衡数据
4. **质量分析**: 运行`run_quality_analyzer.bat`评估数据质量
5. **模型训练**: 运行`run_enhanced_training.bat`开始训练
6. **API部署**: 启动API服务进行实时预测

### 支持的日志格式
- **Gateway日志**: 自动识别和处理
- **MQTT日志**: 支持消息队列日志
- **系统日志**: 通用系统日志格式
- **应用日志**: 自定义应用日志格式

## 技术特点

### 🎯 智能化处理
- 基于机器学习的自动标签系统
- 智能数据清洗和去重
- 多级分类和异常检测

### ⚡ 高性能计算
- Intel Arc GPU加速支持
- 分阶段训练策略
- 内存优化和并行处理

### 🔧 模块化设计
- 清晰的功能分离
- 可扩展的架构
- 完整的工具链

### 📊 可视化支持
- 训练过程监控
- 结果可视化
- 性能指标分析

## 开发团队
AI日志分析项目团队

**项目维护**: 持续更新和优化，支持多种日志格式和机器学习模型
