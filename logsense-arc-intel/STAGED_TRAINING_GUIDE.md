# Intel Arc GPU 分阶段训练指南

## 🎯 训练策略

采用分阶段训练策略，先使用小数据集快速验证系统功能，再使用完整数据集进行正式训练。

## 📊 数据集配置

### 小数据集（快速验证）
- **每类样本数**: 100条
- **总样本数**: ~300条
- **训练轮数**: 3 epochs
- **用途**: 快速验证系统功能、调试问题

### 完整数据集（正式训练）
- **每类样本数**: 1500条
- **总样本数**: ~4500条
- **训练轮数**: 10 epochs
- **用途**: 正式训练、获得最佳性能

## 🚀 快速开始

### 一键启动（推荐）
```bash
cd logsense-arc-intel
run_staged_training.bat
```

### 分步执行

#### 1. 环境检查
```bash
python quick_start.py
```

#### 2. 准备分阶段数据
```bash
python scripts/prepare_data_staged.py
```

#### 3. 小数据集快速验证
```bash
python staged_training.py --skip_large
```

#### 4. 完整数据集正式训练
```bash
python staged_training.py --skip_small --skip_data_prep
```

## 📈 训练流程

### 阶段1: 小数据集验证
```bash
# 自动执行
python staged_training.py --skip_large

# 或手动执行
python scripts/train.py --model textcnn --data data/processed_logs_small.csv --epochs 3 --save_dir results/models_small
```

**预期结果**:
- 训练时间: 1-3分钟
- 准确率: 70-85%
- 内存使用: <2GB

### 阶段2: 完整数据集训练
```bash
# 自动执行
python staged_training.py --skip_small --skip_data_prep

# 或手动执行
python scripts/train.py --model textcnn --data data/processed_logs_large.csv --epochs 10 --save_dir results/models_large
```

**预期结果**:
- 训练时间: 10-20分钟
- 准确率: 85-95%
- 内存使用: <4GB

## 🔧 高级配置

### 自定义数据集大小
```bash
python scripts/prepare_data_staged.py --small_samples 50 --large_samples 1000
```

### 跳过特定阶段
```bash
# 只运行小数据集训练
python staged_training.py --skip_large

# 只运行完整数据集训练
python staged_training.py --skip_small --skip_data_prep

# 跳过数据准备
python staged_training.py --skip_data_prep
```

### 使用不同模型
```bash
# 使用FastText模型
python scripts/train.py --model fasttext --data data/processed_logs_small.csv --epochs 3
```

## 📊 性能监控

### 内存监控
```bash
# 监控训练过程
python scripts/memory_monitor.py --duration 300 --interval 10
```

### 训练进度
- 小数据集: 每epoch约30秒
- 完整数据集: 每epoch约2-3分钟

## 🛠️ 故障排除

### 常见问题

#### 1. 小数据集训练失败
**检查项**:
- 数据文件是否存在
- 数据格式是否正确
- GPU内存是否充足

**解决方案**:
```bash
# 检查数据
python scripts/prepare_data_staged.py --small_samples 50

# 使用更小的批次
python scripts/train.py --model textcnn --data data/processed_logs_small.csv --epochs 3 --batch_size 8
```

#### 2. 完整数据集OOM
**解决方案**:
```bash
# 减少每类样本数
python scripts/prepare_data_staged.py --large_samples 1000

# 使用更小的模型配置
python scripts/train_optimized.py --model textcnn --data data/processed_logs_large.csv --batch_size 8
```

#### 3. 训练时间过长
**优化方案**:
- 减少epoch数量
- 使用更小的模型
- 增加批次大小（如果内存允许）

## 📁 输出文件

### 数据文件
```
data/
├── processed_logs_small.csv    # 小数据集
└── processed_logs_large.csv    # 完整数据集
```

### 模型文件
```
results/
├── models_small/               # 小数据集模型
│   ├── arc_gpu_model_textcnn_best_*.pth
│   └── arc_gpu_model_textcnn_final_*.pth
└── models_large/               # 完整数据集模型
    ├── arc_gpu_model_textcnn_best_*.pth
    └── arc_gpu_model_textcnn_final_*.pth
```

## 🎯 最佳实践

### 1. 验证阶段
- 使用小数据集快速验证
- 检查GPU内存使用
- 确认数据加载正常
- 验证模型保存功能

### 2. 正式训练
- 使用完整数据集
- 监控训练进度
- 保存最佳模型
- 记录训练日志

### 3. 结果对比
- 比较两个阶段的准确率
- 分析训练时间差异
- 评估内存使用情况
- 选择最佳模型部署

## 📞 技术支持

如遇问题，请按以下顺序检查：

1. **环境检查**: `python quick_start.py`
2. **数据检查**: 确认CSV文件格式正确
3. **GPU检查**: 确认Intel Arc GPU可用
4. **内存检查**: 监控GPU内存使用情况

---

**开始你的分阶段训练之旅！** 🚀 