# LogSense GPU - 多平台日志分析模型

## 项目概述

LogSense GPU是一个支持多平台计算的日志分析模型训练框架，专门用于日志分类和异常检测。

### 主要特性

- 🖥️ **多平台支持**: 自动检测GPU/CPU环境，优化计算配置
- 🧪 **小样本验证**: 支持小样本实验，快速验证模型效果
- 🤖 **多种模型**: 支持GradientBoosting、LightGBM等模型
- 📊 **可视化分析**: 自动生成混淆矩阵和性能图表
- ⚡ **性能监控**: 实时监控训练性能和资源使用

## 目录结构

```
logsense-gpu/
├── scripts/              # 训练脚本
│   └── baseline_model.py # 小样本验证 + Base模型训练
├── utils/                # 工具脚本
│   └── platform_utils.py # 多平台计算支持
├── models/               # 保存训练好的模型
├── data/                 # 数据文件
├── config/               # 配置文件
└── results/              # 实验结果和图表
```

## 快速开始

### 1. 环境准备

```bash
# 安装基础依赖
pip install pandas scikit-learn matplotlib seaborn

# 可选：安装LightGBM（用于更快的训练）
pip install lightgbm

# 可选：安装PyTorch（用于GPU检测）
pip install torch
```

### 2. 运行小样本验证

```bash
# 使用批处理脚本（推荐）
batch-scripts/run_baseline_model.bat

# 或直接运行Python脚本
python logsense-gpu/scripts/baseline_model.py --sample-size 500 --model-type gradient_boosting
```

### 3. 查看结果

训练完成后，结果将保存在：
- `logsense-gpu/results/models/` - 训练好的模型
- `logsense-gpu/results/plots/` - 可视化图表
- `logsense-gpu/results/` - 实验结果JSON文件

## 详细使用说明

### 小样本验证实验

```bash
# 基本用法
python logsense-gpu/scripts/baseline_model.py

# 自定义参数
python logsense-gpu/scripts/baseline_model.py \
    --sample-size 300 \
    --model-type lightgbm \
    --data-file DATA_OUTPUT/training_dataset_20250802_013437.csv
```

### 参数说明

- `--sample-size N`: 每类样本数（默认500）
- `--model-type TYPE`: 模型类型（gradient_boosting/lightgbm）
- `--data-file PATH`: 训练数据文件路径
- `--output-dir PATH`: 输出目录
- `--gpu`: 启用GPU加速

### 支持的类别

当前支持5个主要日志类别：
1. `stack_exception` - 堆栈异常
2. `connection_issue` - 连接问题
3. `database_exception` - 数据库异常
4. `auth_authorization` - 认证授权
5. `memory_performance` - 内存性能

## 多平台计算支持

### GPU检测

系统会自动检测NVIDIA GPU并优化配置：

```python
from logsense-gpu.utils.platform_utils import setup_environment

# 检测系统环境
detector, optimizer, config = setup_environment()

# 获取推荐配置
print(f"计算设备: {config['device']}")
print(f"批处理大小: {config['batch_size']}")
print(f"工作进程数: {config['num_workers']}")
```

### 性能监控

```python
from logsense-gpu.utils.platform_utils import PerformanceMonitor

# 创建性能监控器
monitor = PerformanceMonitor()
monitor.start_monitoring()

# 训练过程中记录指标
monitor.record_metrics()

# 训练结束后查看摘要
monitor.print_summary()
```

## 模型训练流程

### 1. 数据加载和采样

```python
# 加载训练数据
df = pd.read_csv('training_dataset.csv')

# 小样本采样（每类500条）
sampled_data = []
for class_name in target_classes:
    class_data = df[df['label'] == class_name]
    sampled = class_data.sample(n=500, random_state=42)
    sampled_data.append(sampled)

df_sampled = pd.concat(sampled_data, ignore_index=True)
```

### 2. 特征工程

```python
# TF-IDF向量化
vectorizer = TfidfVectorizer(
    max_features=3000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
```

### 3. 模型训练

```python
# GradientBoosting模型
model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)

model.fit(X_train_vec, y_train)
```

### 4. 模型评估

```python
# 预测和评估
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"准确率: {accuracy:.4f}")
print(report)
```

## 实验结果示例

### 训练过程输出

```
🧪 开始小样本验证实验
============================================================
📂 加载数据文件: DATA_OUTPUT/training_dataset_20250802_013437.csv
📊 原始数据: 4764 条记录
🔍 过滤后数据: 2500 条记录

📈 最终类别分布:
  stack_exception: 500 条 (20.0%)
  connection_issue: 500 条 (20.0%)
  database_exception: 500 条 (20.0%)
  auth_authorization: 500 条 (20.0%)
  memory_performance: 500 条 (20.0%)

📊 训练集: 2000 条记录
📊 测试集: 500 条记录

🚀 开始训练模型...
📝 向量化训练数据...
📊 特征维度: 3000
🏋️ 训练模型...
⏱️ 训练时间: 15.23 秒

📊 模型评估结果:
准确率: 0.8920
宏平均F1: 0.8915
加权平均F1: 0.8920
```

### 生成的图表

- **混淆矩阵**: 显示各类别的预测准确度
- **F1分数图**: 各类别的F1分数对比
- **性能监控**: 训练过程中的资源使用情况

## 扩展功能

### 添加新的模型类型

```python
# 在baseline_model.py中添加新的模型
def create_custom_model(self):
    from sklearn.ensemble import RandomForestClassifier
    
    self.model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
```

### 自定义特征工程

```python
# 添加自定义特征
def extract_custom_features(self, text):
    features = {}
    features['length'] = len(text)
    features['word_count'] = len(text.split())
    features['has_error'] = 'error' in text.lower()
    return features
```

## 故障排除

### 常见问题

1. **GPU不可用**
   - 检查NVIDIA驱动是否正确安装
   - 确认PyTorch支持CUDA

2. **内存不足**
   - 减少`sample_size`参数
   - 降低`max_features`参数

3. **训练速度慢**
   - 使用LightGBM替代GradientBoosting
   - 启用GPU加速
   - 调整批处理大小

### 性能优化建议

- **GPU训练**: 使用NVIDIA GPU可显著提升训练速度
- **内存优化**: 根据系统内存调整批处理大小
- **并行处理**: 利用多核CPU进行特征工程

## 贡献指南

欢迎提交Issue和Pull Request来改进项目！

### 开发环境设置

```bash
# 克隆项目
git clone <repository-url>
cd AILogForSpring

# 安装开发依赖
pip install -r requirements-dev.txt

# 运行测试
python -m pytest tests/
```

## 许可证

本项目采用MIT许可证。 