# Baseline日志分类模型

本项目实现了基于TF-IDF + LightGBM和FastText的日志分类baseline模型，适配Intel XPU环境。

## 功能特性

- **TF-IDF + LightGBM**: 使用TF-IDF特征提取和LightGBM分类器
- **FastText**: 使用FastText进行文本分类
- **Intel XPU支持**: 适配Intel Arc GPU环境
- **自动数据加载**: 从分类好的日志目录自动加载数据
- **模型评估**: 提供详细的评估指标和可视化
- **模型保存**: 自动保存训练好的模型和向量器

## 环境要求

### 基础依赖
```bash
pip install -r requirements.txt
```

### Intel XPU环境（可选）
```bash
# 安装Intel Extension for PyTorch
pip install intel-extension-for-pytorch
```

## 使用方法

### 1. 快速开始

使用批处理脚本运行训练：
```bash
run_baseline_training.bat
```

### 2. 命令行训练

训练LightGBM模型：
```bash
python train_baseline.py --model_type lightgbm --data_dir ../../DATA_OUTPUT --use_xpu
```

训练FastText模型：
```bash
python train_baseline.py --model_type fasttext --data_dir ../../DATA_OUTPUT --use_xpu
```

### 3. 参数说明

- `--data_dir`: 数据目录路径（默认: ../../DATA_OUTPUT）
- `--model_type`: 模型类型，lightgbm或fasttext（默认: lightgbm）
- `--test_size`: 测试集比例（默认: 0.2）
- `--random_state`: 随机种子（默认: 42）
- `--use_xpu`: 是否使用Intel XPU
- `--output_dir`: 结果输出目录（默认: results）

### 4. 编程接口

```python
from core.baseline_model import BaselineLogClassifier

# 创建分类器
classifier = BaselineLogClassifier(model_type="lightgbm", use_xpu=True)

# 训练模型
results = classifier.train(data_dir="../../DATA_OUTPUT")

# 预测
texts = ["日志消息1", "日志消息2"]
predictions = classifier.predict(texts)

# 保存模型
classifier.save_model("models")

# 加载模型
classifier.load_model("models", "20250802_140000")
```

## 数据格式

模型期望的数据结构：
```
DATA_OUTPUT/
├── 01_堆栈异常_stack_exception/
│   ├── stack_exception_堆栈异常_20250802_205628.csv
│   └── ...
├── 02_数据库异常_database_exception/
│   └── ...
└── ...
```

CSV文件应包含以下列：
- `original_log`: 原始日志内容
- `log_level`: 日志级别（可选）
- `content_type`: 内容类型（可选）

## 模型配置

### TF-IDF参数
```python
tfidf_params = {
    'max_features': 10000,      # 最大特征数
    'ngram_range': (1, 2),      # n-gram范围
    'min_df': 2,                # 最小文档频率
    'max_df': 0.95              # 最大文档频率
}
```

### LightGBM参数
```python
lightgbm_params = {
    'objective': 'multiclass',   # 多分类目标
    'metric': 'multi_logloss',   # 评估指标
    'boosting_type': 'gbdt',     # 提升类型
    'num_leaves': 31,            # 叶子节点数
    'learning_rate': 0.05,       # 学习率
    'feature_fraction': 0.9,     # 特征采样比例
    'bagging_fraction': 0.8,     # 数据采样比例
    'bagging_freq': 5,           # 采样频率
    'verbose': -1,               # 静默模式
    'random_state': 42           # 随机种子
}
```

### FastText参数
```python
fasttext_params = {
    'lr': 0.1,                  # 学习率
    'epoch': 25,                 # 训练轮数
    'wordNgrams': 2,             # 词n-gram
    'minCount': 2,               # 最小词频
    'minn': 3,                   # 最小字符n-gram
    'maxn': 6,                   # 最大字符n-gram
    'verbose': 2                 # 详细程度
}
```

## 输出结果

训练完成后，会在`results`目录下生成：

1. **模型文件**：
   - `lightgbm_model_YYYYMMDD_HHMMSS.txt`
   - `tfidf_vectorizer_YYYYMMDD_HHMMSS.joblib`
   - `label_encoder_YYYYMMDD_HHMMSS.joblib`

2. **结果文件**：
   - `baseline_results_YYYYMMDD_HHMMSS.json`: 训练结果和评估指标
   - `confusion_matrix_YYYYMMDD_HHMMSS.png`: 混淆矩阵可视化

3. **评估指标**：
   - 准确率 (Accuracy)
   - 精确率 (Precision)
   - 召回率 (Recall)
   - F1分数 (F1-Score)

## 性能优化

### Intel XPU加速
- 确保安装了Intel Extension for PyTorch
- 使用`--use_xpu`参数启用XPU加速
- 模型会自动检测XPU可用性

### 内存优化
- 调整`max_features`参数控制TF-IDF特征数量
- 使用`min_df`和`max_df`过滤低频和高频词汇
- 适当调整LightGBM的`num_leaves`参数

## 故障排除

### 常见问题

1. **Intel XPU不可用**
   ```
   解决方案: 检查Intel GPU驱动和Intel Extension for PyTorch安装
   ```

2. **内存不足**
   ```
   解决方案: 减少max_features或使用更小的数据集
   ```

3. **数据加载失败**
   ```
   解决方案: 检查数据目录结构和CSV文件格式
   ```

4. **模型训练缓慢**
   ```
   解决方案: 启用XPU加速或减少训练数据量
   ```

## 扩展功能

### 自定义特征提取
可以继承`BaselineLogClassifier`类并重写特征提取方法：

```python
class CustomClassifier(BaselineLogClassifier):
    def extract_custom_features(self, text):
        # 实现自定义特征提取
        pass
```

### 集成其他模型
可以轻松添加其他分类器：

```python
# 添加SVM分类器
from sklearn.svm import SVC
self.model = SVC(kernel='rbf', probability=True)
```

## 许可证

本项目遵循MIT许可证。 