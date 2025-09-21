# 🔧 工具脚本集合

本目录包含各种数据处理、模型检查和验证工具脚本。

## 📋 工具列表

### 数据处理工具

- **`adapt_issue_data.py`** - 数据适配工具
  - 将issue日志数据转换为验证脚本期望的格式
  - 使用方法: `python adapt_issue_data.py`

- **`improved_data_processor.py`** - 改进的数据处理脚本
  - 统一标签体系并提升数据质量
  - 使用方法: `python improved_data_processor.py`

- **`prepare_issue_data.py`** - 问题数据准备工具
  - 准备和预处理问题日志数据
  - 使用方法: `python prepare_issue_data.py`

- **`validation_data_adapter.py`** - 验证数据适配工具
  - 适配验证数据格式
  - 使用方法: `python validation_data_adapter.py`

### 模型检查工具

- **`check_model.py`** - 模型检查工具
  - 检查模型文件是否正确保存了权重和词汇表
  - 使用方法: `python check_model.py`

- **`check_weights.py`** - 权重检查工具
  - 检查模型权重维度
  - 使用方法: `python check_weights.py`

### 验证工具

- **`simple_text_validator.py`** - 简单文本验证工具
  - 对文本数据进行简单验证
  - 使用方法: `python simple_text_validator.py`

- **`simple_validation_runner.py`** - 简单验证运行工具
  - 运行简单的验证流程
  - 使用方法: `python simple_validation_runner.py`

- **`fixed_model_runner.py`** - 修复的模型运行工具
  - 运行修复后的模型
  - 使用方法: `python fixed_model_runner.py`

### 标签处理工具

- **`filter_known_labels.py`** - 标签过滤工具
  - 过滤掉训练时没有见过的标签，只保留已知的类别
  - 使用方法: `python filter_known_labels.py`

## 🚀 快速使用

### 通过主入口文件使用

```bash
# 检查模型
python main.py --mode check --model_path "results/models/best_model.pth"
```

### 直接使用工具脚本

```bash
# 进入工具目录
cd tools/

# 运行特定工具
python adapt_issue_data.py
python check_model.py
python improved_data_processor.py
```

## 📝 注意事项

1. 所有工具脚本都支持命令行参数，使用 `--help` 查看详细用法
2. 确保在正确的目录下运行脚本
3. 某些工具需要特定的输入文件，请参考各脚本的文档说明
4. 建议先运行数据准备工具，再进行模型训练和验证

## 🔗 相关文档

- [主项目README](../README.md)
- [训练总结](../TRAINING_SUMMARY.md)
- [最终验证报告](../FINAL_VALIDATION_REPORT.md)
