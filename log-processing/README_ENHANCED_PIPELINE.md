# 增强的日志半自动分类流水线

## 概述

这是一个遵循最佳实践的日志半自动分类系统，实现了完整的日志处理流水线，包括：

1. **日志模板化** - 使用类似Drain3的方法将相似日志归并为模板
2. **噪声去除** - 自动去除时间戳、线程ID、UUID等噪声
3. **特征工程** - 提取结构特征+语义特征
4. **机器学习分类** - 支持TF-IDF + LightGBM/朴素贝叶斯
5. **人工审查** - 提供交互式审查工具
6. **质量分析** - 自动评估分类质量

## 最佳实践实现

### 🔑 1. 日志结构化（减少噪声）

#### 日志模板化
- 使用类似Drain3的算法将相似日志归并为模板
- 自动分配模板ID，便于后续处理
- 支持多种日志格式（Java、Spring Boot等）

```bash
# 示例：原始日志
Connection failed to 192.168.1.10:3306
Connection failed to 10.0.0.5:3306

# 模板化后
Connection failed to <IP>:<PORT>
模板ID: T_a1b2c3d4
```

#### 噪声去除
- 时间戳：`2024-01-01 12:00:00` → `<TIMESTAMP>`
- 线程ID：`[thread-123]` → `<THREAD_ID>`
- UUID：`550e8400-e29b-41d4-a716-446655440000` → `<UUID>`
- 请求ID：`request-id: abc123` → `<REQUEST_ID>`
- IP地址：`192.168.1.1` → `<IP>`
- 端口号：`:3306` → `<PORT>`

#### 异常关键字提取
- 自动扫描并扩充异常字典
- 支持Java、Spring、数据库、网络等异常类型
- 生成异常关键字特征

### 🔑 2. 特征工程（半自动化）

#### 结构特征（强约束）
- `log_level`: INFO/WARN/ERROR/FATAL
- `contains_stack`: 是否包含堆栈跟踪
- `exception_type`: 异常类名（字典化）
- `file_path`: 归一化为模块名
- `function_name`: 函数名
- `line_number`: 行号

#### 语义特征（弱约束）
- **TF-IDF**: 经典文本向量化，轻量级模型必备
- **模板ID embedding**: 模板ID → one-hot编码
- **异常关键字 embedding**: 是否命中异常字典

#### 特征组合（提升效果）
- `log_level + contains_stack`: 辅助区分error/warn
- `template_id + tfidf`: 保证模板归一化后还能有语义区分
- `exception_type + function`: 细化到某个调用栈的异常

### 🔑 3. 半自动最佳实践流程

#### 第一步：模板化
```bash
python log_templater.py --input-file logs.csv --output-dir output/
```

#### 第二步：特征工程
```bash
python feature_engineer.py --input-file templated_logs.csv --output-dir output/
```

#### 第三步：预分类
```bash
python enhanced_pre_classifier.py single --input-file logs.csv --output-dir output/
```

#### 第四步：自动标签
```bash
python auto_labeler.py logs.csv --use-ml
```

#### 第五步：人工审查
```bash
python log_reviewer.py labeled_logs.csv
```

#### 第六步：质量分析
```bash
python quality_analyzer.py analyze --file final_logs.csv
```

## 使用方法

### 完整流水线
```bash
# 运行完整的半自动分类流水线
python enhanced_pipeline.py --input-file logs.csv --mode full

# 跳过人工审查（全自动模式）
python enhanced_pipeline.py --input-file logs.csv --mode full --skip-human-review

# 仅模板化
python enhanced_pipeline.py --input-file logs.csv --mode template-only
```

### 批量处理
```bash
# 批量处理目录中的所有日志文件
python enhanced_pipeline.py --input-dir logs/ --mode batch

# 批量处理，跳过某些步骤
python enhanced_pipeline.py --input-dir logs/ --mode batch --skip-human-review --skip-quality-analysis
```

### 单独使用各个组件

#### 1. 日志模板化
```bash
# 单文件处理
python log_templater.py --input-file logs.csv --output-dir output/

# 批量处理
python log_templater.py --batch --input-dir logs/ --output-dir output/
```

#### 2. 特征工程
```bash
# 处理模板化后的日志
python feature_engineer.py --input-file templated_logs.csv --output-dir output/

# 批量处理
python feature_engineer.py --batch --input-dir templated_logs/ --output-dir output/
```

#### 3. 预分类
```bash
# 单文件处理
python enhanced_pre_classifier.py single --input-file logs.csv --output-dir output/

# 批量处理
python enhanced_pre_classifier.py batch --input-dir logs/ --output-dir output/
```

#### 4. 自动标签
```bash
# 使用规则分类
python auto_labeler.py logs.csv

# 使用机器学习
python auto_labeler.py logs.csv --use-ml

# 批量处理
python auto_labeler.py --batch
```

#### 5. 人工审查
```bash
# 审查已标注的日志
python log_reviewer.py labeled_logs.csv --output-dir output/
```

#### 6. 质量分析
```bash
# 分析单个文件
python quality_analyzer.py analyze --file logs.csv

# 比较两个文件
python quality_analyzer.py compare --file1 logs1.csv --file2 logs2.csv
```

## 输出文件说明

### 模板化输出
- `*_templated_*.csv`: 包含模板ID和处理后日志
- `*_templates_*.json`: 模板详细信息
- `*_template_report_*.txt`: 模板统计报告

### 特征工程输出
- `*_features_*.csv`: 包含所有特征的完整数据集
- `*_model_*.pkl`: 训练好的机器学习模型
- `*_model_report_*.json`: 模型评估报告
- `*_feature_report_*.txt`: 特征统计报告

### 分类输出
- `*_classified_*.csv`: 预分类结果
- `*_labeled_*.csv`: 自动标签结果
- `*_reviewed_*.csv`: 人工审查结果

### 质量分析输出
- `*_quality_analysis_report_*.txt`: 质量分析报告
- `*_quality_analysis_data_*.json`: 详细质量数据
- `quality_analysis_charts.png`: 可视化图表

## 配置说明

### 流水线配置 (`pipeline_config.json`)
```json
{
  "pipeline_config": {
    "enable_templating": true,
    "enable_feature_engineering": true,
    "enable_ml_classification": true,
    "enable_human_review": true,
    "enable_quality_analysis": true,
    "max_per_class": 500,
    "confidence_threshold": 0.7
  }
}
```

### 模板化配置
- 可配置需要去除的噪声类型
- 可配置异常关键字类别
- 支持自定义正则表达式模式

### 特征工程配置
- 可配置TF-IDF参数
- 可配置特征组合策略
- 支持自定义特征提取规则

## 性能优化建议

### 1. 大数据集处理
- 使用批量处理模式
- 调整`max_per_class`参数控制内存使用
- 考虑分块处理大文件

### 2. 模型训练优化
- 使用LightGBM获得更好的性能
- 调整TF-IDF参数平衡准确率和速度
- 定期重新训练模型

### 3. 人工审查效率
- 优先审查低置信度的记录
- 使用快捷键提高审查速度
- 定期保存进度避免重复工作

## 故障排除

### 常见问题

1. **内存不足**
   - 减少`max_per_class`参数
   - 使用批量处理模式
   - 增加系统内存

2. **分类准确率低**
   - 检查异常关键字字典
   - 调整分类规则
   - 增加训练数据

3. **模板化效果差**
   - 检查噪声模式配置
   - 调整正则表达式
   - 增加异常关键字

4. **特征工程失败**
   - 检查输入数据格式
   - 确保必要列存在
   - 检查数据类型

### 日志和调试
- 所有组件都提供详细的进度信息
- 错误信息会保存到报告中
- 可以使用`--verbose`参数获取更多调试信息

## 扩展和定制

### 添加新的噪声模式
在`log_templater.py`中的`noise_patterns`字典中添加新的正则表达式模式。

### 添加新的异常关键字
在`log_templater.py`中的`exception_keywords`字典中添加新的异常类型。

### 自定义分类规则
在`enhanced_pre_classifier.py`中的`classification_rules`字典中添加新的分类规则。

### 添加新的特征
在`feature_engineer.py`中添加新的特征提取函数。

## 依赖要求

### 必需依赖
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

### 可选依赖
- lightgbm (推荐，用于更好的分类性能)

### 安装命令
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
pip install lightgbm  # 可选，但推荐
```

## 许可证

本项目遵循MIT许可证。
