# 数据清洗脚本 (Data Cleaner)

## 功能概述

数据清洗脚本是一个专门用于处理已分类日志数据的工具，主要功能包括：

1. **去除"other"类别** - 自动过滤掉标记为"other"的日志条目
2. **数据平衡** - 控制每类别的数据量，避免数据偏斜
3. **生成训练数据集** - 创建适合机器学习模型训练的数据格式
4. **支持多种数据格式** - 自动检测不同的列名和数据结构
5. **批量处理** - 支持处理多个数据文件并合并

## 主要特性

### 自动列检测
脚本会自动检测以下列名：
- **标签列**: `content_type`, `final_label`, `label`, `category`
- **文本列**: `original_log`, `message`, `content`, `text`

### 数据平衡
- 支持设置每类别最大数据量（默认500条）
- 避免某些类别数据过多导致模型偏斜
- 保持各类别数据相对均衡

### 输出格式
- **清洗数据**: 保留所有列，仅过滤other类别
- **训练数据**: 只保留文本和标签列，重命名为`text`和`label`

## 使用方法

### 基本用法

```bash
# 处理DATA_OUTPUT目录下的所有CSV文件并创建合并数据集
python data_cleaner.py --combined

# 处理指定文件
python data_cleaner.py your_file.csv

# 设置每类最多保留条数
python data_cleaner.py --combined --max-per-class 300

# 指定输出目录
python data_cleaner.py --combined --output-dir my_output
```

### Windows批处理脚本

```bash
# 使用批处理脚本（推荐）
batch-scripts/run_data_cleaner.bat
```

## 输出文件

### 合并数据集模式
脚本会生成以下文件：

1. **合并数据集** - `combined_dataset_{时间戳}.csv`
   - 包含所有非"other"类别的数据
   - 按类别平衡后的数据

2. **训练数据集** - `training_dataset_{时间戳}.csv`
   - 只包含`text`和`label`列
   - 适合机器学习模型训练

### 单文件处理模式
脚本会为每个输入文件生成：
- `{原文件名}_cleaned_{时间戳}.csv` - 清洗后的数据
- `{原文件名}_training_{时间戳}.csv` - 训练数据

## 实际示例

### 输入数据
```csv
line_number,log_level,content_type,priority,manual_annotation_needed,original_log
1,UNKNOWN,stack_exception,1,True,"Exception in thread main..."
2,ERROR,connection_issue,2,True,"Connection timeout..."
3,INFO,other,999,False,"Normal log message..."
```

### 输出数据
```csv
text,label
"Exception in thread main...",stack_exception
"Connection timeout...",connection_issue
```

### 处理过程示例

```
🔄 创建合并数据集...
📊 加载数据: 1509 条记录
  ✓ connection_issue_连接问题_20250802_011450.csv: 1509 条记录
📊 加载数据: 1143 条记录
  ✓ monitoring_heartbeat_监控心跳_20250802_011450.csv: 969 条记录

📊 合并后总数据: 153468 条记录
⚖️ 平衡后数据: 4764 条记录
💾 已保存合并数据集到: DATA_OUTPUT\combined_dataset_20250802_013437.csv
💾 已保存训练数据到: DATA_OUTPUT\training_dataset_20250802_013437.csv

📈 最终类别分布:
  auth_authorization: 500 条 (10.5%)
  business_logic: 500 条 (10.5%)
  config_environment: 500 条 (10.5%)
  connection_issue: 500 条 (10.5%)
  database_exception: 500 条 (10.5%)
  memory_performance: 500 条 (10.5%)
  monitoring_heartbeat: 500 条 (10.5%)
  normal_operation: 500 条 (10.5%)
  stack_exception: 500 条 (10.5%)
  spring_boot_startup_failure: 258 条 (5.4%)
  timeout: 6 条 (0.1%)
```

## 参数说明

- `--combined`: 创建合并的数据集（推荐）
- `--max-per-class N`: 每类别最多保留N条数据（默认500）
- `--output-dir PATH`: 指定输出目录（默认DATA_OUTPUT）
- `input_file`: 指定要处理的单个文件

## 类别支持

脚本支持以下日志类别：
- `stack_exception` - 堆栈异常
- `spring_boot_startup_failure` - Spring Boot启动失败
- `auth_authorization` - 认证授权
- `database_exception` - 数据库异常
- `connection_issue` - 连接问题
- `timeout` - 超时错误
- `memory_performance` - 内存性能
- `config_environment` - 配置环境
- `business_logic` - 业务逻辑
- `normal_operation` - 正常操作
- `monitoring_heartbeat` - 监控心跳

## 注意事项

1. 脚本会自动过滤掉`content_type`为"other"的条目
2. 如果某类别数据量不足，会保留所有可用数据
3. 所有输出文件都使用UTF-8编码
4. 时间戳格式为`YYYYMMDD_HHMMSS`
5. 建议使用`--combined`模式处理多个文件

## 错误处理

- 如果输入文件不存在或格式错误，脚本会显示相应的错误信息
- 如果缺少必要的列，脚本会提示并退出
- 所有错误都会在控制台显示详细信息 