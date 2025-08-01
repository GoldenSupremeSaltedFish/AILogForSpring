# 日志分类整理器使用示例

## 快速开始

### 1. 基本使用

处理dataset-ready目录下的所有CSV文件：

```bash
python log_categorizer.py
```

### 2. 处理特定文件

```bash
python log_categorizer.py dataset-ready/your_log_file.csv
```

### 3. 使用批处理脚本（Windows）

```bash
batch-scripts/run_log_categorizer.bat
```

## 实际示例

假设您有一个包含以下数据的CSV文件：

```csv
timestamp,level,message,final_label
2025-06-13 02:42:24.807,DEBUG,开始验证令牌: eyJhbGciOi...,auth_error
2025-06-13 02:42:24.808,DEBUG,令牌验证成功 - 用户名: admin,auth_error
2025-06-13 02:42:24.810,DEBUG,Route matched: user_route,ignore
2025-06-13 02:42:33.385,DEBUG,临时禁用JWT认证，放行所有请求,other
2025-06-13 02:42:33.386,DEBUG,开始验证令牌: eyJhbGciOi...,other
```

运行脚本后，您将得到：

### 输出文件

1. **主分类文件**: `your_log_file_categorized_20250802_010628.csv`
   - 包含所有非"other"的日志，按类别排序

2. **统计摘要**: `your_log_file_categorized_20250802_010628_summary.txt`
   ```
   📈 分类统计报告
   ==================================================
   总记录数: 3
   
   可忽略 (ignore): 1 条 (33.3%)
   认证错误 (auth_error): 2 条 (66.7%)
   ==================================================
   ```

3. **类别文件**:
   - `auth_error_认证错误_20250802_010628.csv` (2条记录)
   - `ignore_可忽略_20250802_010628.csv` (1条记录)

### 控制台输出

```
🔄 开始处理文件: dataset-ready/your_log_file.csv
------------------------------------------------------------
📊 原始数据: 5 条记录
🔍 过滤后数据: 3 条记录 (移除了 2 条'other'记录)
💾 已保存分类数据到: dataset-ready/your_log_file_categorized_20250802_010628.csv
📋 已保存统计摘要到: dataset-ready/your_log_file_categorized_20250802_010628_summary.txt
📁 已保存 认证错误 类别数据: auth_error_认证错误_20250802_010628.csv (2 条)
📁 已保存 可忽略 类别数据: ignore_可忽略_20250802_010628.csv (1 条)

📈 分类统计报告
==================================================
总记录数: 3

认证错误 (auth_error): 2 条 (66.7%)
可忽略 (ignore): 1 条 (33.3%)
==================================================
```

## 高级用法

### 指定输出目录

```bash
python log_categorizer.py input.csv --output-dir my_output_folder
```

### 不创建类别文件

```bash
python log_categorizer.py input.csv --no-category-files
```

### 批量处理

```bash
# 处理dataset-ready目录下的所有CSV文件
python log_categorizer.py
```

## 类别优先级说明

脚本按照以下优先级对日志进行排序：

1. **auth_error** (认证错误) - 最高优先级
2. **system_error** (系统错误)
3. **db_error** (数据库错误)
4. **timeout** (超时错误)
5. **api_success** (API成功)
6. **ignore** (可忽略的心跳检测)
7. **other** (其他) - 会被过滤掉

## 文件命名规则

- 主文件: `{原文件名}_categorized_{时间戳}.csv`
- 摘要文件: `{原文件名}_categorized_{时间戳}_summary.txt`
- 类别文件: `{类别名}_{中文描述}_{时间戳}.csv`

## 注意事项

1. 确保输入CSV文件包含`final_label`列
2. 脚本会自动过滤掉`final_label`为"other"的条目
3. 所有输出文件都使用UTF-8编码
4. 时间戳格式为`YYYYMMDD_HHMMSS` 