# 数据模板

这个目录包含了示例数据文件的模板，用于展示项目所需的数据格式。

## 文件说明

- `sample_logs.csv` - 日志数据格式示例
- `sample_labels.csv` - 标签映射格式示例

## 使用方法

1. 将您的实际数据文件放入 `../data/` 目录
2. 确保数据格式与这里的模板一致
3. 实际的 `data/` 目录已在 `.gitignore` 中被忽略，不会上传到GitHub

## 数据格式要求

### logs.csv 格式
- `timestamp`: 时间戳 (YYYY-MM-DD HH:MM:SS)
- `level`: 日志级别 (DEBUG/INFO/WARN/ERROR/FATAL)
- `message`: 日志消息内容
- `source`: 日志来源系统

### labels.csv 格式  
- `category`: 分类标签
- `description`: 分类描述 