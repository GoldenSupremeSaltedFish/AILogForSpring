# 自动化日志分类器 - 快速启动指南

## 🚀 一键启动

### 1. API服务模式（推荐）
```bash
# 启动API服务
python start_classifier_service.py --mode api

# 或使用批处理脚本
batch-scripts\start_classifier_service.bat --mode api
```

### 2. 单文件分类
```bash
# 分类单个文件
python start_classifier_service.py --mode file --input-file your_logs.csv

# 或使用批处理脚本
batch-scripts\start_classifier_service.bat --mode file --input-file your_logs.csv
```

### 3. 批量分类
```bash
# 批量分类目录中的所有日志文件
python start_classifier_service.py --mode batch --input-dir logs_directory/

# 或使用批处理脚本
batch-scripts\start_classifier_service.bat --mode batch --input-dir logs_directory/
```

### 4. 交互式分类
```bash
# 交互式分类（逐条输入日志）
python start_classifier_service.py --mode interactive

# 或使用批处理脚本
batch-scripts\start_classifier_service.bat --mode interactive
```

## 📊 支持的日志类别

系统自动识别以下11种日志类别：

1. **堆栈异常** (stack_exception) - 最高优先级
2. **Spring Boot启动失败** (spring_boot_startup_failure)
3. **认证授权** (auth_authorization)
4. **数据库异常** (database_exception)
5. **连接问题** (connection_issue)
6. **超时错误** (timeout)
7. **内存性能** (memory_performance)
8. **配置环境** (config_environment)
9. **业务逻辑** (business_logic)
10. **正常操作** (normal_operation)
11. **监控心跳** (monitoring_heartbeat)

## 🔧 配置选项

### 使用配置文件
```bash
# 使用自定义配置文件
python start_classifier_service.py --config classifier_config.json
```

### 常用参数
- `--no-ml`: 仅使用规则分类，不使用机器学习
- `--debug`: 启用调试模式
- `--host 127.0.0.1`: 指定API服务主机地址
- `--port 8080`: 指定API服务端口

## 📁 数据存储结构

```
AILogForSpring/
├── DATA_OUTPUT/                    # 原始数据目录
│   └── 原始项目数据_original/      # 原始日志文件
├── log-processing-OUTPUT/          # 分类结果输出目录
├── logsense-xpu/models/            # 机器学习模型文件
└── classifier_config.json          # 配置文件
```

## 🌐 API接口使用

### 启动API服务后，可通过以下接口使用：

#### 1. 健康检查
```bash
curl http://localhost:5000/health
```

#### 2. 单条日志分类
```bash
curl -X POST http://localhost:5000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "ERROR: java.lang.NullPointerException at com.example.Controller.handleRequest"}'
```

#### 3. 批量日志分类
```bash
curl -X POST http://localhost:5000/batch_classify \
  -H "Content-Type: application/json" \
  -d '{"texts": ["ERROR: Connection timeout", "INFO: Application started"]}'
```

#### 4. 服务统计
```bash
curl http://localhost:5000/stats
```

## 📈 输出结果

### 分类结果包含：
- **original_log**: 原始日志内容
- **category**: 分类结果
- **confidence**: 置信度 (0-1)
- **log_level**: 日志级别 (ERROR/WARN/INFO/DEBUG)
- **method**: 分类方法 (rules/ml)
- **needs_manual_annotation**: 是否需要人工标注
- **timestamp**: 处理时间戳

### 统计报告包含：
- 总日志数量
- 各类别分布统计
- 平均置信度
- 需要人工标注的比例
- 分类覆盖率

## ⚡ 性能特点

- **混合分类**: 规则分类 + 机器学习，提高准确率
- **自动优先级**: 按重要性自动排序分类结果
- **批量处理**: 支持大规模日志文件处理
- **实时服务**: API服务支持实时分类
- **质量保证**: 自动识别需要人工审核的日志

## 🔍 故障排除

### 常见问题：

1. **模型加载失败**
   - 检查 `logsense-xpu/models/` 目录是否存在模型文件
   - 系统会自动降级为仅使用规则分类

2. **文件路径错误**
   - 确保输入文件存在且可读
   - 检查输出目录权限

3. **API服务无法启动**
   - 检查端口是否被占用
   - 确保Flask已安装：`pip install flask flask-cors`

4. **分类准确率低**
   - 尝试使用 `--no-ml` 参数仅使用规则分类
   - 检查日志格式是否符合预期

## 📞 技术支持

如遇到问题，请检查：
1. Python版本 (推荐 3.7+)
2. 必要依赖包是否安装
3. 数据文件路径是否正确
4. 配置文件格式是否正确
