# Gateway处理工具

## 脚本说明

### gateway_processor.py
Gateway日志处理器核心模块。

### process_gateway.py
Gateway日志处理脚本，支持Spring Boot Gateway格式。

### process_gateway_logs.py
Gateway日志批量处理工具。

**使用方法:**
```bash
python process_gateway_logs.py <Gateway日志目录>
```

## 支持格式
- Spring Boot Gateway标准格式
- 自定义时间戳格式
- 多线程和进程ID解析
