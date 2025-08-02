# LogSense GPU API 使用指南

## 🚀 快速开始

### 1. 启动API服务器

```bash
# 方法1: 使用批处理脚本（推荐）
start_api.bat

# 方法2: 直接运行Python脚本
python api_server.py
```

### 2. 验证服务状态

访问 http://localhost:5000/health 或使用Web界面测试

## 📋 API接口列表

| 接口 | 方法 | 描述 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/model/info` | GET | 模型信息 |
| `/classes` | GET | 支持的类别 |
| `/predict` | POST | 单条预测 |
| `/predict/batch` | POST | 批量预测 |

## 🔧 安装依赖

```bash
pip install -r requirements.txt
```

## 📖 使用示例

### Python示例

```python
import requests

# 单条预测
response = requests.post('http://localhost:5000/predict', json={
    'log_text': 'java.lang.NullPointerException: Cannot invoke "String.length()" because "str" is null',
    'top_k': 3
})
result = response.json()
print(f"预测类别: {result['predicted_class']}")
print(f"置信度: {result['confidence_percentage']:.2f}%")

# 批量预测
response = requests.post('http://localhost:5000/predict/batch', json={
    'log_texts': [
        'Connection refused: connect to database server failed',
        'OutOfMemoryError: Java heap space'
    ],
    'top_k': 2
})
results = response.json()
for pred in results['predictions']:
    print(f"日志 {pred['index'] + 1}: {pred['predicted_class']}")
```

### cURL示例

```bash
# 健康检查
curl http://localhost:5000/health

# 单条预测
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "log_text": "java.lang.NullPointerException: Cannot invoke \"String.length()\" because \"str\" is null",
    "top_k": 3
  }'

# 批量预测
curl -X POST http://localhost:5000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "log_texts": [
      "Connection refused: connect to database server failed",
      "OutOfMemoryError: Java heap space"
    ],
    "top_k": 2
  }'
```

## 🧪 测试

### 1. 运行测试脚本

```bash
python test_api.py
```

### 2. 使用Web界面

打开 `web_interface.html` 文件在浏览器中测试API功能。

## 📊 支持的日志类别

- `stack_exception` - 堆栈异常
- `connection_issue` - 连接问题
- `database_exception` - 数据库异常
- `auth_authorization` - 认证授权
- `memory_performance` - 内存性能

## 🔍 响应格式

### 单条预测响应

```json
{
  "predicted_class": "stack_exception",
  "confidence": 0.95,
  "confidence_percentage": 95.0,
  "top_k_predictions": [
    {
      "class": "stack_exception",
      "confidence": 0.95,
      "percentage": 95.0
    },
    {
      "class": "memory_performance",
      "confidence": 0.03,
      "percentage": 3.0
    }
  ],
  "timestamp": "2025-08-02T14:30:00",
  "model_info": {
    "model_type": "gradient_boosting",
    "feature_dim": 3000,
    "target_classes": ["stack_exception", "connection_issue", "database_exception", "auth_authorization", "memory_performance"]
  }
}
```

### 批量预测响应

```json
{
  "predictions": [
    {
      "index": 0,
      "predicted_class": "stack_exception",
      "confidence": 0.95,
      "confidence_percentage": 95.0,
      "top_k_predictions": [...],
      "timestamp": "2025-08-02T14:30:00",
      "model_info": {...}
    }
  ],
  "total_count": 1,
  "timestamp": "2025-08-02T14:30:00"
}
```

## ⚙️ 配置

### 环境变量

- `PORT`: API服务器端口（默认5000）
- `FLASK_DEBUG`: 调试模式（默认False）

### 模型配置

API会自动加载最新的训练模型：
- 优先加载增强模型（.joblib格式）
- 如果没有增强模型，加载基线模型（.pkl格式）

## 🐛 故障排除

### 常见问题

1. **模型加载失败**
   - 检查 `logsense-gpu/results/models/` 目录是否存在模型文件
   - 确保模型文件完整且可读

2. **端口被占用**
   - 修改 `PORT` 环境变量
   - 或使用 `netstat -ano | findstr :5000` 查看占用进程

3. **依赖安装失败**
   - 确保Python版本 >= 3.7
   - 使用虚拟环境：`python -m venv venv && venv\Scripts\activate`

### 日志查看

API服务器会输出详细的日志信息，包括：
- 模型加载状态
- 请求处理过程
- 错误信息

## 📈 性能优化

1. **模型预加载**: 服务启动时加载模型，避免重复加载
2. **批量处理**: 使用批量接口提高效率
3. **错误处理**: 完善的错误处理机制
4. **日志记录**: 详细的日志记录便于调试

## 🔒 安全建议

1. **生产环境部署**: 使用Gunicorn或uWSGI
2. **负载均衡**: 使用Nginx进行负载均衡
3. **访问控制**: 配置防火墙和访问控制
4. **HTTPS**: 在生产环境中使用HTTPS

## 📞 支持

如有问题，请查看：
- API文档：`API_DOCUMENTATION.md`
- 测试脚本：`test_api.py`
- Web界面：`web_interface.html` 