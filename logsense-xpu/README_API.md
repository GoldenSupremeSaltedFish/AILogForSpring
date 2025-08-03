# 日志分类API服务文档

## 概述

这是一个基于Flask + joblib的REST API服务，用于日志分类。服务会自动加载训练好的TF-IDF + LightGBM模型，提供实时日志分类功能。

## 功能特性

- ✅ 自动加载最新训练的模型
- ✅ 单个日志预测
- ✅ 批量日志预测
- ✅ 模型信息查询
- ✅ 健康检查
- ✅ 模型热重载
- ✅ 跨域支持
- ✅ 详细的错误处理
- ✅ 性能监控

## 快速开始

### 1. 安装依赖

```bash
pip install flask flask-cors requests
```

### 2. 启动服务

```bash
# Windows
start_api.bat

# 或者直接运行
python api_server.py
```

### 3. 测试服务

```bash
python test_api.py
```

## API端点

### 基础信息

- **服务地址**: `http://localhost:5000`
- **内容类型**: `application/json`
- **字符编码**: `UTF-8`

### 1. 健康检查

**GET** `/health`

检查服务状态和模型加载情况。

**响应示例**:
```json
{
  "status": "healthy",
  "timestamp": "2025-08-02T21:50:51.475",
  "model_loaded": true
}
```

### 2. 模型信息

**GET** `/model/info`

获取当前加载模型的信息。

**响应示例**:
```json
{
  "model_type": "TF-IDF + LightGBM",
  "model_timestamp": "20250802_215107",
  "categories": [
    "堆栈异常_stack_exception",
    "数据库异常_database_exception",
    "连接问题_connection_issue",
    "认证授权_auth_authorization",
    "配置环境_config_environment",
    "业务逻辑_business_logic",
    "正常操作_normal_operation",
    "监控心跳_monitoring_heartbeat",
    "内存性能_memory_performance",
    "超时错误_timeout",
    "SpringBoot启动失败_spring_boot_startup_failure"
  ],
  "num_categories": 11,
  "vectorizer_features": 10000,
  "status": "loaded"
}
```

### 3. 单个预测

**POST** `/predict`

对单个日志文本进行分类预测。

**请求体**:
```json
{
  "text": "ERROR: java.lang.NullPointerException at com.example.Controller.handleRequest"
}
```

**响应示例**:
```json
{
  "success": true,
  "text": "ERROR: java.lang.NullPointerException at com.example.Controller.handleRequest",
  "cleaned_text": "ERROR java lang NullPointerException at com example Controller handleRequest",
  "prediction": {
    "category_id": 0,
    "category_name": "堆栈异常_stack_exception",
    "confidence": 0.9876,
    "all_probabilities": [0.9876, 0.0023, 0.0012, ...]
  },
  "timestamp": "2025-08-02T21:50:51.475"
}
```

### 4. 批量预测

**POST** `/predict/batch`

对多个日志文本进行批量分类预测。

**请求体**:
```json
{
  "texts": [
    "ERROR: java.lang.NullPointerException at com.example.Controller.handleRequest",
    "INFO: Application started successfully on port 8080",
    "WARN: Database connection timeout, retrying..."
  ]
}
```

**响应示例**:
```json
{
  "success": true,
  "total_count": 3,
  "predictions": [
    {
      "success": true,
      "index": 0,
      "text": "ERROR: java.lang.NullPointerException at com.example.Controller.handleRequest",
      "cleaned_text": "ERROR java lang NullPointerException at com example Controller handleRequest",
      "prediction": {
        "category_id": 0,
        "category_name": "堆栈异常_stack_exception",
        "confidence": 0.9876,
        "all_probabilities": [0.9876, 0.0023, 0.0012, ...]
      },
      "timestamp": "2025-08-02T21:50:51.475"
    },
    {
      "success": true,
      "index": 1,
      "text": "INFO: Application started successfully on port 8080",
      "cleaned_text": "INFO Application started successfully on port 8080",
      "prediction": {
        "category_id": 6,
        "category_name": "正常操作_normal_operation",
        "confidence": 0.9234,
        "all_probabilities": [0.0012, 0.0023, 0.0012, ..., 0.9234]
      },
      "timestamp": "2025-08-02T21:50:51.476"
    }
  ],
  "timestamp": "2025-08-02T21:50:51.476"
}
```

### 5. 模型重载

**POST** `/reload`

重新加载最新的模型文件。

**响应示例**:
```json
{
  "success": true,
  "message": "模型重新加载成功",
  "timestamp": "2025-08-02T21:50:51.475"
}
```

## 错误处理

### 常见错误码

- **400 Bad Request**: 请求参数错误
- **500 Internal Server Error**: 服务器内部错误

### 错误响应格式

```json
{
  "success": false,
  "error": "错误描述信息"
}
```

### 常见错误

1. **缺少必要字段**
   ```json
   {
     "success": false,
     "error": "缺少text字段"
   }
   ```

2. **文本为空**
   ```json
   {
     "success": false,
     "error": "文本为空或预处理后为空",
     "prediction": null,
     "confidence": 0.0
   }
   ```

3. **批量预测超限**
   ```json
   {
     "success": false,
     "error": "批量预测数量不能超过1000"
   }
   ```

## 使用示例

### Python客户端示例

```python
import requests
import json

# 服务地址
base_url = "http://localhost:5000"

# 单个预测
def predict_single(text):
    response = requests.post(f"{base_url}/predict", json={"text": text})
    return response.json()

# 批量预测
def predict_batch(texts):
    response = requests.post(f"{base_url}/predict/batch", json={"texts": texts})
    return response.json()

# 使用示例
if __name__ == "__main__":
    # 单个预测
    result = predict_single("ERROR: java.lang.NullPointerException")
    print(f"预测结果: {result['prediction']['category_name']}")
    
    # 批量预测
    texts = [
        "ERROR: Database connection failed",
        "INFO: Application started",
        "WARN: Memory usage high"
    ]
    results = predict_batch(texts)
    for pred in results['predictions']:
        print(f"文本: {pred['text'][:50]}... -> {pred['prediction']['category_name']}")
```

### cURL示例

```bash
# 健康检查
curl -X GET http://localhost:5000/health

# 模型信息
curl -X GET http://localhost:5000/model/info

# 单个预测
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "ERROR: java.lang.NullPointerException"}'

# 批量预测
curl -X POST http://localhost:5000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["ERROR: Database failed", "INFO: Started"]}'
```

### JavaScript示例

```javascript
// 单个预测
async function predictSingle(text) {
    const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: text })
    });
    return await response.json();
}

// 批量预测
async function predictBatch(texts) {
    const response = await fetch('http://localhost:5000/predict/batch', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ texts: texts })
    });
    return await response.json();
}

// 使用示例
predictSingle("ERROR: java.lang.NullPointerException")
    .then(result => {
        console.log('预测结果:', result.prediction.category_name);
    });
```

## 性能优化

### 1. 批量处理

对于大量日志，建议使用批量预测接口，可以减少网络开销。

### 2. 连接复用

使用HTTP连接池或保持连接，可以提高性能。

### 3. 异步处理

对于高并发场景，可以考虑使用异步框架如FastAPI。

## 部署建议

### 1. 生产环境

- 使用Gunicorn或uWSGI作为WSGI服务器
- 配置Nginx作为反向代理
- 设置适当的超时和并发限制

### 2. 容器化

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["python", "api_server.py"]
```

### 3. 监控

- 添加日志记录
- 监控API响应时间
- 监控模型预测准确率

## 故障排除

### 1. 模型加载失败

- 检查模型文件是否存在
- 确认模型文件路径正确
- 检查依赖包是否安装完整

### 2. 预测失败

- 检查输入文本格式
- 确认模型已正确加载
- 查看服务器日志

### 3. 性能问题

- 检查服务器资源使用情况
- 考虑使用批量预测
- 优化模型加载时间

## 更新日志

- **v1.0.0**: 初始版本，支持基本的预测功能
- 支持单个和批量预测
- 支持模型热重载
- 添加健康检查和模型信息接口 