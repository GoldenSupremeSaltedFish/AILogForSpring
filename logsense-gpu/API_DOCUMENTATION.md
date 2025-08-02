# LogSense GPU API 文档

## 概述

LogSense GPU API 是一个基于Flask的RESTful API服务，提供日志分类预测功能。该API加载训练好的机器学习模型，支持单条和批量日志分类预测。

## 快速开始

### 1. 启动API服务器

```bash
# 方法1: 使用批处理脚本（推荐）
start_api.bat

# 方法2: 直接运行Python脚本
python api_server.py
```

### 2. 验证服务状态

```bash
curl http://localhost:5000/health
```

响应示例：
```json
{
  "status": "healthy",
  "timestamp": "2025-08-02T14:30:00",
  "model_loaded": true
}
```

## API接口

### 1. 健康检查

**GET** `/health`

检查API服务器和模型状态。

**响应示例：**
```json
{
  "status": "healthy",
  "timestamp": "2025-08-02T14:30:00",
  "model_loaded": true
}
```

### 2. 模型信息

**GET** `/model/info`

获取加载的模型详细信息。

**响应示例：**
```json
{
  "model_type": "gradient_boosting",
  "target_classes": [
    "stack_exception",
    "connection_issue", 
    "database_exception",
    "auth_authorization",
    "memory_performance"
  ],
  "feature_dim": 3000,
  "sample_size": 500,
  "training_time": "2025-08-02T14:08:47",
  "model_loaded": true,
  "vectorizer_loaded": true,
  "label_encoder_loaded": true
}
```

### 3. 支持的类别

**GET** `/classes`

获取所有支持的日志类别。

**响应示例：**
```json
{
  "classes": [
    "stack_exception",
    "connection_issue",
    "database_exception", 
    "auth_authorization",
    "memory_performance"
  ],
  "count": 5
}
```

### 4. 单条预测

**POST** `/predict`

对单条日志进行分类预测。

**请求格式：**
```json
{
  "log_text": "java.lang.NullPointerException: Cannot invoke \"String.length()\" because \"str\" is null",
  "top_k": 3
}
```

**参数说明：**
- `log_text` (必需): 要分类的日志文本
- `top_k` (可选): 返回前K个预测结果，默认3

**响应示例：**
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
    },
    {
      "class": "database_exception",
      "confidence": 0.02, 
      "percentage": 2.0
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

### 5. 批量预测

**POST** `/predict/batch`

对多条日志进行批量分类预测。

**请求格式：**
```json
{
  "log_texts": [
    "java.lang.NullPointerException: Cannot invoke \"String.length()\" because \"str\" is null",
    "Connection refused: connect to database server failed",
    "OutOfMemoryError: Java heap space"
  ],
  "top_k": 2
}
```

**参数说明：**
- `log_texts` (必需): 要分类的日志文本列表
- `top_k` (可选): 返回前K个预测结果，默认3

**响应示例：**
```json
{
  "predictions": [
    {
      "index": 0,
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
          "confidence": 0.05,
          "percentage": 5.0
        }
      ],
      "timestamp": "2025-08-02T14:30:00",
      "model_info": {...}
    },
    {
      "index": 1,
      "predicted_class": "connection_issue",
      "confidence": 0.88,
      "confidence_percentage": 88.0,
      "top_k_predictions": [...],
      "timestamp": "2025-08-02T14:30:00",
      "model_info": {...}
    }
  ],
  "total_count": 2,
  "timestamp": "2025-08-02T14:30:00"
}
```

## 使用示例

### Python示例

```python
import requests
import json

# API基础URL
BASE_URL = "http://localhost:5000"

# 单条预测
def predict_single(log_text):
    response = requests.post(f"{BASE_URL}/predict", json={
        "log_text": log_text,
        "top_k": 3
    })
    return response.json()

# 批量预测
def predict_batch(log_texts):
    response = requests.post(f"{BASE_URL}/predict/batch", json={
        "log_texts": log_texts,
        "top_k": 2
    })
    return response.json()

# 使用示例
if __name__ == "__main__":
    # 测试单条预测
    result = predict_single("java.lang.NullPointerException: Cannot invoke \"String.length()\" because \"str\" is null")
    print(f"预测类别: {result['predicted_class']}")
    print(f"置信度: {result['confidence_percentage']:.2f}%")
    
    # 测试批量预测
    logs = [
        "Connection refused: connect to database server failed",
        "OutOfMemoryError: Java heap space"
    ]
    batch_result = predict_batch(logs)
    for pred in batch_result['predictions']:
        print(f"日志 {pred['index'] + 1}: {pred['predicted_class']}")
```

### cURL示例

```bash
# 健康检查
curl http://localhost:5000/health

# 获取模型信息
curl http://localhost:5000/model/info

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

### JavaScript示例

```javascript
// 单条预测
async function predictSingle(logText) {
    const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            log_text: logText,
            top_k: 3
        })
    });
    return await response.json();
}

// 批量预测
async function predictBatch(logTexts) {
    const response = await fetch('http://localhost:5000/predict/batch', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            log_texts: logTexts,
            top_k: 2
        })
    });
    return await response.json();
}

// 使用示例
predictSingle("java.lang.NullPointerException: Cannot invoke \"String.length()\" because \"str\" is null")
    .then(result => {
        console.log(`预测类别: ${result.predicted_class}`);
        console.log(`置信度: ${result.confidence_percentage}%`);
    });
```

## 错误处理

### 常见错误码

- `400 Bad Request`: 请求参数错误
- `500 Internal Server Error`: 服务器内部错误

### 错误响应格式

```json
{
  "error": "错误描述信息",
  "timestamp": "2025-08-02T14:30:00"
}
```

### 常见错误

1. **模型未加载**
   ```json
   {
     "error": "模型未加载",
     "timestamp": "2025-08-02T14:30:00"
   }
   ```

2. **请求参数错误**
   ```json
   {
     "error": "日志文本不能为空",
     "timestamp": "2025-08-02T14:30:00"
   }
   ```

## 测试

运行测试脚本验证API功能：

```bash
python test_api.py
```

测试脚本会验证：
- 健康检查接口
- 模型信息接口
- 类别列表接口
- 单条预测接口
- 批量预测接口
- 错误处理

## 配置

### 环境变量

- `PORT`: API服务器端口（默认5000）
- `FLASK_DEBUG`: 调试模式（默认False）

### 模型配置

API会自动加载最新的训练模型，支持：
- 增强模型（.joblib格式）
- 基线模型（.pkl格式）

## 性能优化

1. **模型加载**: 服务启动时加载模型，避免重复加载
2. **批量预测**: 使用批量接口提高效率
3. **错误处理**: 完善的错误处理机制
4. **日志记录**: 详细的日志记录便于调试

## 部署建议

1. **生产环境**: 使用Gunicorn或uWSGI部署
2. **负载均衡**: 使用Nginx进行负载均衡
3. **监控**: 添加健康检查和性能监控
4. **安全**: 配置防火墙和访问控制 