# 增强版日志分类API服务器

## 概述

这是一个支持CPU和Intel Arc GPU双模式的日志分类API服务器，基于TF-IDF + LightGBM模型，提供REST API接口。

## 功能特点

- ✅ **双模式支持**: CPU和GPU两种计算模式
- ✅ **Intel Arc GPU加速**: 支持Intel Arc A750等XPU设备
- ✅ **命令行参数**: 启动时指定使用CPU或GPU
- ✅ **设备监控**: 实时查看GPU内存使用情况
- ✅ **动态切换**: 运行时切换设备模式
- ✅ **并发支持**: 支持高并发请求处理

## 快速开始

### 1. 启动CPU模式

```bash
# 方法1: 使用批处理脚本
start_api_cpu.bat

# 方法2: 直接命令行
python api_server_enhanced.py --device cpu
```

### 2. 启动GPU模式

```bash
# 方法1: 使用批处理脚本
start_api_gpu.bat

# 方法2: 直接命令行
python api_server_enhanced.py --device gpu
```

### 3. 自定义参数

```bash
# 指定端口
python api_server_enhanced.py --device gpu --port 5001

# 指定主机地址
python api_server_enhanced.py --device cpu --host 127.0.0.1

# 指定模型目录
python api_server_enhanced.py --device gpu --model-dir models
```

## 命令行参数

| 参数 | 选项 | 默认值 | 说明 |
|------|------|--------|------|
| `--device` | `cpu` / `gpu` | `gpu` | 指定使用的设备 |
| `--host` | 任意IP地址 | `0.0.0.0` | 服务器主机地址 |
| `--port` | 端口号 | `5000` | 服务器端口 |
| `--model-dir` | 目录路径 | `models` | 模型文件目录 |

## API端点

### 基础端点

- `GET /` - API根路径，显示服务信息
- `GET /health` - 健康检查
- `GET /model/info` - 获取模型信息
- `GET /device/info` - 获取设备信息

### 预测端点

- `POST /predict` - 单个日志预测
- `POST /predict/batch` - 批量日志预测

### 管理端点

- `POST /reload` - 重新加载模型
- `POST /switch_device` - 切换设备模式

## 使用示例

### 1. 单个预测

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "ERROR: java.lang.NullPointerException at com.example.Controller.handleRequest"}'
```

### 2. 批量预测

```bash
curl -X POST http://localhost:5000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "ERROR: java.lang.NullPointerException at com.example.Controller.handleRequest",
      "INFO: Application started successfully on port 8080",
      "WARN: Database connection timeout, retrying..."
    ]
  }'
```

### 3. 切换设备模式

```bash
# 切换到CPU模式
curl -X POST http://localhost:5000/switch_device \
  -H "Content-Type: application/json" \
  -d '{"use_gpu": false}'

# 切换到GPU模式
curl -X POST http://localhost:5000/switch_device \
  -H "Content-Type: application/json" \
  -d '{"use_gpu": true}'
```

## 性能测试

### 并发测试

使用提供的并发测试脚本：

```bash
python concurrent_test.py
```

测试配置：
- 每秒10个请求
- 每个请求20行日志
- 持续60秒
- 总请求数：600个
- 总日志数：12,000行

### 性能指标

- **响应时间**: 平均140ms (GPU模式)
- **吞吐量**: 约7.14请求/秒
- **成功率**: 100%
- **设备使用**: 100% GPU (xpu:0)

## 设备要求

### CPU模式
- Python 3.7+
- 依赖包: flask, flask-cors, lightgbm, joblib, numpy, pandas

### GPU模式
- Intel Arc GPU (A750/A770等)
- PyTorch XPU支持
- 依赖包: torch (XPU版本), intel-extension-for-pytorch

## 故障排除

### 1. GPU不可用
```
⚠️  GPU不可用，强制使用CPU模式
```
**解决方案**: 检查PyTorch XPU安装和GPU驱动

### 2. 模型加载失败
```
❌ 模型加载失败，请检查模型文件
```
**解决方案**: 确保models目录下有训练好的模型文件

### 3. 端口被占用
```
Address already in use
```
**解决方案**: 使用 `--port` 参数指定其他端口

## 开发说明

### 项目结构
```
logsense-xpu/
├── api_server_enhanced.py    # 增强版API服务器
├── start_api_cpu.bat         # CPU模式启动脚本
├── start_api_gpu.bat         # GPU模式启动脚本
├── concurrent_test.py         # 并发测试脚本
├── models/                   # 模型文件目录
└── README_API_ENHANCED.md   # 本文档
```

### 扩展功能
- 支持更多模型类型
- 添加模型版本管理
- 实现负载均衡
- 添加监控和日志

## 版本信息

- **版本**: 2.0.0
- **更新日期**: 2025-08-03
- **支持设备**: Intel Arc A750/A770
- **Python版本**: 3.7+ 