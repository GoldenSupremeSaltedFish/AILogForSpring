# AILogForSpring 项目

## 项目概述
基于Spring Boot的AI日志处理和分析项目，支持多种日志格式的处理、清洗、去重和分类。

## 目录结构

```
AILogForSpring/
├── README.md                    # 项目说明
├── log-processing/             # 日志处理脚本
│   ├── log_cleaner.py          # 通用日志清洗器
│   ├── universal_log_cleaner.py # 通用日志清洗器（策略模式）
│   ├── clean_and_filter_logs.py # 日志清洗和过滤
│   └── process_logs.py         # 日志处理脚本
├── gateway-tools/              # Gateway处理工具
│   ├── gateway_processor.py    # Gateway处理器
│   ├── process_gateway.py      # Gateway处理脚本
│   └── process_gateway_logs.py # Gateway日志处理
├── batch-scripts/              # 批处理脚本
│   └── run_gateway_process.bat # Gateway处理批处理
├── utilities/                  # 通用工具
├── documentation/              # 项目文档
└── logsense-xpu/              # LogSense-XPU子项目
    ├── main.py                # 主入口
    ├── core/                  # 核心模块
    ├── scripts/               # 功能脚本
    ├── data/                  # 数据目录
    └── docs/                  # 文档
```

## 快速开始

### 1. 环境准备
```bash
cd logsense-xpu
pip install -r config/requirements.txt
```

### 2. 日志处理
```bash
# 使用通用日志清洗器
python log-processing/universal_log_cleaner.py <日志目录>

# 处理Gateway日志
python gateway-tools/process_gateway_logs.py <Gateway日志目录>
```

### 3. LogSense-XPU子项目
```bash
cd logsense-xpu
python main.py  # 查看可用功能
```

## 功能特点

- **多格式支持**: 支持Gateway、MQTT、User等多种日志格式
- **智能处理**: 基于策略模式的智能日志识别和处理
- **高效去重**: 精确去重和模糊去重两种模式
- **机器学习**: 基于Intel XPU的日志分类和异常检测
- **模块化设计**: 清晰的功能分离和代码组织

## 子项目说明

### LogSense-XPU
高级日志分析子项目，包含：
- 数据预处理和清洗
- embedding向量构造
- 机器学习模型训练
- 异常检测和分类

详细使用说明请参考 `logsense-xpu/docs/` 目录。

## 开发团队
日志处理和AI分析项目团队
