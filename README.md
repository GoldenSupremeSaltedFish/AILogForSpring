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

## 项目开发日志

### 📅 2025-06-21 (今日更新)

**🎯 今日完成**:
- ✅ **半自动日志标签辅助器**: 开发了智能日志标签系统
  - 支持8种标签类型：auth_error, db_error, timeout, api_success, ignore, system_error, network_error, performance
  - 实现"机器预标 + 人工校正"的半自动化流程
  - 集成scikit-learn实现智能分类
- ✅ **优化输出路径**: 统一输出至`log-processing-OUTPUT`目录
- ✅ **自动化处理**: 支持自动处理`DATA_OUTPUT`目录下的所有CSV文件
- ✅ **批处理支持**: 新增`run_auto_labeler.bat`实现一键处理

**📊 处理效果**:
- 示例数据测试: 6条日志实现100%正确分类
- 真实数据验证: 成功处理536条Gateway日志
- 分类分布: 
  - 认证错误: 28.9%
  - 心跳检测: 5.8%
  - 其他类型: 65.3%

**🔧 技术特点**:
- 基于机器学习的智能预分类
- 支持批量处理多个CSV文件
- 灵活的标签类型系统
- 人机协作的校正机制

**📈 项目进度**:
- [x] 数据采集和预处理 (100%)
- [x] 日志清洗和去重 (100%)
- [x] 自动标签系统 (100%)
- [ ] embedding向量构造 (计划中)
- [ ] 模型训练和评估 (计划中)

**🎉 里程碑**: 完成半自动标签系统，显著提升日志分类效率！

---

### 📅 项目历史记录

**2025-06-20**:
- 🚀 项目启动
- 📁 创建LogSense-XPU子项目基础结构
- 🔧 开发日志文件重命名工具
- 📝 识别不同日志格式差异

**未来计划** (下次更新):
- 🎯 开始embedding向量构造模块开发
- 🤖 集成Intel XPU支持
- 📊 开发数据可视化工具

---

## 开发团队
日志处理和AI分析项目团队

**项目维护**: 每日更新进度，记录技术决策和解决方案
