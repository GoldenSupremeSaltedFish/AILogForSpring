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
- ✅ **完成Gateway日志处理**: 成功处理58个日志文件，提取218万条记录
- ✅ **实现智能去重功能**: 开发了精确去重和模糊去重两种算法
  - 精确去重: 从218万条减少到9.9万条 (减少95.5%)  
  - 模糊去重: 从218万条减少到8,458条 (减少99.6%)
- ✅ **项目结构重组**: 按功能区分文件，创建了清晰的目录结构
- ✅ **根目录整理**: 将散乱的脚本文件分类到功能目录中

**📊 数据处理成果**:
- 原始日志文件: 58个
- 处理后数据: 522MB → 几MB (大幅压缩)
- 去重效果: 99.6%的重复日志被智能识别并清理
- 输出格式: 结构化CSV文件，便于后续分析

**🔧 技术亮点**:
- 基于正则表达式的多格式日志解析
- 策略模式的通用日志处理器
- 智能去重算法(替换ID、时间戳、IP等动态内容)
- 模块化的项目架构设计

**📈 项目进度**:
- [x] 数据采集和预处理 (100%)
- [x] 日志清洗和去重 (100%) 
- [x] 项目结构化组织 (100%)
- [ ] embedding向量构造 (0% - 下一步)
- [ ] 模型训练和评估 (0%)
- [ ] 预测和部署接口 (0%)

**🎉 里程碑**: 第一阶段(数据处理)完美收官！为机器学习管道奠定了坚实基础。

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
