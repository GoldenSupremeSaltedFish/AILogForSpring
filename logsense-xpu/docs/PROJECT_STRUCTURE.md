# LogSense-XPU 项目结构

## 项目概述
基于Intel XPU的日志分类和异常检测项目

## 目录结构

```
logsense-xpu/
├── main.py                    # 项目主入口
├── core/                      # 核心模块
│   ├── preprocessing.py       # 数据预处理
│   ├── utils.py              # 工具函数
│   ├── embed.py              # 向量化模块
│   ├── train.py              # 模型训练
│   └── predict.py            # 预测接口
├── scripts/                   # 各类脚本
│   ├── gateway/              # Gateway日志处理
│   │   └── gateway_simple.py
│   ├── preprocessing/        # 数据预处理脚本
│   │   ├── log_cleaner.py
│   │   └── universal_log_cleaner.py
│   ├── deduplication/        # 去重相关
│   │   └── deduplicate_logs.py
│   └── utils/                # 工具脚本
│       ├── run_dedup.py
│       ├── verify_xpu.py
│       ├── rename_logs.bat
│       └── rename_logs.ps1
├── data/                     # 数据目录
│   ├── input/               # 原始数据 (gitignored)
│   └── output/              # 处理结果
├── data_template/           # 数据格式模板
├── config/                  # 配置文件
│   └── requirements.txt
└── docs/                    # 文档
    └── README.md
```

## 使用指南

### 1. 环境准备
```bash
pip install -r config/requirements.txt
```

### 2. Gateway日志处理
```bash
python scripts/gateway/gateway_simple.py <日志目录>
```

### 3. 日志去重
```bash
python scripts/deduplication/deduplicate_logs.py <CSV目录> --mode both
```

### 4. 数据预处理
```bash
python -c "from core.preprocessing import LogPreprocessor; LogPreprocessor().process()"
```

### 5. 向量化和训练
```bash
python core/embed.py      # 生成向量
python core/train.py      # 训练模型
python core/predict.py    # 预测
```

## 功能特点

- **多格式日志支持**: 支持Gateway、MQTT等不同格式
- **智能去重**: 精确去重和模糊去重两种模式
- **数据安全**: 完整的.gitignore保护敏感数据
- **模块化设计**: 清晰的功能分离和代码组织
- **Intel XPU支持**: 针对Intel XPU优化的机器学习管道

## 开发进度

- [x] 数据采集和预处理
- [x] 日志清洗和去重
- [ ] embedding向量构造
- [ ] 模型训练和评估
- [ ] 预测和部署接口
