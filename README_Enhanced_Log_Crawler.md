# Spring Boot 日志抓取工具 - 增强版

## 🎯 项目目标

扩充Spring Boot项目的日志数据，覆盖多种应用场景，提升日志分类模型的性能和泛化能力。

## 📊 数据覆盖范围

### 按应用类型分类

| 类别 | 仓库示例 | 典型日志类型 |
|------|----------|-------------|
| **电商/业务系统** | macrozheng/mall, lenve/vhr | 业务校验异常、支付接口错误、库存不足 |
| **微服务架构** | alibaba/spring-cloud-alibaba, zhoutaoo/SpringCloud | 服务发现失败、负载均衡错误、熔断器触发 |
| **安全/权限系统** | spring-projects/spring-security | 认证失败、权限不足、Token过期 |
| **数据处理/ETL** | spring-projects/spring-batch | 批量处理异常、数据转换错误、调度失败 |
| **消息队列集成** | spring-projects/spring-kafka, spring-projects/spring-amqp | 消息消费失败、序列化错误、连接超时 |
| **数据库驱动集成** | baomidou/mybatis-plus-spring-boot-starter | SQL异常、连接池溢出、事务冲突 |
| **分布式事务** | dromara/Raincat, dromara/Hmily | 事务回滚、补偿失败、状态不一致 |

### 增强的日志识别

支持识别以下类型的日志：

- **基础异常**: Exception, RuntimeException, NullPointerException
- **Spring框架异常**: BeanCreationException, NoSuchBeanDefinitionException
- **数据库异常**: SQLException, DataIntegrityViolationException
- **安全异常**: AuthenticationException, AccessDeniedException
- **HTTP异常**: HttpMessageNotReadableException, MethodArgumentNotValidException
- **消息队列异常**: SerializationException, ConnectionException
- **分布式异常**: TransactionException, LockException

## 🚀 快速开始

### 1. 配置文件

使用 `config_extended.json` 配置文件：

```json
{
    "github_token": "your_github_token",
    "repositories": [
        "alibaba/spring-cloud-alibaba",
        "macrozheng/mall",
        "spring-projects/spring-boot"
    ],
    "output_directory": "issue-logs",
    "max_pages": 20,
    "categories": {
        "ecommerce": ["macrozheng/mall"],
        "microservices": ["alibaba/spring-cloud-alibaba"],
        "core_framework": ["spring-projects/spring-boot"]
    }
}
```

### 2. 运行脚本

#### 方法一：使用批处理脚本
```bash
run_enhanced_log_crawler.bat
```

#### 方法二：直接运行Python脚本
```bash
python issue-helper-enhanced.py
```

### 3. 查看结果

脚本运行完成后，检查 `issue-logs` 目录中的CSV文件：

```
issue-logs/
├── alibaba_spring-cloud-alibaba_logs_20250111_143022_microservices.csv
├── macrozheng_mall_logs_20250111_143156_ecommerce.csv
└── spring-projects_spring-boot_logs_20250111_143245_core_framework.csv
```

## 📈 功能特点

### 1. 智能日志识别
- 支持代码块内的日志提取
- 支持内联文本的日志识别
- 增强的正则表达式匹配
- 多种异常类型识别

### 2. 分类管理
- 按应用场景分类存储
- 支持自定义分类规则
- 便于后续数据分析和模型训练

### 3. 详细统计
- 按类别统计日志数量
- 成功/失败仓库统计
- 详细的错误信息记录

### 4. 错误处理
- 网络异常重试机制
- API限流处理
- 优雅的错误恢复

## 🔧 配置说明

### GitHub Token
1. 访问 [GitHub Settings > Personal Access Tokens](https://github.com/settings/tokens)
2. 生成新的Token，选择 `repo` 权限
3. 将Token添加到配置文件中

### 仓库配置
- `repositories`: 要抓取的GitHub仓库列表
- `max_pages`: 每个仓库最大抓取页数（每页100个issues）
- `output_directory`: 输出目录路径

### 分类配置
```json
"categories": {
    "category_name": ["repo1", "repo2"],
    "another_category": ["repo3", "repo4"]
}
```

## 📊 输出格式

CSV文件包含以下字段：

| 字段 | 说明 |
|------|------|
| timestamp | 抓取时间戳 |
| message | 日志内容 |
| label | 标签（默认为unknown，需要手动标注） |
| source | 数据源（github_issue） |
| repository | 仓库名称 |
| issue_number | Issue编号 |
| issue_title | Issue标题 |
| issue_url | Issue链接 |
| category | 应用类别 |

## 🔄 工作流程

1. **配置检查**: 验证GitHub Token和配置文件
2. **仓库抓取**: 按类别抓取不同仓库的issues
3. **日志提取**: 从issue正文和评论中提取日志
4. **数据保存**: 按类别保存为CSV文件
5. **统计报告**: 生成详细的抓取统计

## 💡 最佳实践

### 1. 分批抓取
- 避免一次性抓取过多仓库
- 建议每次抓取5-10个仓库
- 注意GitHub API限制

### 2. 数据质量
- 使用 `log_reviewer.py` 进行日志标注
- 过滤掉重复和低质量的日志
- 确保日志的完整性和准确性

### 3. 模型训练
- 按类别平衡训练数据
- 使用交叉验证评估模型性能
- 定期更新训练数据

## 🛠️ 故障排除

### 常见问题

1. **API限流**
   - 减少并发请求
   - 增加请求间隔
   - 检查Token权限

2. **网络超时**
   - 检查网络连接
   - 增加超时时间
   - 重试机制

3. **权限错误**
   - 验证GitHub Token
   - 检查Token权限范围
   - 确认仓库访问权限

### 日志分析
- 查看控制台输出的详细错误信息
- 检查API响应状态码
- 分析失败仓库的具体原因

## 📚 相关工具

- `log_reviewer.py`: 日志标注和验证工具
- `log_categorizer.py`: 日志分类工具
- `data_cleaner.py`: 数据清洗工具

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个工具！

## 📄 许可证

MIT License
