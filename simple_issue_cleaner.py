# -*- coding: utf-8 -*-
"""
简单的Issue日志数据清洗脚本
"""

import pandas as pd
import re
from pathlib import Path
from datetime import datetime

def clean_message(message):
    """清洗日志消息"""
    if pd.isna(message) or not isinstance(message, str):
        return ""
    
    # 移除HTML标签
    message = re.sub(r'<[^>]+>', '', message)
    
    # 移除Markdown格式
    message = re.sub(r'\*\*([^*]+)\*\*', r'\1', message)
    message = re.sub(r'\*([^*]+)\*', r'\1', message)
    message = re.sub(r'`([^`]+)`', r'\1', message)
    
    # 移除URL
    message = re.sub(r'http[s]?://[^\s]+', '', message)
    
    # 移除多余空白字符
    message = re.sub(r'\s+', ' ', message)
    message = message.strip()
    
    return message

def should_filter_message(message):
    """判断是否应该过滤掉这条消息"""
    if pd.isna(message) or not isinstance(message, str):
        return True
    
    # 过滤掉太短的消息
    if len(message.strip()) < 10:
        return True
    
    # 过滤掉包含特定模式的消息
    filter_patterns = [
        r'### What did you expect to happen',
        r'### What happened',
        r'### What did I do',
        r'@dependabot',
        r'Dependabot compatibility score',
        r'<li>.*</li>',
        r'<p>.*</p>',
        r'<code>.*</code>',
    ]
    
    for pattern in filter_patterns:
        if re.search(pattern, message, re.IGNORECASE):
            return True
    
    return False

def classify_log(message):
    """根据日志内容自动分类"""
    if pd.isna(message) or not isinstance(message, str):
        return 'unknown'
    
    message_lower = message.lower()
    
    # 定义分类规则
    patterns = {
        'stack_exception': [
            r'Exception|Error|Throwable|RuntimeException|NullPointerException',
            r'java\.lang\..*Exception',
            r'org\.springframework\..*Exception',
            r'Caused by:|at |Stack trace',
            r'java: cannot find symbol',
            r'compilation failed'
        ],
        'startup_failure': [
            r'Failed to start|Startup failed|Application startup failed',
            r'BeanCreationException|ContextLoadException',
            r'Port already in use|Address already in use',
            r'Database connection failed|Connection refused',
            r'Unable to start|Cannot start'
        ],
        'auth_error': [
            r'Authentication failed|Authorization failed',
            r'Access denied|Permission denied',
            r'Invalid token|Token expired',
            r'Unauthorized|Forbidden',
            r'Login failed|Password incorrect',
            r'LDAP.*error|OAuth.*error'
        ],
        'db_error': [
            r'SQLException|DatabaseException',
            r'Connection.*failed|Connection.*refused',
            r'Table.*not found|Column.*not found',
            r'Duplicate entry|Constraint violation',
            r'Deadlock|Lock timeout',
            r'MySQL|PostgreSQL|Oracle.*error'
        ],
        'connection_issue': [
            r'Connection.*timeout|Connection.*refused',
            r'Network.*unreachable|Host.*unreachable',
            r'Connection.*closed|Socket.*closed',
            r'Unable to connect|Failed to connect',
            r'Zookeeper.*connection|Redis.*connection'
        ],
        'timeout': [
            r'Timeout|timeout|TIMEOUT',
            r'Request.*timeout|Response.*timeout',
            r'Read.*timeout|Write.*timeout',
            r'Operation.*timeout|Execution.*timeout'
        ],
        'performance': [
            r'OutOfMemoryError|Memory.*full',
            r'GC.*overhead|Garbage collection',
            r'Performance.*issue|Slow.*query',
            r'CPU.*high|Memory.*leak'
        ],
        'config': [
            r'Configuration.*error|Config.*not found',
            r'Property.*not found|Environment.*variable',
            r'Invalid.*configuration|Missing.*property',
            r'YAML.*error|Properties.*error'
        ],
        'business': [
            r'Business.*error|Logic.*error',
            r'Validation.*failed|Invalid.*input',
            r'Business.*exception|Service.*exception'
        ],
        'normal': [
            r'INFO.*Started|INFO.*Running',
            r'DEBUG.*|TRACE.*',
            r'Application.*started|Server.*started',
            r'Health.*check|Heartbeat'
        ]
    }
    
    # 按优先级检查各类别
    for category, category_patterns in patterns.items():
        for pattern in category_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                return category
    
    return 'unknown'

def main():
    print("🚀 开始处理Issue日志数据...")
    
    # 查找CSV文件
    input_dir = Path("issue-logs")
    output_dir = Path("DATA_OUTPUT")
    output_dir.mkdir(exist_ok=True)
    
    csv_files = list(input_dir.glob("*.csv"))
    print(f"📁 找到 {len(csv_files)} 个CSV文件")
    
    if not csv_files:
        print("❌ 未找到CSV文件")
        return
    
    # 合并所有数据
    all_data = []
    for csv_file in csv_files:
        print(f"📊 加载: {csv_file.name}")
        try:
            df = pd.read_csv(csv_file)
            print(f"  - {len(df)} 条记录")
            all_data.append(df)
        except Exception as e:
            print(f"  ❌ 读取失败: {e}")
    
    if not all_data:
        print("❌ 没有有效数据")
        return
    
    # 合并数据
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"📊 合并后总数据: {len(combined_df)} 条记录")
    
    # 清洗数据
    print("🧹 开始数据清洗...")
    
    # 清洗消息内容
    combined_df['cleaned_message'] = combined_df['message'].apply(clean_message)
    
    # 过滤掉不需要的消息
    original_count = len(combined_df)
    combined_df = combined_df[~combined_df['cleaned_message'].apply(should_filter_message)]
    filtered_count = len(combined_df)
    
    print(f"📊 过滤前: {original_count} 条")
    print(f"📊 过滤后: {filtered_count} 条")
    print(f"🗑️ 过滤掉: {original_count - filtered_count} 条")
    
    # 自动分类
    print("🏷️ 开始自动分类...")
    combined_df['auto_label'] = combined_df['cleaned_message'].apply(classify_log)
    
    # 统计分类结果
    label_counts = combined_df['auto_label'].value_counts()
    print("📈 自动分类结果:")
    for label, count in label_counts.items():
        print(f"  {label}: {count} 条")
    
    # 平衡数据
    print("⚖️ 开始数据平衡...")
    max_per_class = 500
    balanced_dfs = []
    
    for category in combined_df['auto_label'].unique():
        category_df = combined_df[combined_df['auto_label'] == category]
        if len(category_df) > max_per_class:
            category_df = category_df.sample(n=max_per_class, random_state=42)
        balanced_dfs.append(category_df)
    
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    print(f"📊 平衡后总数据: {len(balanced_df)} 条")
    
    # 准备训练数据
    training_df = balanced_df[['cleaned_message', 'auto_label']].copy()
    training_df.columns = ['text', 'label']
    training_df = training_df[training_df['text'].str.len() > 0]
    
    # 保存文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    combined_file = output_dir / f"issue_logs_combined_{timestamp}.csv"
    balanced_df.to_csv(combined_file, index=False, encoding='utf-8')
    print(f"💾 已保存合并数据集到: {combined_file}")
    
    training_file = output_dir / f"issue_logs_training_{timestamp}.csv"
    training_df.to_csv(training_file, index=False, encoding='utf-8')
    print(f"💾 已保存训练数据到: {training_file}")
    
    print(f"\n📈 最终类别分布:")
    final_counts = balanced_df['auto_label'].value_counts()
    for label, count in final_counts.items():
        percentage = (count / len(balanced_df)) * 100
        print(f"  {label}: {count} 条 ({percentage:.1f}%)")
    
    print("\n✅ 数据清洗完成！")

if __name__ == "__main__":
    main()
