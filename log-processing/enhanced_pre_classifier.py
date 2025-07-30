#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强的日志预分类器
支持批量处理和详细的日志分类规则
"""

import os
import sys
import json
import csv
import re
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

class EnhancedPreClassifier:
    def __init__(self):
        self.log_levels = {
            'ERROR': ['ERROR', 'FATAL', 'SEVERE'],
            'WARN': ['WARN', 'WARNING'],
            'INFO': ['INFO'],
            'DEBUG': ['DEBUG', 'TRACE']
        }
        
        # 内容分类规则
        self.classification_rules = {
            'stack_exception': {
                'keywords': ['Exception', 'Error', 'at java.', 'at org.', 'at com.', 'Caused by', 'stack trace'],
                'patterns': [r'\w+Exception:', r'\w+Error:', r'at \w+\.\w+\.\w+'],
                'priority': 1
            },
            'connection_issue': {
                'keywords': ['Connection', 'refused', 'timeout', 'unreachable', 'network', 'socket'],
                'patterns': [r'Connection.*refused', r'Connection.*timeout', r'Network.*unreachable'],
                'priority': 2
            },
            'database_exception': {
                'keywords': ['SQLException', 'database', 'DB', 'mysql', 'oracle', 'postgresql', 'jdbc'],
                'patterns': [r'SQL.*Exception', r'Database.*error', r'JDBC.*error'],
                'priority': 3
            },
            'timeout': {
                'keywords': ['timeout', 'timed out', 'TimeoutException', 'read timeout', 'connect timeout'],
                'patterns': [r'timeout.*exceeded', r'\d+ms.*timeout', r'TimeoutException'],
                'priority': 4
            },
            'spring_boot_startup_failure': {
                'keywords': ['APPLICATION FAILED TO START', 'SpringApplication', 'startup failed', 'bean creation'],
                'patterns': [r'APPLICATION FAILED TO START', r'Error creating bean', r'BeanCreationException'],
                'priority': 5
            },
            'auth_authorization': {
                'keywords': ['authentication', 'authorization', 'login', 'token', 'permission', 'access denied'],
                'patterns': [r'Authentication.*failed', r'Access.*denied', r'Unauthorized'],
                'priority': 6
            },
            'memory_performance': {
                'keywords': ['OutOfMemoryError', 'memory', 'heap', 'GC', 'garbage collection'],
                'patterns': [r'OutOfMemoryError', r'GC.*overhead', r'heap.*space'],
                'priority': 7
            },
            'config_environment': {
                'keywords': ['configuration', 'property', 'environment', 'profile', 'yaml', 'properties'],
                'patterns': [r'Property.*not.*found', r'Configuration.*error'],
                'priority': 8
            },
            'business_logic': {
                'keywords': ['business', 'validation', 'rule', 'constraint', 'invalid'],
                'patterns': [r'Validation.*failed', r'Business.*rule', r'Constraint.*violation'],
                'priority': 9
            },
            'normal_operation': {
                'keywords': ['started', 'completed', 'success', 'finished', 'initialized'],
                'patterns': [r'Started.*in.*seconds', r'Completed.*successfully'],
                'priority': 10
            },
            'monitoring_heartbeat': {
                'keywords': ['health', 'heartbeat', 'ping', 'status', 'alive'],
                'patterns': [r'Health.*check', r'Heartbeat.*received'],
                'priority': 11
            }
        }
    
    def classify_log_level(self, log_line: str) -> str:
        """分类日志级别"""
        log_upper = log_line.upper()
        for level, keywords in self.log_levels.items():
            if any(keyword in log_upper for keyword in keywords):
                return level
        return 'UNKNOWN'
    
    def classify_content_type(self, log_line: str) -> Tuple[str, int]:
        """分类内容类型，返回(类型, 优先级)"""
        log_lower = log_line.lower()
        
        # 按优先级排序检查
        for category, rules in sorted(self.classification_rules.items(), 
                                    key=lambda x: x[1]['priority']):
            # 检查关键词
            if any(keyword.lower() in log_lower for keyword in rules['keywords']):
                return category, rules['priority']
            
            # 检查正则模式
            for pattern in rules['patterns']:
                if re.search(pattern, log_line, re.IGNORECASE):
                    return category, rules['priority']
        
        return 'other', 999
    
    def needs_manual_annotation(self, log_level: str, content_type: str, priority: int) -> bool:
        """判断是否需要人工标注"""
        # 高优先级问题需要人工标注
        if priority <= 5:
            return True
        
        # ERROR级别日志需要人工标注
        if log_level == 'ERROR':
            return True
        
        # 特定类型需要人工标注
        manual_types = ['stack_exception', 'database_exception', 'spring_boot_startup_failure']
        if content_type in manual_types:
            return True
        
        return False
    
    def process_log_file(self, input_file: str, output_dir: str) -> Dict[str, Any]:
        """处理单个日志文件"""
        results = {
            'file_name': os.path.basename(input_file),
            'total_lines': 0,
            'classified_lines': 0,
            'manual_annotation_needed': 0,
            'classifications': {},
            'processed_logs': []
        }
        
        try:
            with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    results['total_lines'] += 1
                    
                    # 分类
                    log_level = self.classify_log_level(line)
                    content_type, priority = self.classify_content_type(line)
                    manual_needed = self.needs_manual_annotation(log_level, content_type, priority)
                    
                    # 统计
                    if content_type != 'other':
                        results['classified_lines'] += 1
                    
                    if manual_needed:
                        results['manual_annotation_needed'] += 1
                    
                    # 更新分类统计
                    if content_type not in results['classifications']:
                        results['classifications'][content_type] = 0
                    results['classifications'][content_type] += 1
                    
                    # 保存处理结果
                    log_entry = {
                        'line_number': line_num,
                        'original_log': line,
                        'log_level': log_level,
                        'content_type': content_type,
                        'priority': priority,
                        'manual_annotation_needed': manual_needed,
                        'timestamp': datetime.now().isoformat()
                    }
                    results['processed_logs'].append(log_entry)
        
        except Exception as e:
            print(f"处理文件 {input_file} 时出错: {str(e)}")
            return results
        
        # 保存结果
        self.save_results(results, input_file, output_dir)
        return results
    
    def save_results(self, results: Dict[str, Any], input_file: str, output_dir: str):
        """保存分类结果"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        
        # 创建时间戳子目录
        timestamped_dir = os.path.join(output_dir, f"classification_{timestamp}")
        os.makedirs(timestamped_dir, exist_ok=True)
        
        # 保存详细结果 (JSON)
        json_file = os.path.join(timestamped_dir, f"{base_name}_classified.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 保存CSV格式
        csv_file = os.path.join(timestamped_dir, f"{base_name}_classified.csv")
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['line_number', 'log_level', 'content_type', 'priority', 
                           'manual_annotation_needed', 'original_log'])
            
            for log_entry in results['processed_logs']:
                writer.writerow([
                    log_entry['line_number'],
                    log_entry['log_level'],
                    log_entry['content_type'],
                    log_entry['priority'],
                    log_entry['manual_annotation_needed'],
                    log_entry['original_log']
                ])
        
        # 保存统计报告
        report_file = os.path.join(timestamped_dir, f"{base_name}_report.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"日志分类报告\n")
            f.write(f"文件: {results['file_name']}\n")
            f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"总行数: {results['total_lines']}\n")
            f.write(f"已分类行数: {results['classified_lines']}\n")
            f.write(f"需要人工标注: {results['manual_annotation_needed']}\n\n")
            f.write("分类统计:\n")
            for category, count in results['classifications'].items():
                f.write(f"  {category}: {count}\n")
        
        print(f"结果已保存到: {timestamped_dir}")
    
    def batch_process(self, input_dir: str, output_dir: str):
        """批量处理目录中的所有日志文件"""
        if not os.path.exists(input_dir):
            print(f"输入目录不存在: {input_dir}")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 添加 .csv 扩展名支持
        log_extensions = ['.log', '.txt', '.out', '.err', '.csv']
        processed_files = 0
        
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in log_extensions):
                    input_file = os.path.join(root, file)
                    print(f"正在处理: {input_file}")
                    
                    try:
                        self.process_log_file(input_file, output_dir)
                        processed_files += 1
                        print(f"✓ 完成处理: {file}")
                    except Exception as e:
                        print(f"✗ 处理失败: {file} - {str(e)}")
        
        print(f"\n批量处理完成，共处理 {processed_files} 个文件")

def main():
    parser = argparse.ArgumentParser(description='增强的日志预分类器')
    parser.add_argument('mode', choices=['single', 'batch'], help='处理模式')
    parser.add_argument('--input-file', help='单文件模式：输入日志文件路径')
    parser.add_argument('--input-dir', help='批量模式：输入目录路径')
    parser.add_argument('--output-dir', help='输出目录路径')
    
    args = parser.parse_args()
    
    classifier = EnhancedPreClassifier()
    
    if args.mode == 'single':
        if not args.input_file or not args.output_dir:
            print("单文件模式需要指定 --input-file 和 --output-dir")
            return
        
        if not os.path.exists(args.input_file):
            print(f"输入文件不存在: {args.input_file}")
            return
        
        classifier.process_log_file(args.input_file, args.output_dir)
    
    elif args.mode == 'batch':
        if not args.input_dir or not args.output_dir:
            print("批量模式需要指定 --input-dir 和 --output-dir")
            return
        
        classifier.batch_process(args.input_dir, args.output_dir)

if __name__ == '__main__':
    main()