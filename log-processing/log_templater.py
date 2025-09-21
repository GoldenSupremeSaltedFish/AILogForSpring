#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日志模板化工具
实现类似Drain3的日志模板化功能，将相似结构的日志归并为模板
支持噪声去除、异常字典生成和模板ID分配

使用方法:
python log_templater.py --input-file logs.csv --output-dir output/
python log_templater.py --batch --input-dir logs/ --output-dir output/
"""

import pandas as pd
import numpy as np
import re
import json
import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, Counter
import hashlib
import warnings
warnings.filterwarnings('ignore')

class LogTemplater:
    """日志模板化器"""
    
    def __init__(self):
        # 噪声模式定义
        self.noise_patterns = {
            'timestamp': [
                r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d{3})?(?:Z|[+-]\d{2}:\d{2})?',  # ISO格式
                r'\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}(?:\.\d{3})?',  # 标准格式
                r'\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}',  # 美式格式
                r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}',  # 带毫秒
            ],
            'thread_id': [
                r'\[thread-\d+\]',
                r'\[Thread-\d+\]', 
                r'\[T-\d+\]',
                r'thread-\d+',
                r'Thread-\d+',
            ],
            'uuid': [
                r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
                r'[0-9a-f]{32}',
            ],
            'request_id': [
                r'request-id:\s*[a-zA-Z0-9-]+',
                r'req-id:\s*[a-zA-Z0-9-]+',
                r'requestId:\s*[a-zA-Z0-9-]+',
            ],
            'ip_address': [
                r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
                r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b',  # IPv6
            ],
            'port_number': [
                r':\d{1,5}\b',
            ],
            'file_path': [
                r'[a-zA-Z]:\\[^:]+',
                r'/[^:]+',
                r'[a-zA-Z0-9_.-]+\.(java|py|js|ts|go|rs|cpp|c|h)',
            ],
            'line_number': [
                r':\d+\)',
                r'line \d+',
                r'at line \d+',
            ],
            'memory_address': [
                r'0x[0-9a-fA-F]+',
            ],
            'session_id': [
                r'session[_-]?id:\s*[a-zA-Z0-9-]+',
                r'sid:\s*[a-zA-Z0-9-]+',
            ]
        }
        
        # 异常关键字字典
        self.exception_keywords = {
            'java': [
                'NullPointerException', 'IllegalArgumentException', 'RuntimeException',
                'IndexOutOfBoundsException', 'ClassCastException', 'UnsupportedOperationException',
                'ConcurrentModificationException', 'NoSuchElementException', 'IllegalStateException',
                'OutOfMemoryError', 'StackOverflowError', 'NoClassDefFoundError',
                'ClassNotFoundException', 'InstantiationException', 'IllegalAccessException'
            ],
            'spring': [
                'BeanCreationException', 'BeanDefinitionStoreException', 'BeanInstantiationException',
                'NoSuchBeanDefinitionException', 'NoUniqueBeanDefinitionException',
                'TransactionSystemException', 'DataAccessException', 'DataIntegrityViolationException'
            ],
            'database': [
                'SQLException', 'DataAccessException', 'DataIntegrityViolationException',
                'DeadlockLoserDataAccessException', 'DuplicateKeyException',
                'DataRetrievalFailureException', 'InvalidDataAccessApiUsageException'
            ],
            'network': [
                'ConnectException', 'SocketTimeoutException', 'UnknownHostException',
                'BindException', 'NoRouteToHostException', 'ConnectionRefusedException'
            ],
            'web': [
                'HttpRequestMethodNotSupportedException', 'HttpMediaTypeNotSupportedException',
                'HttpMessageNotReadableException', 'MethodArgumentNotValidException',
                'MissingServletRequestParameterException', 'ServletRequestBindingException'
            ]
        }
        
        # 模板存储
        self.templates = {}
        self.template_counter = 0
        self.template_stats = defaultdict(int)
        
        # 输出目录配置
        self.output_base_dir = Path(__file__).parent.parent / "log-processing-OUTPUT"
        
    def clean_noise(self, log_line: str) -> Tuple[str, Dict[str, int]]:
        """去除日志中的噪声"""
        cleaned_line = log_line
        noise_count = {}
        
        for noise_type, patterns in self.noise_patterns.items():
            count = 0
            for pattern in patterns:
                matches = re.findall(pattern, cleaned_line, re.IGNORECASE)
                count += len(matches)
                # 替换为占位符
                if noise_type == 'timestamp':
                    cleaned_line = re.sub(pattern, '<TIMESTAMP>', cleaned_line)
                elif noise_type == 'thread_id':
                    cleaned_line = re.sub(pattern, '<THREAD_ID>', cleaned_line)
                elif noise_type == 'uuid':
                    cleaned_line = re.sub(pattern, '<UUID>', cleaned_line)
                elif noise_type == 'request_id':
                    cleaned_line = re.sub(pattern, '<REQUEST_ID>', cleaned_line)
                elif noise_type == 'ip_address':
                    cleaned_line = re.sub(pattern, '<IP>', cleaned_line)
                elif noise_type == 'port_number':
                    cleaned_line = re.sub(pattern, '<PORT>', cleaned_line)
                elif noise_type == 'file_path':
                    cleaned_line = re.sub(pattern, '<FILE_PATH>', cleaned_line)
                elif noise_type == 'line_number':
                    cleaned_line = re.sub(pattern, '<LINE>', cleaned_line)
                elif noise_type == 'memory_address':
                    cleaned_line = re.sub(pattern, '<MEMORY_ADDR>', cleaned_line)
                elif noise_type == 'session_id':
                    cleaned_line = re.sub(pattern, '<SESSION_ID>', cleaned_line)
            
            noise_count[noise_type] = count
        
        return cleaned_line, noise_count
    
    def extract_exception_keywords(self, log_line: str) -> List[str]:
        """提取异常关键字"""
        found_exceptions = []
        log_lower = log_line.lower()
        
        for category, exceptions in self.exception_keywords.items():
            for exception in exceptions:
                if exception.lower() in log_lower:
                    found_exceptions.append(f"{category}:{exception}")
        
        return found_exceptions
    
    def generate_template_id(self, cleaned_line: str) -> str:
        """生成模板ID"""
        # 使用hash生成模板ID
        template_hash = hashlib.md5(cleaned_line.encode('utf-8')).hexdigest()[:8]
        return f"T_{template_hash}"
    
    def create_template(self, cleaned_line: str, original_line: str) -> Dict:
        """创建日志模板"""
        template_id = self.generate_template_id(cleaned_line)
        
        if template_id not in self.templates:
            self.templates[template_id] = {
                'template_id': template_id,
                'template': cleaned_line,
                'count': 0,
                'examples': [],
                'exception_keywords': set(),
                'noise_stats': defaultdict(int),
                'created_at': datetime.now().isoformat()
            }
            self.template_counter += 1
        
        # 更新模板统计
        template = self.templates[template_id]
        template['count'] += 1
        self.template_stats[template_id] += 1
        
        # 保存示例（最多保存5个）
        if len(template['examples']) < 5:
            template['examples'].append(original_line)
        
        return template
    
    def process_log_line(self, log_line: str) -> Dict:
        """处理单行日志"""
        # 去除噪声
        cleaned_line, noise_count = self.clean_noise(log_line)
        
        # 提取异常关键字
        exception_keywords = self.extract_exception_keywords(log_line)
        
        # 创建或更新模板
        template = self.create_template(cleaned_line, log_line)
        
        # 更新异常关键字
        template['exception_keywords'].update(exception_keywords)
        
        # 更新噪声统计
        for noise_type, count in noise_count.items():
            template['noise_stats'][noise_type] += count
        
        return {
            'original_log': log_line,
            'cleaned_log': cleaned_line,
            'template_id': template['template_id'],
            'template': template['template'],
            'noise_count': noise_count,
            'exception_keywords': exception_keywords,
            'has_stack_trace': 'at ' in log_line.lower() or 'caused by' in log_line.lower(),
            'log_length': len(log_line),
            'cleaned_length': len(cleaned_line)
        }
    
    def process_file(self, input_file: str, output_dir: Path) -> Dict:
        """处理单个日志文件"""
        print(f"🔄 处理文件: {Path(input_file).name}")
        
        try:
            # 读取数据
            if input_file.endswith('.csv'):
                df = pd.read_csv(input_file, encoding='utf-8-sig')
                # 尝试不同的列名
                log_column = None
                for col in ['original_log', 'message', 'content', 'text', 'log']:
                    if col in df.columns:
                        log_column = col
                        break
                
                if log_column is None:
                    print(f"❌ 未找到日志列，可用列: {list(df.columns)}")
                    return {}
                
                log_lines = df[log_column].fillna('').astype(str).tolist()
            else:
                # 纯文本文件
                with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
                    log_lines = [line.strip() for line in f if line.strip()]
            
            print(f"📊 加载了 {len(log_lines)} 条日志")
            
            # 处理日志
            processed_logs = []
            for i, log_line in enumerate(log_lines):
                if i % 1000 == 0:
                    print(f"  处理进度: {i}/{len(log_lines)}")
                
                result = self.process_log_line(log_line)
                processed_logs.append(result)
            
            # 保存结果
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            input_name = Path(input_file).stem
            
            # 保存处理后的日志
            output_file = output_dir / f"{input_name}_templated_{timestamp}.csv"
            result_df = pd.DataFrame(processed_logs)
            result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            
            # 保存模板信息
            template_file = output_dir / f"{input_name}_templates_{timestamp}.json"
            template_data = {
                'metadata': {
                    'input_file': input_file,
                    'processed_at': datetime.now().isoformat(),
                    'total_logs': len(log_lines),
                    'total_templates': len(self.templates)
                },
                'templates': {}
            }
            
            # 转换模板数据为JSON可序列化格式
            for template_id, template in self.templates.items():
                template_data['templates'][template_id] = {
                    'template_id': template['template_id'],
                    'template': template['template'],
                    'count': template['count'],
                    'examples': template['examples'],
                    'exception_keywords': list(template['exception_keywords']),
                    'noise_stats': dict(template['noise_stats']),
                    'created_at': template['created_at']
                }
            
            with open(template_file, 'w', encoding='utf-8') as f:
                json.dump(template_data, f, ensure_ascii=False, indent=2)
            
            # 生成统计报告
            self.generate_template_report(output_dir, input_name, timestamp)
            
            print(f"✅ 处理完成: {output_file}")
            print(f"📊 生成了 {len(self.templates)} 个模板")
            
            return {
                'input_file': input_file,
                'output_file': str(output_file),
                'template_file': str(template_file),
                'total_logs': len(log_lines),
                'total_templates': len(self.templates)
            }
            
        except Exception as e:
            print(f"❌ 处理文件失败: {e}")
            return {}
    
    def generate_template_report(self, output_dir: Path, input_name: str, timestamp: str):
        """生成模板统计报告"""
        report_file = output_dir / f"{input_name}_template_report_{timestamp}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("日志模板化统计报告\n")
            f.write("=" * 50 + "\n")
            f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"输入文件: {input_name}\n")
            f.write(f"总日志数: {sum(self.template_stats.values())}\n")
            f.write(f"模板数量: {len(self.templates)}\n\n")
            
            # 模板使用频率统计
            f.write("模板使用频率 (Top 20):\n")
            f.write("-" * 30 + "\n")
            sorted_templates = sorted(self.template_stats.items(), key=lambda x: x[1], reverse=True)
            for template_id, count in sorted_templates[:20]:
                template = self.templates[template_id]
                f.write(f"{template_id}: {count} 次\n")
                f.write(f"  模板: {template['template'][:100]}...\n")
                f.write(f"  异常关键字: {', '.join(template['exception_keywords'])}\n\n")
            
            # 异常关键字统计
            f.write("异常关键字统计:\n")
            f.write("-" * 30 + "\n")
            all_exceptions = []
            for template in self.templates.values():
                all_exceptions.extend(template['exception_keywords'])
            
            exception_counts = Counter(all_exceptions)
            for exception, count in exception_counts.most_common(20):
                f.write(f"{exception}: {count} 次\n")
        
        print(f"📄 统计报告: {report_file}")
    
    def batch_process(self, input_dir: str, output_dir: Path):
        """批量处理目录中的所有日志文件"""
        input_path = Path(input_dir)
        if not input_path.exists():
            print(f"❌ 输入目录不存在: {input_dir}")
            return
        
        # 查找日志文件
        log_extensions = ['.log', '.txt', '.csv', '.out', '.err']
        log_files = []
        
        for ext in log_extensions:
            log_files.extend(input_path.rglob(f"*{ext}"))
        
        if not log_files:
            print("❌ 未找到日志文件")
            return
        
        print(f"📁 找到 {len(log_files)} 个日志文件")
        
        # 创建输出目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        batch_output_dir = output_dir / f"batch_templated_{timestamp}"
        batch_output_dir.mkdir(exist_ok=True, parents=True)
        
        # 处理每个文件
        results = []
        for i, log_file in enumerate(log_files, 1):
            print(f"\n{'='*50}")
            print(f"处理进度: {i}/{len(log_files)}")
            
            result = self.process_file(str(log_file), batch_output_dir)
            if result:
                results.append(result)
        
        # 生成批量处理摘要
        self.generate_batch_summary(batch_output_dir, results)
        
        print(f"\n🎉 批量处理完成！")
        print(f"📁 结果保存在: {batch_output_dir}")
        print(f"📊 成功处理: {len(results)}/{len(log_files)} 个文件")
    
    def generate_batch_summary(self, output_dir: Path, results: List[Dict]):
        """生成批量处理摘要"""
        summary_file = output_dir / "batch_processing_summary.txt"
        
        total_logs = sum(r.get('total_logs', 0) for r in results)
        total_templates = sum(r.get('total_templates', 0) for r in results)
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("批量日志模板化处理摘要\n")
            f.write("=" * 50 + "\n")
            f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"处理文件数: {len(results)}\n")
            f.write(f"总日志数: {total_logs}\n")
            f.write(f"总模板数: {total_templates}\n\n")
            
            f.write("处理结果:\n")
            f.write("-" * 30 + "\n")
            for result in results:
                f.write(f"文件: {Path(result['input_file']).name}\n")
                f.write(f"  日志数: {result['total_logs']}\n")
                f.write(f"  模板数: {result['total_templates']}\n")
                f.write(f"  输出: {Path(result['output_file']).name}\n\n")
        
        print(f"📋 批量处理摘要: {summary_file}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='日志模板化工具')
    parser.add_argument('--input-file', help='输入日志文件路径')
    parser.add_argument('--input-dir', help='输入目录路径（批量模式）')
    parser.add_argument('--output-dir', help='输出目录路径')
    parser.add_argument('--batch', action='store_true', help='批量处理模式')
    
    args = parser.parse_args()
    
    # 创建模板化器
    templater = LogTemplater()
    
    # 设置输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = templater.output_base_dir / "templated_logs"
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    if args.batch or args.input_dir:
        # 批量处理模式
        if not args.input_dir:
            print("❌ 批量模式需要指定 --input-dir")
            return
        
        templater.batch_process(args.input_dir, output_dir)
    
    elif args.input_file:
        # 单文件模式
        if not Path(args.input_file).exists():
            print(f"❌ 输入文件不存在: {args.input_file}")
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        single_output_dir = output_dir / f"single_templated_{timestamp}"
        single_output_dir.mkdir(exist_ok=True, parents=True)
        
        result = templater.process_file(args.input_file, single_output_dir)
        if result:
            print(f"\n🎉 处理完成！")
            print(f"📁 结果保存在: {single_output_dir}")
    
    else:
        print("❌ 请指定 --input-file 或使用 --batch --input-dir 进行批量处理")
        parser.print_help()

if __name__ == "__main__":
    main()
