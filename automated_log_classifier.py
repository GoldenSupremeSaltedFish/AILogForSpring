#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动化日志分类器
整合所有分类规则和存储结构，提供统一的分类服务
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import re
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# 尝试导入机器学习库
try:
    import joblib
    import lightgbm as lgb
    from sklearn.feature_extraction.text import TfidfVectorizer
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠️  机器学习库未安装，将仅使用规则分类")

class AutomatedLogClassifier:
    """自动化日志分类器"""
    
    def __init__(self, config_file: str = None):
        """初始化分类器"""
        self.config = self._load_config(config_file)
        self.classification_rules = self._init_classification_rules()
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        
        # 数据路径配置
        self.data_paths = {
            'raw': Path("DATA_OUTPUT/原始项目数据_original"),
            'processed': Path("DATA_OUTPUT"),
            'models': Path("logsense-xpu/models"),
            'output': Path("log-processing-OUTPUT")
        }
        
        # 确保输出目录存在
        self.data_paths['output'].mkdir(exist_ok=True, parents=True)
        
        # 加载模型（如果可用）
        self._load_model()
    
    def _load_config(self, config_file: str = None) -> Dict:
        """加载配置文件"""
        if config_file and Path(config_file).exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # 默认配置
        return {
            "classification_rules": {
                "priority_order": [
                    "stack_exception",
                    "spring_boot_startup_failure", 
                    "auth_authorization",
                    "database_exception",
                    "connection_issue",
                    "timeout",
                    "memory_performance",
                    "config_environment",
                    "business_logic",
                    "normal_operation",
                    "monitoring_heartbeat"
                ]
            },
            "quality_thresholds": {
                "min_classification_coverage": 80.0,
                "max_manual_annotation_ratio": 30.0,
                "confidence_threshold": 0.7
            }
        }
    
    def _init_classification_rules(self) -> Dict:
        """初始化分类规则"""
        return {
            'stack_exception': {
                'keywords': ['Exception', 'Error', 'at java.', 'at org.', 'at com.', 'Caused by', 'stack trace', 'NullPointerException', 'RuntimeException'],
                'patterns': [r'\w+Exception:', r'\w+Error:', r'at \w+\.\w+\.\w+', r'Caused by:'],
                'priority': 1,
                'description': '堆栈异常'
            },
            'spring_boot_startup_failure': {
                'keywords': ['APPLICATION FAILED TO START', 'SpringApplication', 'startup failed', 'bean creation', 'BeanCreationException'],
                'patterns': [r'APPLICATION FAILED TO START', r'Error creating bean', r'BeanCreationException'],
                'priority': 2,
                'description': 'Spring Boot启动失败'
            },
            'auth_authorization': {
                'keywords': ['authentication', 'authorization', 'login', 'token', 'permission', 'access denied', 'unauthorized', '401', '403'],
                'patterns': [r'Authentication.*failed', r'Access.*denied', r'Unauthorized', r'401', r'403'],
                'priority': 3,
                'description': '认证授权'
            },
            'database_exception': {
                'keywords': ['SQLException', 'database', 'DB', 'mysql', 'oracle', 'postgresql', 'jdbc', 'DataAccessException'],
                'patterns': [r'SQL.*Exception', r'Database.*error', r'JDBC.*error', r'DataAccessException'],
                'priority': 4,
                'description': '数据库异常'
            },
            'connection_issue': {
                'keywords': ['Connection', 'refused', 'timeout', 'unreachable', 'network', 'socket', 'ConnectException'],
                'patterns': [r'Connection.*refused', r'Connection.*timeout', r'Network.*unreachable', r'ConnectException'],
                'priority': 5,
                'description': '连接问题'
            },
            'timeout': {
                'keywords': ['timeout', 'timed out', 'TimeoutException', 'read timeout', 'connect timeout'],
                'patterns': [r'timeout.*exceeded', r'\d+ms.*timeout', r'TimeoutException'],
                'priority': 6,
                'description': '超时错误'
            },
            'memory_performance': {
                'keywords': ['OutOfMemoryError', 'memory', 'heap', 'GC', 'garbage collection', 'performance'],
                'patterns': [r'OutOfMemoryError', r'GC.*overhead', r'heap.*space'],
                'priority': 7,
                'description': '内存性能'
            },
            'config_environment': {
                'keywords': ['configuration', 'property', 'environment', 'profile', 'yaml', 'properties'],
                'patterns': [r'Property.*not.*found', r'Configuration.*error'],
                'priority': 8,
                'description': '配置环境'
            },
            'business_logic': {
                'keywords': ['business', 'validation', 'rule', 'constraint', 'invalid'],
                'patterns': [r'Validation.*failed', r'Business.*rule', r'Constraint.*violation'],
                'priority': 9,
                'description': '业务逻辑'
            },
            'normal_operation': {
                'keywords': ['started', 'completed', 'success', 'finished', 'initialized', 'INFO'],
                'patterns': [r'Started.*in.*seconds', r'Completed.*successfully'],
                'priority': 10,
                'description': '正常操作'
            },
            'monitoring_heartbeat': {
                'keywords': ['health', 'heartbeat', 'ping', 'status', 'alive', 'actuator'],
                'patterns': [r'Health.*check', r'Heartbeat.*received'],
                'priority': 11,
                'description': '监控心跳'
            }
        }
    
    def _load_model(self):
        """加载机器学习模型"""
        if not ML_AVAILABLE:
            return
        
        try:
            # 查找最新的模型文件
            model_files = list(self.data_paths['models'].glob("lightgbm_model_*.txt"))
            if not model_files:
                print("⚠️  未找到模型文件，将仅使用规则分类")
                return
            
            # 获取最新的模型文件
            latest_model = max(model_files, key=os.path.getctime)
            timestamp = latest_model.stem.split('_')[-1]
            
            # 加载模型组件
            model_file = self.data_paths['models'] / f"lightgbm_model_{timestamp}.txt"
            vectorizer_file = self.data_paths['models'] / f"tfidf_vectorizer_{timestamp}.joblib"
            encoder_file = self.data_paths['models'] / f"label_encoder_{timestamp}.joblib"
            
            if all(f.exists() for f in [model_file, vectorizer_file, encoder_file]):
                self.model = lgb.Booster(model_file=str(model_file))
                self.vectorizer = joblib.load(vectorizer_file)
                self.label_encoder = joblib.load(encoder_file)
                print(f"✅ 成功加载模型: {timestamp}")
            else:
                print("⚠️  模型文件不完整，将仅使用规则分类")
                
        except Exception as e:
            print(f"⚠️  模型加载失败: {e}，将仅使用规则分类")
    
    def classify_log_level(self, log_line: str) -> str:
        """分类日志级别"""
        log_upper = log_line.upper()
        if 'ERROR' in log_upper or 'FATAL' in log_upper or 'SEVERE' in log_upper:
            return 'ERROR'
        elif 'WARN' in log_upper or 'WARNING' in log_upper:
            return 'WARN'
        elif 'INFO' in log_upper:
            return 'INFO'
        elif 'DEBUG' in log_upper or 'TRACE' in log_upper:
            return 'DEBUG'
        else:
            return 'UNKNOWN'
    
    def classify_by_rules(self, log_line: str) -> Tuple[str, float, str]:
        """使用规则分类日志"""
        log_lower = log_line.lower()
        
        # 按优先级排序检查
        for category, rules in sorted(self.classification_rules.items(), 
                                    key=lambda x: x[1]['priority']):
            # 检查关键词
            for keyword in rules['keywords']:
                if keyword.lower() in log_lower:
                    return category, 0.8, f"keyword: {keyword}"
            
            # 检查正则模式
            for pattern in rules['patterns']:
                if re.search(pattern, log_line, re.IGNORECASE):
                    return category, 0.8, f"pattern: {pattern}"
        
        return 'other', 0.0, 'no match'
    
    def classify_by_ml(self, log_line: str) -> Tuple[str, float]:
        """使用机器学习模型分类"""
        if not self.model or not self.vectorizer or not self.label_encoder:
            return 'other', 0.0
        
        try:
            # 特征提取
            features = self.vectorizer.transform([log_line])
            
            # 预测
            prediction = self.model.predict(features)
            probabilities = self.model.predict(features, pred_leaf=False)
            
            # 获取最高概率的类别
            max_prob_idx = np.argmax(probabilities)
            confidence = probabilities[0][max_prob_idx]
            
            # 解码标签
            if hasattr(self.label_encoder, 'classes_'):
                predicted_class = self.label_encoder.classes_[max_prob_idx]
            else:
                predicted_class = str(max_prob_idx)
            
            return predicted_class, confidence
            
        except Exception as e:
            print(f"⚠️  ML分类失败: {e}")
            return 'other', 0.0
    
    def classify_single_log(self, log_line: str, use_ml: bool = True) -> Dict:
        """分类单条日志"""
        # 基础信息提取
        log_level = self.classify_log_level(log_line)
        
        # 规则分类
        rule_category, rule_confidence, rule_reason = self.classify_by_rules(log_line)
        
        # 机器学习分类
        if use_ml and ML_AVAILABLE:
            ml_category, ml_confidence = self.classify_by_ml(log_line)
        else:
            ml_category, ml_confidence = 'other', 0.0
        
        # 选择最终分类结果
        if rule_confidence > 0.7:
            final_category = rule_category
            final_confidence = rule_confidence
            method = 'rules'
        elif ml_confidence > 0.5:
            final_category = ml_category
            final_confidence = ml_confidence
            method = 'ml'
        else:
            final_category = rule_category
            final_confidence = rule_confidence
            method = 'rules_fallback'
        
        # 判断是否需要人工标注
        needs_manual = self._needs_manual_annotation(log_level, final_category, final_confidence)
        
        return {
            'original_log': log_line,
            'log_level': log_level,
            'category': final_category,
            'confidence': final_confidence,
            'method': method,
            'rule_reason': rule_reason,
            'needs_manual_annotation': needs_manual,
            'timestamp': datetime.now().isoformat()
        }
    
    def _needs_manual_annotation(self, log_level: str, category: str, confidence: float) -> bool:
        """判断是否需要人工标注"""
        # 高优先级问题需要人工标注
        if category in ['stack_exception', 'spring_boot_startup_failure', 'auth_authorization', 'database_exception']:
            return True
        
        # ERROR级别日志需要人工标注
        if log_level == 'ERROR':
            return True
        
        # 低置信度需要人工标注
        if confidence < self.config['quality_thresholds']['confidence_threshold']:
            return True
        
        return False
    
    def classify_file(self, input_file: str, output_file: str = None, use_ml: bool = True) -> Dict:
        """分类整个文件"""
        print(f"🔄 开始分类文件: {Path(input_file).name}")
        
        try:
            # 读取文件
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
            
            # 分类每条日志
            results = []
            for i, log_line in enumerate(log_lines):
                if i % 1000 == 0:
                    print(f"  处理进度: {i}/{len(log_lines)}")
                
                result = self.classify_single_log(log_line, use_ml)
                results.append(result)
            
            # 保存结果
            if output_file is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                input_name = Path(input_file).stem
                output_file = self.data_paths['output'] / f"{input_name}_classified_{timestamp}.csv"
            
            result_df = pd.DataFrame(results)
            result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            
            # 生成统计报告
            stats = self._generate_classification_stats(results)
            self._save_classification_report(output_file, stats)
            
            print(f"✅ 分类完成: {output_file}")
            return {
                'input_file': input_file,
                'output_file': str(output_file),
                'total_logs': len(log_lines),
                'stats': stats
            }
            
        except Exception as e:
            print(f"❌ 分类失败: {e}")
            return {}
    
    def _generate_classification_stats(self, results: List[Dict]) -> Dict:
        """生成分类统计"""
        total_logs = len(results)
        category_counts = {}
        confidence_stats = []
        manual_needed = 0
        
        for result in results:
            category = result['category']
            confidence = result['confidence']
            
            category_counts[category] = category_counts.get(category, 0) + 1
            confidence_stats.append(confidence)
            
            if result['needs_manual_annotation']:
                manual_needed += 1
        
        return {
            'total_logs': total_logs,
            'category_distribution': category_counts,
            'avg_confidence': np.mean(confidence_stats),
            'manual_annotation_needed': manual_needed,
            'manual_annotation_ratio': (manual_needed / total_logs) * 100,
            'classification_coverage': ((total_logs - category_counts.get('other', 0)) / total_logs) * 100
        }
    
    def _save_classification_report(self, output_file: str, stats: Dict):
        """保存分类报告"""
        report_file = Path(output_file).with_suffix('.txt')
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("日志分类统计报告\n")
            f.write("=" * 50 + "\n")
            f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总日志数: {stats['total_logs']}\n")
            f.write(f"平均置信度: {stats['avg_confidence']:.3f}\n")
            f.write(f"分类覆盖率: {stats['classification_coverage']:.1f}%\n")
            f.write(f"需要人工标注: {stats['manual_annotation_needed']} 条 ({stats['manual_annotation_ratio']:.1f}%)\n\n")
            
            f.write("类别分布:\n")
            f.write("-" * 30 + "\n")
            for category, count in sorted(stats['category_distribution'].items(), key=lambda x: x[1], reverse=True):
                percentage = (count / stats['total_logs']) * 100
                description = self.classification_rules.get(category, {}).get('description', category)
                f.write(f"{description}: {count} 条 ({percentage:.1f}%)\n")
        
        print(f"📄 统计报告: {report_file}")
    
    def batch_classify(self, input_dir: str, output_dir: str = None, use_ml: bool = True) -> Dict:
        """批量分类目录中的所有日志文件"""
        input_path = Path(input_dir)
        if not input_path.exists():
            print(f"❌ 输入目录不存在: {input_dir}")
            return {}
        
        # 设置输出目录
        if output_dir is None:
            output_dir = self.data_paths['output'] / f"batch_classified_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # 查找日志文件
        log_extensions = ['.log', '.txt', '.csv', '.out', '.err']
        log_files = []
        
        for ext in log_extensions:
            log_files.extend(input_path.rglob(f"*{ext}"))
        
        if not log_files:
            print("❌ 未找到日志文件")
            return {}
        
        print(f"📁 找到 {len(log_files)} 个日志文件")
        
        # 处理每个文件
        results = []
        for i, log_file in enumerate(log_files, 1):
            print(f"\n{'='*50}")
            print(f"处理进度: {i}/{len(log_files)} - {log_file.name}")
            print('='*50)
            
            try:
                result = self.classify_file(str(log_file), str(output_dir / f"{log_file.stem}_classified.csv"), use_ml)
                if result:
                    results.append(result)
            except Exception as e:
                print(f"❌ 处理文件失败: {e}")
        
        # 生成批量处理报告
        self._save_batch_report(output_dir, results)
        
        success_count = len(results)
        print(f"\n🎉 批量分类完成！")
        print(f"📊 成功处理: {success_count}/{len(log_files)} 个文件")
        print(f"📁 结果保存在: {output_dir}")
        
        return {
            'total_files': len(log_files),
            'success_count': success_count,
            'results': results,
            'output_dir': str(output_dir)
        }
    
    def _save_batch_report(self, output_dir: Path, results: List[Dict]):
        """保存批量处理报告"""
        report_file = output_dir / "batch_classification_report.txt"
        
        total_logs = sum(r.get('total_logs', 0) for r in results)
        total_categories = {}
        
        for result in results:
            stats = result.get('stats', {})
            for category, count in stats.get('category_distribution', {}).items():
                total_categories[category] = total_categories.get(category, 0) + count
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("批量日志分类报告\n")
            f.write("=" * 50 + "\n")
            f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"处理文件数: {len(results)}\n")
            f.write(f"总日志数: {total_logs}\n\n")
            
            f.write("总体类别分布:\n")
            f.write("-" * 30 + "\n")
            for category, count in sorted(total_categories.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_logs) * 100 if total_logs > 0 else 0
                description = self.classification_rules.get(category, {}).get('description', category)
                f.write(f"{description}: {count} 条 ({percentage:.1f}%)\n")
        
        print(f"📄 批量处理报告: {report_file}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='自动化日志分类器')
    parser.add_argument('--input-file', help='输入日志文件路径')
    parser.add_argument('--input-dir', help='输入目录路径（批量模式）')
    parser.add_argument('--output-file', help='输出文件路径')
    parser.add_argument('--output-dir', help='输出目录路径')
    parser.add_argument('--config', help='配置文件路径')
    parser.add_argument('--no-ml', action='store_true', help='不使用机器学习分类')
    parser.add_argument('--batch', action='store_true', help='批量处理模式')
    
    args = parser.parse_args()
    
    # 创建分类器
    classifier = AutomatedLogClassifier(args.config)
    
    use_ml = not args.no_ml
    
    if args.batch or args.input_dir:
        # 批量处理模式
        if not args.input_dir:
            print("❌ 批量模式需要指定 --input-dir")
            return
        
        classifier.batch_classify(args.input_dir, args.output_dir, use_ml)
    
    elif args.input_file:
        # 单文件模式
        if not Path(args.input_file).exists():
            print(f"❌ 输入文件不存在: {args.input_file}")
            return
        
        classifier.classify_file(args.input_file, args.output_file, use_ml)
    
    else:
        print("❌ 请指定 --input-file 或使用 --batch --input-dir 进行批量处理")
        parser.print_help()

if __name__ == "__main__":
    main()
