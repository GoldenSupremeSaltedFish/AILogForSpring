#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
半自动日志标签辅助器
功能：
1. 使用关键词规则进行初步分类
2. 可选地使用TF-IDF + 简单分类器
3. 输出带有predicted_label列的CSV文件
4. 支持人工校正后的迭代训练
5. 不带参数时自动批量处理DATA_OUTPUT目录下的所有CSV文件

使用方法:
python auto_labeler.py                           # 自动批量处理DATA_OUTPUT目录
python auto_labeler.py <输入CSV文件路径>          # 处理指定文件
python auto_labeler.py <文件路径> --use-ml       # 使用机器学习
"""

import pandas as pd
import numpy as np
import re
import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# 尝试导入机器学习库
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠️  scikit-learn未安装，将仅使用规则分类")


class LogAutoLabeler:
    """日志自动标签器"""
    
    def __init__(self):
        # 定义标签分类规则
        self.label_rules = {
            'auth_error': {
                'keywords': [
                    'token', 'unauthorized', 'authentication', 'auth', 'jwt', 
                    'login', 'logout', 'permission', 'access denied', '401', 
                    '403', 'forbidden', 'credentials', '认证', '授权', '令牌',
                    'security', 'authz', 'authn', 'oauth', 'credential'
                ],
                'description': '登录、权限异常'
            },
            'db_error': {
                'keywords': [
                    'sqlexception', 'dataaccess', 'database', 'connection', 'sql',
                    'hibernate', 'mybatis', 'jdbc', 'mysql', 'oracle', 'postgres',
                    'deadlock', 'constraint', '数据库', 'DB', 'datasource',
                    'connectionpool', 'transaction', 'rollback', 'commit'
                ],
                'description': '数据库异常'
            },
            'timeout': {
                'keywords': [
                    'timeout', 'timed out', 'time out', 'timeoutexception',
                    'read timeout', 'connect timeout', 'socket timeout',
                    'connection timeout', '超时', 'socketread', 'sockettimeout'
                ],
                'description': '超时类异常'
            },
            'api_success': {
                'keywords': [
                    'response=success', 'status=200', 'completed successfully',
                    'request processed', 'operation successful', '成功',
                    'response success', '200 ok', 'successfully processed'
                ],
                'level_filter': ['INFO'],
                'description': '正常API请求'
            },
            'ignore': {
                'keywords': [
                    'heartbeat', 'healthcheck', 'ping', 'metrics', 'actuator',
                    'health', 'monitoring', 'probe', 'keepalive', '心跳',
                    'status check', 'alive', 'health-check', 'prometheus'
                ],
                'description': '可忽略的心跳检测'
            },
            'system_error': {
                'keywords': [
                    'nullpointerexception', 'nullpointer', 'npe', '500', 'exception',
                    'error', 'runtimeexception', 'illegalargument', 'outofmemory',
                    'stacktrace', 'caused by', 'java.lang', '系统错误',
                    'internal server error', 'server error'
                ],
                'level_filter': ['ERROR', 'FATAL'],
                'description': '系统级错误'
            },
            'network_error': {
                'keywords': [
                    'connection refused', 'connection reset', 'network', 'socket',
                    'host unreachable', 'connection failed', '连接失败', '网络',
                    'connectexception', 'unknownhost', 'no route to host'
                ],
                'description': '网络连接异常'
            },
            'performance': {
                'keywords': [
                    'slow query', 'performance', 'latency', 'response time',
                    'execution time', 'memory usage', 'cpu', '性能', '缓慢',
                    'gc', 'garbage collection', 'memory leak'
                ],
                'description': '性能相关'
            }
        }
        
        self.ml_model = None
        self.vectorizer = None
        
        # 设置输出目录
        self.output_base_dir = Path(__file__).parent.parent / "log-processing-OUTPUT"
        self.data_output_dir = Path(__file__).parent.parent / "DATA_OUTPUT"
        
    def classify_by_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """使用规则对日志进行分类"""
        print("🔍 开始基于规则的分类...")
        
        # 创建predicted_label列
        df['predicted_label'] = 'other'
        df['confidence'] = 0.0
        df['rule_matched'] = ''
        
        # 预处理文本用于匹配
        df['text_for_matching'] = ''
        for col in ['message', 'classpath', 'level']:
            if col in df.columns:
                df['text_for_matching'] += ' ' + df[col].fillna('').astype(str)
        df['text_for_matching'] = df['text_for_matching'].str.lower()
        
        # 统计各标签命中次数
        label_counts = {}
        
        # 按优先级顺序应用规则（先处理特殊情况）
        priority_order = ['ignore', 'api_success', 'auth_error', 'db_error', 
                         'timeout', 'network_error', 'performance', 'system_error']
        
        for label in priority_order:
            if label not in self.label_rules:
                continue
                
            rules = self.label_rules[label]
            keywords = rules['keywords']
            level_filter = rules.get('level_filter', [])
            
            # 创建关键词匹配条件
            keyword_pattern = '|'.join(re.escape(kw) for kw in keywords)
            mask = df['text_for_matching'].str.contains(keyword_pattern, regex=True, na=False)
            
            # 如果有级别过滤，添加级别条件
            if level_filter and 'level' in df.columns:
                level_mask = df['level'].isin(level_filter)
                mask = mask & level_mask
            
            # 只更新还未分类的记录
            update_mask = mask & (df['predicted_label'] == 'other')
            df.loc[update_mask, 'predicted_label'] = label
            df.loc[update_mask, 'confidence'] = 0.8  # 规则匹配给较高置信度
            df.loc[update_mask, 'rule_matched'] = f"keywords: {', '.join(keywords[:3])}"
            
            count = update_mask.sum()
            label_counts[label] = count
            if count > 0:
                print(f"  ✅ {label}: {count} 条")
        
        # 统计结果
        other_count = (df['predicted_label'] == 'other').sum()
        label_counts['other'] = other_count
        
        print(f"\n📊 规则分类结果:")
        for label, count in label_counts.items():
            percentage = count / len(df) * 100
            print(f"  {label}: {count} 条 ({percentage:.1f}%)")
        
        return df
    
    def train_ml_model(self, train_df: pd.DataFrame, text_column: str = 'message') -> bool:
        """训练机器学习模型"""
        if not ML_AVAILABLE:
            print("❌ scikit-learn未安装，无法使用机器学习功能")
            return False
            
        if 'label' not in train_df.columns:
            print("❌ 训练数据必须包含'label'列")
            return False
        
        print("🤖 开始训练机器学习模型...")
        
        # 准备训练数据
        texts = train_df[text_column].fillna('').astype(str)
        labels = train_df['label']
        
        # 过滤掉'other'标签（数量可能太多）
        mask = labels != 'other'
        texts = texts[mask]
        labels = labels[mask]
        
        if len(texts) < 10:
            print("❌ 训练数据太少，无法训练模型")
            return False
        
        # 分割训练和测试数据
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                texts, labels, test_size=0.2, random_state=42, stratify=labels
            )
        except ValueError:
            print("⚠️  某些类别样本太少，无法分割数据，使用全部数据训练")
            X_train, y_train = texts, labels
            X_test, y_test = texts[:min(10, len(texts))], labels[:min(10, len(labels))]
        
        # 创建管道
        self.ml_model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))),
            ('classifier', MultinomialNB(alpha=0.1))
        ])
        
        # 训练模型
        self.ml_model.fit(X_train, y_train)
        
        # 评估模型
        y_pred = self.ml_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"  ✅ 模型训练完成，准确率: {accuracy:.3f}")
        print(f"  📊 训练样本: {len(X_train)} 条，测试样本: {len(X_test)} 条")
        
        return True
    
    def predict_with_ml(self, df: pd.DataFrame, text_column: str = 'message') -> pd.DataFrame:
        """使用机器学习模型预测"""
        if not self.ml_model:
            print("❌ 模型未训练，无法进行ML预测")
            return df
        
        print("🤖 使用机器学习模型预测...")
        
        # 只对规则未分类的记录使用ML
        other_mask = df['predicted_label'] == 'other'
        other_texts = df.loc[other_mask, text_column].fillna('').astype(str)
        
        if len(other_texts) == 0:
            print("  ℹ️  所有记录已通过规则分类，无需ML预测")
            return df
        
        # 预测
        ml_predictions = self.ml_model.predict(other_texts)
        ml_probabilities = self.ml_model.predict_proba(other_texts)
        
        # 更新预测结果
        df.loc[other_mask, 'predicted_label'] = ml_predictions
        df.loc[other_mask, 'confidence'] = ml_probabilities.max(axis=1)
        df.loc[other_mask, 'rule_matched'] = 'ML_model'
        
        ml_count = len(ml_predictions)
        print(f"  ✅ ML模型预测了 {ml_count} 条记录")
        
        return df
    
    def generate_label_summary(self, df: pd.DataFrame, file_name: str = "") -> str:
        """生成标签分类摘要"""
        summary = []
        summary.append("=" * 60)
        summary.append("📊 日志自动标签分类摘要")
        summary.append("=" * 60)
        if file_name:
            summary.append(f"文件: {file_name}")
        summary.append(f"总日志数: {len(df)}")
        summary.append(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary.append("")
        
        # 标签分布
        label_counts = df['predicted_label'].value_counts()
        summary.append("标签分布:")
        summary.append("-" * 30)
        for label, count in label_counts.items():
            percentage = count / len(df) * 100
            description = self.label_rules.get(label, {}).get('description', '其他类型')
            summary.append(f"{label:<15} {count:>6} 条 ({percentage:>5.1f}%) - {description}")
        
        summary.append("")
        
        # 置信度统计
        summary.append("置信度统计:")
        summary.append("-" * 30)
        high_conf = (df['confidence'] >= 0.7).sum()
        medium_conf = ((df['confidence'] >= 0.4) & (df['confidence'] < 0.7)).sum()
        low_conf = (df['confidence'] < 0.4).sum()
        
        summary.append(f"高置信度 (≥0.7): {high_conf} 条 ({high_conf/len(df)*100:.1f}%)")
        summary.append(f"中置信度 (0.4-0.7): {medium_conf} 条 ({medium_conf/len(df)*100:.1f}%)")
        summary.append(f"低置信度 (<0.4): {low_conf} 条 ({low_conf/len(df)*100:.1f}%)")
        
        summary.append("")
        summary.append("💡 建议:")
        summary.append("- 重点检查低置信度的记录")
        summary.append("- 'other'类型可能需要补充分类规则")
        summary.append("- 人工校正后可用于训练更好的模型")
        
        return "\n".join(summary)
    
    def process_single_file(self, input_file: str, output_dir: Path, 
                           use_ml: bool = False, train_data_file: str = None) -> bool:
        """处理单个文件"""
        try:
            print(f"\n📁 处理文件: {Path(input_file).name}")
            
            # 读取数据
            df = pd.read_csv(input_file, encoding='utf-8-sig')
            print(f"📊 加载了 {len(df)} 条日志记录")
            
            if len(df) == 0:
                print("⚠️  文件为空，跳过处理")
                return False
            
            # 如果有训练数据，先训练ML模型
            if use_ml and train_data_file and Path(train_data_file).exists():
                try:
                    train_df = pd.read_csv(train_data_file, encoding='utf-8-sig')
                    print(f"📚 加载训练数据: {len(train_df)} 条")
                    self.train_ml_model(train_df)
                except Exception as e:
                    print(f"⚠️  加载训练数据失败，仅使用规则分类: {e}")
            
            # 规则分类
            df = self.classify_by_rules(df)
            
            # ML分类（如果可用）
            if use_ml and self.ml_model:
                df = self.predict_with_ml(df)
            
            # 生成输出文件名
            input_path = Path(input_file)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = output_dir / f"{input_path.stem}_labeled_{timestamp}.csv"
            
            # 保存结果
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"✅ 结果已保存: {output_file}")
            
            # 生成摘要
            summary = self.generate_label_summary(df, input_path.name)
            
            # 保存摘要文件
            summary_file = output_dir / f"{input_path.stem}_labeled_{timestamp}_summary.txt"
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(summary)
            print(f"📄 摘要已保存: {summary_file}")
            
            return True
            
        except Exception as e:
            print(f"❌ 处理文件失败: {e}")
            return False
    
    def find_csv_files(self) -> List[Path]:
        """查找DATA_OUTPUT目录下的所有CSV文件"""
        csv_files = []
        
        if not self.data_output_dir.exists():
            print(f"❌ DATA_OUTPUT目录不存在: {self.data_output_dir}")
            return csv_files
        
        # 递归查找所有CSV文件
        for csv_file in self.data_output_dir.rglob("*.csv"):
            # 排除已经标注过的文件
            if "_labeled_" not in csv_file.name:
                csv_files.append(csv_file)
        
        return csv_files
    
    def batch_process(self, use_ml: bool = False, train_data_file: str = None):
        """批量处理DATA_OUTPUT目录下的所有CSV文件"""
        print("🚀 启动批量日志标签处理...")
        print(f"📁 扫描目录: {self.data_output_dir}")
        
        # 查找CSV文件
        csv_files = self.find_csv_files()
        
        if not csv_files:
            print("❌ 未找到任何可处理的CSV文件")
            return
        
        print(f"📊 找到 {len(csv_files)} 个CSV文件待处理")
        
        # 创建输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self.output_base_dir / f"batch_labeled_{timestamp}"
        output_dir.mkdir(exist_ok=True, parents=True)
        print(f"📁 输出目录: {output_dir}")
        
        # 处理每个文件
        success_count = 0
        total_logs = 0
        
        for i, csv_file in enumerate(csv_files, 1):
            print(f"\n{'='*50}")
            print(f"处理进度: {i}/{len(csv_files)}")
            
            if self.process_single_file(str(csv_file), output_dir, use_ml, train_data_file):
                success_count += 1
                # 简单统计日志数量
                try:
                    df = pd.read_csv(csv_file, encoding='utf-8-sig')
                    total_logs += len(df)
                except:
                    pass
        
        # 生成总体摘要
        self.generate_batch_summary(output_dir, success_count, len(csv_files), total_logs)
        
        print(f"\n{'='*60}")
        print(f"🎉 批量处理完成！")
        print(f"📊 成功处理: {success_count}/{len(csv_files)} 个文件")
        print(f"📊 总日志数: {total_logs} 条")
        print(f"📁 结果保存在: {output_dir}")
    
    def generate_batch_summary(self, output_dir: Path, success_count: int, 
                              total_count: int, total_logs: int):
        """生成批量处理摘要"""
        summary = []
        summary.append("=" * 60)
        summary.append("📊 批量日志标签处理摘要")
        summary.append("=" * 60)
        summary.append(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary.append(f"扫描目录: {self.data_output_dir}")
        summary.append(f"输出目录: {output_dir}")
        summary.append(f"成功处理: {success_count}/{total_count} 个文件")
        summary.append(f"总日志数: {total_logs} 条")
        summary.append("")
        
        # 列出处理的文件
        labeled_files = list(output_dir.glob("*_labeled_*.csv"))
        summary.append("处理结果文件:")
        summary.append("-" * 30)
        for labeled_file in labeled_files:
            summary.append(f"- {labeled_file.name}")
        
        summary.append("")
        summary.append("💡 后续步骤:")
        summary.append("1. 检查各个 *_labeled_*.csv 文件")
        summary.append("2. 人工校正错误的标签")
        summary.append("3. 使用校正后的数据进行ML训练")
        summary.append("4. 重复处理以提高准确率")
        
        # 保存摘要
        summary_file = output_dir / "batch_processing_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(summary))
        
        print(f"📋 批量处理摘要: {summary_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='半自动日志标签辅助器')
    parser.add_argument('input_file', nargs='?', help='输入CSV文件路径（可选，不提供则批量处理）')
    parser.add_argument('--use-ml', action='store_true', help='使用机器学习模型')
    parser.add_argument('--train-data', help='训练数据文件路径（用于ML模型）')
    parser.add_argument('--batch', action='store_true', help='强制批量处理模式')
    
    args = parser.parse_args()
    
    # 创建标签器
    labeler = LogAutoLabeler()
    
    # 确定运行模式
    if args.input_file and not args.batch:
        # 单文件模式
        if not Path(args.input_file).exists():
            print(f"❌ 输入文件不存在: {args.input_file}")
            return
        
        print("🚀 启动单文件日志标签器...")
        
        # 创建输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = labeler.output_base_dir / f"single_labeled_{timestamp}"
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # 处理文件
        success = labeler.process_single_file(
            args.input_file, output_dir, args.use_ml, args.train_data
        )
        
        if success:
            print("\n🎉 处理完成！")
            print(f"📁 结果保存在: {output_dir}")
    else:
        # 批量处理模式
        labeler.batch_process(args.use_ml, args.train_data)
    
    print("\n💡 接下来的步骤:")
    print("1. 检查生成的 *_labeled_*.csv 文件")
    print("2. 在Excel中人工校正错误标签")
    print("3. 将校正后的数据作为训练数据改进模型")
    print("4. 重复此过程以持续优化标签质量")


if __name__ == "__main__":
    main() 