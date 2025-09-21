#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强的特征工程工具
实现结构特征+语义特征的双重特征提取
支持模板ID embedding、异常关键字embedding、TF-IDF等

使用方法:
python feature_engineer.py --input-file templated_logs.csv --output-dir output/
python feature_engineer.py --batch --input-dir templated_logs/ --output-dir output/
"""

import pandas as pd
import numpy as np
import re
import json
import pickle
import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# 尝试导入机器学习库
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠️  scikit-learn未安装，将仅使用基础特征工程")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️  LightGBM未安装，将使用其他分类器")

class FeatureEngineer:
    """特征工程器"""
    
    def __init__(self):
        # 结构特征定义
        self.structural_features = {
            'log_level': ['ERROR', 'WARN', 'INFO', 'DEBUG', 'TRACE', 'FATAL'],
            'contains_stack': [True, False],
            'exception_type': [],  # 动态填充
            'file_path': [],  # 动态填充
            'function_name': [],  # 动态填充
        }
        
        # 语义特征配置
        self.semantic_config = {
            'tfidf_max_features': 1000,
            'tfidf_ngram_range': (1, 2),
            'template_id_buckets': 100,
            'exception_keyword_buckets': 50,
        }
        
        # 特征存储
        self.feature_encoders = {}
        self.feature_stats = {}
        self.template_id_mapping = {}
        self.exception_keyword_mapping = {}
        
        # 输出目录配置
        self.output_base_dir = Path(__file__).parent.parent / "log-processing-OUTPUT"
    
    def extract_structural_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """提取结构特征"""
        print("🔧 提取结构特征...")
        
        # 1. 日志级别特征
        df['log_level'] = df['original_log'].apply(self._extract_log_level)
        
        # 2. 堆栈跟踪特征
        df['contains_stack'] = df['original_log'].apply(self._has_stack_trace)
        
        # 3. 异常类型特征
        df['exception_type'] = df['original_log'].apply(self._extract_exception_type)
        
        # 4. 文件路径特征
        df['file_path'] = df['original_log'].apply(self._extract_file_path)
        
        # 5. 函数名特征
        df['function_name'] = df['original_log'].apply(self._extract_function_name)
        
        # 6. 行号特征
        df['line_number'] = df['original_log'].apply(self._extract_line_number)
        
        # 7. 日志长度特征
        df['log_length'] = df['original_log'].str.len()
        df['cleaned_length'] = df['cleaned_log'].str.len()
        df['compression_ratio'] = df['cleaned_length'] / (df['log_length'] + 1e-6)
        
        # 8. 特殊字符特征
        df['has_quotes'] = df['original_log'].str.contains(r'["\']')
        df['has_brackets'] = df['original_log'].str.contains(r'[\[\]{}()]')
        df['has_numbers'] = df['original_log'].str.contains(r'\d')
        df['has_urls'] = df['original_log'].str.contains(r'https?://')
        df['has_emails'] = df['original_log'].str.contains(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        
        print(f"  ✅ 提取了 {len([col for col in df.columns if col.startswith(('log_', 'contains_', 'has_', 'exception_', 'file_', 'function_', 'line_', 'compression_'))])} 个结构特征")
        
        return df
    
    def extract_semantic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """提取语义特征"""
        print("🧠 提取语义特征...")
        
        # 1. 模板ID特征
        df = self._create_template_id_features(df)
        
        # 2. 异常关键字特征
        df = self._create_exception_keyword_features(df)
        
        # 3. TF-IDF特征
        if ML_AVAILABLE:
            df = self._create_tfidf_features(df)
        
        # 4. 文本统计特征
        df = self._create_text_statistics_features(df)
        
        print(f"  ✅ 提取了语义特征")
        
        return df
    
    def _extract_log_level(self, log_line: str) -> str:
        """提取日志级别"""
        log_upper = log_line.upper()
        for level in ['ERROR', 'FATAL', 'WARN', 'INFO', 'DEBUG', 'TRACE']:
            if level in log_upper:
                return level
        return 'UNKNOWN'
    
    def _has_stack_trace(self, log_line: str) -> bool:
        """检查是否包含堆栈跟踪"""
        stack_indicators = ['at ', 'caused by', 'stack trace', 'exception in thread']
        return any(indicator in log_line.lower() for indicator in stack_indicators)
    
    def _extract_exception_type(self, log_line: str) -> str:
        """提取异常类型"""
        # 匹配Java异常
        java_exception = re.search(r'(\w+Exception|\w+Error):', log_line)
        if java_exception:
            return java_exception.group(1)
        
        # 匹配其他异常模式
        exception_patterns = [
            r'(\w+Exception)',
            r'(\w+Error)',
            r'Exception:\s*(\w+)',
            r'Error:\s*(\w+)'
        ]
        
        for pattern in exception_patterns:
            match = re.search(pattern, log_line)
            if match:
                return match.group(1)
        
        return 'None'
    
    def _extract_file_path(self, log_line: str) -> str:
        """提取文件路径"""
        # 匹配Java文件路径
        java_path = re.search(r'at\s+([a-zA-Z0-9_.]+\.java)', log_line)
        if java_path:
            return java_path.group(1)
        
        # 匹配其他文件路径
        path_patterns = [
            r'([a-zA-Z0-9_.]+\.(java|py|js|ts|go|rs|cpp|c|h))',
            r'([a-zA-Z]:\\[^:]+)',
            r'(/[^:]+)'
        ]
        
        for pattern in path_patterns:
            match = re.search(pattern, log_line)
            if match:
                return match.group(1)
        
        return 'None'
    
    def _extract_function_name(self, log_line: str) -> str:
        """提取函数名"""
        # 匹配Java方法调用
        java_method = re.search(r'at\s+[a-zA-Z0-9_.]+\.([a-zA-Z0-9_]+)', log_line)
        if java_method:
            return java_method.group(1)
        
        # 匹配其他函数模式
        function_patterns = [
            r'([a-zA-Z0-9_]+)\s*\(',
            r'function\s+([a-zA-Z0-9_]+)',
            r'method\s+([a-zA-Z0-9_]+)'
        ]
        
        for pattern in function_patterns:
            match = re.search(pattern, log_line)
            if match:
                return match.group(1)
        
        return 'None'
    
    def _extract_line_number(self, log_line: str) -> int:
        """提取行号"""
        line_patterns = [
            r':(\d+)\)',
            r'line\s+(\d+)',
            r'at\s+line\s+(\d+)'
        ]
        
        for pattern in line_patterns:
            match = re.search(pattern, log_line)
            if match:
                return int(match.group(1))
        
        return 0
    
    def _create_template_id_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建模板ID特征"""
        # 模板ID one-hot编码
        if 'template_id' in df.columns:
            # 创建模板ID映射
            unique_templates = df['template_id'].unique()
            self.template_id_mapping = {template: i for i, template in enumerate(unique_templates)}
            
            # 创建模板ID特征
            df['template_id_encoded'] = df['template_id'].map(self.template_id_mapping)
            
            # 模板频率特征
            template_counts = df['template_id'].value_counts()
            df['template_frequency'] = df['template_id'].map(template_counts)
            df['template_frequency_log'] = np.log1p(df['template_frequency'])
        
        return df
    
    def _create_exception_keyword_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建异常关键字特征"""
        if 'exception_keywords' in df.columns:
            # 解析异常关键字
            all_keywords = set()
            for keywords_str in df['exception_keywords'].fillna('[]'):
                try:
                    if isinstance(keywords_str, str):
                        keywords = eval(keywords_str) if keywords_str.startswith('[') else [keywords_str]
                    else:
                        keywords = keywords_str
                    all_keywords.update(keywords)
                except:
                    continue
            
            # 创建异常关键字映射
            self.exception_keyword_mapping = {kw: i for i, kw in enumerate(all_keywords)}
            
            # 创建异常关键字特征
            df['exception_count'] = df['exception_keywords'].apply(self._count_exceptions)
            df['has_exception'] = df['exception_count'] > 0
            
            # 为每个异常关键字创建二进制特征
            for keyword in list(all_keywords)[:20]:  # 限制前20个最常见的
                df[f'exception_{keyword.replace(":", "_")}'] = df['exception_keywords'].apply(
                    lambda x: self._has_exception_keyword(x, keyword)
                )
        
        return df
    
    def _create_tfidf_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建TF-IDF特征"""
        if not ML_AVAILABLE:
            return df
        
        print("  📊 创建TF-IDF特征...")
        
        # 使用清理后的日志进行TF-IDF
        texts = df['cleaned_log'].fillna('').astype(str)
        
        # 创建TF-IDF向量化器
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.semantic_config['tfidf_max_features'],
            ngram_range=self.semantic_config['tfidf_ngram_range'],
            stop_words='english',
            min_df=2,
            max_df=0.95
        )
        
        # 拟合和转换
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        
        # 将TF-IDF特征添加到DataFrame
        feature_names = [f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names, index=df.index)
        
        # 合并特征
        df = pd.concat([df, tfidf_df], axis=1)
        
        print(f"  ✅ 创建了 {tfidf_matrix.shape[1]} 个TF-IDF特征")
        
        return df
    
    def _create_text_statistics_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建文本统计特征"""
        # 词数统计
        df['word_count'] = df['cleaned_log'].str.split().str.len()
        df['char_count'] = df['cleaned_log'].str.len()
        df['avg_word_length'] = df['char_count'] / (df['word_count'] + 1e-6)
        
        # 特殊字符统计
        df['digit_count'] = df['cleaned_log'].str.count(r'\d')
        df['uppercase_count'] = df['cleaned_log'].str.count(r'[A-Z]')
        df['lowercase_count'] = df['cleaned_log'].str.count(r'[a-z]')
        df['special_char_count'] = df['cleaned_log'].str.count(r'[^a-zA-Z0-9\s]')
        
        # 标点符号统计
        df['comma_count'] = df['cleaned_log'].str.count(',')
        df['period_count'] = df['cleaned_log'].str.count('\.')
        df['colon_count'] = df['cleaned_log'].str.count(':')
        df['semicolon_count'] = df['cleaned_log'].str.count(';')
        
        return df
    
    def _count_exceptions(self, keywords_str) -> int:
        """计算异常关键字数量"""
        try:
            if isinstance(keywords_str, str):
                keywords = eval(keywords_str) if keywords_str.startswith('[') else [keywords_str]
            else:
                keywords = keywords_str
            return len(keywords) if keywords else 0
        except:
            return 0
    
    def _has_exception_keyword(self, keywords_str, keyword) -> bool:
        """检查是否包含特定异常关键字"""
        try:
            if isinstance(keywords_str, str):
                keywords = eval(keywords_str) if keywords_str.startswith('[') else [keywords_str]
            else:
                keywords = keywords_str
            return keyword in keywords if keywords else False
        except:
            return False
    
    def create_feature_combinations(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建特征组合"""
        print("🔗 创建特征组合...")
        
        # 1. 日志级别 + 堆栈跟踪
        df['error_with_stack'] = (df['log_level'] == 'ERROR') & df['contains_stack']
        df['warn_with_stack'] = (df['log_level'] == 'WARN') & df['contains_stack']
        
        # 2. 模板ID + TF-IDF (如果有的话)
        if 'template_id_encoded' in df.columns and any(col.startswith('tfidf_') for col in df.columns):
            # 选择最重要的TF-IDF特征进行组合
            tfidf_cols = [col for col in df.columns if col.startswith('tfidf_')]
            if tfidf_cols:
                top_tfidf = df[tfidf_cols].sum(axis=1)
                df['template_tfidf_interaction'] = df['template_id_encoded'] * top_tfidf
        
        # 3. 异常类型 + 函数名
        df['exception_function'] = df['exception_type'] + '_' + df['function_name']
        
        # 4. 日志长度 + 压缩比
        df['length_compression'] = df['log_length'] * df['compression_ratio']
        
        print(f"  ✅ 创建了特征组合")
        
        return df
    
    def prepare_ml_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """准备机器学习特征"""
        print("🤖 准备机器学习特征...")
        
        # 选择数值特征
        numeric_features = []
        categorical_features = []
        
        # 结构特征
        structural_cols = ['log_length', 'cleaned_length', 'compression_ratio', 'word_count', 
                          'char_count', 'avg_word_length', 'digit_count', 'uppercase_count',
                          'lowercase_count', 'special_char_count', 'comma_count', 'period_count',
                          'colon_count', 'semicolon_count', 'line_number']
        
        for col in structural_cols:
            if col in df.columns:
                numeric_features.append(col)
        
        # 布尔特征
        boolean_cols = ['contains_stack', 'has_quotes', 'has_brackets', 'has_numbers', 
                       'has_urls', 'has_emails', 'has_exception', 'error_with_stack', 'warn_with_stack']
        
        for col in boolean_cols:
            if col in df.columns:
                numeric_features.append(col)
        
        # 分类特征
        categorical_cols = ['log_level', 'exception_type', 'file_path', 'function_name']
        
        for col in categorical_cols:
            if col in df.columns:
                categorical_features.append(col)
        
        # TF-IDF特征
        tfidf_features = [col for col in df.columns if col.startswith('tfidf_')]
        numeric_features.extend(tfidf_features)
        
        # 异常关键字特征
        exception_features = [col for col in df.columns if col.startswith('exception_') and col != 'exception_count']
        numeric_features.extend(exception_features)
        
        # 模板特征
        template_features = ['template_id_encoded', 'template_frequency', 'template_frequency_log']
        for col in template_features:
            if col in df.columns:
                numeric_features.append(col)
        
        # 创建特征矩阵
        feature_cols = numeric_features + categorical_features
        available_cols = [col for col in feature_cols if col in df.columns]
        
        print(f"  ✅ 选择了 {len(available_cols)} 个特征用于机器学习")
        print(f"  📊 数值特征: {len([col for col in available_cols if col in numeric_features])}")
        print(f"  📊 分类特征: {len([col for col in available_cols if col in categorical_features])}")
        
        return df[available_cols], available_cols
    
    def train_classifier(self, df: pd.DataFrame, target_column: str = 'content_type') -> Dict:
        """训练分类器"""
        if not ML_AVAILABLE:
            print("❌ scikit-learn未安装，无法训练分类器")
            return {}
        
        print("🎯 训练分类器...")
        
        # 准备特征
        X, feature_names = self.prepare_ml_features(df)
        y = df[target_column]
        
        # 过滤掉'other'类别
        mask = y != 'other'
        X = X[mask]
        y = y[mask]
        
        if len(X) < 10:
            print("❌ 训练数据太少，无法训练分类器")
            return {}
        
        # 处理分类特征
        categorical_features = [col for col in feature_names if col in ['log_level', 'exception_type', 'file_path', 'function_name']]
        
        # 编码分类特征
        for col in categorical_features:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.feature_encoders[f'{col}_encoder'] = le
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 训练模型
        if LIGHTGBM_AVAILABLE:
            # 使用LightGBM
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
            
            params = {
                'objective': 'multiclass',
                'num_class': len(y.unique()),
                'metric': 'multi_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1
            }
            
            model = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                num_boost_round=100,
                callbacks=[lgb.early_stopping(10)]
            )
            
            # 预测
            y_pred = model.predict(X_test)
            y_pred_labels = [list(y.unique())[i] for i in np.argmax(y_pred, axis=1)]
            
        else:
            # 使用朴素贝叶斯
            from sklearn.naive_bayes import MultinomialNB
            from sklearn.pipeline import Pipeline
            
            model = Pipeline([
                ('classifier', MultinomialNB(alpha=0.1))
            ])
            
            model.fit(X_train, y_train)
            y_pred_labels = model.predict(X_test)
        
        # 评估
        from sklearn.metrics import accuracy_score, classification_report
        
        accuracy = accuracy_score(y_test, y_pred_labels)
        report = classification_report(y_test, y_pred_labels, output_dict=True)
        
        print(f"  ✅ 模型训练完成，准确率: {accuracy:.3f}")
        
        return {
            'model': model,
            'feature_names': feature_names,
            'accuracy': accuracy,
            'classification_report': report,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred_labels
        }
    
    def process_file(self, input_file: str, output_dir: Path) -> Dict:
        """处理单个文件"""
        print(f"🔄 处理文件: {Path(input_file).name}")
        
        try:
            # 读取数据
            df = pd.read_csv(input_file, encoding='utf-8-sig')
            print(f"📊 加载了 {len(df)} 条记录")
            
            # 提取结构特征
            df = self.extract_structural_features(df)
            
            # 提取语义特征
            df = self.extract_semantic_features(df)
            
            # 创建特征组合
            df = self.create_feature_combinations(df)
            
            # 训练分类器（如果有标签列）
            model_results = {}
            if 'content_type' in df.columns:
                model_results = self.train_classifier(df)
            
            # 保存结果
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            input_name = Path(input_file).stem
            
            # 保存特征数据
            output_file = output_dir / f"{input_name}_features_{timestamp}.csv"
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            
            # 保存模型和编码器
            if model_results:
                model_file = output_dir / f"{input_name}_model_{timestamp}.pkl"
                with open(model_file, 'wb') as f:
                    pickle.dump({
                        'model': model_results['model'],
                        'feature_encoders': self.feature_encoders,
                        'template_id_mapping': self.template_id_mapping,
                        'exception_keyword_mapping': self.exception_keyword_mapping,
                        'feature_names': model_results['feature_names'],
                        'tfidf_vectorizer': getattr(self, 'tfidf_vectorizer', None)
                    }, f)
                
                # 保存模型评估报告
                report_file = output_dir / f"{input_name}_model_report_{timestamp}.json"
                with open(report_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'accuracy': model_results['accuracy'],
                        'classification_report': model_results['classification_report'],
                        'feature_count': len(model_results['feature_names']),
                        'training_time': datetime.now().isoformat()
                    }, f, ensure_ascii=False, indent=2)
            
            # 生成特征统计报告
            self.generate_feature_report(df, output_dir, input_name, timestamp)
            
            print(f"✅ 处理完成: {output_file}")
            
            return {
                'input_file': input_file,
                'output_file': str(output_file),
                'total_records': len(df),
                'feature_count': len(df.columns),
                'model_results': model_results
            }
            
        except Exception as e:
            print(f"❌ 处理文件失败: {e}")
            return {}
    
    def generate_feature_report(self, df: pd.DataFrame, output_dir: Path, input_name: str, timestamp: str):
        """生成特征统计报告"""
        report_file = output_dir / f"{input_name}_feature_report_{timestamp}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("特征工程统计报告\n")
            f.write("=" * 50 + "\n")
            f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"输入文件: {input_name}\n")
            f.write(f"总记录数: {len(df)}\n")
            f.write(f"特征数量: {len(df.columns)}\n\n")
            
            # 特征类型统计
            f.write("特征类型统计:\n")
            f.write("-" * 30 + "\n")
            
            structural_features = [col for col in df.columns if any(col.startswith(prefix) for prefix in 
                ['log_', 'contains_', 'has_', 'exception_', 'file_', 'function_', 'line_', 'compression_'])]
            
            semantic_features = [col for col in df.columns if col.startswith('tfidf_')]
            
            template_features = [col for col in df.columns if 'template' in col]
            
            f.write(f"结构特征: {len(structural_features)}\n")
            f.write(f"语义特征 (TF-IDF): {len(semantic_features)}\n")
            f.write(f"模板特征: {len(template_features)}\n")
            f.write(f"其他特征: {len(df.columns) - len(structural_features) - len(semantic_features) - len(template_features)}\n\n")
            
            # 特征重要性（如果有模型）
            if hasattr(self, 'feature_importance'):
                f.write("特征重要性 (Top 20):\n")
                f.write("-" * 30 + "\n")
                for feature, importance in self.feature_importance[:20]:
                    f.write(f"{feature}: {importance:.4f}\n")
        
        print(f"📄 特征报告: {report_file}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='特征工程工具')
    parser.add_argument('--input-file', help='输入文件路径')
    parser.add_argument('--input-dir', help='输入目录路径（批量模式）')
    parser.add_argument('--output-dir', help='输出目录路径')
    parser.add_argument('--batch', action='store_true', help='批量处理模式')
    
    args = parser.parse_args()
    
    # 创建特征工程器
    engineer = FeatureEngineer()
    
    # 设置输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = engineer.output_base_dir / "feature_engineered"
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    if args.batch or args.input_dir:
        # 批量处理模式
        if not args.input_dir:
            print("❌ 批量模式需要指定 --input-dir")
            return
        
        input_path = Path(args.input_dir)
        if not input_path.exists():
            print(f"❌ 输入目录不存在: {args.input_dir}")
            return
        
        # 查找CSV文件
        csv_files = list(input_path.rglob("*.csv"))
        if not csv_files:
            print("❌ 未找到CSV文件")
            return
        
        print(f"📁 找到 {len(csv_files)} 个CSV文件")
        
        # 创建批量输出目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        batch_output_dir = output_dir / f"batch_features_{timestamp}"
        batch_output_dir.mkdir(exist_ok=True, parents=True)
        
        # 处理每个文件
        results = []
        for i, csv_file in enumerate(csv_files, 1):
            print(f"\n{'='*50}")
            print(f"处理进度: {i}/{len(csv_files)}")
            
            result = engineer.process_file(str(csv_file), batch_output_dir)
            if result:
                results.append(result)
        
        print(f"\n🎉 批量处理完成！")
        print(f"📁 结果保存在: {batch_output_dir}")
        print(f"📊 成功处理: {len(results)}/{len(csv_files)} 个文件")
    
    elif args.input_file:
        # 单文件模式
        if not Path(args.input_file).exists():
            print(f"❌ 输入文件不存在: {args.input_file}")
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        single_output_dir = output_dir / f"single_features_{timestamp}"
        single_output_dir.mkdir(exist_ok=True, parents=True)
        
        result = engineer.process_file(args.input_file, single_output_dir)
        if result:
            print(f"\n🎉 处理完成！")
            print(f"📁 结果保存在: {single_output_dir}")
    
    else:
        print("❌ 请指定 --input-file 或使用 --batch --input-dir 进行批量处理")
        parser.print_help()

if __name__ == "__main__":
    main()
