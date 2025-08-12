# -*- coding: utf-8 -*-
"""
改进的数据处理脚本
统一标签体系并提升数据质量
"""

import pandas as pd
import re
import logging
from pathlib import Path
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImprovedDataProcessor:
    """改进的数据处理器"""
    
    def __init__(self):
        # 定义训练时的9个标准类别
        self.standard_categories = [
            'database_exception',
            'business_logic', 
            'connection_issue',
            'stack_exception',
            'auth_authorization',
            'config_environment',
            'normal_operation',
            'memory_performance',
            'monitoring_heartbeat'
        ]
        
        # 标签映射规则
        self.label_mapping = {
            # 直接映射
            'stack_exception': 'stack_exception',
            'connection_issue': 'connection_issue',
            'normal': 'normal_operation',
            
            # 数据库相关
            'db_error': 'database_exception',
            
            # 认证授权相关
            'auth_error': 'auth_authorization',
            
            # 配置环境相关
            'config': 'config_environment',
            'startup_failure': 'config_environment',
            
            # 性能相关
            'performance': 'memory_performance',
            'timeout': 'memory_performance',
            
            # 业务逻辑相关
            'business': 'business_logic',
            
            # unknown需要根据内容重新分类
            'unknown': None
        }
        
        # 关键词模式用于重新分类unknown标签
        self.category_keywords = {
            'database_exception': [
                'sql', 'database', 'connection', 'jdbc', 'hibernate', 'mybatis',
                'mysql', 'postgresql', 'oracle', 'mongodb', 'redis',
                'transaction', 'deadlock', 'timeout', 'connection pool',
                'datasource', 'jpa', 'entity', 'repository'
            ],
            'business_logic': [
                'business', 'validation', 'rule', 'logic', 'service',
                'controller', 'request', 'response', 'api', 'rest',
                'parameter', 'argument', 'invalid', 'illegal'
            ],
            'connection_issue': [
                'connection', 'network', 'socket', 'http', 'https',
                'timeout', 'refused', 'unreachable', 'dns', 'proxy',
                'gateway', 'load balancer', 'service discovery'
            ],
            'stack_exception': [
                'exception', 'error', 'stack trace', 'caused by',
                'nullpointer', 'illegalargument', 'runtimeexception',
                'classnotfound', 'nosuchmethod', 'nosuchfield'
            ],
            'auth_authorization': [
                'authentication', 'authorization', 'security', 'token',
                'jwt', 'oauth', 'permission', 'access denied', 'unauthorized',
                'login', 'password', 'credential', 'role', 'privilege'
            ],
            'config_environment': [
                'configuration', 'property', 'environment', 'profile',
                'application.yml', 'application.properties', 'bootstrap',
                'spring.config', 'server.port', 'database.url'
            ],
            'normal_operation': [
                'info', 'debug', 'started', 'running', 'success',
                'completed', 'finished', 'initialized', 'ready'
            ],
            'memory_performance': [
                'memory', 'heap', 'gc', 'performance', 'slow',
                'timeout', 'outofmemory', 'leak', 'cpu', 'thread'
            ],
            'monitoring_heartbeat': [
                'health', 'monitor', 'heartbeat', 'status', 'alive',
                'check', 'probe', 'metrics', 'actuator'
            ]
        }
    
    def clean_text(self, text):
        """清洗文本内容"""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # 移除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        
        # 移除URL
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # 移除代码块标记
        text = re.sub(r'```[\s\S]*?```', '', text)
        
        # 移除行内代码标记
        text = re.sub(r'`[^`]+`', '', text)
        
        # 移除特殊字符但保留基本的标点
        text = re.sub(r'[^\w\s\.\,\!\?\:\;\-\(\)\[\]\{\}]', '', text)
        
        # 移除多余空白
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def is_valid_log(self, text):
        """判断是否为有效的日志内容"""
        if not text or len(text.strip()) < 10:
            return False
        
        # 检查是否包含日志特征
        log_indicators = [
            'exception', 'error', 'warn', 'info', 'debug',
            'trace', 'stack', 'caused by', 'at ',
            'java.lang.', 'org.springframework.',
            'failed', 'cannot', 'unable', 'null'
        ]
        
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in log_indicators)
    
    def classify_unknown(self, text):
        """对unknown标签进行重新分类"""
        if not text:
            return 'normal_operation'
        
        text_lower = text.lower()
        scores = {}
        
        # 计算每个类别的匹配分数
        for category, keywords in self.category_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    score += 1
            scores[category] = score
        
        # 找到得分最高的类别
        if scores:
            best_category = max(scores, key=scores.get)
            if scores[best_category] > 0:
                return best_category
        
        # 如果没有匹配，默认为normal_operation
        return 'normal_operation'
    
    def map_labels(self, df):
        """映射标签到标准类别"""
        logger.info("🏷️ 开始标签映射...")
        
        # 创建新的标签列
        df['mapped_label'] = df['label'].map(self.label_mapping)
        
        # 处理unknown标签
        unknown_mask = df['mapped_label'].isna()
        if unknown_mask.any():
            logger.info(f"🔍 发现 {unknown_mask.sum()} 个unknown标签，开始重新分类...")
            
            for idx in df[unknown_mask].index:
                text = df.loc[idx, 'message']
                new_label = self.classify_unknown(text)
                df.loc[idx, 'mapped_label'] = new_label
        
        # 统计映射结果
        logger.info("📊 标签映射结果:")
        for category in self.standard_categories:
            count = (df['mapped_label'] == category).sum()
            if count > 0:
                logger.info(f"  {category}: {count} 条")
        
        return df
    
    def filter_and_clean(self, df):
        """过滤和清洗数据"""
        logger.info("🧹 开始数据清洗...")
        original_count = len(df)
        
        # 清洗文本
        df['cleaned_text'] = df['message'].apply(self.clean_text)
        
        # 过滤无效数据
        df = df[df['cleaned_text'].apply(self.is_valid_log)]
        
        # 移除重复数据
        df = df.drop_duplicates(subset=['cleaned_text'])
        
        # 移除过短或过长的文本
        df = df[df['cleaned_text'].str.len() >= 20]
        df = df[df['cleaned_text'].str.len() <= 2000]
        
        cleaned_count = len(df)
        logger.info(f"📊 数据清洗完成: {original_count} -> {cleaned_count} 条")
        logger.info(f"🗑️ 过滤掉: {original_count - cleaned_count} 条")
        
        return df
    
    def balance_data(self, df, max_per_class=500):
        """平衡各类别数据量"""
        logger.info(f"⚖️ 开始数据平衡 (每类最多 {max_per_class} 条)...")
        
        balanced_dfs = []
        for category in self.standard_categories:
            category_df = df[df['mapped_label'] == category]
            if len(category_df) > max_per_class:
                category_df = category_df.sample(n=max_per_class, random_state=42)
                logger.info(f"  {category}: 采样 {len(category_df)} 条 (原始 {len(df[df['mapped_label'] == category])} 条)")
            else:
                logger.info(f"  {category}: 使用全部 {len(category_df)} 条")
            balanced_dfs.append(category_df)
        
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        logger.info(f"📊 数据平衡完成: {len(balanced_df)} 条记录")
        
        return balanced_df
    
    def process_data(self, input_file, output_file):
        """处理数据的主函数"""
        logger.info(f"🚀 开始处理数据: {input_file}")
        
        try:
            # 加载数据
            df = pd.read_csv(input_file)
            logger.info(f"📊 原始数据: {len(df)} 条记录")
            
            # 数据清洗
            df = self.filter_and_clean(df)
            
            # 标签映射
            df = self.map_labels(df)
            
            # 数据平衡
            df = self.balance_data(df)
            
            # 准备最终输出
            final_df = df[['cleaned_text', 'mapped_label']].copy()
            final_df.columns = ['text', 'label']
            
            # 保存结果
            final_df.to_csv(output_file, index=False, encoding='utf-8')
            logger.info(f"💾 处理完成，已保存到: {output_file}")
            
            # 显示最终统计
            logger.info("📈 最终类别分布:")
            for category in self.standard_categories:
                count = (final_df['label'] == category).sum()
                percentage = (count / len(final_df)) * 100
                logger.info(f"  {category}: {count} 条 ({percentage:.1f}%)")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 数据处理失败: {e}")
            return False

def main():
    """主函数"""
    processor = ImprovedDataProcessor()
    
    # 处理GitHub抓取的日志数据
    input_file = "../DATA_OUTPUT/issue_logs_combined_20250812_001907.csv"
    output_file = "data/improved_validation_data.csv"
    
    success = processor.process_data(input_file, output_file)
    
    if success:
        print("\n" + "=" * 60)
        print("✅ 数据处理完成！")
        print(f"📁 输出文件: {output_file}")
        print("=" * 60)
    else:
        print("\n❌ 数据处理失败！")

if __name__ == "__main__":
    main()
