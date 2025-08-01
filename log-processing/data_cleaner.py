#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据清洗脚本
功能：
1. 读取DATA_OUTPUT目录中的分类数据文件
2. 去除'other'类别数据
3. 平衡各类别数据量（避免数据偏斜）
4. 生成用于模型训练的数据集
5. 支持多种数据格式和来源

使用方法:
python data_cleaner.py                           # 处理DATA_OUTPUT目录下的所有CSV文件
python data_cleaner.py <输入文件路径>            # 处理指定文件
python data_cleaner.py --max-per-class 500      # 设置每类最多保留条数
python data_cleaner.py --output-dir <输出目录>   # 指定输出目录
"""

import pandas as pd
import numpy as np
import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class DataCleaner:
    """数据清洗器"""
    
    def __init__(self):
        # 定义类别映射（用于统一类别名称）
        self.category_mapping = {
            'stack_exception': 'stack_exception',
            'spring_boot_startup_failure': 'startup_failure',
            'auth_authorization': 'auth_error',
            'database_exception': 'db_error',
            'connection_issue': 'connection_issue',
            'timeout': 'timeout',
            'memory_performance': 'performance',
            'config_environment': 'config',
            'business_logic': 'business',
            'normal_operation': 'normal',
            'monitoring_heartbeat': 'heartbeat',
            'other': 'other'  # 将被过滤
        }
        
        # 类别中文描述
        self.category_descriptions = {
            'stack_exception': '堆栈异常',
            'startup_failure': '启动失败',
            'auth_error': '认证错误',
            'db_error': '数据库错误',
            'connection_issue': '连接问题',
            'timeout': '超时错误',
            'performance': '性能问题',
            'config': '配置问题',
            'business': '业务逻辑',
            'normal': '正常操作',
            'heartbeat': '监控心跳'
        }
    
    def load_data(self, input_file: str) -> pd.DataFrame:
        """加载数据文件"""
        try:
            df = pd.read_csv(input_file)
            print(f"📊 加载数据: {len(df)} 条记录")
            return df
        except Exception as e:
            print(f"❌ 读取文件失败: {e}")
            return pd.DataFrame()
    
    def detect_label_column(self, df: pd.DataFrame) -> str:
        """检测标签列名"""
        possible_labels = ['content_type', 'final_label', 'label', 'category']
        for col in possible_labels:
            if col in df.columns:
                return col
        return None
    
    def clean_data(self, df: pd.DataFrame, max_per_class: int = None) -> pd.DataFrame:
        """清洗数据"""
        if df.empty:
            return df
        
        # 检测标签列
        label_column = self.detect_label_column(df)
        if not label_column:
            print("❌ 未找到标签列")
            return pd.DataFrame()
        
        print(f"🔍 使用标签列: {label_column}")
        
        # 过滤掉'other'类别
        original_count = len(df)
        df_cleaned = df[df[label_column] != 'other'].copy()
        filtered_count = len(df_cleaned)
        
        print(f"🔍 过滤后数据: {filtered_count} 条记录 (移除了 {original_count - filtered_count} 条'other'记录)")
        
        # 统计各类别数量
        category_counts = df_cleaned[label_column].value_counts()
        print("\n📈 类别分布:")
        for category, count in category_counts.items():
            percentage = (count / filtered_count) * 100
            print(f"  {category}: {count} 条 ({percentage:.1f}%)")
        
        # 平衡数据（如果指定了max_per_class）
        if max_per_class and max_per_class > 0:
            df_balanced = df_cleaned.groupby(label_column).head(max_per_class).copy()
            balanced_count = len(df_balanced)
            
            print(f"\n⚖️ 平衡后数据: {balanced_count} 条记录 (每类最多 {max_per_class} 条)")
            
            # 重新统计
            balanced_counts = df_balanced[label_column].value_counts()
            print("\n📈 平衡后类别分布:")
            for category, count in balanced_counts.items():
                percentage = (count / balanced_count) * 100
                print(f"  {category}: {count} 条 ({percentage:.1f}%)")
            
            return df_balanced
        
        return df_cleaned
    
    def prepare_training_data(self, df: pd.DataFrame, output_file: str) -> bool:
        """准备训练数据"""
        try:
            # 检测必要的列
            label_column = self.detect_label_column(df)
            text_column = None
            
            # 检测文本列
            possible_text_columns = ['original_log', 'message', 'content', 'text']
            for col in possible_text_columns:
                if col in df.columns:
                    text_column = col
                    break
            
            if not text_column:
                print("❌ 未找到文本列")
                return False
            
            # 选择用于训练的列
            training_columns = [text_column, label_column]
            
            # 确保所有必要的列都存在
            missing_columns = [col for col in training_columns if col not in df.columns]
            if missing_columns:
                print(f"❌ 缺少必要的列: {missing_columns}")
                return False
            
            # 创建训练数据集
            training_df = df[training_columns].copy()
            
            # 重命名列以便模型训练
            training_df.columns = ['text', 'label']
            
            # 保存训练数据
            training_df.to_csv(output_file, index=False, encoding='utf-8')
            print(f"💾 已保存训练数据到: {output_file}")
            
            return True
            
        except Exception as e:
            print(f"❌ 准备训练数据失败: {e}")
            return False
    
    def process_single_file(self, input_file: str, output_dir: Path, 
                          max_per_class: int = None) -> bool:
        """处理单个文件"""
        print(f"\n🔄 开始处理文件: {input_file}")
        print("-" * 60)
        
        # 加载数据
        df = self.load_data(input_file)
        if df.empty:
            return False
        
        # 清洗数据
        df_cleaned = self.clean_data(df, max_per_class)
        if df_cleaned.empty:
            return False
        
        # 生成输出文件名
        input_path = Path(input_file)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        cleaned_file = output_dir / f"{input_path.stem}_cleaned_{timestamp}.csv"
        training_file = output_dir / f"{input_path.stem}_training_{timestamp}.csv"
        
        # 保存清洗后的数据
        df_cleaned.to_csv(cleaned_file, index=False, encoding='utf-8')
        print(f"💾 已保存清洗数据到: {cleaned_file}")
        
        # 准备训练数据
        success = self.prepare_training_data(df_cleaned, str(training_file))
        
        return success
    
    def find_data_files(self) -> List[Path]:
        """查找DATA_OUTPUT目录下的数据文件"""
        data_dir = Path("DATA_OUTPUT")
        if not data_dir.exists():
            print(f"❌ 目录不存在: {data_dir}")
            return []
        
        # 查找所有CSV文件
        csv_files = list(data_dir.glob("*.csv"))
        return csv_files
    
    def batch_process(self, output_dir: Path = None, max_per_class: int = None):
        """批量处理所有数据文件"""
        if output_dir is None:
            output_dir = Path("DATA_OUTPUT")
        
        data_files = self.find_data_files()
        if not data_files:
            print("❌ 未找到数据文件")
            return
        
        print(f"📁 找到 {len(data_files)} 个数据文件")
        print("=" * 60)
        
        success_count = 0
        total_records = 0
        
        for data_file in data_files:
            try:
                if self.process_single_file(str(data_file), output_dir, max_per_class):
                    success_count += 1
                    # 统计记录数
                    df = pd.read_csv(data_file)
                    label_column = self.detect_label_column(df)
                    if label_column:
                        total_records += len(df[df[label_column] != 'other'])
            except Exception as e:
                print(f"❌ 处理文件 {data_file} 时出错: {e}")
        
        print("\n" + "=" * 60)
        print(f"✅ 批量处理完成: {success_count}/{len(data_files)} 个文件处理成功")
        print(f"📊 总共处理了 {total_records} 条有效记录")
    
    def create_combined_dataset(self, output_dir: Path, max_per_class: int = None):
        """创建合并的数据集"""
        print("\n🔄 创建合并数据集...")
        
        data_files = self.find_data_files()
        if not data_files:
            print("❌ 未找到数据文件")
            return
        
        all_data = []
        
        for data_file in data_files:
            try:
                df = self.load_data(str(data_file))
                if not df.empty:
                    label_column = self.detect_label_column(df)
                    if label_column:
                        # 过滤other类别
                        df_cleaned = df[df[label_column] != 'other'].copy()
                        all_data.append(df_cleaned)
                        print(f"  ✓ {data_file.name}: {len(df_cleaned)} 条记录")
            except Exception as e:
                print(f"  ❌ {data_file.name}: {e}")
        
        if not all_data:
            print("❌ 没有有效数据")
            return
        
        # 合并所有数据
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"\n📊 合并后总数据: {len(combined_df)} 条记录")
        
        # 平衡数据
        if max_per_class and max_per_class > 0:
            label_column = self.detect_label_column(combined_df)
            combined_df = combined_df.groupby(label_column).head(max_per_class).copy()
            print(f"⚖️ 平衡后数据: {len(combined_df)} 条记录")
        
        # 保存合并数据集
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        combined_file = output_dir / f"combined_dataset_{timestamp}.csv"
        training_file = output_dir / f"training_dataset_{timestamp}.csv"
        
        combined_df.to_csv(combined_file, index=False, encoding='utf-8')
        print(f"💾 已保存合并数据集到: {combined_file}")
        
        # 准备训练数据
        self.prepare_training_data(combined_df, str(training_file))
        
        # 显示统计信息
        label_column = self.detect_label_column(combined_df)
        if label_column:
            category_counts = combined_df[label_column].value_counts()
            print("\n📈 最终类别分布:")
            for category, count in category_counts.items():
                percentage = (count / len(combined_df)) * 100
                print(f"  {category}: {count} 条 ({percentage:.1f}%)")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="数据清洗脚本")
    parser.add_argument("input_file", nargs="?", help="输入文件路径")
    parser.add_argument("--output-dir", help="输出目录路径")
    parser.add_argument("--max-per-class", type=int, default=500, 
                       help="每类最多保留条数 (默认: 500)")
    parser.add_argument("--combined", action="store_true", 
                       help="创建合并的数据集")
    
    args = parser.parse_args()
    
    cleaner = DataCleaner()
    
    # 设置输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("DATA_OUTPUT")
    
    # 确保输出目录存在
    output_dir.mkdir(exist_ok=True)
    
    if args.combined:
        # 创建合并数据集
        cleaner.create_combined_dataset(output_dir, args.max_per_class)
    elif args.input_file:
        # 处理指定文件
        success = cleaner.process_single_file(
            args.input_file, 
            output_dir, 
            args.max_per_class
        )
        sys.exit(0 if success else 1)
    else:
        # 批量处理
        cleaner.batch_process(output_dir, args.max_per_class)


if __name__ == "__main__":
    main() 