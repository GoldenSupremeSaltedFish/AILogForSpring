#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日志分类整理器
功能：
1. 去除日志中的"others"条目
2. 按照类别对现有条目进行归类排序
3. 生成按类别分组的CSV文件
4. 生成分类统计报告

使用方法:
python log_categorizer.py                           # 自动处理dataset-ready目录下的所有CSV文件
python log_categorizer.py <输入CSV文件路径>          # 处理指定文件
python log_categorizer.py <文件路径> --output-dir <输出目录>  # 指定输出目录
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


class LogCategorizer:
    """日志分类整理器"""
    
    def __init__(self):
        # 定义类别优先级顺序（用于排序）
        self.category_priority = {
            'stack_exception': 1,           # 堆栈异常 - 最高优先级
            'spring_boot_startup_failure': 2, # Spring Boot启动失败
            'auth_authorization': 3,        # 认证授权
            'database_exception': 4,        # 数据库异常
            'connection_issue': 5,          # 连接问题
            'timeout': 6,                   # 超时错误
            'memory_performance': 7,        # 内存性能
            'config_environment': 8,        # 配置环境
            'business_logic': 9,            # 业务逻辑
            'normal_operation': 10,         # 正常操作
            'monitoring_heartbeat': 11,     # 监控心跳
            'other': 999                    # 其他类别 - 最低优先级（将被过滤）
        }
        
        # 类别中文描述
        self.category_descriptions = {
            'stack_exception': '堆栈异常',
            'spring_boot_startup_failure': 'Spring Boot启动失败',
            'auth_authorization': '认证授权',
            'database_exception': '数据库异常',
            'connection_issue': '连接问题',
            'timeout': '超时错误',
            'memory_performance': '内存性能',
            'config_environment': '配置环境',
            'business_logic': '业务逻辑',
            'normal_operation': '正常操作',
            'monitoring_heartbeat': '监控心跳',
            'other': '其他'
        }
    
    def load_and_filter_data(self, input_file: str) -> pd.DataFrame:
        """加载数据并过滤掉'other'类别"""
        try:
            df = pd.read_csv(input_file)
            print(f"📊 原始数据: {len(df)} 条记录")
            
            # 检查必要的列是否存在
            required_columns = ['content_type', 'original_log']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                # 尝试检查是否有final_label列（旧格式）
                if 'final_label' in df.columns:
                    required_columns = ['final_label', 'message', 'timestamp']
                    missing_columns = [col for col in required_columns if col not in df.columns]
                    if missing_columns:
                        print(f"❌ 缺少必要的列: {missing_columns}")
                        return pd.DataFrame()
                    
                    # 使用旧格式处理
                    original_count = len(df)
                    df_filtered = df[df['final_label'] != 'other'].copy()
                    filtered_count = len(df_filtered)
                    
                    print(f"🔍 过滤后数据: {filtered_count} 条记录 (移除了 {original_count - filtered_count} 条'other'记录)")
                    return df_filtered
                else:
                    print(f"❌ 缺少必要的列: {missing_columns}")
                    return pd.DataFrame()
            
            # 过滤掉'other'类别
            original_count = len(df)
            df_filtered = df[df['content_type'] != 'other'].copy()
            filtered_count = len(df_filtered)
            
            print(f"🔍 过滤后数据: {filtered_count} 条记录 (移除了 {original_count - filtered_count} 条'other'记录)")
            
            return df_filtered
            
        except Exception as e:
            print(f"❌ 读取文件失败: {e}")
            return pd.DataFrame()
    
    def categorize_and_sort(self, df: pd.DataFrame) -> pd.DataFrame:
        """按照类别对数据进行分类和排序"""
        if df.empty:
            return df
        
        # 确定使用的标签列名
        label_column = 'content_type' if 'content_type' in df.columns else 'final_label'
        
        # 添加类别优先级列用于排序
        df['category_priority'] = df[label_column].map(self.category_priority)
        
        # 确定排序列
        sort_columns = ['category_priority']
        if 'timestamp' in df.columns:
            sort_columns.append('timestamp')
        elif 'line_number' in df.columns:
            sort_columns.append('line_number')
        
        # 按类别优先级和时间戳排序
        df_sorted = df.sort_values(sort_columns).copy()
        
        # 移除临时的优先级列
        df_sorted = df_sorted.drop('category_priority', axis=1)
        
        return df_sorted
    
    def generate_category_summary(self, df: pd.DataFrame) -> str:
        """生成分类统计摘要"""
        if df.empty:
            return "没有数据可统计"
        
        summary_lines = []
        summary_lines.append("📈 分类统计报告")
        summary_lines.append("=" * 50)
        
        # 确定使用的标签列名
        label_column = 'content_type' if 'content_type' in df.columns else 'final_label'
        
        # 按类别统计
        category_counts = df[label_column].value_counts()
        total_count = len(df)
        
        summary_lines.append(f"总记录数: {total_count}")
        summary_lines.append("")
        
        for category, count in category_counts.items():
            percentage = (count / total_count) * 100
            description = self.category_descriptions.get(category, category)
            summary_lines.append(f"{description} ({category}): {count} 条 ({percentage:.1f}%)")
        
        summary_lines.append("")
        summary_lines.append("=" * 50)
        
        return "\n".join(summary_lines)
    
    def save_categorized_data(self, df: pd.DataFrame, output_file: str, 
                            include_summary: bool = True) -> bool:
        """保存分类后的数据"""
        try:
            # 保存主要数据
            df.to_csv(output_file, index=False, encoding='utf-8')
            print(f"💾 已保存分类数据到: {output_file}")
            
            # 生成并保存摘要
            if include_summary:
                summary = self.generate_category_summary(df)
                summary_file = output_file.replace('.csv', '_summary.txt')
                
                with open(summary_file, 'w', encoding='utf-8') as f:
                    f.write(summary)
                    f.write(f"\n\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                print(f"📋 已保存统计摘要到: {summary_file}")
            
            return True
            
        except Exception as e:
            print(f"❌ 保存文件失败: {e}")
            return False
    
    def create_category_files(self, df: pd.DataFrame, output_dir: Path) -> bool:
        """为每个类别创建单独的文件"""
        try:
            # 确定使用的标签列名
            label_column = 'content_type' if 'content_type' in df.columns else 'final_label'
            
            # 按类别分组
            for category in df[label_column].unique():
                if category == 'other':
                    continue
                
                category_df = df[df[label_column] == category].copy()
                category_description = self.category_descriptions.get(category, category)
                
                # 创建类别文件名
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                category_file = output_dir / f"{category}_{category_description}_{timestamp}.csv"
                
                # 保存类别数据
                category_df.to_csv(category_file, index=False, encoding='utf-8')
                print(f"📁 已保存 {category_description} 类别数据: {category_file.name} ({len(category_df)} 条)")
            
            return True
            
        except Exception as e:
            print(f"❌ 创建类别文件失败: {e}")
            return False
    
    def process_single_file(self, input_file: str, output_dir: Path, 
                          create_category_files: bool = True) -> bool:
        """处理单个文件"""
        print(f"\n🔄 开始处理文件: {input_file}")
        print("-" * 60)
        
        # 加载和过滤数据
        df = self.load_and_filter_data(input_file)
        if df.empty:
            return False
        
        # 分类和排序
        df_sorted = self.categorize_and_sort(df)
        
        # 生成输出文件名
        input_path = Path(input_file)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = output_dir / f"{input_path.stem}_categorized_{timestamp}.csv"
        
        # 保存分类后的数据
        success = self.save_categorized_data(df_sorted, str(output_file))
        
        # 创建按类别分组的文件
        if success and create_category_files:
            self.create_category_files(df_sorted, output_dir)
        
        # 显示统计信息
        print("\n" + self.generate_category_summary(df_sorted))
        
        return success
    
    def find_csv_files(self) -> List[Path]:
        """查找dataset-ready目录下的CSV文件"""
        dataset_dir = Path("dataset-ready")
        if not dataset_dir.exists():
            print(f"❌ 目录不存在: {dataset_dir}")
            return []
        
        csv_files = list(dataset_dir.glob("*.csv"))
        return csv_files
    
    def batch_process(self, output_dir: Path = None, create_category_files: bool = True):
        """批量处理所有CSV文件"""
        if output_dir is None:
            output_dir = Path("dataset-ready")
        
        csv_files = self.find_csv_files()
        if not csv_files:
            print("❌ 未找到CSV文件")
            return
        
        print(f"📁 找到 {len(csv_files)} 个CSV文件")
        print("=" * 60)
        
        success_count = 0
        total_logs = 0
        
        for csv_file in csv_files:
            try:
                if self.process_single_file(str(csv_file), output_dir, create_category_files):
                    success_count += 1
                    # 统计日志条数
                    df = pd.read_csv(csv_file)
                    total_logs += len(df[df['final_label'] != 'other'])
            except Exception as e:
                print(f"❌ 处理文件 {csv_file} 时出错: {e}")
        
        print("\n" + "=" * 60)
        print(f"✅ 批量处理完成: {success_count}/{len(csv_files)} 个文件处理成功")
        print(f"📊 总共处理了 {total_logs} 条有效日志记录")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="日志分类整理器")
    parser.add_argument("input_file", nargs="?", help="输入CSV文件路径")
    parser.add_argument("--output-dir", help="输出目录路径")
    parser.add_argument("--no-category-files", action="store_true", 
                       help="不创建按类别分组的文件")
    
    args = parser.parse_args()
    
    categorizer = LogCategorizer()
    
    # 设置输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("dataset-ready")
    
    # 确保输出目录存在
    output_dir.mkdir(exist_ok=True)
    
    if args.input_file:
        # 处理指定文件
        success = categorizer.process_single_file(
            args.input_file, 
            output_dir, 
            not args.no_category_files
        )
        sys.exit(0 if success else 1)
    else:
        # 批量处理
        categorizer.batch_process(output_dir, not args.no_category_files)


if __name__ == "__main__":
    main() 