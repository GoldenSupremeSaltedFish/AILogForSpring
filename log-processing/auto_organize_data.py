#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动化数据整理脚本
功能：
1. 读取DATA_OUTPUT/classified_data中的已分类数据文件
2. 去除other类别，按类别排序
3. 将分类后的数据存储到对应的目录结构中
4. 生成汇总报告

使用方法:
python auto_organize_data.py                           # 处理所有classified_data文件
python auto_organize_data.py <文件名>                  # 处理指定文件
python auto_organize_data.py --update-existing        # 更新现有文件
"""

import pandas as pd
import numpy as np
import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class DataOrganizer:
    """数据整理器"""
    
    def __init__(self):
        # 定义目录结构
        self.base_dir = Path("DATA_OUTPUT")
        self.classified_data_dir = self.base_dir / "classified_data"
        self.categorized_logs_dir = self.base_dir / "categorized_logs"
        self.training_data_dir = self.base_dir / "training_data"
        self.summary_reports_dir = self.base_dir / "summary_reports"
        self.raw_data_dir = self.base_dir / "raw_data"
        
        # 确保目录存在
        self._ensure_directories()
        
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
    
    def _ensure_directories(self):
        """确保所有必要的目录存在"""
        directories = [
            self.classified_data_dir,
            self.categorized_logs_dir,
            self.training_data_dir,
            self.summary_reports_dir,
            self.raw_data_dir
        ]
        
        for directory in directories:
            directory.mkdir(exist_ok=True)
            print(f"✅ 确保目录存在: {directory}")
    
    def find_classified_files(self) -> List[Path]:
        """查找classified_data目录中的已分类文件"""
        if not self.classified_data_dir.exists():
            print(f"❌ 目录不存在: {self.classified_data_dir}")
            return []
        
        # 查找所有CSV文件
        csv_files = list(self.classified_data_dir.glob("*.csv"))
        
        # 过滤掉已经处理过的文件（包含categorized的文件）
        classified_files = [f for f in csv_files if "categorized" not in f.name]
        
        print(f"📁 找到 {len(classified_files)} 个待处理的已分类文件")
        for file in classified_files:
            print(f"  - {file.name}")
        
        return classified_files
    
    def detect_label_column(self, df: pd.DataFrame) -> str:
        """检测标签列名"""
        possible_labels = ['content_type', 'final_label', 'label', 'category']
        for col in possible_labels:
            if col in df.columns:
                return col
        return None
    
    def process_single_file(self, file_path: Path, update_existing: bool = False) -> bool:
        """处理单个文件"""
        print(f"\n🔄 开始处理文件: {file_path.name}")
        print("-" * 60)
        
        try:
            # 读取数据
            df = pd.read_csv(file_path)
            print(f"📊 原始数据: {len(df)} 条记录")
            
            # 检测标签列
            label_column = self.detect_label_column(df)
            if not label_column:
                print(f"❌ 未找到标签列: {file_path.name}")
                return False
            
            print(f"🔍 使用标签列: {label_column}")
            
            # 过滤掉other类别
            original_count = len(df)
            df_filtered = df[df[label_column] != 'other'].copy()
            filtered_count = len(df_filtered)
            
            print(f"🔍 过滤后数据: {filtered_count} 条记录 (移除了 {original_count - filtered_count} 条'other'记录)")
            
            if filtered_count == 0:
                print("⚠️ 过滤后没有有效数据")
                return False
            
            # 统计各类别数量
            category_counts = df_filtered[label_column].value_counts()
            print("\n📈 类别分布:")
            for category, count in category_counts.items():
                percentage = (count / filtered_count) * 100
                description = self.category_descriptions.get(category, category)
                print(f"  {description} ({category}): {count} 条 ({percentage:.1f}%)")
            
            # 生成时间戳
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 1. 保存分类后的主文件到classified_data
            categorized_file = self.classified_data_dir / f"{file_path.stem}_categorized_{timestamp}.csv"
            df_filtered.to_csv(categorized_file, index=False, encoding='utf-8')
            print(f"💾 已保存分类数据到: {categorized_file.name}")
            
            # 2. 按类别分别保存到categorized_logs
            self._save_categorized_files(df_filtered, label_column, timestamp)
            
            # 3. 生成训练数据集到training_data
            self._save_training_data(df_filtered, label_column, timestamp)
            
            # 4. 生成汇总报告到summary_reports
            self._save_summary_report(df_filtered, label_column, file_path.name, timestamp)
            
            return True
            
        except Exception as e:
            print(f"❌ 处理文件失败: {e}")
            return False
    
    def _save_categorized_files(self, df: pd.DataFrame, label_column: str, timestamp: str):
        """按类别分别保存文件"""
        print("\n📁 保存分类文件...")
        
        for category in df[label_column].unique():
            if category == 'other':
                continue
            
            category_df = df[df[label_column] == category].copy()
            description = self.category_descriptions.get(category, category)
            
            # 创建文件名
            category_file = self.categorized_logs_dir / f"{category}_{description}_{timestamp}.csv"
            
            # 保存类别数据
            category_df.to_csv(category_file, index=False, encoding='utf-8')
            print(f"  ✓ {description}: {category_file.name} ({len(category_df)} 条)")
    
    def _save_training_data(self, df: pd.DataFrame, label_column: str, timestamp: str):
        """保存训练数据集"""
        print("\n📚 生成训练数据集...")
        
        # 检测文本列
        text_column = None
        possible_text_columns = ['original_log', 'message', 'content', 'text']
        for col in possible_text_columns:
            if col in df.columns:
                text_column = col
                break
        
        if not text_column:
            print("⚠️ 未找到文本列，跳过训练数据生成")
            return
        
        # 创建训练数据集
        training_df = df[[text_column, label_column]].copy()
        training_df.columns = ['text', 'label']
        
        # 保存训练数据
        training_file = self.training_data_dir / f"training_dataset_{timestamp}.csv"
        training_df.to_csv(training_file, index=False, encoding='utf-8')
        print(f"  ✓ 训练数据集: {training_file.name} ({len(training_df)} 条)")
        
        # 保存合并数据集
        combined_file = self.training_data_dir / f"combined_dataset_{timestamp}.csv"
        df.to_csv(combined_file, index=False, encoding='utf-8')
        print(f"  ✓ 合并数据集: {combined_file.name} ({len(df)} 条)")
    
    def _save_summary_report(self, df: pd.DataFrame, label_column: str, original_filename: str, timestamp: str):
        """生成汇总报告"""
        print("\n📋 生成汇总报告...")
        
        # 统计信息
        total_count = len(df)
        category_counts = df[label_column].value_counts()
        
        # 生成报告内容
        report_lines = []
        report_lines.append("📈 日志分类汇总报告")
        report_lines.append("=" * 50)
        report_lines.append(f"原始文件: {original_filename}")
        report_lines.append(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"总记录数: {total_count}")
        report_lines.append("")
        report_lines.append("类别分布:")
        
        for category, count in category_counts.items():
            percentage = (count / total_count) * 100
            description = self.category_descriptions.get(category, category)
            report_lines.append(f"  {description} ({category}): {count} 条 ({percentage:.1f}%)")
        
        report_lines.append("")
        report_lines.append("=" * 50)
        
        # 保存报告
        report_file = self.summary_reports_dir / f"{Path(original_filename).stem}_categorized_{timestamp}_summary.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"  ✓ 汇总报告: {report_file.name}")
    
    def process_all_files(self, update_existing: bool = False):
        """处理所有文件"""
        classified_files = self.find_classified_files()
        
        if not classified_files:
            print("❌ 没有找到待处理的文件")
            return
        
        print(f"\n🚀 开始批量处理 {len(classified_files)} 个文件")
        print("=" * 60)
        
        success_count = 0
        total_records = 0
        
        for file_path in classified_files:
            try:
                if self.process_single_file(file_path, update_existing):
                    success_count += 1
                    # 统计记录数
                    df = pd.read_csv(file_path)
                    label_column = self.detect_label_column(df)
                    if label_column:
                        total_records += len(df[df[label_column] != 'other'])
            except Exception as e:
                print(f"❌ 处理文件 {file_path.name} 时出错: {e}")
        
        print("\n" + "=" * 60)
        print(f"✅ 批量处理完成: {success_count}/{len(classified_files)} 个文件处理成功")
        print(f"📊 总共处理了 {total_records} 条有效记录")
        
        # 生成总体汇总报告
        self._generate_overall_summary()
    
    def _generate_overall_summary(self):
        """生成总体汇总报告"""
        print("\n📊 生成总体汇总报告...")
        
        # 统计各个目录的文件
        categorized_files = list(self.categorized_logs_dir.glob("*.csv"))
        training_files = list(self.training_data_dir.glob("*.csv"))
        summary_files = list(self.summary_reports_dir.glob("*.txt"))
        
        # 生成总体报告
        overall_report = []
        overall_report.append("📊 数据整理总体汇总")
        overall_report.append("=" * 50)
        overall_report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        overall_report.append("")
        overall_report.append("📁 目录统计:")
        overall_report.append(f"  分类日志文件: {len(categorized_files)} 个")
        overall_report.append(f"  训练数据文件: {len(training_files)} 个")
        overall_report.append(f"  汇总报告文件: {len(summary_files)} 个")
        overall_report.append("")
        overall_report.append("📈 最近处理的文件:")
        
        # 显示最近的文件
        all_files = categorized_files + training_files + summary_files
        all_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        for file in all_files[:5]:  # 显示最近5个文件
            mtime = datetime.fromtimestamp(file.stat().st_mtime)
            overall_report.append(f"  {file.name} ({mtime.strftime('%Y-%m-%d %H:%M')})")
        
        overall_report.append("")
        overall_report.append("=" * 50)
        
        # 保存总体报告
        overall_file = self.summary_reports_dir / f"overall_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(overall_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(overall_report))
        
        print(f"✅ 总体汇总报告已保存到: {overall_file.name}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="自动化数据整理脚本")
    parser.add_argument("file", nargs="?", help="指定要处理的文件")
    parser.add_argument("--update-existing", action="store_true", 
                       help="更新现有文件")
    
    args = parser.parse_args()
    
    organizer = DataOrganizer()
    
    if args.file:
        # 处理指定文件
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"❌ 文件不存在: {file_path}")
            sys.exit(1)
        
        success = organizer.process_single_file(file_path, args.update_existing)
        sys.exit(0 if success else 1)
    else:
        # 处理所有文件
        organizer.process_all_files(args.update_existing)


if __name__ == "__main__":
    main() 