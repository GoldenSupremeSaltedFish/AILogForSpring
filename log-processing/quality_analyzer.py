#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日志分类质量检测分析器
支持自动化质量评估、统计分析和报告生成
"""

import pandas as pd
import numpy as np
import json
import os
import sys
from pathlib import Path
from datetime import datetime
import argparse
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib for Chinese font support
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class QualityAnalyzer:
    """日志分类质量分析器"""
    
    def __init__(self):
        self.classification_mapping = {
            'stack_exception': '堆栈异常',
            'connection_issue': '连接问题', 
            'database_exception': '数据库异常',
            'timeout': '超时',
            'spring_boot_startup_failure': 'Spring Boot启动失败',
            'config_environment': '配置环境',
            'monitoring_heartbeat': '监控心跳',
            'performance_issue': '性能问题',
            'security_auth': '安全认证',
            'api_request_response': 'API请求响应',
            'business_logic': '业务逻辑',
            'other': '其他'
        }
        
        self.priority_levels = {
            1: '极高优先级',
            2: '高优先级', 
            3: '中高优先级',
            4: '中等优先级',
            8: '低优先级',
            11: '极低优先级',
            999: '忽略级别'
        }
        
        # 添加默认输出目录配置
        self.default_output_base = r"c:\Users\30871\Desktop\AILogForSpring\DATA_OUTPUT\质量分析结果"
    
    def convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, dict):
            return {key: self.convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def get_output_directory(self, file_path: str, custom_output_dir: str = None) -> str:
        """获取输出目录，按文件名创建子文件夹"""
        if custom_output_dir:
            base_dir = custom_output_dir
        else:
            base_dir = self.default_output_base
        
        # 从文件路径提取文件名（不含扩展名）
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # 创建以文件名命名的子目录
        output_dir = os.path.join(base_dir, file_name)
        os.makedirs(output_dir, exist_ok=True)
        
        return output_dir

    def load_data(self, file_path: str) -> pd.DataFrame:
        """加载分类数据"""
        try:
            df = pd.read_csv(file_path, encoding='utf-8-sig')
            print(f"✅ 成功加载数据: {len(df)} 条记录")
            return df
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            return None
    
    def basic_statistics(self, df: pd.DataFrame) -> Dict:
        """基础统计分析"""
        stats = {
            'total_records': int(len(df)),  # Convert to int
            'log_level_distribution': df['log_level'].value_counts().to_dict(),
            'content_type_distribution': df['content_type'].value_counts().to_dict(),
            'priority_distribution': df['priority'].value_counts().to_dict(),
            'manual_annotation_needed': {
                'count': int(df['manual_annotation_needed'].sum()),  # Convert to int
                'percentage': float((df['manual_annotation_needed'].sum() / len(df)) * 100)  # Convert to float
            }
        }
        return self.convert_numpy_types(stats)
    
    def quality_metrics(self, df: pd.DataFrame) -> Dict:
        """质量指标计算"""
        metrics = {}
        
        # 分类覆盖率
        classified_count = len(df[df['content_type'] != 'other'])
        metrics['classification_coverage'] = float((classified_count / len(df)) * 100)
        
        # 高优先级比例
        high_priority_count = len(df[df['priority'] <= 4])
        metrics['high_priority_ratio'] = float((high_priority_count / len(df)) * 100)
        
        # 需要人工标注比例
        manual_needed = df['manual_annotation_needed'].sum()
        metrics['manual_annotation_ratio'] = float((manual_needed / len(df)) * 100)
        
        # 日志级别分布均衡性（熵值）
        level_counts = df['log_level'].value_counts(normalize=True)
        metrics['level_distribution_entropy'] = float(-sum(p * np.log2(p) for p in level_counts if p > 0))
        
        # 分类分布均衡性
        type_counts = df['content_type'].value_counts(normalize=True)
        metrics['type_distribution_entropy'] = float(-sum(p * np.log2(p) for p in type_counts if p > 0))
        
        return self.convert_numpy_types(metrics)
    
    def anomaly_detection(self, df: pd.DataFrame) -> Dict:
        """异常检测"""
        anomalies = {}
        
        # 检测异常优先级组合
        priority_content_combinations = df.groupby(['content_type', 'priority']).size()
        unusual_combinations = []
        
        for (content_type, priority), count in priority_content_combinations.items():
            if content_type == 'stack_exception' and priority > 4:
                unusual_combinations.append(f"堆栈异常但优先级低: {content_type} - {priority} ({count}条)")
            elif content_type == 'monitoring_heartbeat' and priority < 8:
                unusual_combinations.append(f"监控心跳但优先级高: {content_type} - {priority} ({count}条)")
        
        anomalies['unusual_priority_combinations'] = unusual_combinations
        
        # 检测空或异常长度的日志
        empty_logs = len(df[df['original_log'].str.len() < 10])
        very_long_logs = len(df[df['original_log'].str.len() > 1000])
        
        anomalies['data_quality_issues'] = {
            'empty_or_short_logs': empty_logs,
            'very_long_logs': very_long_logs
        }
        
        # 检测重复日志
        duplicate_logs = df['original_log'].duplicated().sum()
        anomalies['duplicate_logs'] = duplicate_logs
        
        return anomalies
    
    def generate_recommendations(self, stats: Dict, metrics: Dict, anomalies: Dict) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        # 基于分类覆盖率的建议
        if metrics['classification_coverage'] < 80:
            recommendations.append("🔧 分类覆盖率较低，建议优化分类规则以减少'其他'类别")
        
        # 基于人工标注比例的建议
        if metrics['manual_annotation_ratio'] > 30:
            recommendations.append("⚠️ 需要人工标注的比例过高，建议完善自动分类规则")
        
        # 基于优先级分布的建议
        if metrics['high_priority_ratio'] > 50:
            recommendations.append("📊 高优先级日志比例过高，建议检查优先级分配逻辑")
        elif metrics['high_priority_ratio'] < 10:
            recommendations.append("📊 高优先级日志比例过低，可能遗漏重要问题")
        
        # 基于异常检测的建议
        if anomalies['unusual_priority_combinations']:
            recommendations.append("🚨 发现异常的优先级组合，建议检查分类逻辑")
        
        if anomalies['duplicate_logs'] > 0:
            recommendations.append(f"🔄 发现 {anomalies['duplicate_logs']} 条重复日志，建议进行去重处理")
        
        # 基于数据质量的建议
        data_issues = anomalies['data_quality_issues']
        if data_issues['empty_or_short_logs'] > 0:
            recommendations.append(f"📝 发现 {data_issues['empty_or_short_logs']} 条过短日志，建议检查数据质量")
        
        return recommendations
    
    def create_visualizations(self, df: pd.DataFrame, output_dir: str):
        """创建可视化图表"""
        # 确保中文字体正确显示
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('日志分类质量分析报告', fontsize=16, fontweight='bold')
        
        # 1. 日志级别分布
        level_counts = df['log_level'].value_counts()
        axes[0, 0].pie(level_counts.values, labels=level_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('日志级别分布')
        
        # 2. 内容类型分布
        type_counts = df['content_type'].value_counts().head(10)
        axes[0, 1].bar(range(len(type_counts)), type_counts.values)
        axes[0, 1].set_xticks(range(len(type_counts)))
        axes[0, 1].set_xticklabels([self.classification_mapping.get(t, t) for t in type_counts.index], 
                                  rotation=45, ha='right')
        axes[0, 1].set_title('内容类型分布 (Top 10)')
        axes[0, 1].set_ylabel('数量')
        
        # 3. 优先级分布
        priority_counts = df['priority'].value_counts().sort_index()
        axes[1, 0].bar([self.priority_levels.get(p, str(p)) for p in priority_counts.index], 
                      priority_counts.values)
        axes[1, 0].set_title('优先级分布')
        axes[1, 0].set_ylabel('数量')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. 人工标注需求分布
        manual_counts = df['manual_annotation_needed'].value_counts()
        axes[1, 1].pie(manual_counts.values, 
                      labels=['不需要人工标注', '需要人工标注'], 
                      autopct='%1.1f%%',
                      colors=['lightgreen', 'lightcoral'])
        axes[1, 1].set_title('人工标注需求分布')
        
        plt.tight_layout()
        
        # 保存图表，确保支持中文文件名
        chart_path = os.path.join(output_dir, 'quality_analysis_charts.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"📊 可视化图表已保存: {chart_path}")

    def generate_report(self, file_path: str, output_dir: str = None):
        """生成完整的质量分析报告"""
        # Use new output directory logic
        output_dir = self.get_output_directory(file_path, output_dir)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载数据
        df = self.load_data(file_path)
        if df is None:
            return
        
        # 执行分析
        print("\n🔍 执行基础统计分析...")
        stats = self.basic_statistics(df)
        
        print("📊 计算质量指标...")
        metrics = self.quality_metrics(df)
        
        print("🚨 执行异常检测...")
        anomalies = self.anomaly_detection(df)
        
        print("💡 生成改进建议...")
        recommendations = self.generate_recommendations(stats, metrics, anomalies)
        
        print("📈 创建可视化图表...")
        self.create_visualizations(df, output_dir)
        
        # 生成报告
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = os.path.join(output_dir, f'quality_analysis_report_{timestamp}.txt')
        
        with open(report_file, 'w', encoding='utf-8-sig') as f:
            f.write("="*60 + "\n")
            f.write("日志分类质量分析报告\n")
            f.write("="*60 + "\n")
            f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"数据文件: {os.path.basename(file_path)}\n")
            f.write(f"总记录数: {stats['total_records']}\n\n")
            
            # 基础统计
            f.write("📊 基础统计信息\n")
            f.write("-"*30 + "\n")
            f.write(f"日志级别分布:\n")
            for level, count in stats['log_level_distribution'].items():
                percentage = (count / stats['total_records']) * 100
                f.write(f"  {level}: {count} ({percentage:.1f}%)\n")
            
            f.write(f"\n内容类型分布 (Top 10):\n")
            for i, (ctype, count) in enumerate(list(stats['content_type_distribution'].items())[:10]):
                percentage = (count / stats['total_records']) * 100
                display_name = self.classification_mapping.get(ctype, ctype)
                f.write(f"  {display_name}: {count} ({percentage:.1f}%)\n")
            
            # 质量指标
            f.write(f"\n📈 质量指标\n")
            f.write("-"*30 + "\n")
            f.write(f"分类覆盖率: {metrics['classification_coverage']:.1f}%\n")
            f.write(f"高优先级比例: {metrics['high_priority_ratio']:.1f}%\n")
            f.write(f"需要人工标注比例: {metrics['manual_annotation_ratio']:.1f}%\n")
            f.write(f"日志级别分布熵值: {metrics['level_distribution_entropy']:.2f}\n")
            f.write(f"分类分布熵值: {metrics['type_distribution_entropy']:.2f}\n")
            
            # 异常检测
            f.write(f"\n🚨 异常检测结果\n")
            f.write("-"*30 + "\n")
            if anomalies['unusual_priority_combinations']:
                f.write("异常优先级组合:\n")
                for combo in anomalies['unusual_priority_combinations']:
                    f.write(f"  ⚠️ {combo}\n")
            else:
                f.write("✅ 未发现异常优先级组合\n")
            
            f.write(f"\n数据质量问题:\n")
            f.write(f"  过短日志: {anomalies['data_quality_issues']['empty_or_short_logs']} 条\n")
            f.write(f"  过长日志: {anomalies['data_quality_issues']['very_long_logs']} 条\n")
            f.write(f"  重复日志: {anomalies['duplicate_logs']} 条\n")
            
            # 改进建议
            f.write(f"\n💡 改进建议\n")
            f.write("-"*30 + "\n")
            if recommendations:
                for i, rec in enumerate(recommendations, 1):
                    f.write(f"{i}. {rec}\n")
            else:
                f.write("✅ 当前分类质量良好，暂无特别建议\n")
            
            f.write(f"\n" + "="*60 + "\n")
            f.write("报告生成完成\n")
        
        # 生成JSON格式的详细数据
        json_file = os.path.join(output_dir, f'quality_analysis_data_{timestamp}.json')
        # Apply conversion before JSON serialization
        analysis_data = self.convert_numpy_types({
            'metadata': {
                'file_path': file_path,
                'analysis_time': datetime.now().isoformat(),
                'total_records': stats['total_records']
            },
            'statistics': stats,
            'quality_metrics': metrics,
            'anomalies': anomalies,
            'recommendations': recommendations
        })
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ 质量分析完成!")
        print(f"📄 文本报告: {report_file}")
        print(f"📊 数据文件: {json_file}")
        print(f"📈 图表文件: {os.path.join(output_dir, 'quality_analysis_charts.png')}")
        
        return analysis_data
    
    def compare_files(self, file1: str, file2: str, output_dir: str = None):
        """比较两个分类文件的质量差异"""
        print("🔄 开始比较分析...")
        
        df1 = self.load_data(file1)
        df2 = self.load_data(file2)
        
        if df1 is None or df2 is None:
            return
        
        metrics1 = self.quality_metrics(df1)
        metrics2 = self.quality_metrics(df2)
        
        # 使用新的输出目录逻辑，基于第一个文件名创建目录
        if output_dir is None:
            output_dir = self.get_output_directory(file1)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        comparison_file = os.path.join(output_dir, f'quality_comparison_{timestamp}.txt')
        
        with open(comparison_file, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("分类质量对比分析报告\n")
            f.write("="*60 + "\n")
            f.write(f"文件1: {os.path.basename(file1)} ({len(df1)} 条记录)\n")
            f.write(f"文件2: {os.path.basename(file2)} ({len(df2)} 条记录)\n")
            f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("📊 质量指标对比\n")
            f.write("-"*40 + "\n")
            f.write(f"{'指标':<20} {'文件1':<15} {'文件2':<15} {'差异':<10}\n")
            f.write("-"*40 + "\n")
            
            for metric in ['classification_coverage', 'high_priority_ratio', 'manual_annotation_ratio']:
                val1 = metrics1[metric]
                val2 = metrics2[metric]
                diff = val2 - val1
                f.write(f"{metric:<20} {val1:<15.1f} {val2:<15.1f} {diff:+.1f}\n")
        
        print(f"📊 对比报告已生成: {comparison_file}")

def main():
    parser = argparse.ArgumentParser(description='日志分类质量分析器')
    parser.add_argument('mode', choices=['analyze', 'compare'], help='分析模式')
    parser.add_argument('--file', help='要分析的CSV文件路径')
    parser.add_argument('--file1', help='比较模式：第一个文件')
    parser.add_argument('--file2', help='比较模式：第二个文件')
    parser.add_argument('--output-dir', help='输出目录')
    
    args = parser.parse_args()
    
    analyzer = QualityAnalyzer()
    
    if args.mode == 'analyze':
        if not args.file:
            print("❌ 分析模式需要指定 --file 参数")
            return
        
        analyzer.generate_report(args.file, args.output_dir)
    
    elif args.mode == 'compare':
        if not args.file1 or not args.file2:
            print("❌ 比较模式需要指定 --file1 和 --file2 参数")
            return
        
        analyzer.compare_files(args.file1, args.file2, args.output_dir)

if __name__ == '__main__':
    main()