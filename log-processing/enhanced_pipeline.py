#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强的日志半自动分类流水线
整合模板化、特征工程、机器学习分类和人工审查的完整流程

使用方法:
python enhanced_pipeline.py --input-file logs.csv --mode full
python enhanced_pipeline.py --input-dir logs/ --mode batch --skip-human-review
python enhanced_pipeline.py --input-file logs.csv --mode template-only
"""

import pandas as pd
import numpy as np
import json
import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# 导入自定义模块
from log_templater import LogTemplater
from feature_engineer import FeatureEngineer
from enhanced_pre_classifier import EnhancedPreClassifier
from auto_labeler import LogAutoLabeler
from log_reviewer import LogReviewer
from quality_analyzer import QualityAnalyzer

class EnhancedLogPipeline:
    """增强的日志半自动分类流水线"""
    
    def __init__(self):
        self.pipeline_config = {
            'enable_templating': True,
            'enable_feature_engineering': True,
            'enable_ml_classification': True,
            'enable_human_review': True,
            'enable_quality_analysis': True,
            'max_per_class': 500,
            'confidence_threshold': 0.7
        }
        
        # 初始化各个组件
        self.templater = LogTemplater()
        self.feature_engineer = FeatureEngineer()
        self.pre_classifier = EnhancedPreClassifier()
        self.auto_labeler = LogAutoLabeler()
        self.quality_analyzer = QualityAnalyzer()
        
        # 输出目录配置
        self.output_base_dir = Path(__file__).parent.parent / "log-processing-OUTPUT"
        
        # 流水线状态
        self.pipeline_state = {
            'current_step': 0,
            'total_steps': 0,
            'results': {},
            'errors': []
        }
    
    def configure_pipeline(self, config: Dict):
        """配置流水线参数"""
        self.pipeline_config.update(config)
        print(f"🔧 流水线配置更新: {config}")
    
    def step_1_template_logs(self, input_file: str, output_dir: Path) -> Dict:
        """步骤1: 日志模板化"""
        print("\n" + "="*60)
        print("📋 步骤1: 日志模板化")
        print("="*60)
        
        try:
            result = self.templater.process_file(input_file, output_dir)
            if result:
                self.pipeline_state['results']['templating'] = result
                print(f"✅ 模板化完成: {result['total_templates']} 个模板")
                return result
            else:
                raise Exception("模板化处理失败")
                
        except Exception as e:
            error_msg = f"模板化步骤失败: {e}"
            print(f"❌ {error_msg}")
            self.pipeline_state['errors'].append(error_msg)
            return {}
    
    def step_2_extract_features(self, templated_file: str, output_dir: Path) -> Dict:
        """步骤2: 特征工程"""
        print("\n" + "="*60)
        print("🔧 步骤2: 特征工程")
        print("="*60)
        
        try:
            result = self.feature_engineer.process_file(templated_file, output_dir)
            if result:
                self.pipeline_state['results']['feature_engineering'] = result
                print(f"✅ 特征工程完成: {result['feature_count']} 个特征")
                return result
            else:
                raise Exception("特征工程处理失败")
                
        except Exception as e:
            error_msg = f"特征工程步骤失败: {e}"
            print(f"❌ {error_msg}")
            self.pipeline_state['errors'].append(error_msg)
            return {}
    
    def step_3_pre_classify(self, input_file: str, output_dir: Path) -> Dict:
        """步骤3: 预分类"""
        print("\n" + "="*60)
        print("🏷️ 步骤3: 预分类")
        print("="*60)
        
        try:
            result = self.pre_classifier.process_log_file(input_file, str(output_dir))
            if result:
                self.pipeline_state['results']['pre_classification'] = result
                print(f"✅ 预分类完成: {result['classified_lines']} 条已分类")
                return result
            else:
                raise Exception("预分类处理失败")
                
        except Exception as e:
            error_msg = f"预分类步骤失败: {e}"
            print(f"❌ {error_msg}")
            self.pipeline_state['errors'].append(error_msg)
            return {}
    
    def step_4_auto_label(self, input_file: str, output_dir: Path) -> Dict:
        """步骤4: 自动标签"""
        print("\n" + "="*60)
        print("🤖 步骤4: 自动标签")
        print("="*60)
        
        try:
            # 创建临时输出目录
            temp_output_dir = output_dir / "temp_auto_label"
            temp_output_dir.mkdir(exist_ok=True)
            
            success = self.auto_labeler.process_single_file(input_file, temp_output_dir, use_ml=True)
            if success:
                # 查找生成的文件
                labeled_files = list(temp_output_dir.glob("*_labeled_*.csv"))
                if labeled_files:
                    result = {
                        'output_file': str(labeled_files[0]),
                        'success': True
                    }
                    self.pipeline_state['results']['auto_labeling'] = result
                    print(f"✅ 自动标签完成: {labeled_files[0].name}")
                    return result
            
            raise Exception("自动标签处理失败")
                
        except Exception as e:
            error_msg = f"自动标签步骤失败: {e}"
            print(f"❌ {error_msg}")
            self.pipeline_state['errors'].append(error_msg)
            return {}
    
    def step_5_human_review(self, labeled_file: str, output_dir: Path) -> Dict:
        """步骤5: 人工审查"""
        print("\n" + "="*60)
        print("👥 步骤5: 人工审查")
        print("="*60)
        
        if not self.pipeline_config['enable_human_review']:
            print("⏭️ 跳过人工审查步骤")
            return {'skipped': True}
        
        try:
            print("🔍 启动人工审查工具...")
            print("💡 提示: 在审查工具中按 'q' 可以退出并保存进度")
            
            # 创建审查器
            reviewer = LogReviewer(labeled_file, str(output_dir))
            
            # 运行审查
            reviewer.run()
            
            result = {
                'output_file': str(reviewer.output_file),
                'stats': reviewer.stats,
                'success': True
            }
            
            self.pipeline_state['results']['human_review'] = result
            print(f"✅ 人工审查完成")
            return result
            
        except Exception as e:
            error_msg = f"人工审查步骤失败: {e}"
            print(f"❌ {error_msg}")
            self.pipeline_state['errors'].append(error_msg)
            return {}
    
    def step_6_quality_analysis(self, final_file: str, output_dir: Path) -> Dict:
        """步骤6: 质量分析"""
        print("\n" + "="*60)
        print("📊 步骤6: 质量分析")
        print("="*60)
        
        try:
            result = self.quality_analyzer.generate_report(final_file, str(output_dir))
            if result:
                self.pipeline_state['results']['quality_analysis'] = result
                print(f"✅ 质量分析完成")
                return result
            else:
                raise Exception("质量分析处理失败")
                
        except Exception as e:
            error_msg = f"质量分析步骤失败: {e}"
            print(f"❌ {error_msg}")
            self.pipeline_state['errors'].append(error_msg)
            return {}
    
    def run_full_pipeline(self, input_file: str, output_dir: Path) -> Dict:
        """运行完整的流水线"""
        print("🚀 启动增强的日志半自动分类流水线")
        print(f"📁 输入文件: {input_file}")
        print(f"📁 输出目录: {output_dir}")
        
        # 创建输出目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        pipeline_output_dir = output_dir / f"pipeline_{timestamp}"
        pipeline_output_dir.mkdir(exist_ok=True, parents=True)
        
        # 设置流水线状态
        self.pipeline_state = {
            'current_step': 0,
            'total_steps': 6,
            'results': {},
            'errors': [],
            'start_time': datetime.now().isoformat(),
            'input_file': input_file,
            'output_dir': str(pipeline_output_dir)
        }
        
        try:
            # 步骤1: 模板化
            if self.pipeline_config['enable_templating']:
                self.pipeline_state['current_step'] = 1
                templating_result = self.step_1_template_logs(input_file, pipeline_output_dir)
                if not templating_result:
                    raise Exception("模板化步骤失败")
                
                # 使用模板化结果作为下一步输入
                next_input = templating_result['output_file']
            else:
                next_input = input_file
            
            # 步骤2: 特征工程
            if self.pipeline_config['enable_feature_engineering']:
                self.pipeline_state['current_step'] = 2
                feature_result = self.step_2_extract_features(next_input, pipeline_output_dir)
                if not feature_result:
                    raise Exception("特征工程步骤失败")
            
            # 步骤3: 预分类
            self.pipeline_state['current_step'] = 3
            preclass_result = self.step_3_pre_classify(input_file, pipeline_output_dir)
            if not preclass_result:
                raise Exception("预分类步骤失败")
            
            # 步骤4: 自动标签
            if self.pipeline_config['enable_ml_classification']:
                self.pipeline_state['current_step'] = 4
                # 使用预分类结果
                preclass_files = list(pipeline_output_dir.rglob("*_classified.csv"))
                if preclass_files:
                    auto_label_result = self.step_4_auto_label(str(preclass_files[0]), pipeline_output_dir)
                    if not auto_label_result:
                        raise Exception("自动标签步骤失败")
                    
                    # 使用自动标签结果作为下一步输入
                    review_input = auto_label_result['output_file']
                else:
                    raise Exception("未找到预分类结果文件")
            else:
                # 跳过自动标签，使用预分类结果
                preclass_files = list(pipeline_output_dir.rglob("*_classified.csv"))
                if preclass_files:
                    review_input = str(preclass_files[0])
                else:
                    raise Exception("未找到预分类结果文件")
            
            # 步骤5: 人工审查
            self.pipeline_state['current_step'] = 5
            review_result = self.step_5_human_review(review_input, pipeline_output_dir)
            
            # 确定最终文件
            if review_result.get('success'):
                final_file = review_result['output_file']
            else:
                final_file = review_input
            
            # 步骤6: 质量分析
            if self.pipeline_config['enable_quality_analysis']:
                self.pipeline_state['current_step'] = 6
                quality_result = self.step_6_quality_analysis(final_file, pipeline_output_dir)
            
            # 完成流水线
            self.pipeline_state['end_time'] = datetime.now().isoformat()
            self.pipeline_state['status'] = 'completed'
            
            # 生成流水线报告
            self.generate_pipeline_report(pipeline_output_dir)
            
            print("\n🎉 流水线执行完成！")
            print(f"📁 结果保存在: {pipeline_output_dir}")
            
            return self.pipeline_state
            
        except Exception as e:
            self.pipeline_state['status'] = 'failed'
            self.pipeline_state['end_time'] = datetime.now().isoformat()
            self.pipeline_state['errors'].append(f"流水线执行失败: {e}")
            
            print(f"\n❌ 流水线执行失败: {e}")
            print(f"📁 部分结果保存在: {pipeline_output_dir}")
            
            # 生成错误报告
            self.generate_pipeline_report(pipeline_output_dir)
            
            return self.pipeline_state
    
    def run_template_only(self, input_file: str, output_dir: Path) -> Dict:
        """仅运行模板化步骤"""
        print("🚀 启动模板化流水线")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        pipeline_output_dir = output_dir / f"template_only_{timestamp}"
        pipeline_output_dir.mkdir(exist_ok=True, parents=True)
        
        try:
            result = self.step_1_template_logs(input_file, pipeline_output_dir)
            if result:
                print(f"\n🎉 模板化完成！")
                print(f"📁 结果保存在: {pipeline_output_dir}")
                return result
            else:
                raise Exception("模板化失败")
                
        except Exception as e:
            print(f"\n❌ 模板化失败: {e}")
            return {}
    
    def run_batch_pipeline(self, input_dir: str, output_dir: Path) -> Dict:
        """批量运行流水线"""
        print("🚀 启动批量流水线")
        
        input_path = Path(input_dir)
        if not input_path.exists():
            print(f"❌ 输入目录不存在: {input_dir}")
            return {}
        
        # 查找日志文件
        log_extensions = ['.log', '.txt', '.csv', '.out', '.err']
        log_files = []
        
        for ext in log_extensions:
            log_files.extend(input_path.rglob(f"*{ext}"))
        
        if not log_files:
            print("❌ 未找到日志文件")
            return {}
        
        print(f"📁 找到 {len(log_files)} 个日志文件")
        
        # 创建批量输出目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        batch_output_dir = output_dir / f"batch_pipeline_{timestamp}"
        batch_output_dir.mkdir(exist_ok=True, parents=True)
        
        # 处理每个文件
        results = []
        for i, log_file in enumerate(log_files, 1):
            print(f"\n{'='*60}")
            print(f"处理进度: {i}/{len(log_files)} - {log_file.name}")
            print('='*60)
            
            try:
                result = self.run_full_pipeline(str(log_file), batch_output_dir)
                if result.get('status') == 'completed':
                    results.append({
                        'file': str(log_file),
                        'status': 'success',
                        'result': result
                    })
                else:
                    results.append({
                        'file': str(log_file),
                        'status': 'failed',
                        'errors': result.get('errors', [])
                    })
                    
            except Exception as e:
                results.append({
                    'file': str(log_file),
                    'status': 'error',
                    'error': str(e)
                })
        
        # 生成批量处理报告
        self.generate_batch_report(batch_output_dir, results)
        
        success_count = len([r for r in results if r['status'] == 'success'])
        print(f"\n🎉 批量处理完成！")
        print(f"📊 成功处理: {success_count}/{len(log_files)} 个文件")
        print(f"📁 结果保存在: {batch_output_dir}")
        
        return {
            'total_files': len(log_files),
            'success_count': success_count,
            'results': results,
            'output_dir': str(batch_output_dir)
        }
    
    def generate_pipeline_report(self, output_dir: Path):
        """生成流水线执行报告"""
        report_file = output_dir / "pipeline_execution_report.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("增强日志半自动分类流水线执行报告\n")
            f.write("=" * 60 + "\n")
            f.write(f"执行时间: {self.pipeline_state.get('start_time', 'N/A')} - {self.pipeline_state.get('end_time', 'N/A')}\n")
            f.write(f"输入文件: {self.pipeline_state.get('input_file', 'N/A')}\n")
            f.write(f"输出目录: {self.pipeline_state.get('output_dir', 'N/A')}\n")
            f.write(f"执行状态: {self.pipeline_state.get('status', 'N/A')}\n")
            f.write(f"当前步骤: {self.pipeline_state.get('current_step', 0)}/{self.pipeline_state.get('total_steps', 0)}\n\n")
            
            # 各步骤结果
            f.write("步骤执行结果:\n")
            f.write("-" * 30 + "\n")
            
            steps = [
                ('templating', '模板化'),
                ('feature_engineering', '特征工程'),
                ('pre_classification', '预分类'),
                ('auto_labeling', '自动标签'),
                ('human_review', '人工审查'),
                ('quality_analysis', '质量分析')
            ]
            
            for step_key, step_name in steps:
                if step_key in self.pipeline_state['results']:
                    result = self.pipeline_state['results'][step_key]
                    f.write(f"✅ {step_name}: 成功\n")
                    if isinstance(result, dict):
                        for key, value in result.items():
                            if key not in ['output_file', 'success']:
                                f.write(f"   {key}: {value}\n")
                else:
                    f.write(f"⏭️ {step_name}: 跳过\n")
            
            # 错误信息
            if self.pipeline_state['errors']:
                f.write(f"\n错误信息:\n")
                f.write("-" * 30 + "\n")
                for i, error in enumerate(self.pipeline_state['errors'], 1):
                    f.write(f"{i}. {error}\n")
            
            # 配置信息
            f.write(f"\n流水线配置:\n")
            f.write("-" * 30 + "\n")
            for key, value in self.pipeline_config.items():
                f.write(f"{key}: {value}\n")
        
        print(f"📄 流水线报告: {report_file}")
    
    def generate_batch_report(self, output_dir: Path, results: List[Dict]):
        """生成批量处理报告"""
        report_file = output_dir / "batch_processing_report.txt"
        
        success_count = len([r for r in results if r['status'] == 'success'])
        failed_count = len([r for r in results if r['status'] == 'failed'])
        error_count = len([r for r in results if r['status'] == 'error'])
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("批量流水线处理报告\n")
            f.write("=" * 50 + "\n")
            f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总文件数: {len(results)}\n")
            f.write(f"成功处理: {success_count}\n")
            f.write(f"处理失败: {failed_count}\n")
            f.write(f"执行错误: {error_count}\n\n")
            
            # 详细结果
            f.write("详细结果:\n")
            f.write("-" * 30 + "\n")
            for i, result in enumerate(results, 1):
                f.write(f"{i}. {Path(result['file']).name}\n")
                f.write(f"   状态: {result['status']}\n")
                if result['status'] == 'error':
                    f.write(f"   错误: {result['error']}\n")
                elif result['status'] == 'failed':
                    f.write(f"   错误: {', '.join(result['errors'])}\n")
                f.write("\n")
        
        print(f"📄 批量处理报告: {report_file}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='增强的日志半自动分类流水线')
    parser.add_argument('--input-file', help='输入日志文件路径')
    parser.add_argument('--input-dir', help='输入目录路径（批量模式）')
    parser.add_argument('--output-dir', help='输出目录路径')
    parser.add_argument('--mode', choices=['full', 'template-only', 'batch'], 
                       default='full', help='运行模式')
    parser.add_argument('--skip-human-review', action='store_true', 
                       help='跳过人工审查步骤')
    parser.add_argument('--skip-templating', action='store_true', 
                       help='跳过模板化步骤')
    parser.add_argument('--skip-feature-engineering', action='store_true', 
                       help='跳过特征工程步骤')
    parser.add_argument('--skip-ml', action='store_true', 
                       help='跳过机器学习分类')
    parser.add_argument('--skip-quality-analysis', action='store_true', 
                       help='跳过质量分析')
    
    args = parser.parse_args()
    
    # 创建流水线
    pipeline = EnhancedLogPipeline()
    
    # 配置流水线
    config = {
        'enable_human_review': not args.skip_human_review,
        'enable_templating': not args.skip_templating,
        'enable_feature_engineering': not args.skip_feature_engineering,
        'enable_ml_classification': not args.skip_ml,
        'enable_quality_analysis': not args.skip_quality_analysis
    }
    pipeline.configure_pipeline(config)
    
    # 设置输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = pipeline.output_base_dir / "enhanced_pipeline"
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 根据模式运行
    if args.mode == 'full':
        if not args.input_file:
            print("❌ 完整模式需要指定 --input-file")
            return
        
        if not Path(args.input_file).exists():
            print(f"❌ 输入文件不存在: {args.input_file}")
            return
        
        pipeline.run_full_pipeline(args.input_file, output_dir)
    
    elif args.mode == 'template-only':
        if not args.input_file:
            print("❌ 模板化模式需要指定 --input-file")
            return
        
        if not Path(args.input_file).exists():
            print(f"❌ 输入文件不存在: {args.input_file}")
            return
        
        pipeline.run_template_only(args.input_file, output_dir)
    
    elif args.mode == 'batch':
        if not args.input_dir:
            print("❌ 批量模式需要指定 --input-dir")
            return
        
        if not Path(args.input_dir).exists():
            print(f"❌ 输入目录不存在: {args.input_dir}")
            return
        
        pipeline.run_batch_pipeline(args.input_dir, output_dir)
    
    else:
        print("❌ 无效的运行模式")
        parser.print_help()

if __name__ == "__main__":
    main()
