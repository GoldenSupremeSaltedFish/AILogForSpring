#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日志标签审批脚本
支持逐条审查、进度保存和断点续传
"""

import pandas as pd
import json
import os
import sys
from pathlib import Path
from datetime import datetime
import argparse
from typing import Dict, List, Optional

class LogReviewer:
    """日志标签审批器"""
    
    def __init__(self, input_file: str, output_dir: str = None):
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir) if output_dir else self.input_file.parent
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # 进度文件
        self.progress_file = self.output_dir / f"{self.input_file.stem}_review_progress.json"
        
        # 输出文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_file = self.output_dir / f"{self.input_file.stem}_reviewed_{timestamp}.csv"
        
        # 加载数据
        self.df = pd.read_csv(self.input_file, encoding='utf-8-sig')
        self.total_count = len(self.df)
        
        # 审查状态
        self.current_index = 0
        self.reviewed_data = []
        self.stats = {
            'total': self.total_count,
            'reviewed': 0,
            'correct': 0,
            'modified': 0,
            'skipped': 0
        }
        
        # 标签选项
        self.label_options = {
            '1': 'auth_error',
            '2': 'db_error', 
            '3': 'timeout',
            '4': 'api_success',
            '5': 'ignore',
            '6': 'system_error',
            '7': 'other'
        }
        
        # 加载进度
        self.load_progress()
    
    def load_progress(self):
        """加载审查进度"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    progress = json.load(f)
                    self.current_index = progress.get('current_index', 0)
                    self.stats = progress.get('stats', self.stats)
                    self.reviewed_data = progress.get('reviewed_data', [])
                    
                print(f"📂 加载进度: 已审查 {self.current_index}/{self.total_count} 条")
            except Exception as e:
                print(f"⚠️  加载进度失败: {e}，从头开始")
    
    def save_progress(self):
        """保存审查进度"""
        progress = {
            'current_index': self.current_index,
            'stats': self.stats,
            'reviewed_data': self.reviewed_data,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress, f, ensure_ascii=False, indent=2)
    
    def display_log_entry(self, index: int) -> Dict:
        """显示日志条目"""
        row = self.df.iloc[index]
        
        print("\n" + "="*80)
        print(f"📋 审查进度: {index + 1}/{self.total_count} ({((index + 1)/self.total_count)*100:.1f}%)")
        print("="*80)
        
        print(f"🕐 时间戳: {row.get('timestamp', 'N/A')}")
        print(f"📝 日志级别: {row.get('level', 'N/A')}")
        print(f"💬 消息内容: {row.get('message', 'N/A')[:200]}{'...' if len(str(row.get('message', ''))) > 200 else ''}")
        print(f"🏷️  当前标签: {row.get('predicted_label', 'N/A')}")
        print(f"🎯 置信度: {row.get('confidence', 'N/A')}")
        print(f"📊 匹配规则: {row.get('rule_matched', 'N/A')}")
        
        return row.to_dict()
    
    def show_label_options(self):
        """显示标签选项"""
        print("\n🏷️  标签选项:")
        for key, label in self.label_options.items():
            print(f"  {key}. {label}")
    
    def get_user_input(self) -> tuple:
        """获取用户输入"""
        print("\n" + "-"*50)
        print("操作选项:")
        print("  [Enter] - 确认当前标签正确")
        print("  [1-7]  - 修改为对应标签")
        print("  [s]     - 跳过此条")
        print("  [q]     - 退出并保存")
        print("  [h]     - 显示帮助")
        print("-"*50)
        
        while True:
            user_input = input("请选择操作: ").strip().lower()
            
            if user_input == '':
                return 'confirm', None
            elif user_input in self.label_options:
                return 'modify', self.label_options[user_input]
            elif user_input == 's':
                return 'skip', None
            elif user_input == 'q':
                return 'quit', None
            elif user_input == 'h':
                self.show_help()
                continue
            else:
                print("❌ 无效输入，请重新选择")
    
    def show_help(self):
        """显示帮助信息"""
        print("\n📖 帮助信息:")
        print("- 按 Enter 确认当前标签正确")
        print("- 输入数字 1-7 修改标签:")
        self.show_label_options()
        print("- 输入 's' 跳过当前条目")
        print("- 输入 'q' 退出并保存进度")
        print("- 输入 'h' 显示此帮助")
    
    def process_entry(self, row_data: Dict, action: str, new_label: str = None):
        """处理审查结果"""
        reviewed_row = row_data.copy()
        
        if action == 'confirm':
            reviewed_row['review_status'] = 'confirmed'
            reviewed_row['final_label'] = row_data.get('predicted_label')
            self.stats['correct'] += 1
            print("✅ 标签确认正确")
            
        elif action == 'modify':
            reviewed_row['review_status'] = 'modified'
            reviewed_row['final_label'] = new_label
            reviewed_row['original_label'] = row_data.get('predicted_label')
            self.stats['modified'] += 1
            print(f"🔄 标签已修改: {row_data.get('predicted_label')} → {new_label}")
            
        elif action == 'skip':
            reviewed_row['review_status'] = 'skipped'
            reviewed_row['final_label'] = row_data.get('predicted_label')
            self.stats['skipped'] += 1
            print("⏭️  已跳过")
        
        # 添加审查信息
        reviewed_row['review_timestamp'] = datetime.now().isoformat()
        self.reviewed_data.append(reviewed_row)
        self.stats['reviewed'] += 1
    
    def show_stats(self):
        """显示统计信息"""
        print("\n" + "="*60)
        print("📊 审查统计")
        print("="*60)
        print(f"总计: {self.stats['total']} 条")
        print(f"已审查: {self.stats['reviewed']} 条 ({(self.stats['reviewed']/self.stats['total'])*100:.1f}%)")
        print(f"确认正确: {self.stats['correct']} 条")
        print(f"修改标签: {self.stats['modified']} 条")
        print(f"跳过: {self.stats['skipped']} 条")
        
        if self.stats['reviewed'] > 0:
            accuracy = (self.stats['correct'] / self.stats['reviewed']) * 100
            print(f"当前准确率: {accuracy:.1f}%")
    
    def save_results(self):
        """保存审查结果"""
        if not self.reviewed_data:
            print("⚠️  没有审查数据需要保存")
            return
        
        # 创建结果DataFrame
        result_df = pd.DataFrame(self.reviewed_data)
        
        # 保存到CSV
        result_df.to_csv(self.output_file, index=False, encoding='utf-8-sig')
        print(f"💾 审查结果已保存: {self.output_file}")
        
        # 生成摘要报告
        summary_file = self.output_file.with_suffix('.txt')
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("日志标签审查报告\n")
            f.write("=" * 50 + "\n")
            f.write(f"审查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"输入文件: {self.input_file}\n")
            f.write(f"输出文件: {self.output_file}\n\n")
            
            f.write("统计信息:\n")
            f.write(f"总计: {self.stats['total']} 条\n")
            f.write(f"已审查: {self.stats['reviewed']} 条\n")
            f.write(f"确认正确: {self.stats['correct']} 条\n")
            f.write(f"修改标签: {self.stats['modified']} 条\n")
            f.write(f"跳过: {self.stats['skipped']} 条\n")
            
            if self.stats['reviewed'] > 0:
                accuracy = (self.stats['correct'] / self.stats['reviewed']) * 100
                f.write(f"准确率: {accuracy:.1f}%\n")
        
        print(f"📄 摘要报告已保存: {summary_file}")
    
    def run(self):
        """运行审查流程"""
        print("🚀 日志标签审查工具启动")
        print(f"📁 输入文件: {self.input_file}")
        print(f"📊 总计: {self.total_count} 条日志")
        
        if self.current_index > 0:
            print(f"📂 从第 {self.current_index + 1} 条开始继续审查")
        
        try:
            while self.current_index < self.total_count:
                # 显示当前日志
                row_data = self.display_log_entry(self.current_index)
                
                # 获取用户操作
                action, new_label = self.get_user_input()
                
                if action == 'quit':
                    print("\n🛑 用户选择退出")
                    break
                
                # 处理审查结果
                self.process_entry(row_data, action, new_label)
                
                # 移动到下一条
                self.current_index += 1
                
                # 定期保存进度
                if self.current_index % 10 == 0:
                    self.save_progress()
                    print(f"💾 进度已保存 ({self.current_index}/{self.total_count})")
            
            # 显示最终统计
            self.show_stats()
            
            # 保存结果
            self.save_results()
            
            # 清理进度文件（如果完成）
            if self.current_index >= self.total_count:
                if self.progress_file.exists():
                    self.progress_file.unlink()
                    print("🗑️  进度文件已清理")
                print("🎉 审查完成！")
            else:
                self.save_progress()
                print("💾 进度已保存，下次可继续")
                
        except KeyboardInterrupt:
            print("\n\n⚠️  检测到 Ctrl+C，正在保存进度...")
            self.save_progress()
            self.save_results()
            print("💾 进度和结果已保存")
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")
            self.save_progress()
            self.save_results()
            raise

def main():
    parser = argparse.ArgumentParser(description='日志标签审查工具')
    parser.add_argument('input_file', help='输入的已标注CSV文件路径')
    parser.add_argument('--output-dir', '-o', help='输出目录（默认为输入文件同目录）')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"❌ 输入文件不存在: {args.input_file}")
        sys.exit(1)
    
    reviewer = LogReviewer(args.input_file, args.output_dir)
    reviewer.run()

if __name__ == '__main__':
    main()