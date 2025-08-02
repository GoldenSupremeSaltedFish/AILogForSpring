#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分类结果整理脚本
功能：将DATA_OUTPUT目录中的分类结果按错误类型归类到子文件夹
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
import re

class ClassifiedResultsOrganizer:
    """分类结果整理器"""
    
    def __init__(self):
        self.base_dir = Path("DATA_OUTPUT")
        
        # 定义类别映射和优先级
        self.category_mapping = {
            'stack_exception': {'name': '01_堆栈异常_stack_exception', 'desc': '堆栈异常'},
            'database_exception': {'name': '02_数据库异常_database_exception', 'desc': '数据库异常'},
            'connection_issue': {'name': '03_连接问题_connection_issue', 'desc': '连接问题'},
            'auth_authorization': {'name': '04_认证授权_auth_authorization', 'desc': '认证授权'},
            'config_environment': {'name': '05_配置环境_config_environment', 'desc': '配置环境'},
            'business_logic': {'name': '06_业务逻辑_business_logic', 'desc': '业务逻辑'},
            'normal_operation': {'name': '07_正常操作_normal_operation', 'desc': '正常操作'},
            'monitoring_heartbeat': {'name': '08_监控心跳_monitoring_heartbeat', 'desc': '监控心跳'},
            'memory_performance': {'name': '09_内存性能_memory_performance', 'desc': '内存性能'},
            'timeout': {'name': '10_超时错误_timeout', 'desc': '超时错误'},
            'spring_boot_startup_failure': {'name': '11_SpringBoot启动失败_spring_boot_startup_failure', 'desc': 'Spring Boot启动失败'}
        }
        
        # 特殊目录
        self.special_dirs = {
            'categorized': '主分类文件_categorized',
            'summaries': '统计报告_summaries',
            'original': '原始项目数据_original'
        }
    
    def create_directory_structure(self):
        """创建目录结构"""
        print("🏗️ 创建目录结构...")
        
        # 创建类别目录
        for category, info in self.category_mapping.items():
            category_dir = self.base_dir / info['name']
            category_dir.mkdir(exist_ok=True)
            print(f"  ✓ {info['desc']}: {category_dir.name}")
        
        # 创建特殊目录
        for key, dir_name in self.special_dirs.items():
            special_dir = self.base_dir / dir_name
            special_dir.mkdir(exist_ok=True)
            print(f"  ✓ {dir_name}")
    
    def organize_category_files(self):
        """整理按类别分组的文件"""
        print("\n📁 整理类别文件...")
        
        for category, info in self.category_mapping.items():
            # 查找该类别的文件
            pattern = f"{category}_*.csv"
            category_files = list(self.base_dir.glob(pattern))
            
            if category_files:
                target_dir = self.base_dir / info['name']
                print(f"\n  📂 {info['desc']} ({len(category_files)} 个文件):")
                
                for file in category_files:
                    target_path = target_dir / file.name
                    if not target_path.exists():
                        shutil.move(str(file), str(target_path))
                        print(f"    ✓ 移动: {file.name}")
                    else:
                        print(f"    ⚠️ 已存在: {file.name}")
    
    def organize_categorized_files(self):
        """整理主分类文件"""
        print("\n📋 整理主分类文件...")
        
        # 查找categorized文件
        categorized_files = list(self.base_dir.glob("*_categorized_*.csv"))
        target_dir = self.base_dir / self.special_dirs['categorized']
        
        if categorized_files:
            print(f"  找到 {len(categorized_files)} 个主分类文件:")
            for file in categorized_files:
                target_path = target_dir / file.name
                if not target_path.exists():
                    shutil.move(str(file), str(target_path))
                    print(f"    ✓ 移动: {file.name}")
                else:
                    print(f"    ⚠️ 已存在: {file.name}")
    
    def organize_summary_files(self):
        """整理统计摘要文件"""
        print("\n📊 整理统计摘要文件...")
        
        # 查找summary文件
        summary_files = list(self.base_dir.glob("*_summary.txt"))
        target_dir = self.base_dir / self.special_dirs['summaries']
        
        if summary_files:
            print(f"  找到 {len(summary_files)} 个统计文件:")
            for file in summary_files:
                target_path = target_dir / file.name
                if not target_path.exists():
                    shutil.move(str(file), str(target_path))
                    print(f"    ✓ 移动: {file.name}")
                else:
                    print(f"    ⚠️ 已存在: {file.name}")
    
    def organize_original_project_dirs(self):
        """整理原始项目目录"""
        print("\n🗂️ 整理原始项目目录...")
        
        project_dirs = ['apache-camel', 'jhipster', 'spring-boot', 'spring-cloud', 'spring-security']
        target_dir = self.base_dir / self.special_dirs['original']
        
        for project in project_dirs:
            project_path = self.base_dir / project
            if project_path.exists() and project_path.is_dir():
                target_path = target_dir / project
                if not target_path.exists():
                    shutil.move(str(project_path), str(target_path))
                    print(f"    ✓ 移动目录: {project}")
                else:
                    print(f"    ⚠️ 目录已存在: {project}")
    
    def organize_quality_analysis(self):
        """整理质量分析结果"""
        print("\n🔍 整理质量分析结果...")
        
        quality_dir = self.base_dir / "质量分析结果"
        if quality_dir.exists():
            target_dir = self.base_dir / self.special_dirs['original'] / "质量分析结果"
            if not target_dir.exists():
                shutil.move(str(quality_dir), str(target_dir))
                print(f"    ✓ 移动目录: 质量分析结果")
            else:
                print(f"    ⚠️ 目录已存在: 质量分析结果")
    
    def generate_organization_report(self):
        """生成整理报告"""
        print("\n📋 生成整理报告...")
        
        report_lines = []
        report_lines.append("📁 分类结果整理报告")
        report_lines.append("=" * 50)
        report_lines.append(f"整理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # 统计各目录的文件数量
        report_lines.append("📊 目录统计:")
        
        for category, info in self.category_mapping.items():
            category_dir = self.base_dir / info['name']
            if category_dir.exists():
                file_count = len(list(category_dir.glob("*.csv")))
                report_lines.append(f"  {info['desc']}: {file_count} 个文件")
        
        for key, dir_name in self.special_dirs.items():
            special_dir = self.base_dir / dir_name
            if special_dir.exists():
                if key == 'original':
                    subdir_count = len([d for d in special_dir.iterdir() if d.is_dir()])
                    report_lines.append(f"  {dir_name}: {subdir_count} 个项目目录")
                else:
                    file_count = len(list(special_dir.glob("*.*")))
                    report_lines.append(f"  {dir_name}: {file_count} 个文件")
        
        report_lines.append("")
        report_lines.append("✅ 整理完成！")
        report_lines.append("=" * 50)
        
        # 保存报告
        report_file = self.base_dir / f"整理报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"📄 整理报告已保存: {report_file.name}")
        
        # 打印报告内容
        print("\n" + "\n".join(report_lines))
    
    def organize_all(self):
        """执行完整的整理流程"""
        print("🚀 开始整理分类结果...")
        print("=" * 60)
        
        # 1. 创建目录结构
        self.create_directory_structure()
        
        # 2. 整理各类文件
        self.organize_category_files()
        self.organize_categorized_files()
        self.organize_summary_files()
        self.organize_original_project_dirs()
        self.organize_quality_analysis()
        
        # 3. 生成报告
        self.generate_organization_report()

def main():
    """主函数"""
    organizer = ClassifiedResultsOrganizer()
    organizer.organize_all()

if __name__ == "__main__":
    main()