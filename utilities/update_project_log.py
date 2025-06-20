#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目日志更新脚本
方便每天更新README中的项目进度记录
"""

import re
from datetime import datetime
from pathlib import Path

def get_current_date():
    """获取当前日期"""
    return datetime.now().strftime("%Y-%m-%d")

def update_project_log():
    """更新项目日志"""
    readme_path = Path("README.md")
    
    if not readme_path.exists():
        print("❌ 未找到README.md文件")
        return
    
    current_date = get_current_date()
    
    print(f"📝 更新项目日志 - {current_date}")
    print("=" * 50)
    
    # 获取用户输入
    print("请输入今日完成的工作 (多行输入，空行结束):")
    completed_tasks = []
    while True:
        task = input("- ")
        if not task.strip():
            break
        completed_tasks.append(f"- ✅ {task}")
    
    print("\n请输入技术亮点或重要发现 (多行输入，空行结束):")
    tech_highlights = []
    while True:
        highlight = input("- ")
        if not highlight.strip():
            break
        tech_highlights.append(f"- {highlight}")
    
    print("\n请输入数据统计或处理成果 (多行输入，空行结束):")
    data_results = []
    while True:
        result = input("- ")
        if not result.strip():
            break
        data_results.append(f"- {result}")
    
    milestone = input("\n里程碑或重要成果 (可选): ")
    next_plan = input("明日计划 (可选): ")
    
    # 生成新的日志条目
    new_log_entry = generate_log_entry(
        current_date, completed_tasks, tech_highlights, 
        data_results, milestone, next_plan
    )
    
    # 更新README文件
    update_readme(readme_path, new_log_entry, current_date)
    
    print(f"\n✅ 项目日志已更新！")
    print(f"📅 日期: {current_date}")
    print(f"📁 文件: {readme_path}")

def generate_log_entry(date, tasks, highlights, results, milestone, next_plan):
    """生成日志条目"""
    entry = f"""### 📅 {date} (今日更新)

**🎯 今日完成**:
{chr(10).join(tasks) if tasks else "- 无重大更新"}
"""
    
    if results:
        entry += f"""
**📊 数据处理成果**:
{chr(10).join(results)}
"""
    
    if highlights:
        entry += f"""
**🔧 技术亮点**:
{chr(10).join(highlights)}
"""
    
    if milestone:
        entry += f"""
**🎉 里程碑**: {milestone}
"""
    
    if next_plan:
        entry += f"""
**📋 明日计划**: {next_plan}
"""
    
    return entry

def update_readme(readme_path, new_entry, current_date):
    """更新README文件"""
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找项目日志部分
    log_pattern = r'(## 项目开发日志\s*\n\s*)(### 📅 \d{4}-\d{2}-\d{2} \(今日更新\).*?)(\n---\s*\n### 📅 项目历史记录)'
    
    match = re.search(log_pattern, content, re.DOTALL)
    
    if match:
        # 将旧的"今日更新"移动到历史记录
        old_entry = match.group(2)
        
        # 提取旧条目的日期和内容
        old_date_match = re.search(r'### 📅 (\d{4}-\d{2}-\d{2}) \(今日更新\)', old_entry)
        if old_date_match:
            old_date = old_date_match.group(1)
            # 清理旧条目，移除"今日更新"标记
            historical_entry = old_entry.replace(f"{old_date} (今日更新)", old_date)
            
            # 更新历史记录部分
            history_pattern = r'(### 📅 项目历史记录\s*\n\s*)'
            history_replacement = f'\\1\n{historical_entry}\n\n---\n'
            content = re.sub(history_pattern, history_replacement, content)
        
        # 替换为新的日志条目
        replacement = f'\\1{new_entry}\\3'
        content = re.sub(log_pattern, replacement, content, flags=re.DOTALL)
    else:
        print("⚠️ 未找到项目日志部分，无法更新")
        return
    
    # 保存更新后的内容
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(content)

def quick_update():
    """快速更新模式"""
    print("🚀 快速更新模式")
    print("请选择预设的更新类型:")
    print("1. 代码开发")
    print("2. 数据处理")
    print("3. 测试调试")
    print("4. 文档更新")
    print("5. 架构设计")
    print("6. 自定义")
    
    choice = input("选择 (1-6): ")
    
    presets = {
        "1": {
            "tasks": ["完成核心模块开发", "修复已知bug", "代码重构优化"],
            "highlights": ["改进算法性能", "增强代码可读性"],
            "results": ["代码质量提升", "功能稳定性增强"]
        },
        "2": {
            "tasks": ["处理新的数据集", "优化数据清洗流程", "完成数据验证"],
            "highlights": ["数据处理效率提升", "新的清洗规则"],
            "results": ["数据质量改善", "处理速度提升"]
        },
        "3": {
            "tasks": ["完成单元测试", "修复测试中发现的问题", "性能测试"],
            "highlights": ["测试覆盖率提升", "性能瓶颈识别"],
            "results": ["代码稳定性提升", "性能优化"]
        },
        "4": {
            "tasks": ["更新项目文档", "完善API文档", "添加使用示例"],
            "highlights": ["文档结构优化", "示例代码完善"],
            "results": ["文档完整性提升", "用户体验改善"]
        },
        "5": {
            "tasks": ["完成架构设计", "模块接口定义", "技术选型"],
            "highlights": ["架构灵活性提升", "模块解耦"],
            "results": ["系统可扩展性增强", "维护成本降低"]
        }
    }
    
    if choice in presets:
        preset = presets[choice]
        current_date = get_current_date()
        
        new_entry = generate_log_entry(
            current_date,
            preset["tasks"],
            preset["highlights"], 
            preset["results"],
            "阶段性进展顺利",
            "继续推进下一阶段开发"
        )
        
        readme_path = Path("README.md")
        update_readme(readme_path, new_entry, current_date)
        print(f"✅ 快速更新完成 - {current_date}")
    elif choice == "6":
        update_project_log()
    else:
        print("❌ 无效选择")

def main():
    """主函数"""
    print("📋 项目日志更新工具")
    print("=" * 30)
    print("1. 详细更新")
    print("2. 快速更新")
    
    mode = input("选择模式 (1-2): ")
    
    if mode == "1":
        update_project_log()
    elif mode == "2":
        quick_update()
    else:
        print("❌ 无效选择")

if __name__ == "__main__":
    main() 