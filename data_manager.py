#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据管理工具
用于管理本地数据文件，确保数据安全且不上传到Git仓库
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

class DataManager:
    def __init__(self):
        self.data_dirs = {
            "raw": "data/raw",
            "processed": "data/processed", 
            "training": "data/training",
            "validation": "data/validation",
            "test": "data/test",
            "logs": "logs",
            "outputs": "outputs",
            "models": "models"
        }
        
        # 确保目录存在
        for dir_path in self.data_dirs.values():
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def organize_data(self, source_path, data_type="raw"):
        """整理数据文件到指定目录"""
        if not os.path.exists(source_path):
            print(f"❌ 源路径不存在: {source_path}")
            return False
        
        target_dir = self.data_dirs.get(data_type, "data/raw")
        
        try:
            if os.path.isfile(source_path):
                filename = os.path.basename(source_path)
                target_path = os.path.join(target_dir, filename)
                
                if os.path.exists(target_path):
                    name, ext = os.path.splitext(filename)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{name}_{timestamp}{ext}"
                    target_path = os.path.join(target_dir, filename)
                
                shutil.copy2(source_path, target_path)
                print(f"✅ 已复制: {source_path} -> {target_path}")
                
            elif os.path.isdir(source_path):
                dirname = os.path.basename(source_path)
                target_path = os.path.join(target_dir, dirname)
                
                if os.path.exists(target_path):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    dirname = f"{dirname}_{timestamp}"
                    target_path = os.path.join(target_dir, dirname)
                
                shutil.copytree(source_path, target_path)
                print(f"✅ 已复制目录: {source_path} -> {target_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ 移动文件失败: {e}")
            return False
    
    def list_data_files(self, data_type=None):
        """列出数据文件"""
        print(f"📁 数据文件列表:")
        print("=" * 60)
        
        if data_type:
            dirs_to_show = {data_type: self.data_dirs[data_type]}
        else:
            dirs_to_show = self.data_dirs
        
        total_files = 0
        
        for dir_name, dir_path in dirs_to_show.items():
            if os.path.exists(dir_path):
                print(f"\n📂 {dir_name} ({dir_path}):")
                
                files = list(Path(dir_path).rglob("*"))
                dir_files = [f for f in files if f.is_file()]
                
                if not dir_files:
                    print("  (空目录)")
                    continue
                
                for file_path in dir_files:
                    rel_path = file_path.relative_to(dir_path)
                    print(f"  📄 {rel_path}")
                    total_files += 1
        
        print(f"\n📊 总计: {total_files} 个文件")
    
    def check_git_ignore(self):
        """检查数据文件是否被正确忽略"""
        print("🔍 检查数据文件Git忽略状态...")
        
        if not os.path.exists('.gitignore'):
            print("❌ .gitignore 文件不存在")
            return False
        
        with open('.gitignore', 'r', encoding='utf-8') as f:
            gitignore_content = f.read()
        
        data_patterns = ["*.csv", "*.xlsx", "data/", "logs/", "outputs/", "models/"]
        
        for pattern in data_patterns:
            if pattern in gitignore_content:
                print(f"✅ {pattern}")
            else:
                print(f"⚠️  {pattern} (未忽略)")

def main():
    """主函数"""
    import sys
    
    manager = DataManager()
    
    if len(sys.argv) < 2:
        print("用法: python data_manager.py <command> [options]")
        print("命令:")
        print("  organize <source> [type] - 整理数据文件")
        print("  list [type]              - 列出数据文件")
        print("  check                    - 检查Git忽略状态")
        return
    
    command = sys.argv[1]
    
    if command == "organize" and len(sys.argv) >= 3:
        source = sys.argv[2]
        data_type = sys.argv[3] if len(sys.argv) > 3 else "raw"
        manager.organize_data(source, data_type)
    
    elif command == "list":
        data_type = sys.argv[2] if len(sys.argv) > 2 else None
        manager.list_data_files(data_type)
    
    elif command == "check":
        manager.check_git_ignore()
    
    else:
        print("无效的命令")

if __name__ == "__main__":
    main()
