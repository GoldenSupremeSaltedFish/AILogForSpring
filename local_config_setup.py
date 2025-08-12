#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地配置设置工具
用于创建和管理本地配置文件，确保敏感信息不上传到Git仓库
"""

import os
import json
import sys
from pathlib import Path

class LocalConfigSetup:
    def __init__(self):
        self.template_files = {
            "config_extended.json": "config_template.json",
            "config_test.json": "config_template.json",
            "config_local.json": "config_template.json"
        }
    
    def create_local_configs(self):
        """创建本地配置文件"""
        print("🔧 创建本地配置文件...")
        
        # 检查环境变量
        github_token = os.getenv('GITHUB_TOKEN')
        if not github_token:
            print("⚠️  警告: 环境变量 GITHUB_TOKEN 未设置")
            print("请先设置环境变量:")
            print("  Windows: set GITHUB_TOKEN=your_token_here")
            print("  Linux/Mac: export GITHUB_TOKEN=your_token_here")
            print("\n或者稍后在配置文件中手动设置token")
        
        # 创建配置文件
        for config_file, template_file in self.template_files.items():
            if self.create_config_from_template(config_file, template_file, github_token):
                print(f"✅ 已创建: {config_file}")
            else:
                print(f"❌ 创建失败: {config_file}")
        
        print("\n📝 配置文件说明:")
        print("- 所有配置文件已添加到 .gitignore，不会上传到Git仓库")
        print("- 请根据需要在配置文件中设置真实的GitHub token")
        print("- 建议使用环境变量管理敏感信息")
    
    def create_config_from_template(self, output_file, template_file, github_token=None):
        """从模板创建配置文件"""
        try:
            # 读取模板
            if os.path.exists(template_file):
                with open(template_file, 'r', encoding='utf-8') as f:
                    template = json.load(f)
            else:
                # 如果模板不存在，创建基本模板
                template = self.create_basic_template()
            
            # 设置token
            if github_token:
                template["github_token"] = github_token
            else:
                template["github_token"] = "YOUR_GITHUB_TOKEN_HERE"
            
            # 根据文件名调整配置
            if "test" in output_file:
                template["max_pages"] = 3
                template["repositories"] = template["repositories"][:3]
            elif "local" in output_file:
                template["max_pages"] = 10
            
            # 写入配置文件
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(template, f, indent=4, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            print(f"错误: {e}")
            return False
    
    def create_basic_template(self):
        """创建基本配置模板"""
        return {
            "github_token": "YOUR_GITHUB_TOKEN_HERE",
            "repositories": [
                "spring-projects/spring-boot",
                "spring-projects/spring-security",
                "macrozheng/mall",
                "alibaba/spring-cloud-alibaba"
            ],
            "output_directory": "./issue-logs",
            "max_pages": 20,
            "categories": {
                "core_framework": ["spring-projects/spring-boot"],
                "security": ["spring-projects/spring-security"],
                "ecommerce": ["macrozheng/mall"],
                "microservices": ["alibaba/spring-cloud-alibaba"]
            },
            "log_keywords": [
                "Exception", "Caused by", "at ", "ERROR", "WARN", "INFO", "DEBUG",
                "TRACE", "FATAL", "Stack trace", "java.lang.", "org.springframework.",
                "Stacktrace:", "Error:", "Failed to", "Cannot", "Unable to",
                "NullPointerException", "IllegalArgumentException", "RuntimeException",
                "SQLException", "ConnectionException", "TimeoutException",
                "AuthenticationException", "AuthorizationException", "ValidationException",
                "TransactionException", "SerializationException", "DeserializationException"
            ]
        }
    
    def check_git_status(self):
        """检查Git状态，确保配置文件不会被提交"""
        print("\n🔍 检查Git状态...")
        
        # 检查配置文件是否在.gitignore中
        gitignore_content = ""
        if os.path.exists('.gitignore'):
            with open('.gitignore', 'r', encoding='utf-8') as f:
                gitignore_content = f.read()
        
        config_files = ["config_extended.json", "config_test.json", "config_local.json"]
        ignored_files = []
        
        for config_file in config_files:
            if os.path.exists(config_file):
                if config_file in gitignore_content:
                    print(f"✅ {config_file} 已在 .gitignore 中")
                    ignored_files.append(config_file)
                else:
                    print(f"⚠️  {config_file} 不在 .gitignore 中")
        
        if ignored_files:
            print(f"\n📋 已忽略的配置文件: {', '.join(ignored_files)}")
            print("这些文件不会被提交到Git仓库")
        
        return len(ignored_files) > 0
    
    def create_data_directories(self):
        """创建数据目录结构"""
        print("\n📁 创建数据目录结构...")
        
        directories = [
            "data",
            "data/raw",
            "data/processed", 
            "data/training",
            "data/validation",
            "data/test",
            "logs",
            "outputs",
            "models"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"✅ 已创建: {directory}/")
        
        # 创建README文件说明数据目录
        readme_content = """# 数据目录说明

此目录包含项目的数据文件，已添加到 .gitignore 中，不会上传到Git仓库。

## 目录结构
- `data/raw/` - 原始数据文件
- `data/processed/` - 处理后的数据文件  
- `data/training/` - 训练数据
- `data/validation/` - 验证数据
- `data/test/` - 测试数据
- `logs/` - 日志文件
- `outputs/` - 输出文件
- `models/` - 模型文件

## 注意事项
- 所有CSV、Excel等数据文件都不会上传到Git仓库
- 请将敏感数据文件放在这些目录中
- 建议使用相对路径引用数据文件
"""
        
        with open('data/README.md', 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        print("✅ 已创建: data/README.md")

def main():
    """主函数"""
    setup = LocalConfigSetup()
    
    print("🚀 本地配置设置工具")
    print("=" * 50)
    
    # 创建本地配置文件
    setup.create_local_configs()
    
    # 检查Git状态
    setup.check_git_status()
    
    # 创建数据目录
    setup.create_data_directories()
    
    print("\n" + "=" * 50)
    print("✅ 本地配置设置完成!")
    print("\n📋 下一步操作:")
    print("1. 设置环境变量 GITHUB_TOKEN")
    print("2. 根据需要修改配置文件")
    print("3. 将数据文件放入相应目录")
    print("4. 运行你的程序")

if __name__ == "__main__":
    main()
