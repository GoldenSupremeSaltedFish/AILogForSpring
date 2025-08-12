#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
安全配置管理器
用于安全地管理包含敏感信息的配置文件
"""

import os
import json
import sys
from pathlib import Path

class SecureConfigManager:
    def __init__(self, config_template_path="config_template.json"):
        self.config_template_path = config_template_path
        self.github_token_env_var = "GITHUB_TOKEN"
    
    def load_config(self, config_path):
        """安全地加载配置文件，从环境变量获取敏感信息"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 从环境变量获取GitHub token
            github_token = os.getenv(self.github_token_env_var)
            if not github_token:
                print(f"警告: 环境变量 {self.github_token_env_var} 未设置")
                print("请设置环境变量: set GITHUB_TOKEN=your_token_here (Windows)")
                print("或: export GITHUB_TOKEN=your_token_here (Linux/Mac)")
                return None
            
            # 替换配置中的占位符
            if config.get("github_token") == "YOUR_GITHUB_TOKEN_HERE":
                config["github_token"] = github_token
            
            return config
            
        except FileNotFoundError:
            print(f"错误: 配置文件 {config_path} 不存在")
            return None
        except json.JSONDecodeError as e:
            print(f"错误: 配置文件格式错误 - {e}")
            return None
    
    def create_secure_config(self, output_path, template_path=None):
        """从模板创建安全的配置文件"""
        template_path = template_path or self.config_template_path
        
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                template = json.load(f)
            
            # 检查环境变量
            github_token = os.getenv(self.github_token_env_var)
            if github_token:
                template["github_token"] = github_token
            else:
                print(f"警告: 环境变量 {self.github_token_env_var} 未设置，使用占位符")
            
            # 写入配置文件
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(template, f, indent=4, ensure_ascii=False)
            
            print(f"配置文件已创建: {output_path}")
            return True
            
        except Exception as e:
            print(f"错误: 创建配置文件失败 - {e}")
            return False
    
    def validate_config(self, config):
        """验证配置文件的完整性"""
        required_fields = ["github_token", "repositories", "output_directory"]
        
        for field in required_fields:
            if field not in config:
                print(f"错误: 缺少必需字段 '{field}'")
                return False
        
        if config["github_token"] == "YOUR_GITHUB_TOKEN_HERE":
            print("错误: GitHub token 未正确设置")
            return False
        
        return True

def main():
    """主函数"""
    manager = SecureConfigManager()
    
    if len(sys.argv) < 2:
        print("用法: python secure_config_manager.py <command> [options]")
        print("命令:")
        print("  load <config_path>     - 加载配置文件")
        print("  create <output_path>   - 创建安全配置文件")
        print("  validate <config_path> - 验证配置文件")
        return
    
    command = sys.argv[1]
    
    if command == "load" and len(sys.argv) >= 3:
        config_path = sys.argv[2]
        config = manager.load_config(config_path)
        if config:
            print("配置加载成功:")
            print(f"  GitHub Token: {'*' * 10}...{config['github_token'][-4:]}")
            print(f"  仓库数量: {len(config['repositories'])}")
            print(f"  输出目录: {config['output_directory']}")
    
    elif command == "create" and len(sys.argv) >= 3:
        output_path = sys.argv[2]
        manager.create_secure_config(output_path)
    
    elif command == "validate" and len(sys.argv) >= 3:
        config_path = sys.argv[2]
        config = manager.load_config(config_path)
        if config and manager.validate_config(config):
            print("配置文件验证通过")
        else:
            print("配置文件验证失败")
    
    else:
        print("无效的命令或参数")

if __name__ == "__main__":
    main()
