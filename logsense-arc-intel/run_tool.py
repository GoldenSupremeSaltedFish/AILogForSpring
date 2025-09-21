#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工具脚本兼容性运行器
为了保持向后兼容性，提供从根目录直接运行工具脚本的能力
"""

import sys
import os
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    parser = argparse.ArgumentParser(
        description="工具脚本兼容性运行器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python run_tool.py adapt_issue_data
  python run_tool.py check_model
  python run_tool.py improved_data_processor
  python run_tool.py prepare_issue_data
  python run_tool.py simple_text_validator
  python run_tool.py validation_data_adapter
  python run_tool.py fixed_model_runner
  python run_tool.py filter_known_labels
  python run_tool.py check_weights
        """
    )
    
    parser.add_argument(
        'tool_name',
        help='要运行的工具名称'
    )
    
    parser.add_argument(
        'args',
        nargs='*',
        help='传递给工具的参数'
    )
    
    args = parser.parse_args()
    
    # 工具脚本映射
    tool_mapping = {
        'adapt_issue_data': 'tools.adapt_issue_data',
        'check_model': 'tools.check_model',
        'improved_data_processor': 'tools.improved_data_processor',
        'prepare_issue_data': 'tools.prepare_issue_data',
        'simple_text_validator': 'tools.simple_text_validator',
        'validation_data_adapter': 'tools.validation_data_adapter',
        'fixed_model_runner': 'tools.fixed_model_runner',
        'filter_known_labels': 'tools.filter_known_labels',
        'check_weights': 'tools.check_weights'
    }
    
    if args.tool_name not in tool_mapping:
        print(f"❌ 未知的工具名称: {args.tool_name}")
        print(f"可用的工具: {', '.join(tool_mapping.keys())}")
        sys.exit(1)
    
    try:
        # 动态导入并运行工具
        module_name = tool_mapping[args.tool_name]
        module = __import__(module_name, fromlist=['main'])
        
        # 设置sys.argv以传递参数
        original_argv = sys.argv.copy()
        sys.argv = [args.tool_name] + args.args
        
        print(f"🔧 运行工具: {args.tool_name}")
        print(f"📁 模块路径: {module_name}")
        print("=" * 50)
        
        # 运行工具的main函数
        module.main()
        
        # 恢复原始argv
        sys.argv = original_argv
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 运行错误: {e}")
        sys.exit(1)
    
    print("✅ 工具运行完成!")

if __name__ == "__main__":
    main()
