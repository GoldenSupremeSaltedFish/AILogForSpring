#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行去重任务
"""

import subprocess
import sys

def main():
    input_dir = r"data/output/processed_gateway_20250620_234918"
    cmd = [sys.executable, "deduplicate_logs.py", input_dir, "--mode", "both"]
    
    print("执行去重命令:", " ".join(cmd))
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        print(f"\n返回码: {result.returncode}")
    except Exception as e:
        print(f"执行失败: {e}")

if __name__ == "__main__":
    main() 