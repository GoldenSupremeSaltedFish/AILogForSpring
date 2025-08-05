#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intel Arc GPU 检测脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import ArcGPUDetector


def main():
    """主函数"""
    print("🔍 Intel Arc GPU 检测工具")
    print("=" * 50)
    
    # 检测GPU状态
    ArcGPUDetector.print_gpu_status()
    
    # 获取详细信息
    info = ArcGPUDetector.get_gpu_info()
    
    if info['available']:
        print("\n📊 详细信息:")
        print(f"   设备名称: {info['device_name']}")
        print(f"   设备数量: {info['device_count']}")
        
        for device_id, memory_info in info['memory_info'].items():
            total_gb = memory_info['total_memory'] / (1024**3)
            print(f"   {device_id}: {memory_info['name']} ({total_gb:.1f} GB)")
        
        print("\n✅ Intel Arc GPU 可用，可以开始训练！")
    else:
        print("\n❌ Intel Arc GPU 不可用")
        print("💡 请检查以下项目:")
        print("   1. 是否安装了Intel Extension for PyTorch")
        print("   2. 是否安装了Intel GPU驱动")
        print("   3. 系统是否识别到Intel Arc GPU")


if __name__ == "__main__":
    main() 