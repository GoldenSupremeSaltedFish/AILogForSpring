#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
内存监控脚本
监控Intel Arc GPU内存使用情况
"""

import torch
import psutil
import time
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_gpu_memory_info():
    """获取GPU内存信息"""
    try:
        if torch.xpu.is_available():
            # Intel XPU内存信息
            memory_allocated = torch.xpu.memory_allocated(0)
            memory_reserved = torch.xpu.memory_reserved(0)
            memory_total = torch.xpu.get_device_properties(0).total_memory
            
            return {
                'allocated_mb': memory_allocated / (1024**2),
                'reserved_mb': memory_reserved / (1024**2),
                'total_gb': memory_total / (1024**3),
                'utilization_percent': (memory_allocated / memory_total) * 100
            }
        else:
            return None
    except Exception as e:
        logger.warning(f"获取GPU内存信息失败: {e}")
        return None


def get_system_memory_info():
    """获取系统内存信息"""
    memory = psutil.virtual_memory()
    return {
        'total_gb': memory.total / (1024**3),
        'available_gb': memory.available / (1024**3),
        'used_gb': memory.used / (1024**3),
        'utilization_percent': memory.percent
    }


def monitor_memory(duration_seconds=60, interval_seconds=5):
    """监控内存使用情况"""
    logger.info(f"🔍 开始内存监控 ({duration_seconds}秒)")
    logger.info("=" * 60)
    
    start_time = time.time()
    iteration = 0
    
    while time.time() - start_time < duration_seconds:
        iteration += 1
        current_time = datetime.now().strftime("%H:%M:%S")
        
        print(f"\n[{current_time}] 迭代 {iteration}")
        print("-" * 40)
        
        # GPU内存信息
        gpu_info = get_gpu_memory_info()
        if gpu_info:
            print(f"🖥️  GPU内存:")
            print(f"   已分配: {gpu_info['allocated_mb']:.1f} MB")
            print(f"   已保留: {gpu_info['reserved_mb']:.1f} MB")
            print(f"   总内存: {gpu_info['total_gb']:.1f} GB")
            print(f"   使用率: {gpu_info['utilization_percent']:.1f}%")
        else:
            print("❌ GPU内存信息不可用")
        
        # 系统内存信息
        sys_info = get_system_memory_info()
        print(f"💻 系统内存:")
        print(f"   总内存: {sys_info['total_gb']:.1f} GB")
        print(f"   可用内存: {sys_info['available_gb']:.1f} GB")
        print(f"   已使用: {sys_info['used_gb']:.1f} GB")
        print(f"   使用率: {sys_info['utilization_percent']:.1f}%")
        
        # 清理GPU缓存
        if torch.xpu.is_available():
            torch.xpu.empty_cache()
        
        time.sleep(interval_seconds)
    
    logger.info("✅ 内存监控完成")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="内存监控工具")
    parser.add_argument("--duration", type=int, default=60, help="监控时长（秒）")
    parser.add_argument("--interval", type=int, default=5, help="监控间隔（秒）")
    
    args = parser.parse_args()
    
    logger.info("🎯 Intel Arc GPU 内存监控工具")
    
    try:
        monitor_memory(args.duration, args.interval)
    except KeyboardInterrupt:
        logger.info("⏹️ 监控被用户中断")
    except Exception as e:
        logger.error(f"❌ 监控失败: {e}")


if __name__ == "__main__":
    main() 