#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intel Arc GPU 检测器
"""

import torch
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class ArcGPUDetector:
    """Intel Arc GPU 检测器"""
    
    @staticmethod
    def check_arc_gpu() -> bool:
        """
        检查Intel Arc GPU是否可用
        Returns:
            bool: GPU是否可用
        """
        try:
            import intel_extension_for_pytorch as ipex
            if torch.xpu.is_available():
                device_count = torch.xpu.device_count()
                device_name = torch.xpu.get_device_name(0)
                logger.info(f"✅ 检测到Intel GPU: {device_name}")
                logger.info(f"   GPU数量: {device_count}")
                return True
            else:
                logger.warning("⚠️ 未检测到Intel XPU设备")
                return False
        except ImportError:
            logger.error("❌ Intel Extension for PyTorch未安装")
            return False
        except Exception as e:
            logger.error(f"❌ GPU检测失败: {e}")
            return False
    
    @staticmethod
    def get_device() -> torch.device:
        """
        获取最佳计算设备
        Returns:
            torch.device: 计算设备
        """
        if ArcGPUDetector.check_arc_gpu():
            return torch.device("xpu:0")
        else:
            return torch.device("cpu")
    
    @staticmethod
    def get_gpu_info() -> Dict[str, any]:
        """
        获取GPU信息
        Returns:
            Dict: GPU信息字典
        """
        info = {
            'available': False,
            'device_name': None,
            'device_count': 0,
            'memory_info': {}
        }
        
        try:
            import intel_extension_for_pytorch as ipex
            if torch.xpu.is_available():
                info['available'] = True
                info['device_name'] = torch.xpu.get_device_name(0)
                info['device_count'] = torch.xpu.device_count()
                
                # 获取内存信息
                for i in range(torch.xpu.device_count()):
                    props = torch.xpu.get_device_properties(i)
                    info['memory_info'][f'device_{i}'] = {
                        'total_memory': props.total_memory,
                        'name': props.name
                    }
        except Exception as e:
            logger.warning(f"获取GPU信息失败: {e}")
        
        return info
    
    @staticmethod
    def print_gpu_status():
        """打印GPU状态信息"""
        info = ArcGPUDetector.get_gpu_info()
        
        print("🖥️ GPU状态信息")
        print("=" * 50)
        
        if info['available']:
            print(f"✅ GPU可用: {info['device_name']}")
            print(f"📊 GPU数量: {info['device_count']}")
            
            for device_id, memory_info in info['memory_info'].items():
                total_gb = memory_info['total_memory'] / (1024**3)
                print(f"💾 {device_id}: {memory_info['name']} ({total_gb:.1f} GB)")
        else:
            print("❌ GPU不可用")
            print("💡 建议安装Intel Extension for PyTorch")
        
        print("=" * 50) 