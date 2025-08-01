#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多平台计算支持工具
功能：
1. 检测和配置GPU/CPU环境
2. 自动选择最优计算设备
3. 内存和性能监控
4. 跨平台兼容性支持
"""

import os
import platform
import psutil
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class PlatformDetector:
    """平台检测器"""
    
    def __init__(self):
        self.system_info = self._get_system_info()
        self.gpu_info = self._detect_gpu()
        self.memory_info = self._get_memory_info()
        self.cpu_info = self._get_cpu_info()
    
    def _get_system_info(self) -> Dict:
        """获取系统信息"""
        return {
            'platform': platform.system(),
            'platform_version': platform.version(),
            'architecture': platform.architecture()[0],
            'processor': platform.processor(),
            'python_version': platform.python_version()
        }
    
    def _detect_gpu(self) -> Dict:
        """检测GPU信息"""
        gpu_info = {
            'available': False,
            'type': None,
            'name': None,
            'memory': None,
            'count': 0
        }
        
        # 检测NVIDIA GPU
        try:
            import torch
            if torch.cuda.is_available():
                gpu_info['available'] = True
                gpu_info['type'] = 'NVIDIA'
                gpu_info['name'] = torch.cuda.get_device_name(0)
                gpu_info['memory'] = torch.cuda.get_device_properties(0).total_memory
                gpu_info['count'] = torch.cuda.device_count()
                print(f"✅ 检测到NVIDIA GPU: {gpu_info['name']}")
                print(f"   GPU内存: {gpu_info['memory'] / 1024**3:.1f} GB")
                print(f"   GPU数量: {gpu_info['count']}")
        except ImportError:
            print("⚠️  PyTorch未安装，无法检测NVIDIA GPU")
        except Exception as e:
            print(f"⚠️  GPU检测失败: {e}")
        
        # 检测其他GPU类型
        if not gpu_info['available']:
            # 可以添加其他GPU检测逻辑
            pass
        
        return gpu_info
    
    def _get_memory_info(self) -> Dict:
        """获取内存信息"""
        memory = psutil.virtual_memory()
        return {
            'total': memory.total,
            'available': memory.available,
            'used': memory.used,
            'percent': memory.percent
        }
    
    def _get_cpu_info(self) -> Dict:
        """获取CPU信息"""
        return {
            'count': psutil.cpu_count(),
            'count_logical': psutil.cpu_count(logical=True),
            'freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            'usage': psutil.cpu_percent(interval=1)
        }
    
    def print_system_info(self):
        """打印系统信息"""
        print("🖥️ 系统信息")
        print("=" * 50)
        print(f"操作系统: {self.system_info['platform']} {self.system_info['platform_version']}")
        print(f"架构: {self.system_info['architecture']}")
        print(f"处理器: {self.system_info['processor']}")
        print(f"Python版本: {self.system_info['python_version']}")
        
        print(f"\n💾 内存信息")
        print(f"总内存: {self.memory_info['total'] / 1024**3:.1f} GB")
        print(f"可用内存: {self.memory_info['available'] / 1024**3:.1f} GB")
        print(f"内存使用率: {self.memory_info['percent']:.1f}%")
        
        print(f"\n🖥️ CPU信息")
        print(f"物理核心数: {self.cpu_info['count']}")
        print(f"逻辑核心数: {self.cpu_info['count_logical']}")
        print(f"CPU使用率: {self.cpu_info['usage']:.1f}%")
        
        if self.gpu_info['available']:
            print(f"\n🎮 GPU信息")
            print(f"类型: {self.gpu_info['type']}")
            print(f"名称: {self.gpu_info['name']}")
            print(f"GPU内存: {self.gpu_info['memory'] / 1024**3:.1f} GB")
            print(f"GPU数量: {self.gpu_info['count']}")
        else:
            print(f"\n⚠️  GPU不可用")
    
    def get_optimal_device(self) -> str:
        """获取最优计算设备"""
        if self.gpu_info['available']:
            # 检查GPU内存是否足够
            gpu_memory_gb = self.gpu_info['memory'] / 1024**3
            if gpu_memory_gb >= 4:  # 至少4GB GPU内存
                return 'gpu'
            else:
                print(f"⚠️  GPU内存不足 ({gpu_memory_gb:.1f} GB < 4 GB)，使用CPU")
                return 'cpu'
        else:
            return 'cpu'
    
    def get_recommended_batch_size(self, model_type: str = 'default') -> int:
        """获取推荐的批处理大小"""
        if self.gpu_info['available']:
            gpu_memory_gb = self.gpu_info['memory'] / 1024**3
            if gpu_memory_gb >= 8:
                return 64
            elif gpu_memory_gb >= 4:
                return 32
            else:
                return 16
        else:
            # CPU模式
            cpu_count = self.cpu_info['count']
            if cpu_count >= 8:
                return 16
            elif cpu_count >= 4:
                return 8
            else:
                return 4


class ModelOptimizer:
    """模型优化器"""
    
    def __init__(self, platform_detector: PlatformDetector):
        self.platform = platform_detector
        self.device = platform_detector.get_optimal_device()
    
    def optimize_for_platform(self, model_config: Dict) -> Dict:
        """根据平台优化模型配置"""
        optimized_config = model_config.copy()
        
        if self.device == 'gpu':
            # GPU优化配置
            optimized_config.update({
                'device': 'gpu',
                'batch_size': self.platform.get_recommended_batch_size(),
                'use_mixed_precision': True,
                'num_workers': min(4, self.platform.cpu_info['count'])
            })
        else:
            # CPU优化配置
            optimized_config.update({
                'device': 'cpu',
                'batch_size': self.platform.get_recommended_batch_size(),
                'use_mixed_precision': False,
                'num_workers': min(2, self.platform.cpu_info['count'])
            })
        
        return optimized_config
    
    def get_training_config(self, model_type: str = 'gradient_boosting') -> Dict:
        """获取训练配置"""
        base_config = {
            'model_type': model_type,
            'random_state': 42,
            'test_size': 0.2,
            'stratify': True
        }
        
        if model_type == 'gradient_boosting':
            base_config.update({
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6
            })
        elif model_type == 'lightgbm':
            base_config.update({
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'verbose': -1
            })
        
        return self.optimize_for_platform(base_config)


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.start_time = None
        self.memory_usage = []
        self.cpu_usage = []
    
    def start_monitoring(self):
        """开始监控"""
        import time
        self.start_time = time.time()
        self.memory_usage = []
        self.cpu_usage = []
    
    def record_metrics(self):
        """记录性能指标"""
        if self.start_time is None:
            return
        
        # 记录内存使用
        memory = psutil.virtual_memory()
        self.memory_usage.append({
            'timestamp': time.time() - self.start_time,
            'used': memory.used,
            'percent': memory.percent
        })
        
        # 记录CPU使用
        cpu_percent = psutil.cpu_percent(interval=0.1)
        self.cpu_usage.append({
            'timestamp': time.time() - self.start_time,
            'usage': cpu_percent
        })
    
    def get_summary(self) -> Dict:
        """获取性能摘要"""
        if not self.memory_usage:
            return {}
        
        total_time = time.time() - self.start_time
        avg_memory = sum(m['percent'] for m in self.memory_usage) / len(self.memory_usage)
        avg_cpu = sum(c['usage'] for c in self.cpu_usage) / len(self.cpu_usage)
        max_memory = max(m['percent'] for m in self.memory_usage)
        max_cpu = max(c['usage'] for c in self.cpu_usage)
        
        return {
            'total_time': total_time,
            'avg_memory_usage': avg_memory,
            'avg_cpu_usage': avg_cpu,
            'max_memory_usage': max_memory,
            'max_cpu_usage': max_cpu
        }
    
    def print_summary(self):
        """打印性能摘要"""
        summary = self.get_summary()
        if not summary:
            return
        
        print("\n📊 性能监控摘要")
        print("=" * 30)
        print(f"总运行时间: {summary['total_time']:.2f} 秒")
        print(f"平均内存使用: {summary['avg_memory_usage']:.1f}%")
        print(f"平均CPU使用: {summary['avg_cpu_usage']:.1f}%")
        print(f"最大内存使用: {summary['max_memory_usage']:.1f}%")
        print(f"最大CPU使用: {summary['max_cpu_usage']:.1f}%")


def setup_environment():
    """设置环境"""
    detector = PlatformDetector()
    detector.print_system_info()
    
    optimizer = ModelOptimizer(detector)
    config = optimizer.get_training_config()
    
    print(f"\n⚙️ 推荐配置")
    print("=" * 30)
    print(f"计算设备: {config['device']}")
    print(f"批处理大小: {config['batch_size']}")
    print(f"工作进程数: {config['num_workers']}")
    print(f"混合精度: {config['use_mixed_precision']}")
    
    return detector, optimizer, config


if __name__ == "__main__":
    setup_environment() 