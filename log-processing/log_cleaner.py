#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用日志清洗脚本 - 策略模式实现
用法: python log_cleaner.py <目录路径>
"""

import sys
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd
from datetime import datetime


class LogStrategy(ABC):
    """日志处理策略抽象基类"""
    
    @abstractmethod
    def get_name(self) -> str:
        """获取策略名称"""
        pass
    
    @abstractmethod
    def can_handle(self, sample_lines: List[str]) -> bool:
        """判断是否能处理这种格式的日志"""
        pass
    
    @abstractmethod
    def parse_line(self, line: str) -> Optional[Dict[str, Any]]:
        """解析单行日志"""
        pass
    
    @abstractmethod
    def is_relevant(self, log_data: Dict[str, Any]) -> bool:
        """判断日志是否符合保留条件"""
        pass


class GatewayStrategy(LogStrategy):
    """Gateway日志处理策略"""
    
    def __init__(self):
        self.pattern = re.compile(
            r'^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\s+'
            r'(?P<level>[A-Z]+)\s+(?P<process_id>\d+)\s+---\s+'
            r'\[(?P<thread>[^\]]+)\]\s+(?P<classpath>.+?)\s+:\s+(?P<message>.+)$'
        )
        self.keywords = [
            "user", "auth", "jwt", "token", "security", "gateway",
            "controller", "service", "http", "api", "request", "response",
            "error", "exception", "failed", "success", "验证", "令牌"
        ]
    
    def get_name(self) -> str:
        return "Gateway日志策略"
    
    def can_handle(self, sample_lines: List[str]) -> bool:
        """检查是否为Gateway格式"""
        indicators = 0
        for line in sample_lines:
            if self.pattern.match(line.strip()):
                indicators += 1
            if "gateway" in line.lower() or "---" in line:
                indicators += 1
            if indicators >= 3:
                return True
        return indicators >= 2
    
    def parse_line(self, line: str) -> Optional[Dict[str, Any]]:
        """解析Gateway格式日志"""
        match = self.pattern.match(line.strip())
        if match:
            result = match.groupdict()
            # 统一时间戳格式（去掉毫秒）
            result['timestamp'] = result['timestamp'].split('.')[0]
            return result
        return None
    
    def is_relevant(self, log_data: Dict[str, Any]) -> bool:
        """判断Gateway日志是否相关"""
        if log_data.get('level') not in ['INFO', 'ERROR', 'WARN', 'DEBUG', 'TRACE']:
            return False
        
        classpath = log_data.get('classpath', '').lower()
        message = log_data.get('message', '').lower()
        
        return any(keyword in classpath or keyword in message for keyword in self.keywords)


class ListenerStrategy(LogStrategy):
    """Listener日志处理策略"""
    
    def __init__(self):
        self.pattern = re.compile(
            r'^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+'
            r'\[(?P<thread>[^\]]+)\]\s+(?P<level>[A-Z]+)\s+'
            r'(?P<classpath>.+?)\s+-\s+(?P<message>.+)$'
        )
        self.keywords = [
            "mqtt", "listener", "monitor", "sensor", "data", "message",
            "receive", "send", "publish", "subscribe", "connect",
            "error", "exception", "failed", "success"
        ]
    
    def get_name(self) -> str:
        return "Listener日志策略"
    
    def can_handle(self, sample_lines: List[str]) -> bool:
        """检查是否为Listener格式"""
        indicators = 0
        for line in sample_lines:
            if self.pattern.match(line.strip()):
                indicators += 1
            if any(word in line.lower() for word in ["mqtt", "listener", "monitor"]):
                indicators += 1
            if indicators >= 3:
                return True
        return indicators >= 2
    
    def parse_line(self, line: str) -> Optional[Dict[str, Any]]:
        """解析Listener格式日志"""
        match = self.pattern.match(line.strip())
        if match:
            return match.groupdict()
        return None
    
    def is_relevant(self, log_data: Dict[str, Any]) -> bool:
        """判断Listener日志是否相关"""
        if log_data.get('level') not in ['INFO', 'ERROR', 'WARN', 'DEBUG', 'TRACE']:
            return False
        
        classpath = log_data.get('classpath', '').lower()
        message = log_data.get('message', '').lower()
        
        return any(keyword in classpath or keyword in message for keyword in self.keywords)


class LogProcessor:
    """日志处理器上下文类"""
    
    def __init__(self):
        self.strategies = [
            GatewayStrategy(),
            ListenerStrategy()
        ]
    
    def detect_strategy(self, log_file: Path) -> Optional[LogStrategy]:
        """自动检测日志文件应该使用哪种策略"""
        print(f"🔍 检测日志格式: {log_file.name}")
        
        # 读取前50行作为样本
        sample_lines = []
        try:
            with log_file.open("r", encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f):
                    if i >= 50:
                        break
                    line = line.strip()
                    if line:
                        sample_lines.append(line)
        except Exception as e:
            print(f"  ❌ 读取文件失败: {e}")
            return None
        
        if not sample_lines:
            print(f"  ❌ 文件为空或无法读取")
            return None
        
        # 尝试每种策略
        for strategy in self.strategies:
            if strategy.can_handle(sample_lines):
                print(f"  ✅ 使用策略: {strategy.get_name()}")
                return strategy
        
        print(f"  ⚠️  未找到合适的处理策略，使用默认策略: {self.strategies[0].get_name()}")
        return self.strategies[0]  # 默认使用第一个策略
    
    def process_file(self, log_file: Path, output_dir: Path, strategy: LogStrategy) -> int:
        """使用指定策略处理单个日志文件"""
        print(f"📄 处理文件: {log_file.name} [使用 {strategy.get_name()}]")
        
        parsed_logs = []
        try:
            with log_file.open("r", encoding="utf-8", errors="ignore") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    # 解析日志行
                    log_data = strategy.parse_line(line)
                    if log_data and strategy.is_relevant(log_data):
                        log_data['source_file'] = log_file.name
                        log_data['line_number'] = line_num
                        parsed_logs.append(log_data)
            
            # 保存清洗后的数据
            if parsed_logs:
                df = pd.DataFrame(parsed_logs)
                output_file = output_dir / f"{log_file.stem}_cleaned.csv"
                df.to_csv(output_file, index=False, encoding="utf-8-sig")
                
                count = len(parsed_logs)
                print(f"  ✅ 提取 {count} 条有效日志记录")
                return count
            else:
                print(f"  ⚠️  未找到符合条件的日志记录")
                return 0
        
        except Exception as e:
            print(f"  ❌ 处理失败: {e}")
            return 0


def process_logs(input_dir_path: str):
    """主处理函数"""
    print("🚀 开始处理日志文件...")
    print(f"📁 输入目录: {input_dir_path}")
    
    # 验证输入目录
    input_dir = Path(input_dir_path)
    if not input_dir.exists():
        print(f"❌ 错误：目录不存在 - {input_dir}")
        return
    
    if not input_dir.is_dir():
        print(f"❌ 错误：路径不是目录 - {input_dir}")
        return
    
    # 设置输出目录
    output_base_dir = Path(__file__).parent.parent / "DATA_OUTPUT"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = input_dir.name
    output_dir = output_base_dir / f"processed_{dir_name}_{timestamp}"
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"📁 输出目录: {output_dir}")
    
    # 扫描日志文件
    log_files = list(input_dir.glob("*.log"))
    print(f"📊 找到 {len(log_files)} 个 .log 文件")
    
    if not log_files:
        print("❌ 未找到任何 .log 文件")
        return
    
    # 保存文件路径列表
    file_list_path = output_dir / "file_list.txt"
    with file_list_path.open("w", encoding="utf-8") as f:
        f.write("日志文件路径列表\n")
        f.write("=" * 50 + "\n")
        for i, log_file in enumerate(log_files, 1):
            f.write(f"{i:3d}. {log_file}\n")
    print(f"📝 文件路径列表已保存: {file_list_path}")
    
    # 创建处理器
    processor = LogProcessor()
    
    # 处理每个文件
    total_logs = 0
    processed_files = 0
    processing_summary = {}
    strategy_usage = {}
    
    print("-" * 60)
    for log_file in log_files:
        # 自动检测处理策略
        strategy = processor.detect_strategy(log_file)
        if strategy:
            strategy_name = strategy.get_name()
            strategy_usage[strategy_name] = strategy_usage.get(strategy_name, 0) + 1
            
            # 处理文件
            count = processor.process_file(log_file, output_dir, strategy)
            if count > 0:
                total_logs += count
                processed_files += 1
            
            processing_summary[log_file.name] = {
                'count': count,
                'strategy': strategy_name
            }
        else:
            processing_summary[log_file.name] = {
                'count': 0,
                'strategy': '无法识别'
            }
    
    # 生成处理报告
    report_file = output_dir / "processing_report.txt"
    with report_file.open("w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("通用日志处理报告 - 策略模式\n")
        f.write("=" * 60 + "\n")
        f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"输入目录: {input_dir}\n")
        f.write(f"输出目录: {output_dir}\n")
        f.write(f"扫描文件数: {len(log_files)}\n")
        f.write(f"成功处理文件数: {processed_files}\n")
        f.write(f"有效日志总数: {total_logs}\n\n")
        
        f.write("策略使用统计:\n")
        f.write("-" * 30 + "\n")
        for strategy_name, count in strategy_usage.items():
            f.write(f"{strategy_name}: {count} 个文件\n")
        f.write("\n")
        
        f.write("各文件处理详情:\n")
        f.write("-" * 50 + "\n")
        for filename, info in processing_summary.items():
            f.write(f"{filename:<35} {str(info['count']):>6} 条  [{info['strategy']}]\n")
    
    print("-" * 60)
    print(f"🎉 处理完成!")
    print(f"📊 扫描了 {len(log_files)} 个文件")
    print(f"📊 成功处理 {processed_files} 个文件")
    print(f"📊 提取 {total_logs} 条有效日志记录")
    print(f"📁 结果保存在: {output_dir}")
    print(f"📄 处理报告: {report_file}")
    print(f"📝 文件列表: {file_list_path}")
    
    print("\n策略使用统计:")
    for strategy_name, count in strategy_usage.items():
        print(f"  {strategy_name}: {count} 个文件")


def main():
    """主函数"""
    if len(sys.argv) != 2:
        print("通用日志清洗工具 - 策略模式")
        print("=" * 40)
        print("使用方法:")
        print(f"  python {sys.argv[0]} <目录路径>")
        print("\n示例:")
        print(f"  python {sys.argv[0]} C:\\path\\to\\logs")
        print("\n支持的日志格式:")
        print("  - Gateway日志 (Spring Boot Gateway格式)")
        print("  - Listener日志 (MQTT监听器格式)")
        print("\n特性:")
        print("  - 自动识别日志格式")
        print("  - 智能关键词过滤")
        print("  - 生成详细处理报告")
        print("  - 使用策略模式，易于扩展")
        return
    
    input_dir_path = sys.argv[1]
    process_logs(input_dir_path)


if __name__ == "__main__":
    main() 