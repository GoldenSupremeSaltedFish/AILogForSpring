import re
import json
import argparse
from pathlib import Path
from typing import Optional, List, Dict
import pandas as pd
from datetime import datetime
import os
import glob

# ----------------------
# 日志解析正则模式
# ----------------------
# Listener格式: 时间戳 [线程] 级别 类路径 - 消息
LOG_PATTERN_LISTENER = re.compile(
    r'^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+\[(?P<thread>[^\]]+)\]\s+(?P<level>[A-Z]+)\s+(?P<classpath>.+?)\s+-\s+(?P<message>.+)$'
)

# Gateway格式: 时间戳.毫秒 级别 进程ID --- [线程] 类路径 : 消息
LOG_PATTERN_GATEWAY = re.compile(
    r'^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\s+(?P<level>[A-Z]+)\s+\d+\s+---\s+\[(?P<thread>[^\]]+)\]\s+(?P<classpath>.+?)\s+:\s+(?P<message>.+)$'
)

# 支持的日志格式模式列表
LOG_PATTERNS = [LOG_PATTERN_LISTENER, LOG_PATTERN_GATEWAY]

# ----------------------
# 自定义过滤配置
# ----------------------
ALLOWED_LEVELS = {"INFO", "ERROR", "WARN", "DEBUG", "TRACE"}  # 可保留的日志级别
INCLUDE_KEYWORDS = [
    # 用户相关
    "user", "account", "login", "auth", "jwt", "token",
    # 系统组件
    "controller", "service", "repository", "config", "gateway", "security",
    # API相关
    "http", "api", "request", "response", "filter",
    # 数据相关
    "data", "database", "sql", "asset",
    # 监控相关
    "alert", "dashboard", "audit", "runtime",
    # 通用关键词
    "getting", "creating", "updating", "deleting", "processing",
    # Gateway特定关键词
    "validation", "verify", "success", "failed"
]  # 类路径或消息中包含任一关键词即保留


def parse_line(line: str) -> Optional[dict]:
    """将单行日志解析为结构化字段"""
    line_stripped = line.strip()
    
    # 尝试所有支持的日志格式
    for pattern in LOG_PATTERNS:
        match = pattern.match(line_stripped)
        if match:
            result = match.groupdict()
            # 统一时间戳格式（去掉毫秒部分以便统一处理）
            if '.' in result['timestamp']:
                result['timestamp'] = result['timestamp'].split('.')[0]
            return result
    
    return None


def is_relevant(log: dict) -> bool:
    """判断日志是否满足保留条件"""
    if log["level"] not in ALLOWED_LEVELS:
        return False
    
    # 检查类路径或消息中是否包含关键词
    classpath_lower = log["classpath"].lower()
    message_lower = log["message"].lower()
    
    # 更宽松的过滤条件：只要包含任一关键词就保留
    if any(k in classpath_lower or k in message_lower for k in INCLUDE_KEYWORDS):
        return True
    
    # 额外保留一些重要的日志模式
    if any(pattern in message_lower for pattern in [
        "error", "exception", "failed", "timeout", "connection",
        "started", "stopped", "shutdown", "startup", "success"
    ]):
        return True
    
    return False


def clean_log_file(input_path: Path, output_path: Path, to_json: bool = False):
    """主处理函数：解析日志、过滤、输出"""
    parsed_logs = []

    with input_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parsed = parse_line(line)
            if parsed and is_relevant(parsed):
                # 添加源文件信息
                parsed['source_file'] = input_path.name
                parsed['source_directory'] = input_path.parent.name
                parsed_logs.append(parsed)

    if not parsed_logs:
        print(f"❌ {input_path.name}: 未找到符合条件的日志")
        return 0

    df = pd.DataFrame(parsed_logs)

    if to_json:
        df.to_json(output_path, orient="records", force_ascii=False, indent=2)
    else:
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
    
    print(f"✅ {input_path.name}: 处理有效日志 {len(df)} 条")
    return len(df)


def process_gateway_logs():
    """专门处理gateway日志目录"""
    # 设置路径
    script_dir = Path(__file__).parent
    gateway_dir = script_dir.parent / "data" / "gate_way_logs"
    output_base_dir = script_dir.parent / "data" / "output"
    
    # 检查输入目录
    if not gateway_dir.exists():
        print(f"❌ 目录不存在: {gateway_dir}")
        return
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_base_dir / f"processed_gate_way_logs_{timestamp}"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"🚀 开始处理Gateway日志文件...")
    print(f"📁 输入目录: {gateway_dir}")
    print(f"📁 输出目录: {output_dir}")
    print("-" * 50)
    
    # 扫描日志文件
    log_extensions = ['*.log', '*.txt', '*.out']
    log_files = []
    
    for ext in log_extensions:
        files = list(gateway_dir.glob(ext))
        log_files.extend(files)
        if files:
            print(f"📂 找到 {len(files)} 个 {ext} 文件")
    
    print(f"📊 总共找到 {len(log_files)} 个日志文件")
    
    if not log_files:
        print("❌ 未找到任何日志文件")
        return
    
    # 处理每个日志文件
    total_logs = 0
    processed_count = 0
    summary = {}
    
    for log_file in log_files:
        source_name = f"gate_way_logs_{log_file.stem}"
        output_file = output_dir / f"{source_name}_cleaned.csv"
        
        try:
            count = clean_log_file(log_file, output_file, False)
            if count > 0:
                total_logs += count
                processed_count += 1
                summary[source_name] = count
        except Exception as e:
            print(f"❌ 处理文件失败 {log_file.name}: {e}")
    
    # 生成处理报告
    generate_summary_report(output_dir, summary, total_logs, processed_count)
    
    print("-" * 50)
    print(f"🎉 处理完成!")
    print(f"📊 处理了 {processed_count} 个文件，共 {total_logs} 条有效日志")
    print(f"📁 结果保存在: {output_dir}")


def generate_summary_report(output_dir: Path, summary: Dict[str, int], total_logs: int, processed_count: int):
    """生成处理摘要报告"""
    report_file = output_dir / "processing_summary.txt"
    
    with report_file.open("w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("LogSense-XPU Gateway日志处理报告\n")
        f.write("=" * 60 + "\n")
        f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"处理文件数: {processed_count}\n")
        f.write(f"有效日志总数: {total_logs}\n\n")
        
        f.write("各文件处理详情:\n")
        f.write("-" * 40 + "\n")
        for source, count in summary.items():
            f.write(f"{source:<35} {count:>6} 条\n")
        
        f.write("\n过滤条件:\n")
        f.write(f"- 日志级别: {', '.join(ALLOWED_LEVELS)}\n")
        f.write(f"- 关键词: {', '.join(INCLUDE_KEYWORDS)}\n")
    
    print(f"📄 处理报告已生成: {report_file.name}")


if __name__ == "__main__":
    process_gateway_logs() 