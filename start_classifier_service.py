#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日志分类器服务启动器
整合API服务和自动化分类器，提供统一的服务接口
"""

import os
import sys
import json
import argparse
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import subprocess

# 导入分类器
from automated_log_classifier import AutomatedLogClassifier

# 尝试导入Flask
try:
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("⚠️  Flask未安装，将仅提供命令行接口")

class ClassifierService:
    """日志分类器服务"""
    
    def __init__(self, config_file: str = None):
        """初始化服务"""
        self.config = self._load_config(config_file)
        self.classifier = AutomatedLogClassifier(config_file)
        self.app = None
        self.service_status = {
            'started_at': None,
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0
        }
        
        if FLASK_AVAILABLE:
            self._init_flask_app()
    
    def _load_config(self, config_file: str = None) -> Dict:
        """加载配置文件"""
        default_config = {
            "service": {
                "host": "0.0.0.0",
                "port": 5000,
                "debug": False
            },
            "classification": {
                "use_ml": True,
                "confidence_threshold": 0.7,
                "batch_size": 1000
            },
            "data_paths": {
                "input": "DATA_OUTPUT/原始项目数据_original",
                "output": "log-processing-OUTPUT",
                "models": "logsense-xpu/models"
            }
        }
        
        if config_file and Path(config_file).exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                # 合并配置
                for key, value in user_config.items():
                    if key in default_config:
                        default_config[key].update(value)
                    else:
                        default_config[key] = value
        
        return default_config
    
    def _init_flask_app(self):
        """初始化Flask应用"""
        self.app = Flask(__name__)
        CORS(self.app)
        
        @self.app.route('/', methods=['GET'])
        def index():
            return jsonify({
                "service": "日志分类器服务",
                "version": "1.0.0",
                "status": "running",
                "started_at": self.service_status['started_at'],
                "endpoints": {
                    "health": "/health",
                    "classify": "/classify",
                    "batch_classify": "/batch_classify",
                    "stats": "/stats"
                }
            })
        
        @self.app.route('/health', methods=['GET'])
        def health():
            return jsonify({
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "model_loaded": self.classifier.model is not None,
                "service_stats": self.service_status
            })
        
        @self.app.route('/classify', methods=['POST'])
        def classify():
            try:
                self.service_status['total_requests'] += 1
                
                data = request.get_json()
                if not data or 'text' not in data:
                    return jsonify({"error": "缺少text字段"}), 400
                
                log_text = data['text']
                use_ml = data.get('use_ml', self.config['classification']['use_ml'])
                
                result = self.classifier.classify_single_log(log_text, use_ml)
                
                self.service_status['successful_requests'] += 1
                return jsonify(result)
                
            except Exception as e:
                self.service_status['failed_requests'] += 1
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/batch_classify', methods=['POST'])
        def batch_classify():
            try:
                self.service_status['total_requests'] += 1
                
                data = request.get_json()
                if not data or 'texts' not in data:
                    return jsonify({"error": "缺少texts字段"}), 400
                
                texts = data['texts']
                use_ml = data.get('use_ml', self.config['classification']['use_ml'])
                
                results = []
                for text in texts:
                    result = self.classifier.classify_single_log(text, use_ml)
                    results.append(result)
                
                self.service_status['successful_requests'] += 1
                return jsonify({"results": results, "count": len(results)})
                
            except Exception as e:
                self.service_status['failed_requests'] += 1
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/stats', methods=['GET'])
        def stats():
            return jsonify({
                "service_stats": self.service_status,
                "classification_rules": list(self.classifier.classification_rules.keys()),
                "model_info": {
                    "loaded": self.classifier.model is not None,
                    "vectorizer_loaded": self.classifier.vectorizer is not None,
                    "label_encoder_loaded": self.classifier.label_encoder is not None
                }
            })
    
    def start_api_service(self, host: str = None, port: int = None, debug: bool = None):
        """启动API服务"""
        if not FLASK_AVAILABLE:
            print("❌ Flask未安装，无法启动API服务")
            return
        
        if not self.app:
            print("❌ Flask应用未初始化")
            return
        
        host = host or self.config['service']['host']
        port = port or self.config['service']['port']
        debug = debug if debug is not None else self.config['service']['debug']
        
        self.service_status['started_at'] = datetime.now().isoformat()
        
        print("🚀 启动日志分类器API服务")
        print(f"📍 服务地址: http://{host}:{port}")
        print(f"📍 健康检查: http://{host}:{port}/health")
        print(f"📍 分类接口: http://{host}:{port}/classify")
        print(f"📍 批量分类: http://{host}:{port}/batch_classify")
        print(f"📍 服务统计: http://{host}:{port}/stats")
        print("\n按 Ctrl+C 停止服务")
        
        try:
            self.app.run(host=host, port=port, debug=debug)
        except KeyboardInterrupt:
            print("\n🛑 服务已停止")
    
    def classify_file_cli(self, input_file: str, output_file: str = None, use_ml: bool = None):
        """命令行文件分类"""
        use_ml = use_ml if use_ml is not None else self.config['classification']['use_ml']
        
        print(f"🔄 开始分类文件: {Path(input_file).name}")
        result = self.classifier.classify_file(input_file, output_file, use_ml)
        
        if result:
            print(f"✅ 分类完成")
            print(f"📊 总日志数: {result['total_logs']}")
            print(f"📁 输出文件: {result['output_file']}")
            
            stats = result['stats']
            print(f"📈 分类覆盖率: {stats['classification_coverage']:.1f}%")
            print(f"📈 平均置信度: {stats['avg_confidence']:.3f}")
            print(f"📈 需要人工标注: {stats['manual_annotation_needed']} 条 ({stats['manual_annotation_ratio']:.1f}%)")
        else:
            print("❌ 分类失败")
    
    def batch_classify_cli(self, input_dir: str, output_dir: str = None, use_ml: bool = None):
        """命令行批量分类"""
        use_ml = use_ml if use_ml is not None else self.config['classification']['use_ml']
        
        print(f"🔄 开始批量分类目录: {input_dir}")
        result = self.classifier.batch_classify(input_dir, output_dir, use_ml)
        
        if result:
            print(f"✅ 批量分类完成")
            print(f"📊 处理文件数: {result['success_count']}/{result['total_files']}")
            print(f"📁 输出目录: {result['output_dir']}")
        else:
            print("❌ 批量分类失败")
    
    def interactive_classify(self):
        """交互式分类"""
        print("🔍 交互式日志分类器")
        print("输入日志内容进行分类，输入 'quit' 退出")
        print("-" * 50)
        
        while True:
            try:
                log_text = input("\n请输入日志内容: ").strip()
                
                if log_text.lower() in ['quit', 'exit', 'q']:
                    print("👋 再见！")
                    break
                
                if not log_text:
                    continue
                
                result = self.classifier.classify_single_log(log_text)
                
                print(f"\n📊 分类结果:")
                print(f"  类别: {result['category']} ({self.classifier.classification_rules.get(result['category'], {}).get('description', '未知')})")
                print(f"  置信度: {result['confidence']:.3f}")
                print(f"  方法: {result['method']}")
                print(f"  日志级别: {result['log_level']}")
                print(f"  需要人工标注: {'是' if result['needs_manual_annotation'] else '否'}")
                
                if result['rule_reason']:
                    print(f"  匹配原因: {result['rule_reason']}")
                
            except KeyboardInterrupt:
                print("\n👋 再见！")
                break
            except Exception as e:
                print(f"❌ 分类失败: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='日志分类器服务')
    parser.add_argument('--mode', choices=['api', 'file', 'batch', 'interactive'], 
                       default='api', help='运行模式')
    parser.add_argument('--input-file', help='输入文件路径（file模式）')
    parser.add_argument('--input-dir', help='输入目录路径（batch模式）')
    parser.add_argument('--output-file', help='输出文件路径')
    parser.add_argument('--output-dir', help='输出目录路径')
    parser.add_argument('--config', help='配置文件路径')
    parser.add_argument('--host', default='0.0.0.0', help='API服务主机地址')
    parser.add_argument('--port', type=int, default=5000, help='API服务端口')
    parser.add_argument('--no-ml', action='store_true', help='不使用机器学习分类')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    
    args = parser.parse_args()
    
    # 创建服务
    service = ClassifierService(args.config)
    
    # 根据模式运行
    if args.mode == 'api':
        # API服务模式
        use_ml = not args.no_ml
        service.start_api_service(args.host, args.port, args.debug)
    
    elif args.mode == 'file':
        # 单文件分类模式
        if not args.input_file:
            print("❌ 文件模式需要指定 --input-file")
            return
        
        if not Path(args.input_file).exists():
            print(f"❌ 输入文件不存在: {args.input_file}")
            return
        
        use_ml = not args.no_ml
        service.classify_file_cli(args.input_file, args.output_file, use_ml)
    
    elif args.mode == 'batch':
        # 批量分类模式
        if not args.input_dir:
            print("❌ 批量模式需要指定 --input-dir")
            return
        
        if not Path(args.input_dir).exists():
            print(f"❌ 输入目录不存在: {args.input_dir}")
            return
        
        use_ml = not args.no_ml
        service.batch_classify_cli(args.input_dir, args.output_dir, use_ml)
    
    elif args.mode == 'interactive':
        # 交互式模式
        service.interactive_classify()

if __name__ == "__main__":
    main()
