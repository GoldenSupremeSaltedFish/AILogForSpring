#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日志分类器测试脚本
用于验证分类器的功能和性能
"""

import sys
import time
from pathlib import Path
from automated_log_classifier import AutomatedLogClassifier

def test_single_log_classification():
    """测试单条日志分类"""
    print("🧪 测试单条日志分类")
    print("-" * 40)
    
    classifier = AutomatedLogClassifier()
    
    # 测试用例
    test_logs = [
        "ERROR: java.lang.NullPointerException at com.example.Controller.handleRequest(Controller.java:45)",
        "WARN: Connection timeout to database server 192.168.1.100:3306",
        "INFO: Application started successfully in 2.5 seconds",
        "ERROR: SQLException: Connection refused to database",
        "DEBUG: Health check endpoint accessed",
        "ERROR: Authentication failed for user admin",
        "INFO: Memory usage: 512MB / 1024MB",
        "ERROR: BeanCreationException: Error creating bean 'dataSource'",
        "WARN: Configuration property 'server.port' not found, using default 8080",
        "INFO: Business validation passed for order #12345"
    ]
    
    for i, log in enumerate(test_logs, 1):
        print(f"\n测试 {i}: {log[:50]}...")
        
        result = classifier.classify_single_log(log)
        
        print(f"  类别: {result['category']} ({classifier.classification_rules.get(result['category'], {}).get('description', '未知')})")
        print(f"  置信度: {result['confidence']:.3f}")
        print(f"  方法: {result['method']}")
        print(f"  日志级别: {result['log_level']}")
        print(f"  需要人工标注: {'是' if result['needs_manual_annotation'] else '否'}")

def test_performance():
    """测试性能"""
    print("\n🚀 测试性能")
    print("-" * 40)
    
    classifier = AutomatedLogClassifier()
    
    # 生成测试数据
    test_logs = [
        "ERROR: java.lang.NullPointerException at com.example.Controller.handleRequest",
        "WARN: Connection timeout to database server",
        "INFO: Application started successfully",
        "ERROR: SQLException: Connection refused",
        "DEBUG: Health check endpoint accessed"
    ] * 100  # 500条测试日志
    
    print(f"测试数据: {len(test_logs)} 条日志")
    
    # 测试规则分类性能
    start_time = time.time()
    for log in test_logs:
        classifier.classify_by_rules(log)
    rules_time = time.time() - start_time
    
    print(f"规则分类: {rules_time:.3f}秒 ({len(test_logs)/rules_time:.0f} 条/秒)")
    
    # 测试机器学习分类性能（如果可用）
    if classifier.model:
        start_time = time.time()
        for log in test_logs:
            classifier.classify_by_ml(log)
        ml_time = time.time() - start_time
        
        print(f"机器学习分类: {ml_time:.3f}秒 ({len(test_logs)/ml_time:.0f} 条/秒)")
    else:
        print("机器学习分类: 模型未加载")

def test_file_classification():
    """测试文件分类"""
    print("\n📁 测试文件分类")
    print("-" * 40)
    
    classifier = AutomatedLogClassifier()
    
    # 创建测试文件
    test_file = Path("test_logs.csv")
    test_data = [
        "timestamp,level,message",
        "2024-01-01 10:00:00,ERROR,java.lang.NullPointerException at com.example.Controller.handleRequest",
        "2024-01-01 10:01:00,WARN,Connection timeout to database server",
        "2024-01-01 10:02:00,INFO,Application started successfully",
        "2024-01-01 10:03:00,ERROR,SQLException: Connection refused to database",
        "2024-01-01 10:04:00,DEBUG,Health check endpoint accessed"
    ]
    
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(test_data))
    
    print(f"创建测试文件: {test_file}")
    
    # 测试文件分类
    result = classifier.classify_file(str(test_file))
    
    if result:
        print(f"✅ 文件分类成功")
        print(f"📊 总日志数: {result['total_logs']}")
        print(f"📁 输出文件: {result['output_file']}")
        
        stats = result['stats']
        print(f"📈 分类覆盖率: {stats['classification_coverage']:.1f}%")
        print(f"📈 平均置信度: {stats['avg_confidence']:.3f}")
        print(f"📈 需要人工标注: {stats['manual_annotation_needed']} 条 ({stats['manual_annotation_ratio']:.1f}%)")
        
        print("\n类别分布:")
        for category, count in stats['category_distribution'].items():
            description = classifier.classification_rules.get(category, {}).get('description', category)
            percentage = (count / stats['total_logs']) * 100
            print(f"  {description}: {count} 条 ({percentage:.1f}%)")
    else:
        print("❌ 文件分类失败")
    
    # 清理测试文件
    if test_file.exists():
        test_file.unlink()
        print(f"🗑️ 清理测试文件: {test_file}")

def test_model_loading():
    """测试模型加载"""
    print("\n🤖 测试模型加载")
    print("-" * 40)
    
    classifier = AutomatedLogClassifier()
    
    print(f"模型加载状态:")
    print(f"  模型: {'✅ 已加载' if classifier.model else '❌ 未加载'}")
    print(f"  向量化器: {'✅ 已加载' if classifier.vectorizer else '❌ 未加载'}")
    print(f"  标签编码器: {'✅ 已加载' if classifier.label_encoder else '❌ 未加载'}")
    
    if classifier.model:
        print(f"  模型类型: {type(classifier.model).__name__}")
    
    if classifier.vectorizer:
        print(f"  特征数量: {classifier.vectorizer.max_features if hasattr(classifier.vectorizer, 'max_features') else '未知'}")

def main():
    """主测试函数"""
    print("🧪 日志分类器功能测试")
    print("=" * 50)
    
    try:
        # 测试模型加载
        test_model_loading()
        
        # 测试单条日志分类
        test_single_log_classification()
        
        # 测试性能
        test_performance()
        
        # 测试文件分类
        test_file_classification()
        
        print("\n🎉 所有测试完成！")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
