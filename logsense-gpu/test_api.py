#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LogSense API 测试脚本
"""

import requests
import json
import time

# API服务器地址
BASE_URL = "http://localhost:5000"

def test_health_check():
    """测试健康检查接口"""
    print("🔍 测试健康检查接口...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"状态码: {response.status_code}")
        print(f"响应: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ 健康检查失败: {e}")
        return False

def test_model_info():
    """测试模型信息接口"""
    print("\n🔍 测试模型信息接口...")
    try:
        response = requests.get(f"{BASE_URL}/model/info")
        print(f"状态码: {response.status_code}")
        print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ 模型信息获取失败: {e}")
        return False

def test_classes():
    """测试类别列表接口"""
    print("\n🔍 测试类别列表接口...")
    try:
        response = requests.get(f"{BASE_URL}/classes")
        print(f"状态码: {response.status_code}")
        print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ 类别列表获取失败: {e}")
        return False

def test_single_predict():
    """测试单条预测接口"""
    print("\n🔍 测试单条预测接口...")
    
    # 测试日志样本
    test_logs = [
        "java.lang.NullPointerException: Cannot invoke \"String.length()\" because \"str\" is null",
        "Connection refused: connect to database server failed",
        "OutOfMemoryError: Java heap space",
        "Authentication failed: invalid credentials",
        "Database connection timeout after 30 seconds"
    ]
    
    for i, log_text in enumerate(test_logs, 1):
        print(f"\n📝 测试日志 {i}: {log_text[:50]}...")
        
        try:
            data = {
                "log_text": log_text,
                "top_k": 3
            }
            
            response = requests.post(f"{BASE_URL}/predict", json=data)
            print(f"状态码: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"预测类别: {result['predicted_class']}")
                print(f"置信度: {result['confidence_percentage']:.2f}%")
                print("Top-3预测结果:")
                for pred in result['top_k_predictions']:
                    print(f"  - {pred['class']}: {pred['percentage']:.2f}%")
            else:
                print(f"❌ 预测失败: {response.text}")
                
        except Exception as e:
            print(f"❌ 预测请求失败: {e}")

def test_batch_predict():
    """测试批量预测接口"""
    print("\n🔍 测试批量预测接口...")
    
    test_logs = [
        "java.lang.ArrayIndexOutOfBoundsException: Index 5 out of bounds for length 3",
        "MySQL connection pool exhausted",
        "Memory usage exceeded 90% threshold",
        "User authentication failed: password expired",
        "Database query timeout: SELECT * FROM large_table"
    ]
    
    try:
        data = {
            "log_texts": test_logs,
            "top_k": 2
        }
        
        response = requests.post(f"{BASE_URL}/predict/batch", json=data)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"批量预测完成，共 {result['total_count']} 条日志")
            
            for pred in result['predictions']:
                print(f"\n日志 {pred['index'] + 1}:")
                print(f"  预测类别: {pred['predicted_class']}")
                print(f"  置信度: {pred['confidence_percentage']:.2f}%")
                print("  Top-2预测结果:")
                for top_pred in pred['top_k_predictions']:
                    print(f"    - {top_pred['class']}: {top_pred['percentage']:.2f}%")
        else:
            print(f"❌ 批量预测失败: {response.text}")
            
    except Exception as e:
        print(f"❌ 批量预测请求失败: {e}")

def test_error_cases():
    """测试错误情况"""
    print("\n🔍 测试错误情况...")
    
    # 测试空文本
    print("测试空文本:")
    try:
        response = requests.post(f"{BASE_URL}/predict", json={"log_text": ""})
        print(f"状态码: {response.status_code}")
        print(f"响应: {response.json()}")
    except Exception as e:
        print(f"❌ 请求失败: {e}")
    
    # 测试无效的top_k
    print("\n测试无效的top_k:")
    try:
        response = requests.post(f"{BASE_URL}/predict", json={
            "log_text": "test log",
            "top_k": -1
        })
        print(f"状态码: {response.status_code}")
        print(f"响应: {response.json()}")
    except Exception as e:
        print(f"❌ 请求失败: {e}")

def main():
    """主测试函数"""
    print("🚀 开始LogSense API测试")
    print("=" * 50)
    
    # 等待服务器启动
    print("⏳ 等待服务器启动...")
    time.sleep(2)
    
    # 运行测试
    tests = [
        test_health_check,
        test_model_info,
        test_classes,
        test_single_predict,
        test_batch_predict,
        test_error_cases
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ 测试异常: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("✅ 所有测试通过！")
    else:
        print("❌ 部分测试失败")

if __name__ == "__main__":
    main() 