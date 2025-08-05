#!/usr/bin/env python3
import requests
import json

def test_enhanced_api():
    base_url = "http://localhost:5000"
    
    print("🚀 增强版API测试结果:")
    print("=" * 60)
    
    # 1. 健康检查
    print("1. 健康检查:")
    try:
        response = requests.get(f"{base_url}/health")
        result = response.json()
        print(f"   状态: {result['status']}")
        print(f"   模型加载: {result['model_loaded']}")
        print(f"   时间戳: {result['timestamp']}")
        print(f"   设备信息: {result.get('device_info', {})}")
    except Exception as e:
        print(f"   错误: {e}")
    print()
    
    # 2. 设备信息
    print("2. 设备信息:")
    try:
        response = requests.get(f"{base_url}/device/info")
        result = response.json()
        print(f"   CPU核心数: {result.get('cpu_count', 'N/A')}")
        print(f"   GPU可用: {result.get('gpu_available', 'N/A')}")
        print(f"   当前设备: {result.get('current_device', 'N/A')}")
        print(f"   使用GPU: {result.get('use_gpu', 'N/A')}")
        if result.get('gpu_available'):
            print(f"   GPU数量: {result.get('gpu_count', 'N/A')}")
            print(f"   GPU名称: {result.get('gpu_names', 'N/A')}")
    except Exception as e:
        print(f"   错误: {e}")
    print()
    
    # 3. 模型信息
    print("3. 模型信息:")
    try:
        response = requests.get(f"{base_url}/model/info")
        result = response.json()
        print(f"   模型类型: {result.get('model_type', 'N/A')}")
        print(f"   时间戳: {result.get('model_timestamp', 'N/A')}")
        print(f"   类别数: {result.get('num_categories', 'N/A')}")
        print(f"   特征数: {result.get('vectorizer_features', 'N/A')}")
        print(f"   状态: {result.get('status', 'N/A')}")
        device_info = result.get('device_info', {})
        print(f"   当前设备: {device_info.get('current_device', 'N/A')}")
        print(f"   GPU名称: {device_info.get('gpu_name', 'N/A')}")
    except Exception as e:
        print(f"   错误: {e}")
    print()
    
    # 4. 单个预测
    print("4. 单个预测:")
    test_text = "ERROR: java.lang.NullPointerException at com.example.Controller.handleRequest"
    try:
        data = {"text": test_text}
        response = requests.post(f"{base_url}/predict", json=data)
        result = response.json()
        
        if result.get("success"):
            prediction = result.get("prediction", {})
            print(f"   输入文本: {test_text[:50]}...")
            print(f"   预测类别: {prediction.get('category_name')}")
            print(f"   置信度: {prediction.get('confidence', 0):.4f}")
            print(f"   类别ID: {prediction.get('category_id')}")
            print(f"   使用设备: {result.get('device_used', 'N/A')}")
        else:
            print(f"   预测失败: {result.get('error')}")
    except Exception as e:
        print(f"   错误: {e}")
    print()
    
    # 5. 批量预测
    print("5. 批量预测:")
    test_texts = [
        "ERROR: java.lang.NullPointerException at com.example.Controller.handleRequest",
        "INFO: Application started successfully on port 8080",
        "WARN: Database connection timeout, retrying..."
    ]
    try:
        data = {"texts": test_texts}
        response = requests.post(f"{base_url}/predict/batch", json=data)
        result = response.json()
        
        if result.get("success"):
            predictions = result.get("predictions", [])
            print(f"   总数量: {result.get('total_count')}")
            print(f"   使用设备: {result.get('device_used', 'N/A')}")
            for i, pred in enumerate(predictions):
                if pred.get("success"):
                    prediction = pred.get("prediction", {})
                    print(f"   [{i+1}] {pred['text'][:40]}... -> {prediction.get('category_name')} (置信度: {prediction.get('confidence', 0):.3f})")
        else:
            print(f"   批量预测失败: {result.get('error')}")
    except Exception as e:
        print(f"   错误: {e}")
    print()
    
    # 6. 切换设备模式测试
    print("6. 设备切换测试:")
    try:
        # 切换到CPU模式
        data = {"use_gpu": False}
        response = requests.post(f"{base_url}/switch_device", json=data)
        result = response.json()
        print(f"   切换到CPU: {result.get('success')} - {result.get('message')}")
        
        # 再切换回GPU模式
        data = {"use_gpu": True}
        response = requests.post(f"{base_url}/switch_device", json=data)
        result = response.json()
        print(f"   切换回GPU: {result.get('success')} - {result.get('message')}")
    except Exception as e:
        print(f"   错误: {e}")
    print()
    
    print("✅ 增强版API测试完成！")

if __name__ == "__main__":
    test_enhanced_api() 