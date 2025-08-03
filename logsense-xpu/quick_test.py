#!/usr/bin/env python3
import requests
import json

def test_api():
    base_url = "http://localhost:5000"
    
    print("🚀 API测试结果:")
    print("=" * 50)
    
    # 1. 健康检查
    print("1. 健康检查:")
    response = requests.get(f"{base_url}/health")
    result = response.json()
    print(f"   状态: {result['status']}")
    print(f"   模型加载: {result['model_loaded']}")
    print(f"   时间戳: {result['timestamp']}")
    print()
    
    # 2. 模型信息
    print("2. 模型信息:")
    response = requests.get(f"{base_url}/model/info")
    result = response.json()
    print(f"   模型类型: {result['model_type']}")
    print(f"   时间戳: {result['model_timestamp']}")
    print(f"   类别数: {result['num_categories']}")
    print(f"   特征数: {result['vectorizer_features']}")
    print(f"   状态: {result['status']}")
    print()
    
    # 3. 单个预测
    print("3. 单个预测:")
    test_text = "ERROR: java.lang.NullPointerException at com.example.Controller.handleRequest"
    data = {"text": test_text}
    response = requests.post(f"{base_url}/predict", json=data)
    result = response.json()
    
    if result.get("success"):
        prediction = result.get("prediction", {})
        print(f"   输入文本: {test_text[:50]}...")
        print(f"   预测类别: {prediction.get('category_name')}")
        print(f"   置信度: {prediction.get('confidence', 0):.4f}")
        print(f"   类别ID: {prediction.get('category_id')}")
    else:
        print(f"   预测失败: {result.get('error')}")
    print()
    
    # 4. 批量预测
    print("4. 批量预测:")
    test_texts = [
        "ERROR: java.lang.NullPointerException at com.example.Controller.handleRequest",
        "INFO: Application started successfully on port 8080",
        "WARN: Database connection timeout, retrying..."
    ]
    data = {"texts": test_texts}
    response = requests.post(f"{base_url}/predict/batch", json=data)
    result = response.json()
    
    if result.get("success"):
        predictions = result.get("predictions", [])
        print(f"   总数量: {result.get('total_count')}")
        for i, pred in enumerate(predictions):
            if pred.get("success"):
                prediction = pred.get("prediction", {})
                print(f"   [{i+1}] {pred['text'][:40]}... -> {prediction.get('category_name')} (置信度: {prediction.get('confidence', 0):.3f})")
    else:
        print(f"   批量预测失败: {result.get('error')}")
    print()
    
    # 5. 模型重载
    print("5. 模型重载:")
    response = requests.post(f"{base_url}/reload")
    result = response.json()
    print(f"   成功: {result.get('success')}")
    print(f"   消息: {result.get('message')}")
    print()
    
    print("✅ 所有API测试完成！")

if __name__ == "__main__":
    test_api() 