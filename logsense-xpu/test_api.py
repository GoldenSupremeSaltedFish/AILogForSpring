#!/usr/bin/env python3
"""
API测试脚本
用于测试日志分类API服务的各个功能
"""

import requests
import json
import time
from typing import List, Dict

class APITester:
    """API测试类"""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def test_health(self) -> Dict:
        """测试健康检查接口"""
        print("🔍 测试健康检查接口...")
        try:
            response = self.session.get(f"{self.base_url}/health")
            result = response.json()
            print(f"✅ 健康检查成功: {result}")
            return result
        except Exception as e:
            print(f"❌ 健康检查失败: {e}")
            return {"success": False, "error": str(e)}
    
    def test_model_info(self) -> Dict:
        """测试模型信息接口"""
        print("🔍 测试模型信息接口...")
        try:
            response = self.session.get(f"{self.base_url}/model/info")
            result = response.json()
            print(f"✅ 模型信息获取成功:")
            print(f"  模型类型: {result.get('model_type')}")
            print(f"  时间戳: {result.get('model_timestamp')}")
            print(f"  类别数: {result.get('num_categories')}")
            print(f"  状态: {result.get('status')}")
            return result
        except Exception as e:
            print(f"❌ 模型信息获取失败: {e}")
            return {"success": False, "error": str(e)}
    
    def test_single_predict(self, text: str) -> Dict:
        """测试单个预测接口"""
        print(f"🔍 测试单个预测接口...")
        print(f"   输入文本: {text[:100]}{'...' if len(text) > 100 else ''}")
        
        try:
            data = {"text": text}
            response = self.session.post(f"{self.base_url}/predict", json=data)
            result = response.json()
            
            if result.get("success"):
                prediction = result.get("prediction", {})
                print(f"✅ 预测成功:")
                print(f"   类别: {prediction.get('category_name')}")
                print(f"   置信度: {prediction.get('confidence', 0):.4f}")
                print(f"   类别ID: {prediction.get('category_id')}")
            else:
                print(f"❌ 预测失败: {result.get('error')}")
            
            return result
        except Exception as e:
            print(f"❌ 预测请求失败: {e}")
            return {"success": False, "error": str(e)}
    
    def test_batch_predict(self, texts: List[str]) -> Dict:
        """测试批量预测接口"""
        print(f"🔍 测试批量预测接口...")
        print(f"   输入文本数量: {len(texts)}")
        
        try:
            data = {"texts": texts}
            response = self.session.post(f"{self.base_url}/predict/batch", json=data)
            result = response.json()
            
            if result.get("success"):
                predictions = result.get("predictions", [])
                print(f"✅ 批量预测成功:")
                print(f"   总数量: {result.get('total_count')}")
                
                # 统计预测结果
                category_counts = {}
                for pred in predictions:
                    if pred.get("success"):
                        category = pred.get("prediction", {}).get("category_name", "unknown")
                        category_counts[category] = category_counts.get(category, 0) + 1
                
                print(f"   预测分布:")
                for category, count in category_counts.items():
                    print(f"     {category}: {count}")
            else:
                print(f"❌ 批量预测失败: {result.get('error')}")
            
            return result
        except Exception as e:
            print(f"❌ 批量预测请求失败: {e}")
            return {"success": False, "error": str(e)}
    
    def test_reload_model(self) -> Dict:
        """测试模型重新加载接口"""
        print("🔍 测试模型重新加载接口...")
        try:
            response = self.session.post(f"{self.base_url}/reload")
            result = response.json()
            
            if result.get("success"):
                print(f"✅ 模型重新加载成功: {result.get('message')}")
            else:
                print(f"❌ 模型重新加载失败: {result.get('error')}")
            
            return result
        except Exception as e:
            print(f"❌ 重新加载请求失败: {e}")
            return {"success": False, "error": str(e)}
    
    def run_all_tests(self):
        """运行所有测试"""
        print("🚀 开始API测试...")
        print("=" * 50)
        
        # 测试数据
        test_texts = [
            "ERROR: java.lang.NullPointerException at com.example.Controller.handleRequest",
            "INFO: Application started successfully on port 8080",
            "WARN: Database connection timeout, retrying...",
            "ERROR: Failed to connect to database: Connection refused",
            "INFO: User authentication successful for user: admin",
            "ERROR: Stack overflow in recursive function",
            "WARN: Memory usage is high: 85%",
            "INFO: Health check passed",
            "ERROR: Service unavailable due to maintenance",
            "INFO: Request processed successfully"
        ]
        
        # 运行测试
        tests = [
            ("健康检查", lambda: self.test_health()),
            ("模型信息", lambda: self.test_model_info()),
            ("单个预测", lambda: self.test_single_predict(test_texts[0])),
            ("批量预测", lambda: self.test_batch_predict(test_texts[:5])),
            ("模型重载", lambda: self.test_reload_model())
        ]
        
        results = {}
        for test_name, test_func in tests:
            print(f"\n📋 测试: {test_name}")
            print("-" * 30)
            start_time = time.time()
            result = test_func()
            end_time = time.time()
            
            results[test_name] = {
                "success": result.get("success", False),
                "duration": end_time - start_time,
                "result": result
            }
            
            print(f"⏱️  耗时: {end_time - start_time:.3f}秒")
        
        # 输出测试总结
        print("\n" + "=" * 50)
        print("📊 测试总结:")
        print("=" * 50)
        
        passed = 0
        total = len(tests)
        
        for test_name, test_result in results.items():
            status = "✅ 通过" if test_result["success"] else "❌ 失败"
            duration = test_result["duration"]
            print(f"{test_name}: {status} ({duration:.3f}s)")
            if test_result["success"]:
                passed += 1
        
        print(f"\n🎯 测试结果: {passed}/{total} 通过")
        
        if passed == total:
            print("🎉 所有测试通过！API服务运行正常。")
        else:
            print("⚠️  部分测试失败，请检查API服务状态。")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='API测试脚本')
    parser.add_argument('--url', type=str, default='http://localhost:5000',
                        help='API服务器地址')
    
    args = parser.parse_args()
    
    # 创建测试器
    tester = APITester(args.url)
    
    # 运行测试
    tester.run_all_tests()

if __name__ == "__main__":
    main() 