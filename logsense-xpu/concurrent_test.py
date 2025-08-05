#!/usr/bin/env python3
"""
并发测试脚本
模拟每秒10条请求，每条请求20行日志
"""

import requests
import json
import time
import threading
import random
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics

class ConcurrentTester:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.results = []
        self.lock = threading.Lock()
        
        # 测试日志样本
        self.log_samples = [
            "ERROR: java.lang.NullPointerException at com.example.Controller.handleRequest",
            "INFO: Application started successfully on port 8080",
            "WARN: Database connection timeout, retrying...",
            "ERROR: Failed to connect to database server",
            "INFO: User login successful - username: admin",
            "WARN: High memory usage detected: 85%",
            "ERROR: Authentication failed for user: invalid_user",
            "INFO: File uploaded successfully: document.pdf",
            "WARN: Slow query detected: SELECT * FROM users WHERE id = ?",
            "ERROR: Network timeout while connecting to external API",
            "INFO: Cache miss for key: user_profile_123",
            "WARN: Disk space low: 10% remaining",
            "ERROR: Invalid JSON format in request body",
            "INFO: Background job completed: email_sender",
            "WARN: CPU usage high: 90%",
            "ERROR: Permission denied: user cannot access /admin",
            "INFO: Session created for user: john_doe",
            "WARN: Connection pool exhausted",
            "ERROR: File not found: /var/log/app.log",
            "INFO: Configuration reloaded successfully"
        ]
    
    def generate_test_logs(self, count=20):
        """生成测试日志"""
        logs = []
        for _ in range(count):
            log = random.choice(self.log_samples)
            # 添加一些随机变化
            if "ERROR" in log:
                log = log.replace("com.example", f"com.example.{random.randint(1, 10)}")
            elif "INFO" in log:
                log = log.replace("8080", str(random.randint(8000, 9000)))
            elif "WARN" in log:
                log = log.replace("85%", f"{random.randint(70, 95)}%")
            logs.append(log)
        return logs
    
    def single_request(self, request_id):
        """单个请求测试"""
        start_time = time.time()
        
        try:
            # 生成20行测试日志
            test_logs = self.generate_test_logs(20)
            
            # 发送批量预测请求
            data = {"texts": test_logs}
            response = requests.post(f"{self.base_url}/predict/batch", json=data, timeout=30)
            
            end_time = time.time()
            response_time = (end_time - start_time) * 1000  # 转换为毫秒
            
            result = {
                "request_id": request_id,
                "success": response.status_code == 200,
                "response_time_ms": response_time,
                "status_code": response.status_code,
                "timestamp": datetime.now().isoformat()
            }
            
            if response.status_code == 200:
                response_data = response.json()
                result.update({
                    "total_predictions": response_data.get("total_count", 0),
                    "device_used": response_data.get("device_used", "N/A"),
                    "successful_predictions": sum(1 for pred in response_data.get("predictions", []) if pred.get("success", False))
                })
            else:
                result["error"] = response.text
            
            return result
            
        except Exception as e:
            end_time = time.time()
            response_time = (end_time - start_time) * 1000
            
            return {
                "request_id": request_id,
                "success": False,
                "response_time_ms": response_time,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def run_concurrent_test(self, requests_per_second=10, duration_seconds=60):
        """运行并发测试"""
        print(f"🚀 开始并发测试")
        print(f"�� 配置: {requests_per_second} 请求/秒, 持续 {duration_seconds} 秒")
        print(f"📝 每条请求: 20行日志")
        print(f"🎯 总请求数: {requests_per_second * duration_seconds}")
        print("=" * 60)
        
        start_time = time.time()
        request_id = 0
        results = []
        
        with ThreadPoolExecutor(max_workers=requests_per_second * 2) as executor:
            # 提交初始请求
            futures = []
            for _ in range(requests_per_second):
                request_id += 1
                future = executor.submit(self.single_request, request_id)
                futures.append(future)
            
            # 持续发送请求
            while time.time() - start_time < duration_seconds:
                # 等待一秒
                time.sleep(1)
                
                # 提交新的请求
                for _ in range(requests_per_second):
                    request_id += 1
                    future = executor.submit(self.single_request, request_id)
                    futures.append(future)
                
                # 收集已完成的结果
                for future in as_completed(futures):
                    if future.done():
                        result = future.result()
                        results.append(result)
                        futures.remove(future)
                        
                        # 实时显示进度
                        if len(results) % 10 == 0:
                            elapsed = time.time() - start_time
                            success_count = sum(1 for r in results if r.get("success", False))
                            print(f"⏱️  已处理 {len(results)} 个请求 ({elapsed:.1f}s), 成功: {success_count}")
        
        # 等待所有请求完成
        for future in futures:
            result = future.result()
            results.append(result)
        
        return results
    
    def analyze_results(self, results):
        """分析测试结果"""
        if not results:
            print("❌ 没有测试结果")
            return
        
        print("\n" + "=" * 60)
        print("📈 测试结果分析")
        print("=" * 60)
        
        # 基本统计
        total_requests = len(results)
        successful_requests = sum(1 for r in results if r.get("success", False))
        failed_requests = total_requests - successful_requests
        
        print(f"📊 总请求数: {total_requests}")
        print(f"✅ 成功请求: {successful_requests}")
        print(f"❌ 失败请求: {failed_requests}")
        print(f"📈 成功率: {successful_requests/total_requests*100:.2f}%")
        
        # 响应时间统计
        response_times = [r.get("response_time_ms", 0) for r in results if r.get("success", False)]
        if response_times:
            print(f"\n⏱️  响应时间统计 (毫秒):")
            print(f"   平均响应时间: {statistics.mean(response_times):.2f}ms")
            print(f"   中位数响应时间: {statistics.median(response_times):.2f}ms")
            print(f"   最小响应时间: {min(response_times):.2f}ms")
            print(f"   最大响应时间: {max(response_times):.2f}ms")
            print(f"   标准差: {statistics.stdev(response_times):.2f}ms")
        
        # 设备使用统计
        devices = {}
        for r in results:
            if r.get("success") and "device_used" in r:
                device = r["device_used"]
                devices[device] = devices.get(device, 0) + 1
        
        if devices:
            print(f"\n🖥️  设备使用统计:")
            for device, count in devices.items():
                print(f"   {device}: {count} 次 ({count/total_requests*100:.1f}%)")
        
        # 吞吐量计算
        if response_times:
            avg_response_time = statistics.mean(response_times)
            throughput = 1000 / avg_response_time if avg_response_time > 0 else 0
            print(f"\n🚀 性能指标:")
            print(f"   平均吞吐量: {throughput:.2f} 请求/秒")
            print(f"   理论最大吞吐量: {1000/min(response_times):.2f} 请求/秒")
        
        # 错误分析
        errors = {}
        for r in results:
            if not r.get("success") and "error" in r:
                error_type = type(r["error"]).__name__
                errors[error_type] = errors.get(error_type, 0) + 1
        
        if errors:
            print(f"\n❌ 错误分析:")
            for error_type, count in errors.items():
                print(f"   {error_type}: {count} 次")
        
        print("\n" + "=" * 60)
        print("✅ 并发测试完成！")

def main():
    tester = ConcurrentTester()
    
    # 运行并发测试
    print("开始并发测试...")
    results = tester.run_concurrent_test(requests_per_second=10, duration_seconds=60)
    
    # 分析结果
    tester.analyze_results(results)

if __name__ == "__main__":
    main()