#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试增强版日志抓取工具
"""

import json
import os
import sys

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 直接导入函数
import importlib.util
spec = importlib.util.spec_from_file_location("issue_helper_enhanced", "issue-helper-enhanced.py")
if spec is None or spec.loader is None:
    print("❌ 无法加载 issue-helper-enhanced.py 文件")
    sys.exit(1)

issue_helper_enhanced = importlib.util.module_from_spec(spec)
spec.loader.exec_module(issue_helper_enhanced)

# 获取需要的函数
load_config = issue_helper_enhanced.load_config
is_log_text_enhanced = issue_helper_enhanced.is_log_text_enhanced
clean_log_text = issue_helper_enhanced.clean_log_text

def test_config_loading():
    """测试配置文件加载"""
    print("🧪 测试配置文件加载...")
    
    # 测试扩展配置
    config = load_config('config_extended.json')
    if config:
        print("✅ 扩展配置文件加载成功")
        print(f"   📊 仓库数量: {len(config['repositories'])}")
        print(f"   🏷️  类别数量: {len(config['categories'])}")
    else:
        print("❌ 扩展配置文件加载失败")
    
    # 测试测试配置
    config = load_config('config_test.json')
    if config:
        print("✅ 测试配置文件加载成功")
        print(f"   📊 仓库数量: {len(config['repositories'])}")
        print(f"   🏷️  类别数量: {len(config['categories'])}")
    else:
        print("❌ 测试配置文件加载失败")

def test_log_recognition():
    """测试日志识别功能"""
    print("\n🧪 测试日志识别功能...")
    
    # 测试用例
    test_cases = [
        {
            "name": "Spring Boot异常",
            "text": """
            org.springframework.beans.factory.BeanCreationException: Error creating bean with name 'userService': Injection of autowired dependencies failed; nested exception is org.springframework.beans.factory.BeanCreationException: Could not autowire field: private com.example.repository.UserRepository com.example.service.UserService.userRepository; nested exception is org.springframework.beans.factory.NoSuchBeanDefinitionException: No qualifying bean of type [com.example.repository.UserRepository] found for dependency: expected at least 1 bean which qualifies as autowire candidate for this dependency. Dependency annotations: {@org.springframework.beans.factory.annotation.Autowired(required=true)}
            """,
            "expected": True
        },
        {
            "name": "数据库异常",
            "text": """
            org.springframework.dao.DataIntegrityViolationException: could not execute statement; SQL [n/a]; constraint [uk_user_email]; nested exception is org.hibernate.exception.ConstraintViolationException: could not execute statement
            """,
            "expected": True
        },
        {
            "name": "安全异常",
            "text": """
            org.springframework.security.authentication.BadCredentialsException: Bad credentials
            """,
            "expected": True
        },
        {
            "name": "普通文本",
            "text": """
            这是一个普通的文本，不包含任何异常信息。
            只是用来测试日志识别功能是否正常工作。
            """,
            "expected": False
        },
        {
            "name": "HTTP异常",
            "text": """
            org.springframework.web.bind.MethodArgumentNotValidException: Validation failed for argument [0] in public org.springframework.http.ResponseEntity<com.example.dto.UserResponse> com.example.controller.UserController.createUser(com.example.dto.UserRequest): [Field error in object 'userRequest' on field 'email': rejected value [invalid-email]; codes [Email.userRequest.email,Email.email,Email.java.lang.String,Email]; arguments [org.springframework.context.support.DefaultMessageSourceResolvable: codes [userRequest.email,email]; arguments []; default message [email],[Ljavax.validation.constraints.Pattern$Flag;@12345678]; default message [must be a well-formed email address]]
            """,
            "expected": True
        }
    ]
    
    # 加载配置获取关键词
    config = load_config('config_test.json')
    if not config:
        print("❌ 无法加载配置文件，跳过日志识别测试")
        return
    
    log_keywords = config.get('log_keywords', [])
    
    passed = 0
    total = len(test_cases)
    
    for test_case in test_cases:
        result = is_log_text_enhanced(test_case['text'], log_keywords)
        if result == test_case['expected']:
            print(f"✅ {test_case['name']}: 通过")
            passed += 1
        else:
            print(f"❌ {test_case['name']}: 失败 (期望: {test_case['expected']}, 实际: {result})")
    
    print(f"\n📊 日志识别测试结果: {passed}/{total} 通过")

def test_text_cleaning():
    """测试文本清理功能"""
    print("\n🧪 测试文本清理功能...")
    
    test_cases = [
        {
            "name": "代码块清理",
            "input": """```java
            Exception in thread "main" java.lang.NullPointerException
            at com.example.Main.main(Main.java:10)
            ```""",
            "expected_contains": "Exception in thread"
        },
        {
            "name": "多余空行清理",
            "input": """ERROR: Database connection failed


            Caused by: java.sql.SQLException""",
            "expected_contains": "ERROR: Database connection failed"
        },
        {
            "name": "首尾空白清理",
            "input": """
            
            WARN: Service unavailable
            
            """,
            "expected_contains": "WARN: Service unavailable"
        }
    ]
    
    passed = 0
    total = len(test_cases)
    
    for test_case in test_cases:
        cleaned = clean_log_text(test_case['input'])
        if test_case['expected_contains'] in cleaned:
            print(f"✅ {test_case['name']}: 通过")
            passed += 1
        else:
            print(f"❌ {test_case['name']}: 失败")
            print(f"   期望包含: {test_case['expected_contains']}")
            print(f"   实际结果: {cleaned[:100]}...")
    
    print(f"\n📊 文本清理测试结果: {passed}/{total} 通过")

def test_output_directory():
    """测试输出目录"""
    print("\n🧪 测试输出目录...")
    
    config = load_config('config_test.json')
    if not config:
        print("❌ 无法加载配置文件")
        return
    
    output_dir = config['output_directory']
    
    if os.path.exists(output_dir):
        print(f"✅ 输出目录存在: {output_dir}")
        
        # 检查目录权限
        try:
            test_file = os.path.join(output_dir, 'test_write_permission.txt')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            print("✅ 输出目录可写")
        except Exception as e:
            print(f"❌ 输出目录不可写: {e}")
    else:
        print(f"⚠️  输出目录不存在: {output_dir}")
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"✅ 已创建输出目录: {output_dir}")
        except Exception as e:
            print(f"❌ 无法创建输出目录: {e}")

def main():
    """主测试函数"""
    print("🚀 增强版日志抓取工具 - 功能测试")
    print("=" * 50)
    
    # 运行各项测试
    test_config_loading()
    test_log_recognition()
    test_text_cleaning()
    test_output_directory()
    
    print("\n" + "=" * 50)
    print("🎉 测试完成！")
    print("\n💡 下一步:")
    print("1. 如果所有测试都通过，可以运行完整的抓取脚本")
    print("2. 使用 run_enhanced_log_crawler.bat 开始抓取")
    print("3. 或者直接运行: python issue-helper-enhanced.py")

if __name__ == "__main__":
    main()
