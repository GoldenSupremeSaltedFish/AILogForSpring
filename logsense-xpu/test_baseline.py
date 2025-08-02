#!/usr/bin/env python3
"""
Baseline模型测试脚本
用于验证模型的基本功能
"""
import os
import sys
import tempfile
import shutil
from datetime import datetime

# 添加core目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

def test_data_loading():
    """测试数据加载功能"""
    print("=== 测试数据加载 ===")
    
    try:
        from baseline_model import BaselineLogClassifier
        
        # 创建分类器
        classifier = BaselineLogClassifier(model_type="lightgbm", use_xpu=False)
        
        # 测试数据加载
        data_dir = "../../DATA_OUTPUT"
        if os.path.exists(data_dir):
            df, categories = classifier.load_data_from_categories(data_dir)
            print(f"✓ 成功加载数据: {len(df)} 条记录, {len(categories)} 个类别")
            print(f"  类别: {categories}")
            return True
        else:
            print(f"✗ 数据目录不存在: {data_dir}")
            return False
            
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        return False

def test_preprocessing():
    """测试数据预处理功能"""
    print("\n=== 测试数据预处理 ===")
    
    try:
        from baseline_model import BaselineLogClassifier
        import pandas as pd
        
        # 创建测试数据
        test_data = pd.DataFrame({
            'original_log': [
                "2025-01-01 ERROR: Database connection failed",
                "2025-01-01 INFO: Application started successfully",
                "<p>HTML content</p>",
                "   multiple   spaces   ",
                ""
            ]
        })
        
        # 创建分类器
        classifier = BaselineLogClassifier(model_type="lightgbm", use_xpu=False)
        
        # 测试预处理
        processed_df = classifier.preprocess_data(test_data)
        print(f"✓ 预处理完成: {len(processed_df)} 条记录")
        print(f"  原始记录数: {len(test_data)}")
        print(f"  处理后记录数: {len(processed_df)}")
        
        # 显示清洗后的文本
        print("  清洗后的文本示例:")
        for i, text in enumerate(processed_df['cleaned_log'].head(3)):
            print(f"    {i+1}. {text[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ 预处理失败: {e}")
        return False

def test_model_creation():
    """测试模型创建功能"""
    print("\n=== 测试模型创建 ===")
    
    try:
        from baseline_model import BaselineLogClassifier
        
        # 测试LightGBM分类器
        lgb_classifier = BaselineLogClassifier(model_type="lightgbm", use_xpu=False)
        print("✓ LightGBM分类器创建成功")
        
        # 测试FastText分类器
        ft_classifier = BaselineLogClassifier(model_type="fasttext", use_xpu=False)
        print("✓ FastText分类器创建成功")
        
        return True
        
    except Exception as e:
        print(f"✗ 模型创建失败: {e}")
        return False

def test_small_training():
    """测试小规模训练"""
    print("\n=== 测试小规模训练 ===")
    
    try:
        from baseline_model import BaselineLogClassifier
        import pandas as pd
        import tempfile
        import os
        
        # 创建临时数据目录
        with tempfile.TemporaryDirectory() as temp_dir:
            # 创建测试数据
            test_categories = ["error", "info", "warning"]
            test_data = []
            
            for i, category in enumerate(test_categories):
                category_dir = os.path.join(temp_dir, f"{i+1:02d}_{category}")
                os.makedirs(category_dir, exist_ok=True)
                
                # 创建测试CSV文件
                test_df = pd.DataFrame({
                    'original_log': [
                        f"ERROR: {category} message 1",
                        f"ERROR: {category} message 2",
                        f"INFO: {category} message 3",
                        f"WARN: {category} message 4",
                        f"DEBUG: {category} message 5"
                    ]
                })
                
                csv_file = os.path.join(category_dir, f"{category}_test.csv")
                test_df.to_csv(csv_file, index=False)
            
            # 创建分类器
            classifier = BaselineLogClassifier(model_type="lightgbm", use_xpu=False)
            
                    # 训练模型
            results = classifier.train(temp_dir, test_size=0.1)  # 减小测试集比例
            
            print(f"✓ 小规模训练成功")
            print(f"  模型类型: {results['model_type']}")
            print(f"  类别数量: {len(results['categories'])}")
            print(f"  测试集准确率: {results['test_metrics']['accuracy']:.4f}")
            
            return True
            
    except Exception as e:
        print(f"✗ 小规模训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("Baseline模型功能测试")
    print("=" * 50)
    
    tests = [
        ("数据加载", test_data_loading),
        ("数据预处理", test_preprocessing),
        ("模型创建", test_model_creation),
        ("小规模训练", test_small_training)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name}测试通过")
            else:
                print(f"✗ {test_name}测试失败")
        except Exception as e:
            print(f"✗ {test_name}测试异常: {e}")
    
    print("\n" + "=" * 50)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！Baseline模型可以正常使用。")
    else:
        print("⚠️  部分测试失败，请检查环境配置。")
    
    return passed == total

if __name__ == "__main__":
    main() 