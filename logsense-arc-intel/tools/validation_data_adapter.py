# -*- coding: utf-8 -*-
"""
验证数据适配器
将改进后的数据转换为验证脚本期望的格式
"""

import pandas as pd
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def adapt_validation_data(input_file: str, output_file: str):
    """适配验证数据格式"""
    logger.info(f"🔄 开始适配验证数据格式...")
    logger.info(f"📂 输入文件: {input_file}")
    logger.info(f"📂 输出文件: {output_file}")
    
    try:
        # 加载改进后的数据
        df = pd.read_csv(input_file)
        logger.info(f"✅ 成功加载 {len(df)} 条记录")
        logger.info(f"📋 原始列名: {df.columns.tolist()}")
        
        # 创建适配后的数据框
        adapted_df = pd.DataFrame()
        
        # 映射列名
        if 'text' in df.columns:
            adapted_df['cleaned_log'] = df['text']
        else:
            raise ValueError("未找到文本列")
            
        if 'label' in df.columns:
            adapted_df['category'] = df['label']
        else:
            raise ValueError("未找到标签列")
        
        # 添加其他必要的列
        adapted_df['source'] = 'github_improved'
        adapted_df['timestamp'] = pd.Timestamp.now().isoformat()
        adapted_df['is_augmented'] = False
        
        # 数据清洗
        adapted_df = adapted_df.dropna(subset=['cleaned_log', 'category'])
        adapted_df = adapted_df[adapted_df['cleaned_log'].str.strip() != '']
        
        logger.info(f"✅ 数据适配完成: {len(adapted_df)} 条记录")
        logger.info(f"📋 适配后列名: {adapted_df.columns.tolist()}")
        
        # 显示类别分布
        category_counts = adapted_df['category'].value_counts()
        logger.info("📈 类别分布:")
        for category, count in category_counts.items():
            percentage = (count / len(adapted_df)) * 100
            logger.info(f"  {category}: {count} 条 ({percentage:.1f}%)")
        
        # 保存适配后的数据
        adapted_df.to_csv(output_file, index=False, encoding='utf-8')
        logger.info(f"💾 适配数据已保存到: {output_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 数据适配失败: {e}")
        return False

def main():
    """主函数"""
    input_file = "data/improved_validation_data.csv"
    output_file = "data/improved_validation_adapted.csv"
    
    success = adapt_validation_data(input_file, output_file)
    
    if success:
        print("\n" + "=" * 60)
        print("✅ 验证数据适配完成！")
        print(f"📁 输出文件: {output_file}")
        print("=" * 60)
    else:
        print("\n❌ 验证数据适配失败！")

if __name__ == "__main__":
    main()
