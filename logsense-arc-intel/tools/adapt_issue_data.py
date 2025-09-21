# -*- coding: utf-8 -*-
"""
Issue日志数据适配器
将我们的issue日志数据转换为验证脚本期望的格式
"""

import pandas as pd
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def adapt_issue_data(input_file: str, output_file: str):
    """适配issue日志数据格式"""
    logger.info(f"🔄 开始适配数据格式...")
    logger.info(f"📂 输入文件: {input_file}")
    logger.info(f"📂 输出文件: {output_file}")
    
    try:
        # 读取原始数据
        df = pd.read_csv(input_file)
        logger.info(f"✅ 成功加载 {len(df)} 条记录")
        
        # 显示原始列名
        logger.info(f"📋 原始列名: {df.columns.tolist()}")
        
        # 创建适配后的数据框
        adapted_df = pd.DataFrame()
        
        # 映射列名
        if 'text' in df.columns:
            adapted_df['cleaned_log'] = df['text']
        elif 'cleaned_message' in df.columns:
            adapted_df['cleaned_log'] = df['cleaned_message']
        else:
            raise ValueError("未找到文本列")
        
        if 'label' in df.columns:
            adapted_df['category'] = df['label']
        elif 'auto_label' in df.columns:
            adapted_df['category'] = df['auto_label']
        else:
            raise ValueError("未找到标签列")
        
        # 添加其他必要的列（如果存在）
        for col in ['source', 'timestamp', 'is_augmented']:
            if col in df.columns:
                adapted_df[col] = df[col]
        
        # 添加结构化特征列（如果存在）
        feature_cols = [col for col in df.columns if col not in ['text', 'label', 'cleaned_message', 'auto_label', 'source', 'timestamp', 'is_augmented']]
        for col in feature_cols:
            adapted_df[col] = df[col]
        
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
    logger.info("🚀 Issue日志数据适配器启动...")
    
    # 输入和输出文件路径
    input_file = "data/processed_logs_issue_enhanced.csv"
    output_file = "data/issue_logs_adapted_for_validation.csv"
    
    # 检查输入文件是否存在
    if not Path(input_file).exists():
        logger.error(f"❌ 输入文件不存在: {input_file}")
        return
    
    # 执行数据适配
    if adapt_issue_data(input_file, output_file):
        logger.info("🎉 数据适配完成！")
        logger.info(f"📁 适配后的文件: {output_file}")
        logger.info("💡 现在可以使用此文件进行模型验证")
    else:
        logger.error("❌ 数据适配失败")

if __name__ == "__main__":
    main()
