# -*- coding: utf-8 -*-
"""
过滤已知标签脚本
过滤掉训练时没有见过的标签，只保留已知的类别
"""

import pandas as pd
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def filter_known_labels(input_file: str, output_file: str):
    """过滤已知标签"""
    logger.info(f"🔍 开始过滤已知标签...")
    logger.info(f"📂 输入文件: {input_file}")
    logger.info(f"📂 输出文件: {output_file}")
    
    try:
        # 读取数据
        df = pd.read_csv(input_file)
        logger.info(f"✅ 成功加载 {len(df)} 条记录")
        
        # 训练时已知的标签（从训练数据中获取）
        known_labels = [
            'database_exception',
            'business_logic', 
            'connection_issue',
            'stack_exception',
            'auth_authorization',
            'config_environment',
            'normal_operation',
            'memory_performance',
            'monitoring_heartbeat'
        ]
        
        logger.info(f"📋 已知标签: {known_labels}")
        
        # 显示原始类别分布
        original_counts = df['category'].value_counts()
        logger.info("📈 原始类别分布:")
        for category, count in original_counts.items():
            logger.info(f"  {category}: {count} 条")
        
        # 过滤数据
        filtered_df = df[df['category'].isin(known_labels)].copy()
        
        logger.info(f"✅ 过滤完成: {len(df)} -> {len(filtered_df)} 条记录")
        
        # 显示过滤后的类别分布
        filtered_counts = filtered_df['category'].value_counts()
        logger.info("📈 过滤后类别分布:")
        for category, count in filtered_counts.items():
            percentage = (count / len(filtered_df)) * 100
            logger.info(f"  {category}: {count} 条 ({percentage:.1f}%)")
        
        # 保存过滤后的数据
        filtered_df.to_csv(output_file, index=False, encoding='utf-8')
        logger.info(f"💾 过滤数据已保存到: {output_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 过滤失败: {e}")
        return False

def main():
    """主函数"""
    logger.info("🚀 已知标签过滤器启动...")
    
    # 输入和输出文件路径
    input_file = "data/issue_logs_adapted_for_validation.csv"
    output_file = "data/issue_logs_filtered_for_validation.csv"
    
    # 检查输入文件是否存在
    if not Path(input_file).exists():
        logger.error(f"❌ 输入文件不存在: {input_file}")
        return
    
    # 执行过滤
    if filter_known_labels(input_file, output_file):
        logger.info("🎉 标签过滤完成！")
        logger.info(f"📁 过滤后的文件: {output_file}")
        logger.info("💡 现在可以使用此文件进行模型验证")
    else:
        logger.error("❌ 标签过滤失败")

if __name__ == "__main__":
    main()
