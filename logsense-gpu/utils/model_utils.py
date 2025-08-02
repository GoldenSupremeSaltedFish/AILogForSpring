#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型工具模块
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import numpy as np

# 可选导入
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


def create_model(model_type: str = 'gradient_boosting'):
    """创建模型"""
    if model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
        model = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=8,
            num_leaves=31,
            random_state=42,
            verbose=-1
        )
        print("🤖 使用LightGBM模型")
    elif model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        print("🤖 使用RandomForest模型")
    else:
        model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=8,
            subsample=0.8,
            random_state=42
        )
        print("🤖 使用GradientBoosting模型")
    
    return model


def create_vectorizer():
    """创建向量化器"""
    return TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.95,
        stop_words='english'
    )


def train_model(model, vectorizer, X_train, y_train, X_test, y_test):
    """训练模型"""
    print("\n🚀 开始训练模型...")
    
    # 特征工程
    print("📝 向量化训练数据...")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print(f"📊 特征维度: {X_train_vec.shape[1]}")
    
    # 训练模型
    print("🏋️ 训练模型...")
    start_time = datetime.now()
    
    model.fit(X_train_vec, y_train)
    
    training_time = (datetime.now() - start_time).total_seconds()
    print(f"⏱️ 训练时间: {training_time:.2f} 秒")
    
    # 预测
    y_pred = model.predict(X_test_vec)
    y_pred_proba = model.predict_proba(X_test_vec)
    
    return y_pred, y_pred_proba, X_test_vec 