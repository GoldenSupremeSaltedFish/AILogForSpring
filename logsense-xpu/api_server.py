#!/usr/bin/env python3
"""
日志分类API服务器
基于Flask + joblib的REST API服务
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LogClassificationAPI:
    """日志分类API服务类"""
    
    def __init__(self, model_dir: str = "models"):
        """
        初始化API服务
        
        Args:
            model_dir: 模型文件目录
        """
        self.model_dir = model_dir
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.categories = []
        self.model_timestamp = None
        
        # 加载最新的模型
        self.load_latest_model()
        
        logger.info("API服务初始化完成")
    
    def load_latest_model(self) -> bool:
        """
        加载最新的模型文件
        
        Returns:
            是否加载成功
        """
        try:
            # 查找最新的模型文件
            model_files = []
            for file in os.listdir(self.model_dir):
                if file.startswith("lightgbm_model_") and file.endswith(".txt"):
                    timestamp = file.replace("lightgbm_model_", "").replace(".txt", "")
                    model_files.append((timestamp, file))
            
            if not model_files:
                logger.error("未找到模型文件")
                return False
            
            # 按时间戳排序，选择最新的
            model_files.sort(reverse=True)
            latest_timestamp, latest_model_file = model_files[0]
            
            # 构建文件路径
            model_path = os.path.join(self.model_dir, latest_model_file)
            vectorizer_path = os.path.join(self.model_dir, f"tfidf_vectorizer_{latest_timestamp}.joblib")
            label_encoder_path = os.path.join(self.model_dir, f"label_encoder_{latest_timestamp}.joblib")
            
            # 加载模型组件
            logger.info(f"加载模型: {model_path}")
            self.model = lgb.Booster(model_file=model_path)
            
            logger.info(f"加载向量器: {vectorizer_path}")
            self.vectorizer = joblib.load(vectorizer_path)
            
            logger.info(f"加载标签编码器: {label_encoder_path}")
            self.label_encoder = joblib.load(label_encoder_path)
            
            # 获取类别列表
            self.categories = list(self.label_encoder.classes_)
            self.model_timestamp = latest_timestamp
            
            logger.info(f"模型加载成功 - 时间戳: {latest_timestamp}, 类别数: {len(self.categories)}")
            return True
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def preprocess_text(self, text: str) -> str:
        """
        预处理文本
        
        Args:
            text: 原始文本
            
        Returns:
            预处理后的文本
        """
        import re
        
        # 移除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        
        # 移除特殊字符，保留中英文数字和常用标点
        text = re.sub(r'[^\w\s\u4e00-\u9fff.,!?;:-]', ' ', text)
        
        # 规范化空白字符
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def predict_single(self, text: str) -> Dict:
        """
        预测单个日志文本
        
        Args:
            text: 日志文本
            
        Returns:
            预测结果字典
        """
        try:
            # 预处理文本
            cleaned_text = self.preprocess_text(text)
            
            if not cleaned_text:
                return {
                    "success": False,
                    "error": "文本为空或预处理后为空",
                    "prediction": None,
                    "confidence": 0.0
                }
            
            # TF-IDF特征提取
            X_tfidf = self.vectorizer.transform([cleaned_text])
            
            # LightGBM预测
            predictions = self.model.predict(X_tfidf)
            predicted_class_id = np.argmax(predictions[0])
            confidence = float(np.max(predictions[0]))
            
            # 获取类别名称
            predicted_category = self.categories[predicted_class_id]
            
            return {
                "success": True,
                "text": text,
                "cleaned_text": cleaned_text,
                "prediction": {
                    "category_id": int(predicted_class_id),
                    "category_name": predicted_category,
                    "confidence": confidence,
                    "all_probabilities": predictions[0].tolist()
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"预测失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "prediction": None
            }
    
    def predict_batch(self, texts: List[str]) -> Dict:
        """
        批量预测日志文本
        
        Args:
            texts: 日志文本列表
            
        Returns:
            批量预测结果
        """
        try:
            if not texts:
                return {
                    "success": False,
                    "error": "输入文本列表为空",
                    "predictions": []
                }
            
            results = []
            for i, text in enumerate(texts):
                result = self.predict_single(text)
                result["index"] = i
                results.append(result)
            
            return {
                "success": True,
                "total_count": len(texts),
                "predictions": results,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"批量预测失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "predictions": []
            }
    
    def get_model_info(self) -> Dict:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        return {
            "model_type": "TF-IDF + LightGBM",
            "model_timestamp": self.model_timestamp,
            "categories": self.categories,
            "num_categories": len(self.categories),
            "vectorizer_features": self.vectorizer.get_feature_names_out().shape[0] if self.vectorizer else 0,
            "status": "loaded" if self.model else "not_loaded"
        }

# 创建Flask应用
app = Flask(__name__)
CORS(app)  # 启用跨域支持

# 创建API服务实例
api_service = LogClassificationAPI()

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": api_service.model is not None
    })

@app.route('/model/info', methods=['GET'])
def get_model_info():
    """获取模型信息"""
    return jsonify(api_service.get_model_info())

@app.route('/predict', methods=['POST'])
def predict_single():
    """单个日志预测接口"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                "success": False,
                "error": "缺少text字段"
            }), 400
        
        text = data['text']
        if not isinstance(text, str):
            return jsonify({
                "success": False,
                "error": "text字段必须是字符串"
            }), 400
        
        result = api_service.predict_single(text)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"预测接口错误: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """批量预测接口"""
    try:
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({
                "success": False,
                "error": "缺少texts字段"
            }), 400
        
        texts = data['texts']
        if not isinstance(texts, list):
            return jsonify({
                "success": False,
                "error": "texts字段必须是列表"
            }), 400
        
        if len(texts) > 1000:  # 限制批量大小
            return jsonify({
                "success": False,
                "error": "批量预测数量不能超过1000"
            }), 400
        
        result = api_service.predict_batch(texts)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"批量预测接口错误: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/reload', methods=['POST'])
def reload_model():
    """重新加载模型"""
    try:
        success = api_service.load_latest_model()
        return jsonify({
            "success": success,
            "message": "模型重新加载成功" if success else "模型重新加载失败",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"重新加载模型错误: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/', methods=['GET'])
def index():
    """API根路径"""
    return jsonify({
        "service": "日志分类API服务",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "model_info": "/model/info",
            "predict": "/predict",
            "predict_batch": "/predict/batch",
            "reload": "/reload"
        },
        "timestamp": datetime.now().isoformat()
    })

if __name__ == '__main__':
    # 检查模型是否加载成功
    if api_service.model is None:
        logger.error("模型加载失败，请检查模型文件")
        exit(1)
    
    # 启动服务器
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')
    
    logger.info(f"启动API服务器 - 地址: {host}:{port}")
    logger.info("API端点:")
    logger.info("  GET  /health - 健康检查")
    logger.info("  GET  /model/info - 模型信息")
    logger.info("  POST /predict - 单个预测")
    logger.info("  POST /predict/batch - 批量预测")
    logger.info("  POST /reload - 重新加载模型")
    
    app.run(host=host, port=port, debug=False) 