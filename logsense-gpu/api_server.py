#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LogSense GPU API 服务器
提供日志分类预测服务
"""

import os
import json
import joblib
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class LogSensePredictor:
    """日志分类预测器"""
    
    def __init__(self, model_dir: str = "logsense-gpu/results/models"):
        self.model_dir = Path(model_dir)
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.target_classes = []
        self.model_config = {}
        self._load_models()
    
    def _load_models(self):
        """加载训练好的模型"""
        try:
            # 查找最新的模型文件
            model_files = list(self.model_dir.glob("enhanced_model_*.joblib"))
            if not model_files:
                model_files = list(self.model_dir.glob("baseline_model_*.pkl"))
            
            if not model_files:
                raise FileNotFoundError("未找到模型文件")
            
            latest_model_file = max(model_files, key=lambda x: x.stat().st_mtime)
            timestamp = latest_model_file.stem.split('_')[-1]
            
            logger.info(f"加载模型文件: {latest_model_file}")
            
            # 加载模型
            if latest_model_file.suffix == '.joblib':
                self.model = joblib.load(latest_model_file)
                
                vectorizer_file = self.model_dir / f"vectorizer_{timestamp}.joblib"
                if vectorizer_file.exists():
                    self.vectorizer = joblib.load(vectorizer_file)
                
                label_encoder_file = self.model_dir / f"label_encoder_{timestamp}.joblib"
                if label_encoder_file.exists():
                    self.label_encoder = joblib.load(label_encoder_file)
            else:
                import pickle
                with open(latest_model_file, 'rb') as f:
                    self.model = pickle.load(f)
            
            # 加载模型配置
            config_file = self.model_dir / f"model_config_{timestamp}.json"
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    self.model_config = json.load(f)
                    self.target_classes = self.model_config.get('target_classes', [])
            
            logger.info(f"模型加载成功，目标类别: {self.target_classes}")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def predict(self, log_text: str, top_k: int = 3) -> Dict:
        """预测日志类别"""
        try:
            if not self.model or not self.vectorizer:
                raise ValueError("模型或向量器未正确加载")
            
            if isinstance(log_text, str):
                log_text = [log_text]
            
            X_vec = self.vectorizer.transform(log_text)
            y_pred = self.model.predict(X_vec)
            y_pred_proba = self.model.predict_proba(X_vec)
            
            predicted_class = y_pred[0]
            confidence = np.max(y_pred_proba[0])
            
            # 获取Top-K预测结果
            top_k_indices = np.argsort(y_pred_proba[0])[-top_k:][::-1]
            top_k_predictions = []
            
            for idx in top_k_indices:
                class_name = self.target_classes[idx] if idx < len(self.target_classes) else f"class_{idx}"
                prob = y_pred_proba[0][idx]
                top_k_predictions.append({
                    "class": class_name,
                    "confidence": float(prob),
                    "percentage": float(prob * 100)
                })
            
            if self.label_encoder:
                predicted_class = self.label_encoder.inverse_transform([predicted_class])[0]
            
            result = {
                "predicted_class": predicted_class,
                "confidence": float(confidence),
                "confidence_percentage": float(confidence * 100),
                "top_k_predictions": top_k_predictions,
                "timestamp": datetime.now().isoformat(),
                "model_info": {
                    "model_type": self.model_config.get("model_type", "unknown"),
                    "feature_dim": self.model_config.get("feature_dim", 0),
                    "target_classes": self.target_classes
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"预测失败: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return {
            "model_type": self.model_config.get("model_type", "unknown"),
            "target_classes": self.target_classes,
            "feature_dim": self.model_config.get("feature_dim", 0),
            "sample_size": self.model_config.get("sample_size", 0),
            "training_time": self.model_config.get("training_time", ""),
            "model_loaded": self.model is not None,
            "vectorizer_loaded": self.vectorizer is not None,
            "label_encoder_loaded": self.label_encoder is not None
        }

# 全局预测器实例
predictor = None

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": predictor is not None and predictor.model is not None
    })

@app.route('/model/info', methods=['GET'])
def get_model_info():
    """获取模型信息"""
    if predictor is None:
        return jsonify({"error": "模型未加载"}), 500
    
    return jsonify(predictor.get_model_info())

@app.route('/predict', methods=['POST'])
def predict():
    """预测接口"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "请求数据为空"}), 400
        
        log_text = data.get('log_text', '')
        if not log_text:
            return jsonify({"error": "日志文本不能为空"}), 400
        
        top_k = data.get('top_k', 3)
        if not isinstance(top_k, int) or top_k < 1:
            top_k = 3
        
        result = predictor.predict(log_text, top_k)
        
        if "error" in result:
            return jsonify(result), 500
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"预测接口错误: {e}")
        return jsonify({"error": str(e), "timestamp": datetime.now().isoformat()}), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """批量预测接口"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "请求数据为空"}), 400
        
        log_texts = data.get('log_texts', [])
        if not log_texts or not isinstance(log_texts, list):
            return jsonify({"error": "日志文本列表不能为空"}), 400
        
        top_k = data.get('top_k', 3)
        if not isinstance(top_k, int) or top_k < 1:
            top_k = 3
        
        results = []
        for i, log_text in enumerate(log_texts):
            result = predictor.predict(log_text, top_k)
            result['index'] = i
            results.append(result)
        
        return jsonify({
            "predictions": results,
            "total_count": len(results),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"批量预测接口错误: {e}")
        return jsonify({"error": str(e), "timestamp": datetime.now().isoformat()}), 500

@app.route('/classes', methods=['GET'])
def get_classes():
    """获取支持的类别列表"""
    if predictor is None:
        return jsonify({"error": "模型未加载"}), 500
    
    return jsonify({
        "classes": predictor.target_classes,
        "count": len(predictor.target_classes)
    })

def init_predictor():
    """初始化预测器"""
    global predictor
    try:
        predictor = LogSensePredictor()
        logger.info("预测器初始化成功")
    except Exception as e:
        logger.error(f"预测器初始化失败: {e}")
        raise

if __name__ == '__main__':
    init_predictor()
    
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"启动LogSense API服务器，端口: {port}")
    app.run(host='0.0.0.0', port=port, debug=debug) 