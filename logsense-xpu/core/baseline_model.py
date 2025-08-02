"""
Baseline模型实现
基于TF-IDF + LightGBM和FastText的日志分类模型
适配Intel XPU环境
"""
import os
import pandas as pd
import numpy as np
import joblib
import time
import psutil
from typing import Dict, List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
# import fasttext  # 暂时注释掉，因为编译问题
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
from tqdm import tqdm
from utils import setup_logging, get_device, check_xpu_availability

class BaselineLogClassifier:
    """Baseline日志分类器"""
    
    def __init__(self, model_type: str = "lightgbm", use_xpu: bool = True):
        """
        初始化分类器
        
        Args:
            model_type: 模型类型 ("lightgbm" 或 "fasttext")
            use_xpu: 是否使用Intel XPU
        """
        self.logger = setup_logging()
        self.model_type = model_type
        self.use_xpu = use_xpu and check_xpu_availability()
        
        # 模型组件
        self.vectorizer = None
        self.model = None
        self.label_encoder = None
        
        # 训练参数
        self.tfidf_params = {
            'max_features': 10000,
            'ngram_range': (1, 2),
            'min_df': 2,
            'max_df': 0.95
        }
        
        self.lightgbm_params = {
            'objective': 'multiclass',
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
        
        self.fasttext_params = {
            'lr': 0.1,
            'epoch': 25,
            'wordNgrams': 2,
            'minCount': 2,
            'minn': 3,
            'maxn': 6,
            'verbose': 2
        }
        
        self.logger.info(f"初始化Baseline分类器 - 模型类型: {model_type}, XPU: {self.use_xpu}")
        
        # 性能监控
        self.start_time = None
        self.memory_usage = []
    
    def _log_performance(self, step_name: str, start_time: float = None):
        """记录性能指标"""
        current_time = time.time()
        memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        self.memory_usage.append((step_name, current_time, memory))
        
        if start_time:
            elapsed = current_time - start_time
            self.logger.info(f"⏱️  {step_name} 完成 - 耗时: {elapsed:.2f}秒, 内存: {memory:.1f}MB")
        else:
            self.logger.info(f"🚀 {step_name} 开始 - 内存: {memory:.1f}MB")
        
        return current_time
    
    def _get_memory_usage(self) -> float:
        """获取当前内存使用量(MB)"""
        return psutil.Process().memory_info().rss / 1024 / 1024
    
    def load_data_from_categories(self, data_dir: str) -> Tuple[pd.DataFrame, List[str]]:
        """
        从分类目录加载数据
        
        Args:
            data_dir: DATA_OUTPUT目录路径
            
        Returns:
            数据框和类别列表
        """
        start_time = self._log_performance("数据加载")
        self.logger.info(f"📁 从目录加载数据: {data_dir}")
        
        all_data = []
        categories = []
        total_files = 0
        processed_files = 0
        
        # 统计文件总数
        for item in os.listdir(data_dir):
            item_path = os.path.join(data_dir, item)
            if os.path.isdir(item_path) and item.startswith(('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')):
                for file in os.listdir(item_path):
                    if file.endswith('.csv'):
                        total_files += 1
        
        self.logger.info(f"📊 发现 {total_files} 个CSV文件")
        
        # 遍历所有分类目录
        for item in tqdm(os.listdir(data_dir), desc="遍历类别目录"):
            item_path = os.path.join(data_dir, item)
            if os.path.isdir(item_path) and item.startswith(('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')):
                # 提取类别名称
                category_name = item.split('_', 1)[1] if '_' in item else item
                categories.append(category_name)
                
                # 读取该目录下的所有CSV文件
                for file in os.listdir(item_path):
                    if file.endswith('.csv'):
                        file_path = os.path.join(item_path, file)
                        processed_files += 1
                        try:
                            df = pd.read_csv(file_path)
                            if 'original_log' in df.columns:
                                df['category'] = category_name
                                df['category_id'] = len(categories) - 1
                                all_data.append(df[['original_log', 'category', 'category_id']])
                                
                                if processed_files % 10 == 0:  # 每处理10个文件输出一次进度
                                    self.logger.info(f"📈 已处理 {processed_files}/{total_files} 个文件")
                        except Exception as e:
                            self.logger.warning(f"❌ 读取文件失败 {file_path}: {e}")
        
        if not all_data:
            raise ValueError("未找到有效的数据文件")
        
        combined_df = pd.concat(all_data, ignore_index=True)
        self.logger.info(f"✅ 数据加载完成 - 总记录: {len(combined_df)}, 类别数: {len(categories)}")
        
        self._log_performance("数据加载", start_time)
        return combined_df, categories
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        预处理数据
        
        Args:
            df: 原始数据框
            
        Returns:
            预处理后的数据框
        """
        start_time = self._log_performance("数据预处理")
        self.logger.info(f"🧹 开始数据预处理... 原始记录数: {len(df)}")
        
        # 清洗文本
        self.logger.info("📝 开始文本清洗...")
        tqdm.pandas(desc="清洗日志文本")
        df['cleaned_log'] = df['original_log'].fillna('').astype(str).progress_apply(self._clean_text)
        
        # 移除空文本
        empty_count = len(df[df['cleaned_log'].str.len() == 0])
        df = df[df['cleaned_log'].str.len() > 0]
        self.logger.info(f"🗑️  移除空文本: {empty_count} 条")
        
        # 移除重复
        original_count = len(df)
        df = df.drop_duplicates(subset=['cleaned_log'])
        duplicate_count = original_count - len(df)
        self.logger.info(f"🔍 移除重复: {duplicate_count} 条")
        
        self.logger.info(f"✅ 预处理完成 - 剩余记录: {len(df)} 条")
        self._log_performance("数据预处理", start_time)
        return df
    
    def _clean_text(self, text: str) -> str:
        """清洗文本"""
        import re
        
        # 移除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        
        # 移除特殊字符，保留中英文数字和常用标点
        text = re.sub(r'[^\w\s\u4e00-\u9fff.,!?;:-]', ' ', text)
        
        # 规范化空白字符
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def train_tfidf_lightgbm(self, X_train: List[str], y_train: np.ndarray, 
                             X_val: Optional[List[str]] = None, 
                             y_val: Optional[np.ndarray] = None) -> None:
        """
        训练TF-IDF + LightGBM模型
        
        Args:
            X_train: 训练文本列表
            y_train: 训练标签
            X_val: 验证文本列表
            y_val: 验证标签
        """
        start_time = self._log_performance("TF-IDF特征提取")
        self.logger.info(f"🔤 开始TF-IDF特征提取... 训练样本数: {len(X_train)}")
        
        # TF-IDF特征提取
        self.vectorizer = TfidfVectorizer(**self.tfidf_params)
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        
        if X_val is not None:
            X_val_tfidf = self.vectorizer.transform(X_val)
            self.logger.info(f"📊 TF-IDF特征维度: {X_train_tfidf.shape}, 验证集: {X_val_tfidf.shape}")
        else:
            self.logger.info(f"📊 TF-IDF特征维度: {X_train_tfidf.shape}")
        
        self._log_performance("TF-IDF特征提取", start_time)
        
        # LightGBM训练
        train_start_time = self._log_performance("LightGBM模型训练")
        self.logger.info(f"🌳 开始LightGBM模型训练... 类别数: {len(np.unique(y_train))}")
        
        # 设置LightGBM参数
        train_params = self.lightgbm_params.copy()
        train_params['num_class'] = len(np.unique(y_train))
        
        # 创建数据集
        train_data = lgb.Dataset(X_train_tfidf, label=y_train)
        if X_val is not None:
            val_data = lgb.Dataset(X_val_tfidf, label=y_val, reference=train_data)
            valid_sets = [train_data, val_data]
            valid_names = ['train', 'valid']
        else:
            valid_sets = [train_data]
            valid_names = ['train']
        
        # 训练模型
        self.model = lgb.train(
            train_params,
            train_data,
            valid_sets=valid_sets,
            valid_names=valid_names,
            num_boost_round=100,
            callbacks=[lgb.log_evaluation(period=10)]
        )
        
        self._log_performance("LightGBM模型训练", train_start_time)
        self.logger.info("✅ TF-IDF + LightGBM模型训练完成")
    
    def train_fasttext(self, X_train: List[str], y_train: np.ndarray,
                       X_val: Optional[List[str]] = None,
                       y_val: Optional[np.ndarray] = None) -> None:
        """
        训练FastText模型
        
        Args:
            X_train: 训练文本列表
            y_train: 训练标签
            X_val: 验证文本列表
            y_val: 验证标签
        """
        self.logger.info("FastText模型暂不可用，请安装Microsoft Visual C++ Build Tools")
        raise NotImplementedError("FastText模型暂不可用，请先安装Microsoft Visual C++ Build Tools")
    
    def train(self, data_dir: str, test_size: float = 0.2, 
              random_state: int = 42) -> Dict:
        """
        训练模型
        
        Args:
            data_dir: 数据目录
            test_size: 测试集比例
            random_state: 随机种子
            
        Returns:
            训练结果字典
        """
        total_start_time = self._log_performance("完整训练流程")
        self.logger.info(f"🎯 开始完整训练流程 - 数据目录: {data_dir}")
        
        # 加载数据
        df, categories = self.load_data_from_categories(data_dir)
        
        # 预处理数据
        df = self.preprocess_data(df)
        
        # 标签编码
        encode_start_time = self._log_performance("标签编码")
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(df['category'])
        self.logger.info(f"🏷️  标签编码完成 - 类别数: {len(categories)}")
        self._log_performance("标签编码", encode_start_time)
        
        # 分割数据
        split_start_time = self._log_performance("数据分割")
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                df['cleaned_log'].tolist(), y,
                test_size=test_size,
                random_state=random_state,
                stratify=y
            )
            
            # 进一步分割验证集
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train,
                test_size=0.2,
                random_state=random_state,
                stratify=y_train
            )
        except ValueError:
            # 如果数据太少无法分层抽样，使用随机分割
            self.logger.warning("⚠️  数据量较少，使用随机分割")
            X_train, X_test, y_train, y_test = train_test_split(
                df['cleaned_log'].tolist(), y,
                test_size=test_size,
                random_state=random_state
            )
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train,
                test_size=0.2,
                random_state=random_state
            )
        
        self.logger.info(f"📊 数据分割完成 - 训练: {len(X_train)}, 验证: {len(X_val)}, 测试: {len(X_test)}")
        self._log_performance("数据分割", split_start_time)
        
        # 训练模型
        if self.model_type == "lightgbm":
            self.train_tfidf_lightgbm(X_train, y_train, X_val, y_val)
        elif self.model_type == "fasttext":
            self.train_fasttext(X_train, y_train, X_val, y_val)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
        
        # 评估模型
        eval_start_time = self._log_performance("模型评估")
        self.logger.info("📈 开始模型评估...")
        train_metrics = self.evaluate(X_train, y_train, "训练集")
        val_metrics = self.evaluate(X_val, y_val, "验证集")
        test_metrics = self.evaluate(X_test, y_test, "测试集")
        self._log_performance("模型评估", eval_start_time)
        
        # 保存模型
        save_start_time = self._log_performance("模型保存")
        self.save_model()
        self._log_performance("模型保存", save_start_time)
        
        # 输出总体性能统计
        self._log_performance("完整训练流程", total_start_time)
        self.logger.info("🎉 训练流程完成！")
        
        return {
            'categories': categories,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'model_type': self.model_type,
            'performance_stats': self.memory_usage
        }
    
    def predict(self, texts: List[str]) -> np.ndarray:
        """
        预测
        
        Args:
            texts: 待预测的文本列表
            
        Returns:
            预测结果
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用train方法")
        
        if self.model_type == "lightgbm":
            # TF-IDF特征提取
            X_tfidf = self.vectorizer.transform(texts)
            # LightGBM预测
            predictions = self.model.predict(X_tfidf)
            return np.argmax(predictions, axis=1)
        
        elif self.model_type == "fasttext":
            # FastText预测
            raise NotImplementedError("FastText模型暂不可用")
            # predictions = []
            # for text in texts:
            #     pred = self.model.predict(text, k=1)
            #     # 提取标签ID
            #     label = pred[0][0].replace('__label__', '')
            #     predictions.append(int(label))
            # return np.array(predictions)
    
    def evaluate(self, X: List[str], y: np.ndarray, dataset_name: str = "") -> Dict:
        """
        评估模型
        
        Args:
            X: 输入文本
            y: 真实标签
            dataset_name: 数据集名称
            
        Returns:
            评估指标
        """
        self.logger.info(f"🔍 开始评估 {dataset_name} - 样本数: {len(X)}")
        
        predictions = self.predict(X)
        
        accuracy = accuracy_score(y, predictions)
        
        # 分类报告
        report = classification_report(y, predictions, output_dict=True)
        
        # 混淆矩阵
        cm = confusion_matrix(y, predictions)
        
        metrics = {
            'accuracy': accuracy,
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1_score': report['weighted avg']['f1-score'],
            'confusion_matrix': cm
        }
        
        self.logger.info(f"📊 {dataset_name}评估结果:")
        self.logger.info(f"   📈 准确率: {accuracy:.4f}")
        self.logger.info(f"   🎯 精确率: {metrics['precision']:.4f}")
        self.logger.info(f"   🔄 召回率: {metrics['recall']:.4f}")
        self.logger.info(f"   ⚖️  F1分数: {metrics['f1_score']:.4f}")
        
        return metrics
    
    def save_model(self, model_dir: str = "models") -> None:
        """保存模型"""
        self.logger.info(f"💾 开始保存模型到目录: {model_dir}")
        os.makedirs(model_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.model_type == "lightgbm":
            # 保存LightGBM模型和TF-IDF向量器
            model_path = os.path.join(model_dir, f"lightgbm_model_{timestamp}.txt")
            vectorizer_path = os.path.join(model_dir, f"tfidf_vectorizer_{timestamp}.joblib")
            label_encoder_path = os.path.join(model_dir, f"label_encoder_{timestamp}.joblib")
            
            self.logger.info(f"💾 保存LightGBM模型: {model_path}")
            self.model.save_model(model_path)
            
            self.logger.info(f"💾 保存TF-IDF向量器: {vectorizer_path}")
            joblib.dump(self.vectorizer, vectorizer_path)
            
            self.logger.info(f"💾 保存标签编码器: {label_encoder_path}")
            joblib.dump(self.label_encoder, label_encoder_path)
            
        elif self.model_type == "fasttext":
            # 保存FastText模型
            raise NotImplementedError("FastText模型暂不可用")
            # model_path = os.path.join(model_dir, f"fasttext_model_{timestamp}.bin")
            # label_encoder_path = os.path.join(model_dir, f"label_encoder_{timestamp}.joblib")
            # 
            # self.model.save_model(model_path)
            # joblib.dump(self.label_encoder, label_encoder_path)
        
        self.logger.info(f"✅ 模型保存完成 - 时间戳: {timestamp}")
    
    def load_model(self, model_dir: str, timestamp: str) -> None:
        """加载模型"""
        if self.model_type == "lightgbm":
            model_path = os.path.join(model_dir, f"lightgbm_model_{timestamp}.txt")
            vectorizer_path = os.path.join(model_dir, f"tfidf_vectorizer_{timestamp}.joblib")
            label_encoder_path = os.path.join(model_dir, f"label_encoder_{timestamp}.joblib")
            
            self.model = lgb.Booster(model_file=model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            self.label_encoder = joblib.load(label_encoder_path)
            
        elif self.model_type == "fasttext":
            raise NotImplementedError("FastText模型暂不可用")
            # model_path = os.path.join(model_dir, f"fasttext_model_{timestamp}.bin")
            # label_encoder_path = os.path.join(model_dir, f"label_encoder_{timestamp}.joblib")
            # 
            # self.model = fasttext.load_model(model_path)
            # self.label_encoder = joblib.load(label_encoder_path)
        
        self.logger.info(f"模型已从 {model_dir} 加载")
    
    def plot_confusion_matrix(self, cm: np.ndarray, categories: List[str], 
                             save_path: str = None) -> None:
        """绘制混淆矩阵"""
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=categories, yticklabels=categories)
        plt.title('混淆矩阵')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """主函数"""
    # 创建分类器
    classifier = BaselineLogClassifier(model_type="lightgbm", use_xpu=True)
    
    # 训练模型
    data_dir = "../../DATA_OUTPUT"  # 相对于logsense-xpu目录的路径
    results = classifier.train(data_dir)
    
    # 打印结果
    print("\n=== 训练结果 ===")
    print(f"模型类型: {results['model_type']}")
    print(f"类别数量: {len(results['categories'])}")
    print(f"测试集准确率: {results['test_metrics']['accuracy']:.4f}")
    print(f"测试集F1分数: {results['test_metrics']['f1_score']:.4f}")
    
    # 绘制混淆矩阵
    classifier.plot_confusion_matrix(
        results['test_metrics']['confusion_matrix'],
        results['categories'],
        "confusion_matrix.png"
    )

if __name__ == "__main__":
    main() 