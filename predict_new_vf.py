import os
import numpy as np
import torch
import warnings
import argparse
import json
import pandas as pd
from torch.utils.data import DataLoader
from plm_train import *

# 忽略警告
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", DeprecationWarning)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

def set_all_seeds(seed=42):
    """设置所有随机种子以确保可重现性"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def load_model(model_path, model_type, config, esm_dim, prot5_dim, input_dim, device, feature_type):
    """加载指定类型和路径的模型"""
    print(f"加载模型: {model_path} (类型: {model_type}, 特征: {feature_type})")
    
    if "dual" in model_type.lower():
        model = DualPathwayFusion(
            esm_dim=esm_dim,
            prot5_dim=prot5_dim,
            hidden_dim=128,
            num_layers=4,
            num_classes=2,
            rank=8,
            steps=1,
            dropout=config["dropout"]
        )
    else:
        if feature_type == "esm2":
            # 加载ESM2模型
            model = create_model(
                classifier_type='delta',
                input_dim=1280,
                hidden_dim=512,
                num_layers=4,
                dropout=0.5357584107527866,
                rank=4,
                steps=2,
            )
        elif feature_type == "prot5":
        # 加载常规模型
            model = create_model(
                classifier_type='delta',
                input_dim=1024,
                hidden_dim=1024,
                num_layers=6,
                dropout=0.13123101867893097,
                rank=2,
                steps=4,
            )
    
    # 加载模型权重
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict_with_model(model, data_loader, device="cpu", is_fusion=False, feature_type="esm2"):
    """使用模型进行预测"""
    model.eval()
    all_labels = []
    all_preds = []
    all_scores = []
    
    with torch.no_grad():
        for batch in data_loader:
            # 处理batch数据
            if is_fusion:
                # 双特征模型
                (esm_features, prot5_features), labels = batch
                esm_features = esm_features.to(device)
                prot5_features = prot5_features.to(device)
                labels = labels.to(device)
                outputs = model(esm_features, prot5_features)
            else:
                # 单特征模型
                features, labels = batch
                if isinstance(features, tuple):
                    # 如果是特定特征类型的模型，选择相应的特征
                    if feature_type == "esm2":
                        features = features[0]
                    else:  # prot5
                        features = features[1]
                features = features.to(device)
                labels = labels.to(device)
                outputs = model(features)
            
            # 获取预测结果
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            # 收集结果
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_scores.extend(probs.cpu().numpy())
    
    return np.array(all_labels), np.array(all_preds), np.array(all_scores)

def ensemble_predictions(all_scores, all_preds, ensemble_method="simple_avg", weights=None, all_labels=None):
    """使用不同的集成方法组合预测结果"""
    num_models = len(all_scores)
    num_samples = len(all_scores[0])
    
    # 如果未提供权重，则使用均等权重
    if weights is None:
        weights = np.ones(num_models) / num_models
    
    # 根据不同的集成方法组合预测
    if ensemble_method == "simple_avg":
        # 简单平均集成
        ensemble_scores = np.zeros_like(all_scores[0])
        for i in range(num_models):
            ensemble_scores += all_scores[i] / num_models
        ensemble_preds = np.argmax(ensemble_scores, axis=1)
        
    elif ensemble_method == "weighted_avg":
        # 加权平均集成
        ensemble_scores = np.zeros_like(all_scores[0])
        for i in range(num_models):
            ensemble_scores += all_scores[i] * weights[i]
        ensemble_preds = np.argmax(ensemble_scores, axis=1)
        
    elif ensemble_method == "majority_vote":
        # 多数投票集成
        votes = np.zeros((num_samples, 2))
        for i in range(num_models):
            for j in range(num_samples):
                votes[j, all_preds[i][j]] += 1
        ensemble_preds = np.argmax(votes, axis=1)
        
        # 对于投票方法，使用投票比例作为置信度
        ensemble_scores = votes / num_models
        
    elif ensemble_method == "stacking":
        from sklearn.linear_model import LogisticRegression
        
        # 检查是否提供了真实标签
        if all_labels is None:
            raise ValueError("Stacking方法需要提供真实标签")
        
        # 准备stacking特征
        X_stack = np.hstack([scores[:, 1].reshape(-1, 1) for scores in all_scores])
        
        # 使用交叉验证训练stacking模型
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        # 预测值
        ensemble_preds = np.zeros(num_samples)
        ensemble_probs = np.zeros((num_samples, 2))
        
        # 使用简单的logistic回归作为元模型
        for train_idx, test_idx in kf.split(X_stack):
            X_train, X_test = X_stack[train_idx], X_stack[test_idx]
            y_train = all_labels[train_idx]
            
            # 训练元模型
            meta_model = LogisticRegression(random_state=42)
            meta_model.fit(X_train, y_train)
            
            # 预测
            ensemble_preds[test_idx] = meta_model.predict(X_test)
            ensemble_probs[test_idx] = meta_model.predict_proba(X_test)
        
        ensemble_scores = ensemble_probs
        
    elif ensemble_method == "gradient_boosted_ensemble":
        from sklearn.ensemble import GradientBoostingClassifier
        
        # 检查是否提供了真实标签
        if all_labels is None:
            raise ValueError("GradientBoostedEnsemble方法需要提供真实标签")
        
        # 准备特征
        X_stack = np.hstack([scores[:, 1].reshape(-1, 1) for scores in all_scores])
        
        # 使用交叉验证训练GBDT模型
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        # 预测值
        ensemble_preds = np.zeros(num_samples)
        ensemble_probs = np.zeros((num_samples, 2))
        
        for train_idx, test_idx in kf.split(X_stack):
            X_train, X_test = X_stack[train_idx], X_stack[test_idx]
            y_train = all_labels[train_idx]
            
            # 训练GBDT模型
            gbdt = GradientBoostingClassifier(random_state=42)
            gbdt.fit(X_train, y_train)
            
            # 预测
            ensemble_preds[test_idx] = gbdt.predict(X_test)
            prob = gbdt.predict_proba(X_test)
            ensemble_probs[test_idx] = prob
        
        ensemble_scores = ensemble_probs
    
    else:
        raise ValueError(f"不支持的集成方法: {ensemble_method}")
    
    return ensemble_preds, ensemble_scores

def main():
    parser = argparse.ArgumentParser(description="使用多种集成方法进行测试并输出预测结果")
    parser.add_argument("--config", default='config.json', type=str, help="配置文件路径")
    parser.add_argument("--output", default='ensemble_test_results.csv', type=str, help="输出CSV文件路径")
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, "r") as f:
        config = json.load(f)
    
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 数据集路径
    test_esm_path = config.get("test_esm_path", 'test_esm2_t33_650M_UR50D_mean.h5')
    test_prot5_path = config.get("test_prot5_path", "test_prot_features_modified.h5")
    
    # 加载测试数据集
    test_dataset = DualEmbeddingDataset(
        esm_h5_path=test_esm_path,
        prot5_h5_path=test_prot5_path,
        split='all'
    )
    collate_function = dual_features_collate_fn
    
    # 获取特征维度
    sample = test_dataset[0]
    esm_features, prot5_features = sample[0]
    esm_dim = esm_features.shape[-1]
    prot5_dim = prot5_features.shape[-1]
    print(f"ESM特征维度: {esm_dim}, ProtT5特征维度: {prot5_dim}")
    input_dim = None

    # 创建数据加载器
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.get("batch_size", 64), 
        shuffle=False,
        collate_fn=collate_function
    )
    
    # 加载所有模型并进行预测
    all_labels = []
    all_model_preds = []
    all_model_scores = []
    model_names = []
    
    for model_config in config["models"]:
        model_type = model_config["type"]
        model_path = model_config["path"]
        feature_type = model_config.get("feature_type", "esm2")
        model_name = model_config.get("name", model_type)
        model_names.append(f"{model_name}_{feature_type}")
        
        # 判断是否是融合模型
        is_fusion = "dual" in model_type.lower() or "fusion" in model_type.lower()
        
        # 加载模型
        model = load_model(
            model_path=model_path,
            model_type=model_type,
            config=config,
            esm_dim=esm_dim,
            prot5_dim=prot5_dim,
            input_dim=input_dim,
            device=device,
            feature_type=feature_type
        )
        
        # 为ProtT5模型准备特定数据加载器
        current_loader = test_loader
        if feature_type == "prot5" and not is_fusion:
            test_dataset_prot5 = H5Dataset(
                h5_path=test_esm_path,
                feature_type='prot5',
                prot5_path=test_prot5_path,
            )
            current_loader = DataLoader(
                test_dataset_prot5, 
                batch_size=config.get("batch_size", 64), 
                shuffle=False,
                collate_fn=collate_fn
            )
        
        # 进行预测
        labels, preds, scores = predict_with_model(model, current_loader, device, is_fusion, feature_type=feature_type)
        
        # 只保存第一个模型的标签，因为所有模型使用相同的测试集
        if len(all_labels) == 0:
            all_labels = labels
        
        all_model_preds.append(preds)
        all_model_scores.append(scores)
        
        print(f"完成模型 {model_name} ({feature_type}) 的预测")
    
    # 获取模型权重
    model_weights = np.array([model.get("weight", 1.0) for model in config["models"]])
    model_weights = model_weights / model_weights.sum()  # 归一化权重
    
    # 定义集成方法
    ensemble_methods = [
        'simple_avg',
        'weighted_avg',
        'majority_vote',
        'stacking',
        'gradient_boosted_ensemble'
    ]
    
    # 创建结果数据框
    results_df = pd.DataFrame({
        "ID": [f"sample_{i}" for i in range(len(all_labels))],
        "True_Label": all_labels
    })
    
    # 添加每个模型的预测结果
    for i, model_name in enumerate(model_names):
        results_df[f"{model_name}_Prediction"] = all_model_preds[i]
        results_df[f"{model_name}_Confidence"] = all_model_scores[i][:, 1]
    
    # 使用每种集成方法并保存结果
    for method in ensemble_methods:
        try:
            print(f"应用集成方法: {method}")
            ensemble_preds, ensemble_scores = ensemble_predictions(
                all_model_scores, all_model_preds,
                ensemble_method=method,
                weights=model_weights,
                all_labels=all_labels
            )
            
            # 添加到结果数据框
            results_df[f"{method}_Prediction"] = ensemble_preds
            results_df[f"{method}_Confidence"] = ensemble_scores[:, 1]
            
        except Exception as e:
            print(f"应用集成方法 {method} 失败: {str(e)}")
    
    # 保存结果到CSV文件
    results_df.to_csv(args.output, index=False)
    print(f"\n所有预测结果已保存到: {args.output}")

if __name__ == "__main__":
    # 设置随机种子
    set_all_seeds(42)
    main()