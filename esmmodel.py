import os
from typing import Counter
import warnings
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_curve, auc
from plm_train import DualEmbeddingDataset, dual_features_collate_fn, H5Dataset, collate_fn, create_model
import json
import argparse
import matplotlib.pyplot as plt
from model_type import DualPathwayFusion

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", DeprecationWarning)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

def calculate_metrics(labels, predictions, scores=None):
    """计算各项评估指标"""
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    sn = tp / (tp + fn) if (tp + fn) > 0 else 0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * precision * sn / (precision + sn) if (precision + sn) > 0 else 0
    mcc_numerator = tp * tn - fp * fn
    mcc_denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = mcc_numerator / mcc_denominator if mcc_denominator > 0 else 0
    
    # 计算AUC和AUPR（如果提供了概率分数）
    auc_score = 0
    aupr_score = 0
    if scores is not None:
        try:
            auc_score = roc_auc_score(labels, scores[:, 1])
            precision, recall, _ = precision_recall_curve(labels, scores[:, 1])
            aupr_score = auc(recall, precision)
        except:
            pass
    
    return sn * 100, sp * 100, acc * 100, f1 * 100, mcc * 100, auc_score * 100, aupr_score * 100

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

def predict_with_model(model, loader, device, is_fusion_model, feature_type):
    """使用模型进行预测并返回结果"""
    all_labels = []
    all_preds = []
    all_scores = []
    
    with torch.no_grad():
        for batch in loader:
            x, y, lengths = batch
            
            if is_fusion_model:
                # 处理双特征输入
                esm_features, prot5_features = x
                esm_features = esm_features.to(device)
                prot5_features = prot5_features.to(device)
                x = (esm_features, prot5_features)

            else:
                if feature_type == "esm2":
                    esm_features, prot5_features = x
                    # 处理ESM2特征
                    x = esm_features.to(device)
                elif feature_type == "prot5":
                # 处理单特征输入
                    x = x.to(device)
            
            y = y.to(device)
            lengths = lengths.to(device) if lengths is not None else None
            
            # 前向传播
            logits = model(x, lengths)
            probs = torch.nn.functional.softmax(logits, dim=1)
            
            # 收集结果
            all_labels.extend(y.cpu().numpy())
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_scores.extend(probs.cpu().numpy())
    
    return np.array(all_labels), np.array(all_preds), np.array(all_scores)

def ensemble_predictions(all_model_scores, all_model_preds=None, ensemble_method='weighted_avg', weights=None):
    """
    集成多个模型的预测
    Args:
        all_model_scores (list): List of numpy arrays, each (num_samples, num_classes) with probabilities.
        all_model_preds (list): List of numpy arrays, each (num_samples,) with class predictions (0 or 1).
        ensemble_method (str): Method to use for ensembling.
        weights (np.ndarray, optional): Weights for weighted average. Defaults to None (uniform).
    """
    num_models = len(all_model_scores)
    num_samples = all_model_scores[0].shape[0]
    
    if weights is None:
        weights = np.ones(num_models) / num_models
    
    ensemble_scores = np.zeros_like(all_model_scores[0]) # Initialize ensemble scores
    
    if ensemble_method == 'weighted_avg':
        # 加权平均概率
        for i, scores in enumerate(all_model_scores):
            ensemble_scores += scores * weights[i]
    
    elif ensemble_method == 'simple_avg':
        # 简单平均概率
        ensemble_scores = np.mean(all_model_scores, axis=0)

    elif ensemble_method == 'majority_vote':
        # 多数投票 (硬投票) - 基于预测类别
        ensemble_preds = np.zeros(num_samples, dtype=int)
        ensemble_scores = np.zeros_like(all_model_scores[0]) # To return scores (proportion of votes)

        for i in range(num_samples):
            sample_preds = [model_preds[i] for model_preds in all_model_preds]
            # Count votes for each class
            vote_counts = Counter(sample_preds)
            # Get the winning class(es). max() handles ties by taking the first one.
            winning_class = max(vote_counts, key=vote_counts.get)
            ensemble_preds[i] = winning_class
            
            # Calculate scores as proportion of votes for each class
            total_votes = num_models
            ensemble_scores[i, 0] = vote_counts.get(0, 0) / total_votes
            ensemble_scores[i, 1] = vote_counts.get(1, 0) / total_votes
            
        # Note: majority_vote calculates preds first, then derives scores.
        # Other methods calculate scores, then derive preds.
        # We return calculated ensemble_preds here for consistency with other methods
        # where preds are derived from scores.
        return ensemble_preds, ensemble_scores

    elif ensemble_method == 'product':
        # 概率乘积后归一化
        # Add a small epsilon to avoid log(0) or product becoming exactly zero
        # 这里需要修改处理方式，因为all_model_scores是一个列表
        product_scores = np.ones_like(all_model_scores[0])
        for scores in all_model_scores:
            product_scores *= scores
        product_scores += 1e-9  # 添加小值以避免零值
        
        # Normalize the scores to sum to 1 for each sample
        sum_scores = np.sum(product_scores, axis=1, keepdims=True)
        ensemble_scores = product_scores / sum_scores

    elif ensemble_method == 'max_vote':
         # Original custom max_vote logic - keeping it as is
         # This uses confidence thresholds and is not standard majority vote
         ensemble_scores = np.zeros_like(all_model_scores[0])
         for i in range(len(all_model_scores[0])):
             class1_confidences = [model_scores[i][1] for model_scores in all_model_scores]
             class0_confidences = [model_scores[i][0] for model_scores in all_model_scores]
             
             if max(class1_confidences) > 0.7:  
                 ensemble_scores[i][1] = max(class1_confidences)
                 ensemble_scores[i][0] = 1 - ensemble_scores[i][1]
             elif min(class0_confidences) > 0.7:
                 ensemble_scores[i][0] = max(class0_confidences) # Should this be max(class0_confidences)? Or min(class0_confidences)? Original code used max. Let's keep it.
                 ensemble_scores[i][1] = 1 - ensemble_scores[i][0]
             else: # Fallback to weighted average
                 for j, scores in enumerate(all_model_scores):
                     ensemble_scores[i] += scores[i] * weights[j]
    
    elif ensemble_method == 'rank_avg':
        # 排序平均集成
        ensemble_scores = np.zeros_like(all_model_scores[0])
        for i in range(num_samples):
            # 获取每个模型对当前样本的正类概率
            pos_probs = [scores[i][1] for scores in all_model_scores]
            # 转换为排名 (排名越高，概率越大)
            ranks = np.argsort(np.argsort(pos_probs))
            # 归一化排名为0-1之间
            norm_ranks = (ranks / (num_models - 1)) if num_models > 1 else ranks
            # 计算加权排名平均
            weighted_rank_avg = np.sum(norm_ranks * weights)
            # 转换回概率值 (使用sigmoid函数)
            ensemble_scores[i][1] = 1.0 / (1.0 + np.exp(-5 * (weighted_rank_avg - 0.5)))
            ensemble_scores[i][0] = 1.0 - ensemble_scores[i][1]
    
    elif ensemble_method == 'confidence_weighted':
        # 置信度加权集成 (给予高置信度预测更高权重)
        ensemble_scores = np.zeros_like(all_model_scores[0])
        for i in range(num_samples):
            # 计算每个模型的置信度 (最高概率值)
            confidences = [max(scores[i]) for scores in all_model_scores]
            # 归一化置信度权重
            conf_weights = np.array(confidences) / np.sum(confidences)
            # 结合原始权重
            combined_weights = weights * conf_weights
            combined_weights = combined_weights / np.sum(combined_weights)
            # 加权平均概率
            for j, scores in enumerate(all_model_scores):
                ensemble_scores[i] += scores[i] * combined_weights[j]
    
    elif ensemble_method == 'soft_voting':
        # 软投票 (平均概率，然后选择最高概率类别)
        pos_probs = np.zeros(num_samples)
        neg_probs = np.zeros(num_samples)
        
        for i, scores in enumerate(all_model_scores):
            pos_probs += scores[:, 1] * weights[i]
            neg_probs += scores[:, 0] * weights[i]
            
        # 归一化
        total_probs = pos_probs + neg_probs
        pos_probs = pos_probs / total_probs
        neg_probs = neg_probs / total_probs
        
        ensemble_scores[:, 1] = pos_probs
        ensemble_scores[:, 0] = neg_probs
    
# ===== 新增高级集成方法 =====
    
    elif ensemble_method == 'stacking':
        """
        堆叠集成：使用逻辑回归作为元分类器
        将每个基础模型的预测概率作为特征，训练元分类器
        """
        from sklearn.linear_model import LogisticRegression
        
        # 准备元特征
        meta_features = np.column_stack([scores[:, 1] for scores in all_model_scores])
        
        # 使用交叉验证训练元模型
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        meta_preds = np.zeros(num_samples)
        meta_probs = np.zeros(num_samples)
        
        for train_idx, test_idx in kf.split(meta_features):
            # 训练元分类器
            meta_clf = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000)
            meta_clf.fit(meta_features[train_idx], all_labels[train_idx])
            
            # 预测
            meta_probs[test_idx] = meta_clf.predict_proba(meta_features[test_idx])[:, 1]
        
        # 最终预测
        meta_preds = (meta_probs > 0.5).astype(int)
        ensemble_scores[:, 1] = meta_probs
        ensemble_scores[:, 0] = 1 - meta_probs
        
        return meta_preds, ensemble_scores
    
    elif ensemble_method == 'dempster_shafer':
        """
        Dempster-Shafer证据理论集成
        基于不确定性理论，更好地处理冲突的证据
        """
        # 计算每个模型的基本概率分配
        def calculate_mass(probabilities):
            belief_pos = probabilities[:, 1]
            belief_neg = probabilities[:, 0]
            uncertainty = np.clip(1 - (belief_pos + belief_neg), 0, 1)
            
            # 归一化
            total = belief_pos + belief_neg + uncertainty
            belief_pos = belief_pos / total
            belief_neg = belief_neg / total
            uncertainty = uncertainty / total
            
            return np.column_stack([belief_neg, belief_pos, uncertainty])
        
        # 计算每个模型的质量函数
        masses = [calculate_mass(scores) for scores in all_model_scores]
        
        # Dempster规则组合
        combined_mass = masses[0]
        for i in range(1, len(masses)):
            m1 = combined_mass
            m2 = masses[i]
            
            # 计算冲突度
            k = 0
            for i in range(num_samples):
                k += m1[i, 0] * m2[i, 1] + m1[i, 1] * m2[i, 0]
            
            if k < 1:  # 避免完全冲突
                # 更新组合质量函数
                combined_mass = np.zeros_like(m1)
                combined_mass[:, 0] = (m1[:, 0] * m2[:, 0] + m1[:, 0] * m2[:, 2] + m1[:, 2] * m2[:, 0]) / (1 - k)
                combined_mass[:, 1] = (m1[:, 1] * m2[:, 1] + m1[:, 1] * m2[:, 2] + m1[:, 2] * m2[:, 1]) / (1 - k)
                combined_mass[:, 2] = (m1[:, 2] * m2[:, 2]) / (1 - k)
        
        # 计算信念和似然函数
        belief = combined_mass[:, 1]
        plausibility = combined_mass[:, 1] + combined_mass[:, 2]
        
        # 生成最终预测概率
        ensemble_scores[:, 1] = (belief + plausibility) / 2
        ensemble_scores[:, 0] = 1 - ensemble_scores[:, 1]
        
        ensemble_preds = (ensemble_scores[:, 1] > 0.5).astype(int)
        return ensemble_preds, ensemble_scores
    
    elif ensemble_method == 'adaboost_ensemble':
        """
        AdaBoost风格的集成
        根据模型在样本上的表现动态调整权重
        """
        # 计算每个模型的错误率和权重
        model_weights = np.zeros(num_models)
        
        for i, preds in enumerate(all_model_preds):
            # 计算误差
            errors = (preds != all_labels).astype(float)
            error_rate = np.sum(errors) / len(errors)
            error_rate = max(error_rate, 1e-10)  # 避免零误差
            
            # 计算模型权重 (AdaBoost公式)
            model_weights[i] = 0.5 * np.log((1 - error_rate) / error_rate)
        
        # 归一化模型权重
        model_weights = np.exp(model_weights)
        model_weights = model_weights / np.sum(model_weights)
        
        # 使用动态权重计算加权平均
        for i, scores in enumerate(all_model_scores):
            ensemble_scores += scores * model_weights[i]
        
        ensemble_preds = np.argmax(ensemble_scores, axis=1)
        return ensemble_preds, ensemble_scores
    
    elif ensemble_method == 'dynamic_selection':
        """
        动态集成选择
        对每个样本选择表现最好的模型子集
        """
        from sklearn.neighbors import NearestNeighbors
        
        # 准备元特征空间
        meta_features = np.column_stack([scores[:, 1] for scores in all_model_scores])
        
        # 为每个测试样本找到最相似的k个训练样本
        k = 10  # 邻居数量
        knn = NearestNeighbors(n_neighbors=k)
        knn.fit(meta_features)
        
        # 对每个样本动态选择模型
        for i in range(num_samples):
            # 找到最相似的样本
            distances, indices = knn.kneighbors([meta_features[i]], n_neighbors=k)
            
            # 评估每个模型在邻居上的性能
            model_scores = []
            for j, model_preds in enumerate(all_model_preds):
                # 计算在邻居样本上的准确率
                acc = np.mean(model_preds[indices[0]] == all_labels[indices[0]])
                model_scores.append(acc)
            
            # 选择前50%性能最好的模型
            top_models = np.argsort(model_scores)[::-1][:max(1, num_models // 2)]
            
            # 使用选定的模型集成预测
            sample_scores = np.zeros(2)
            for model_idx in top_models:
                sample_scores += all_model_scores[model_idx][i]
            
            # 归一化
            sample_scores = sample_scores / len(top_models)
            ensemble_scores[i] = sample_scores
        
        ensemble_preds = np.argmax(ensemble_scores, axis=1)
        return ensemble_preds, ensemble_scores
    
    elif ensemble_method == 'super_ensemble':
        """
        超级集成法
        组合多种集成方法的结果
        """
        # 定义要使用的基础集成方法
        base_methods = ['weighted_avg', 'rank_avg', 'confidence_weighted', 'product']
        
        # 获取每种方法的预测结果
        method_preds = []
        method_scores = []
        
        for method in base_methods:
            preds, scores = ensemble_predictions(
                all_model_scores, all_model_preds, 
                ensemble_method=method, 
                weights=weights
            )
            method_preds.append(preds)
            method_scores.append(scores)
        
        # 使用简单投票组合结果
        for i in range(num_samples):
            # 计算正类投票
            pos_votes = sum(preds[i] == 1 for preds in method_preds)
            
            # 如果大多数方法预测为正类
            if pos_votes > len(base_methods) / 2:
                # 找出预测为正类的方法
                pos_methods = [j for j, preds in enumerate(method_preds) if preds[i] == 1]
                
                # 取平均概率
                pos_probs = [method_scores[j][i, 1] for j in pos_methods]
                ensemble_scores[i, 1] = np.mean(pos_probs)
                ensemble_scores[i, 0] = 1 - ensemble_scores[i, 1]
            else:
                # 找出预测为负类的方法
                neg_methods = [j for j, preds in enumerate(method_preds) if preds[i] == 0]
                
                # 取平均概率
                neg_probs = [method_scores[j][i, 0] for j in neg_methods]
                ensemble_scores[i, 0] = np.mean(neg_probs)
                ensemble_scores[i, 1] = 1 - ensemble_scores[i, 0]
        
        ensemble_preds = np.argmax(ensemble_scores, axis=1)
        return ensemble_preds, ensemble_scores
    
    elif ensemble_method == 'confidence_intervals':
        """
        置信区间集成
        考虑每个模型预测的概率分布和不确定性
        """
        from scipy import stats
        
        # 收集每个样本的正类概率
        all_probs = np.array([scores[:, 1] for scores in all_model_scores]).T
        
        # 计算每个样本概率的均值和标准差
        mean_probs = np.mean(all_probs, axis=1)
        std_probs = np.std(all_probs, axis=1)
        
        # 定义置信度参数
        confidence = 0.95
        z_score = stats.norm.ppf((1 + confidence) / 2)
        
        # 计算置信区间
        lower_bound = mean_probs - z_score * std_probs / np.sqrt(num_models)
        upper_bound = mean_probs + z_score * std_probs / np.sqrt(num_models)
        
        # 使用置信区间中值作为最终预测
        ensemble_scores[:, 1] = (lower_bound + upper_bound) / 2
        ensemble_scores[:, 0] = 1 - ensemble_scores[:, 1]
        
        ensemble_preds = np.argmax(ensemble_scores, axis=1)
        return ensemble_preds, ensemble_scores
        
    elif ensemble_method == 'cross_validation_ensemble':
        """
        交叉验证集成
        使用交叉验证确定最优模型权重
        """
        from scipy.optimize import minimize
        
        # 定义优化目标：最大化加权模型的准确率
        def objective(weights):
            # 归一化权重
            weights = weights / np.sum(weights)
            
            # 计算加权预测
            weighted_scores = np.zeros_like(all_model_scores[0])
            for i, scores in enumerate(all_model_scores):
                weighted_scores += scores * weights[i]
            
            # 计算预测类别
            weighted_preds = np.argmax(weighted_scores, axis=1)
            
            # 计算准确率
            accuracy = np.mean(weighted_preds == all_labels)
            
            # 返回负准确率（因为我们要最小化）
            return -accuracy
        
        # 初始权重（均等）
        initial_weights = np.ones(num_models) / num_models
        
        # 约束：权重和为1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        
        # 边界：每个权重在0到1之间
        bounds = [(0, 1) for _ in range(num_models)]
        
        # 优化权重
        result = minimize(
            objective, 
            initial_weights, 
            method='SLSQP', 
            bounds=bounds,
            constraints=constraints
        )
        
        # 使用优化后的权重
        optimal_weights = result.x / np.sum(result.x)
        
        # 计算最终预测
        for i, scores in enumerate(all_model_scores):
            ensemble_scores += scores * optimal_weights[i]
        
        ensemble_preds = np.argmax(ensemble_scores, axis=1)
        return ensemble_preds, ensemble_scores
    
    elif ensemble_method == 'gradient_boosted_ensemble':
        """
        梯度提升集成
        类似于梯度提升决策树的思路，每个模型学习前一个模型的残差
        """
        from sklearn.ensemble import GradientBoostingClassifier
        
        # 准备元特征
        meta_features = np.column_stack([scores[:, 1] for scores in all_model_scores])
        
        # 初始预测（使用权重平均）
        initial_pred = np.zeros(num_samples)
        for i, scores in enumerate(all_model_scores):
            initial_pred += scores[:, 1] * weights[i]
        
        # 计算残差（真实值 - 预测值）
        residuals = all_labels - initial_pred
        
        # 训练梯度提升模型来学习残差
        gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3,random_state=42)
        gb.fit(meta_features, all_labels)
        
        # 最终预测 = 初始预测 + 梯度提升修正
        boosted_probs = gb.predict_proba(meta_features)[:, 1]
        
        # 结合初始预测和提升预测
        alpha = 0.7  # 平衡因子
        final_probs = alpha * initial_pred + (1 - alpha) * boosted_probs
        
        # 裁剪到[0,1]范围
        final_probs = np.clip(final_probs, 0, 1)
        
        # 设置最终预测
        ensemble_scores[:, 1] = final_probs
        ensemble_scores[:, 0] = 1 - final_probs
        
        ensemble_preds = (final_probs > 0.5).astype(int)
        return ensemble_preds, ensemble_scores
    
    # 默认使用已有方法或者抛出异常
    else:
        raise ValueError(f"Unsupported ensemble method: {ensemble_method}")


    # For methods calculating scores first, derive predictions from scores
    if ensemble_method != 'majority_vote':
       ensemble_preds = np.argmax(ensemble_scores, axis=1)

    return ensemble_preds, ensemble_scores

def compare_ensemble_methods(all_labels, all_model_preds, all_model_scores, weights, methods=None, save_plot=True):
    """比较不同集成方法的性能并生成可视化"""
    if methods is None:
        # 添加高级集成方法
        methods = [
            # 基础方法
            'weighted_avg', 'simple_avg', 'majority_vote', 
            'product', 'max_vote', 'rank_avg', 
            'confidence_weighted', 'soft_voting',
            
            # 高级方法
            'stacking', 'dempster_shafer', 'adaboost_ensemble',
            'dynamic_selection', 'super_ensemble', 'confidence_intervals',
            'cross_validation_ensemble', 'gradient_boosted_ensemble'
        ]
    
    # 其他代码不变...
    
    results = {}
    metrics = ['sn', 'sp', 'acc', 'f1', 'mcc', 'auc', 'aupr']
    
    print("\n===== 集成方法对比 =====")
    print(f"{'方法':<20} {'SN':<8} {'SP':<8} {'ACC':<8} {'F1':<8} {'MCC':<8} {'AUC':<8} {'AUPR':<8}")
    print("-" * 80)
    
    for method in methods:
        try:
            ensemble_preds, ensemble_scores = ensemble_predictions(
                all_model_scores, all_model_preds, 
                ensemble_method=method, 
                weights=weights
            )
            
            sn, sp, acc, f1, mcc, auc_score, aupr_score = calculate_metrics(
                all_labels, ensemble_preds, ensemble_scores
            )
            
            results[method] = {
                'sn': sn, 'sp': sp, 'acc': acc, 'f1': f1, 
                'mcc': mcc, 'auc': auc_score, 'aupr': aupr_score
            }
            
            print(f"{method:<20} {sn:<8.2f} {sp:<8.2f} {acc:<8.2f} {f1:<8.2f} {mcc:<8.2f} {auc_score:<8.2f} {aupr_score:<8.2f}")
        except Exception as e:
            print(f"{method:<20} 评估失败: {str(e)}")
    
    # 可视化比较结果
    if save_plot and len(results) > 1:
        plt.figure(figsize=(15, 10))
        
        # 绘制主要指标的比较图
        key_metrics = ['acc', 'f1', 'mcc', 'auc']
        for i, metric in enumerate(key_metrics):
            plt.subplot(2, 2, i+1)
            
            # 提取性能值并排序
            methods_list = list(results.keys())
            values = [results[m][metric] for m in methods_list]
            
            # 排序
            sorted_indices = np.argsort(values)[::-1]
            sorted_methods = [methods_list[i] for i in sorted_indices]
            sorted_values = [values[i] for i in sorted_indices]
            
            # 绘制柱状图
            bars = plt.bar(range(len(sorted_methods)), sorted_values)
            
            # 添加数值标签
            for j, bar in enumerate(bars):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                        f'{sorted_values[j]:.1f}%', ha='center', va='bottom')
            
            plt.title(f'{metric.upper()} (%)')
            plt.xticks(range(len(sorted_methods)), sorted_methods, rotation=45, ha='right')
            plt.ylim(min(sorted_values) - 5, max(sorted_values) + 5)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('ensemble_methods_comparison.png', dpi=300, bbox_inches='tight')
        print("\n比较结果已保存到 ensemble_methods_comparison.png")
    
    # 返回最佳方法 (基于ACC)
    if results:
        best_method = max(results.keys(), key=lambda m: results[m]['acc'])
        print(f"\n最佳集成方法: {best_method} (ACC: {results[best_method]['acc']:.2f}%)")
        return best_method, results
    
    return None, results

def main():
    parser = argparse.ArgumentParser(description="集成多个模型在测试集上进行验证")
    parser.add_argument("--config",default='/root/VF-pred/config.json', type=str,  help="配置文件路径")
    parser.add_argument("--ensemble_method", type=str, default="weighted_avg", 
                        help="集成方法，可选：weighted_avg, simple_avg, majority_vote, product, max_vote, rank_avg, confidence_weighted, soft_voting")
    parser.add_argument("--compare_all", action="store_true",default=True, help="比较所有集成方法")
    parser.add_argument("--save_results", action="store_true", default=True, help="保存结果到CSV文件")
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, "r") as f:
        config = json.load(f)
    
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 数据集路径
    test_esm_path = config.get("test_esm_path", '/root/autodl-tmp/.autodl/embedding_data/test_esm2_t33_650M_UR50D_mean.h5')
    test_prot5_path = config.get("test_prot5_path", "/root/autodl-tmp/test_prot_features_modified.h5")
    
    # 加载测试数据集
    is_fusion_model = any("dual" in model["type"].lower() or "fusion" in model["type"].lower() 
                          for model in config["models"])
    
    # 使用双特征数据集
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
    global all_labels  # 使 all_labels 在整个模块中可用，供高级集成方法使用
    
    # 加载所有模型并进行预测
    all_labels = []
    all_model_preds = []
    all_model_scores = []
    
    for model_config in config["models"]:
        model_type = model_config["type"]
        model_path = model_config["path"]
        feature_type = model_config.get("feature_type", "esm2")
        model_name = model_config.get("name", model_type)
        
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
        
        # 输出单个模型的性能
        print(f"\n===== {model_name} ({feature_type}) 模型性能 =====")
        sn, sp, acc, f1, mcc, auc_score, aupr_score = calculate_metrics(labels, preds, scores)
        print(f"Sensitivity (SN): {sn:.2f}%")
        print(f"Specificity (SP): {sp:.2f}%")
        print(f"Accuracy (ACC):   {acc:.2f}%")
        print(f"F1 Score:         {f1:.2f}%")
        print(f"MCC:              {mcc:.2f}%")
        print(f"AUC:              {auc_score:.2f}%")
        print(f"AUPR:             {aupr_score:.2f}%")
    
    # 获取模型权重
    model_weights = np.array([model.get("weight", 1.0) for model in config["models"]])
    model_weights = model_weights / model_weights.sum()  # 归一化权重
    
    # 根据参数决定是否比较所有集成方法
    if args.compare_all:
        # 比较所有集成方法
        best_method, results = compare_ensemble_methods(
            all_labels, all_model_preds, all_model_scores, model_weights
        )
        
        # 使用最佳方法的预测结果
        ensemble_preds, ensemble_scores = ensemble_predictions(
            all_model_scores, all_model_preds,
            ensemble_method=best_method,
            weights=model_weights
        )
    else:
        # 只使用指定的集成方法
        ensemble_preds, ensemble_scores = ensemble_predictions(
            all_model_scores, all_model_preds,
            ensemble_method=args.ensemble_method,
            weights=model_weights
        )
        
        # 输出单一集成方法性能
        print(f"\n===== 集成模型性能 ({args.ensemble_method}) =====")
        sn, sp, acc, f1, mcc, auc_score, aupr_score = calculate_metrics(all_labels, ensemble_preds, ensemble_scores)
        print(f"Sensitivity (SN): {sn:.2f}%")
        print(f"Specificity (SP): {sp:.2f}%")
        print(f"Accuracy (ACC):   {acc:.2f}%")
        print(f"F1 Score:         {f1:.2f}%")
        print(f"MCC:              {mcc:.2f}%")
        print(f"AUC:              {auc_score:.2f}%")
        print(f"AUPR:             {aupr_score:.2f}%")
    
    # 额外分析：计算每个模型对SN和SP的贡献
    print("\n===== 模型贡献分析 =====")
    for i, model_config in enumerate(config["models"]):
        model_name = model_config.get("name", model_config["type"])
        feature_type = model_config.get("feature_type", "esm2")
        model_preds = all_model_preds[i]
        model_scores = all_model_scores[i]

        # 计算该模型的性能
        sn, sp, acc, _, _, _, _ = calculate_metrics(all_labels, model_preds, model_scores)
        print(f"{model_name} ({feature_type}): SN={sn:.2f}%, SP={sp:.2f}%, ACC={acc:.2f}%, 权重={model_weights[i]:.3f}")
    
    # 保存集成结果到CSV文件
    if args.save_results:
        import pandas as pd
        results = pd.DataFrame({
            "True_Label": all_labels,
            "Ensemble_Prediction": ensemble_preds,
            "Positive_Probability": ensemble_scores[:, 1]
        })
        
        # 添加每个模型的预测
        for i, model_config in enumerate(config["models"]):
            model_name = model_config.get("name", model_config["type"])
            feature_type = model_config.get("feature_type", "esm2")
            results[f"{model_name}_{feature_type}_Prediction"] = all_model_preds[i]
            results[f"{model_name}_{feature_type}_Probability"] = all_model_scores[i][:, 1]
        
        results.to_csv("ensemble_results.csv", index=False)
        print("\n结果已保存到 ensemble_results.csv")
import os
import random
import numpy as np
import torch
from sklearn.utils import check_random_state

# 添加到文件开头，就在导入语句后面
def set_all_seeds(seed=42):
    """设置所有可能的随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 为sklearn设置全局随机种子
    check_random_state(seed)
if __name__ == "__main__":
    from plm_train import seed_everything
    seed_everything(42)  # 设置随机种子
    set_all_seeds(42)
    
    main()