import argparse
import json
import os
import random
import warnings
from typing import Counter
from model_type import ImprovedDualPathwayFusion
        
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.dummy import check_random_state
from sklearn.metrics import (auc, confusion_matrix, precision_recall_curve,
                             roc_auc_score)
from torch.utils.data import DataLoader

from train import (DualEmbeddingDataset, H5Dataset, collate_fn, create_model,
                   dual_features_collate_fn, seed_everything)

seed_everything(42)  # 设置随机种子
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
set_all_seeds(42)
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
    print(f"模型配置: {config}")
    if "dual" in model_type.lower():

        model = ImprovedDualPathwayFusion(
        esm_dim=esm_dim,
        prot5_dim=prot5_dim,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        num_classes=2,
        rank=config["rank"],
        steps=config["steps"],
        dropout=config["dropout"]
    )

    else:

        model = create_model(
            classifier_type=config["type"],
            input_dim=config["input_dim"],
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
            rank=config["rank"],
            steps=config["steps"],
        )

    # 加载模型权重
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    # 关闭所有dropout和batch norm的训练模式
    for module in model.modules():
        if isinstance(module, (torch.nn.Dropout, torch.nn.BatchNorm1d, torch.nn.LayerNorm)):
            module.eval()
        
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
                elif feature_type == "all":
                    esm_features, prot5_features = x
                    esm_features = esm_features.to(device)
                    prot5_features = prot5_features.to(device)
                    x = (esm_features, prot5_features)
                    # 拼接
                    x = torch.cat((esm_features, prot5_features), dim=-1)
                
            
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
    集成多个模型的预测 - 无需真实标签版本
    Args:
        all_model_scores (list): List of numpy arrays, each (num_samples, num_classes) with probabilities.
        all_model_preds (list): List of numpy arrays, each (num_samples,) with class predictions (0 or 1).
        ensemble_method (str): Method to use for ensembling.
        weights (np.ndarray, optional): Weights for weighted average. Defaults to None (uniform).
    """
    # 在函数开始时重新设置随机种子
    np.random.seed(42)
    random.seed(42)
    num_models = len(all_model_scores)
    num_samples = all_model_scores[0].shape[0]
    
    if weights is None:
        weights = np.ones(num_models) / num_models
    # print(f"使用集成方法: {ensemble_method}，模型数量: {num_models}, 样本数量: {num_samples}")
    # print(f"模型权重: {weights}")
    ensemble_scores = np.zeros_like(all_model_scores[0])
    
    if ensemble_method == 'simple_avg':
        """简单平均"""
        ensemble_scores = np.mean(all_model_scores, axis=0)
    
    elif ensemble_method == 'majority_vote':
        """多数投票"""
        ensemble_preds = np.zeros(num_samples, dtype=int)
        ensemble_scores = np.zeros_like(all_model_scores[0])

        for i in range(num_samples):
            sample_preds = [model_preds[i] for model_preds in all_model_preds]
            vote_counts = Counter(sample_preds)
            winning_class = max(vote_counts, key=vote_counts.get)
            ensemble_preds[i] = winning_class
            
            total_votes = num_models
            ensemble_scores[i, 0] = vote_counts.get(0, 0) / total_votes
            ensemble_scores[i, 1] = vote_counts.get(1, 0) / total_votes
            
        return ensemble_preds, ensemble_scores
    
    elif ensemble_method == 'max_vote':
        """最大置信度投票"""
        ensemble_scores = np.zeros_like(all_model_scores[0])
        for i in range(num_samples):
            class1_confidences = [model_scores[i][1] for model_scores in all_model_scores]
            class0_confidences = [model_scores[i][0] for model_scores in all_model_scores]
            
            if max(class1_confidences) > 0.7:  
                ensemble_scores[i][1] = max(class1_confidences)
                ensemble_scores[i][0] = 1 - ensemble_scores[i][1]
            elif max(class0_confidences) > 0.7:
                ensemble_scores[i][0] = max(class0_confidences)
                ensemble_scores[i][1] = 1 - ensemble_scores[i][0]
            else:
                # 回退到加权平均
                for j, scores in enumerate(all_model_scores):
                    ensemble_scores[i] += scores[i] * weights[j]
    
    elif ensemble_method == 'rank_avg':
        """排序平均集成"""
        ensemble_scores = np.zeros_like(all_model_scores[0])
        for i in range(num_samples):
            pos_probs = [scores[i][1] for scores in all_model_scores]
            ranks = np.argsort(np.argsort(pos_probs))
            norm_ranks = (ranks / (num_models - 1)) if num_models > 1 else ranks
            weighted_rank_avg = np.sum(norm_ranks * weights)
            ensemble_scores[i][1] = 1.0 / (1.0 + np.exp(-5 * (weighted_rank_avg - 0.5)))
            ensemble_scores[i][0] = 1.0 - ensemble_scores[i][1]
    
    elif ensemble_method == 'confidence_weighted':
        """置信度加权集成"""
        ensemble_scores = np.zeros_like(all_model_scores[0])
        for i in range(num_samples):
            confidences = [max(scores[i]) for scores in all_model_scores]
            conf_weights = np.array(confidences) / np.sum(confidences)
            combined_weights = weights * conf_weights
            combined_weights = combined_weights / np.sum(combined_weights)
            
            for j, scores in enumerate(all_model_scores):
                ensemble_scores[i] += scores[i] * combined_weights[j]
    
    elif ensemble_method == 'product':
        """概率乘积后归一化"""
        product_scores = np.ones_like(all_model_scores[0])
        for scores in all_model_scores:
            product_scores *= (scores + 1e-9)  # 避免零值
        
        sum_scores = np.sum(product_scores, axis=1, keepdims=True)
        ensemble_scores = product_scores / sum_scores
    
    elif ensemble_method == 'dempster_shafer':
        """Dempster-Shafer证据理论集成"""
        def calculate_mass(probabilities):
            belief_pos = probabilities[:, 1]
            belief_neg = probabilities[:, 0]
            uncertainty = np.clip(1 - (belief_pos + belief_neg), 0, 1)
            
            total = belief_pos + belief_neg + uncertainty
            belief_pos = belief_pos / total
            belief_neg = belief_neg / total
            uncertainty = uncertainty / total
            
            return np.column_stack([belief_neg, belief_pos, uncertainty])
        
        masses = [calculate_mass(scores) for scores in all_model_scores]
        
        combined_mass = masses[0]
        for i in range(1, len(masses)):
            m1 = combined_mass
            m2 = masses[i]
            
            k = np.sum(m1[:, 0] * m2[:, 1] + m1[:, 1] * m2[:, 0], axis=0)
            k = np.clip(k, 0, 0.99)  # 避免完全冲突
            
            combined_mass = np.zeros_like(m1)
            combined_mass[:, 0] = (m1[:, 0] * m2[:, 0] + m1[:, 0] * m2[:, 2] + m1[:, 2] * m2[:, 0]) / (1 - k)
            combined_mass[:, 1] = (m1[:, 1] * m2[:, 1] + m1[:, 1] * m2[:, 2] + m1[:, 2] * m2[:, 1]) / (1 - k)
            combined_mass[:, 2] = (m1[:, 2] * m2[:, 2]) / (1 - k)
        
        belief = combined_mass[:, 1]
        plausibility = combined_mass[:, 1] + combined_mass[:, 2]
        
        ensemble_scores[:, 1] = (belief + plausibility) / 2
        ensemble_scores[:, 0] = 1 - ensemble_scores[:, 1]
    
    elif ensemble_method == 'confidence_intervals':
        """置信区间集成"""
        from scipy import stats
        
        all_probs = np.array([scores[:, 1] for scores in all_model_scores]).T
        mean_probs = np.mean(all_probs, axis=1)
        std_probs = np.std(all_probs, axis=1)
        
        confidence = 0.95
        z_score = stats.norm.ppf((1 + confidence) / 2)
        
        lower_bound = mean_probs - z_score * std_probs / np.sqrt(num_models)
        upper_bound = mean_probs + z_score * std_probs / np.sqrt(num_models)
        
        ensemble_scores[:, 1] = (lower_bound + upper_bound) / 2
        ensemble_scores[:, 0] = 1 - ensemble_scores[:, 1]
    
    elif ensemble_method == 'gradient_boosted_ensemble':
        """梯度提升集成 - 无监督版本"""
        from sklearn.ensemble import IsolationForest

        # 准备元特征
        meta_features = np.column_stack([scores[:, 1] for scores in all_model_scores])
        
        # 使用无监督方法检测异常点
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomalies = iso_forest.fit_predict(meta_features)
        
        # 基于异常检测调整权重
        adjusted_weights = weights.copy()
        for i in range(num_samples):
            if anomalies[i] == -1:  # 异常点
                # 降低置信度
                sample_probs = [scores[i] for scores in all_model_scores]
                avg_prob = np.mean(sample_probs, axis=0)
                
                # 向平均值回归
                ensemble_scores[i] = 0.7 * avg_prob + 0.3 * np.array([0.5, 0.5])
            else:
                # 正常点，使用加权平均
                for j, scores in enumerate(all_model_scores):
                    ensemble_scores[i] += scores[i] * adjusted_weights[j]
    
    elif ensemble_method == 'super_ensemble':
        """超级集成法"""
        base_methods = ['simple_avg', 'rank_avg', 'confidence_weighted', 'product']
        
        method_scores = []
        for method in base_methods:
            _, scores = ensemble_predictions(
                all_model_scores, all_model_preds, 
                ensemble_method=method, 
                weights=weights
            )
            method_scores.append(scores)
        
        # 使用简单平均组合结果
        ensemble_scores = np.mean(method_scores, axis=0)
    
    # 堆叠集成方法 - 使用元学习器
    elif ensemble_method in ['bayesian_stacking', 'svm_stacking', 'mlp_stacking', 'xgboost_stacking', 'catboost_stacking']:
        """各种堆叠集成方法"""
        meta_features = np.column_stack([scores[:, 1] for scores in all_model_scores])
        
        # 使用无监督聚类作为伪标签
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        pseudo_labels = kmeans.fit_predict(meta_features)
        
        # 根据聚类结果调整标签
        cluster_centers = kmeans.cluster_centers_
        if np.mean(cluster_centers[0]) > np.mean(cluster_centers[1]):
            pseudo_labels = 1 - pseudo_labels  # 翻转标签
        
        # 选择元学习器
        if ensemble_method == 'bayesian_stacking':
            from sklearn.naive_bayes import GaussianNB
            meta_clf = GaussianNB()
        elif ensemble_method == 'svm_stacking':
            from sklearn.svm import SVC
            meta_clf = SVC(probability=True, kernel='rbf', random_state=42)
        elif ensemble_method == 'mlp_stacking':
            from sklearn.neural_network import MLPClassifier
            meta_clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
        elif ensemble_method == 'xgboost_stacking':
            try:
                import xgboost as xgb
                meta_clf = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
            except ImportError:
                print("XGBoost未安装，使用随机森林替代")
                from sklearn.ensemble import RandomForestClassifier
                meta_clf = RandomForestClassifier(random_state=42)

        # 训练元学习器
        meta_clf.fit(meta_features, pseudo_labels)
        
        # 预测
        ensemble_probs = meta_clf.predict_proba(meta_features)
        
        # 确保概率维度正确
        if ensemble_probs.shape[1] == 2:
            ensemble_scores[:, 0] = ensemble_probs[:, 0]
            ensemble_scores[:, 1] = ensemble_probs[:, 1]
        else:
            # 如果只有一个类别，回退到简单平均
            ensemble_scores = np.mean(all_model_scores, axis=0)
    
    else:
        # 默认使用加权平均
        for i, scores in enumerate(all_model_scores):
            ensemble_scores += scores * weights[i]
    
    # 计算最终预测
    if ensemble_method != 'majority_vote':
        ensemble_preds = np.argmax(ensemble_scores, axis=1)
    
    return ensemble_preds, ensemble_scores

def compare_ensemble_methods(all_labels, all_model_preds, all_model_scores, weights, methods=None, save_plot=True):
    """比较不同集成方法的性能"""
    if methods is None:
        methods = [
            'simple_avg',
            'majority_vote', 
            'max_vote',
            'rank_avg',
            'confidence_weighted',
            'product',
            'dempster_shafer',
            'confidence_intervals',
            'gradient_boosted_ensemble',
            'super_ensemble',
            'bayesian_stacking',
            'svm_stacking',
            'mlp_stacking',
            'xgboost_stacking',
            'catboost_stacking'
        ]
    
    results = {}
    curve_data = {}
    
    print("\n===== 集成方法对比 =====")
    print(f"{'方法':<25} {'SN':<8} {'SP':<8} {'ACC':<8} {'F1':<8} {'MCC':<8} {'AUC':<8} {'AUPR':<8}")
    print("-" * 85)
    
    for method in methods:
        try:
            
            ensemble_preds, ensemble_scores = ensemble_predictions(
                all_model_scores, all_model_preds, 
                ensemble_method=method, 
                weights=weights
            )
            
            # 计算性能指标
            sn, sp, acc, f1, mcc, auc_score, aupr_score = calculate_metrics(
                all_labels, ensemble_preds, ensemble_scores
            )
            
            results[method] = {
                'sn': sn, 'sp': sp, 'acc': acc, 'f1': f1, 
                'mcc': mcc, 'auc': auc_score, 'aupr': aupr_score
            }
            
            print(f"{method:<25} {sn:<8.2f} {sp:<8.2f} {acc:<8.2f} {f1:<8.2f} {mcc:<8.2f} {auc_score:<8.2f} {aupr_score:<8.2f}")
            
            # 计算ROC和PR曲线数据
            from sklearn.metrics import precision_recall_curve, roc_curve
            
            fpr, tpr, _ = roc_curve(all_labels, ensemble_scores[:, 1])
            precision, recall, _ = precision_recall_curve(all_labels, ensemble_scores[:, 1])
            
            curve_data[method] = {
                'roc': {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'auc': auc_score
                },
                'pr': {
                    'precision': precision.tolist(),
                    'recall': recall.tolist(),
                    'aupr': aupr_score
                },
                'predictions': ensemble_preds.tolist(),
                'scores': ensemble_scores[:, 1].tolist()
            }
            
        except Exception as e:
            print(f"{method:<25} 评估失败: {str(e)}")
    
    # 绘制性能比较图
    if save_plot and len(results) > 1:
        import json
        with open('ensemble_curve_data.json', 'w') as f:
            json.dump(curve_data, f, indent=2)
        print("\n曲线数据已保存到 ensemble_curve_data.json")
        
        # 绘制性能指标条形图
        plt.figure(figsize=(20, 12))
        
        key_metrics = ['acc', 'f1', 'mcc', 'auc']
        for i, metric in enumerate(key_metrics):
            plt.subplot(2, 2, i+1)
            
            methods_list = list(results.keys())
            values = [results[m][metric] for m in methods_list]
            
            sorted_indices = np.argsort(values)[::-1]
            sorted_methods = [methods_list[i] for i in sorted_indices]
            sorted_values = [values[i] for i in sorted_indices]
            
            bars = plt.bar(range(len(sorted_methods)), sorted_values)
            
            for j, bar in enumerate(bars):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                        f'{sorted_values[j]:.1f}%', ha='center', va='bottom', fontsize=8)
            
            plt.title(f'{metric.upper()} (%)', fontsize=12)
            plt.xticks(range(len(sorted_methods)), sorted_methods, rotation=45, ha='right', fontsize=10)
            plt.ylim(min(sorted_values) - 5, max(sorted_values) + 5)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('ensemble_methods_comparison.png', dpi=300, bbox_inches='tight')
        print("比较结果已保存到 ensemble_methods_comparison.png")
        
        # 绘制ROC和PR曲线
        plt.figure(figsize=(15, 12))
        
        # ROC曲线
        plt.subplot(2, 1, 1)
        colors = plt.cm.tab20(np.linspace(0, 1, len(methods)))
        
        for i, method in enumerate(methods):
            if method in curve_data:
                plt.plot(
                    curve_data[method]['roc']['fpr'], 
                    curve_data[method]['roc']['tpr'],
                    label=f"{method} (AUC={curve_data[method]['roc']['auc']:.2f}%)",
                    color=colors[i],
                    linewidth=2
                )
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves', fontsize=14)
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # PR曲线
        plt.subplot(2, 1, 2)
        for i, method in enumerate(methods):
            if method in curve_data:
                plt.plot(
                    curve_data[method]['pr']['recall'], 
                    curve_data[method]['pr']['precision'],
                    label=f"{method} (AUPR={curve_data[method]['pr']['aupr']:.2f}%)",
                    color=colors[i],
                    linewidth=2
                )
        
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves', fontsize=14)
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('ensemble_roc_pr_curves.png', dpi=300, bbox_inches='tight')
        print("ROC和PR曲线已保存到 ensemble_roc_pr_curves.png")
    
    # 返回最佳方法
    if results:
        best_method = max(results.keys(), key=lambda m: results[m]['acc'])
        print(f"\n最佳集成方法: {best_method} (ACC: {results[best_method]['acc']:.2f}%)")
        return best_method, results
    
    return None, results
