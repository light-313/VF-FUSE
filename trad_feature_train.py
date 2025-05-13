import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef
from sklearn.model_selection import KFold
import pandas as pd
import warnings
from itertools import combinations
from model_type import *
from trad_feature_extraction import *
from sklearn.preprocessing import MinMaxScaler
from Bio import SeqIO
import time
from functools import lru_cache
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# 忽略警告
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置随机种子
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 读取FASTA文件（添加缓存提高速度）
@lru_cache(maxsize=4)
def read_fasta(file_path_tuple, max_length=200000):
    sequences, labels, ids = [], [], []
    for file_path in file_path_tuple:
        for record in SeqIO.parse(file_path, "fasta"):
            seq = str(record.seq)
            if len(seq) > max_length:
                continue
            label = int(record.description.split("|")[1])
            sequences.append(seq)
            labels.append(label)
            ids.append(record.description)
    return sequences, labels, ids

# 特征提取 - 使用线程池并行处理
def extract_features_parallel(sequences, methods, k=[1, 2]):
    # 创建一个空的列表来存储所有提取的特征
    results = []

    for method in methods:
        if method == "kmer":
            results.append(_extract_single_feature(sequences, "kmer", k))
        elif method in ["dde", "paac", "dpc", "qso", "aac", "seqsim"]:
            results.append(_extract_single_feature(sequences, method, None))
    
    # 拼接所有特征，假设每个特征的形状是相同的，可以按列拼接
    return np.concatenate(results, axis=1)

# def extract_features_parallel(sequences, methods, k=[1, 2], n_jobs=1):
#     if n_jobs is None:
#         n_jobs = max(1, cpu_count() - 1)  # 保留一个CPU核心
    
#     # 创建参数列表
#     params = []
#     for method in methods:
#         if method == "kmer":
#             params.append((sequences, "kmer", k))
#         elif method in ["dde", "paac", "dpc", "qso", "aac", "seqsim"]:
#             params.append((sequences, method, None))
    
#     # 使用进程池并行处理
#     with Pool(processes=n_jobs) as pool:
#         results = list(pool.starmap(_extract_single_feature, params))
    
#     return np.concatenate(results, axis=1)

# 单个特征提取辅助函数
def _extract_single_feature(sequences, method, k=None):
    if method == "kmer":
        return kmer_feature_extraction(sequences, k)
    elif method == "dde":
        return dde_feature_extraction(sequences)
    elif method == "paac":
        return paac_feature_extraction(sequences)
    elif method == "dpc":
        return dpc_feature_extraction(sequences)
    elif method == "qso":
        return qso_feature_extraction(sequences)
    elif method == "aac":
        return aac_feature_extraction(sequences)
    elif method == "seqsim":
        return seqsim_feature_extraction(sequences)
    else:
        raise ValueError(f"不支持的特征提取方法: {method}")

# 使用缓存优化的seqsim特征提取
@lru_cache(maxsize=2)
def _load_seqsim_csv(csv_path):
    df = pd.read_csv(csv_path)
    return df

def seqsim_feature_extraction(sequences):
    # 读取CSV文件 - 使用缓存
    if len(sequences) > 4000:
        csv_path = "/root/VF-pred/raw_data/train-balance_seqsim_features.csv"
    else:
        csv_path = "/root/VF-pred/raw_data/test_seqsim_features.csv"
    
    df = _load_seqsim_csv(csv_path)
    
    # 创建序列到特征向量的映射
    seq_features_dict = {row["Sequence"]: row[["Positive Bitscore", "Negative Bitscore"]].values 
                        for _, row in df.iterrows()}
    
    # 提取特征
    features = np.zeros((len(sequences), 2), dtype=np.float32)
    for i, seq in enumerate(sequences):
        if seq in seq_features_dict:
            features[i] = seq_features_dict[seq]
    
    # 使用sklearn的MinMaxScaler进行归一化
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(features)
    normalized_features=normalized_features.astype(np.float32)
    # features.astype(np.float32)
    
    return normalized_features

# 优化的数据集类 - 使用numpy数组而不是PyTorch张量
class SequenceDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features.astype(np.float32)
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.from_numpy(self.features[idx]).float(), torch.tensor(self.labels[idx], dtype=torch.long)

# 创建模型
def create_model(classifier_type, input_dim, hidden_dim, num_layers, dropout, num_classes):
    if classifier_type.lower() == "cnn":
        model = BalancedCNN(input_dim=input_dim, num_classes=num_classes)
        return model
    elif classifier_type.lower() == "bilstm":
        model = BalancedBiLSTM(input_dim, hidden_dim, num_classes, dropout)
        return model
    elif classifier_type.lower() == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(use_label_encoder=False, eval_metric="logloss", 
                            n_jobs=-1)  # 使用全部CPU核心
    elif classifier_type.lower() == "mlp":
        model = BalancedMLP(input_dim, num_classes, dropout)
        return model
    else:
        raise ValueError(f"未知的分类器类型: {classifier_type}")

# 计算指标
def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # 避免除零错误
    sn = tp / (tp + fn) if (tp + fn) > 0 else 0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        "acc": (tp + tn) / (tp + tn + fp + fn),
        "sn": sn,
        "sp": sp,
        "f1": f1_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }

# 优化的模型评估函数
def evaluate_model(model, loader, criterion=None):
    if model.__class__.__name__ == "XGBClassifier":
        # 批量处理XGBoost预测
        all_inputs = []
        all_labels = []
        for inputs, labels in loader:
            all_inputs.append(inputs.numpy())
            all_labels.append(labels.numpy())
        
        # 合并为一个大批量
        inputs = np.vstack(all_inputs)
        labels = np.concatenate(all_labels)
        
        # 进行预测
        preds = model.predict(inputs)
        metrics = calculate_metrics(labels, preds)
        if criterion:
            metrics["loss"] = 0
        return metrics
    
    # 优化的PyTorch模型评估
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            if criterion:
                total_loss += criterion(outputs, labels).item() * inputs.size(0)
            _, predicted = outputs.max(1)
            all_preds.append(predicted.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # 合并预测结果
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    metrics = calculate_metrics(all_labels, all_preds)
    if criterion:
        metrics["loss"] = total_loss / len(all_labels)
    return metrics

# 创建数据加载器 - 减少worker数以避免过多进程
def create_dataloader(dataset, batch_size, shuffle=False, num_workers=2, sampler=None):
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle if sampler is None else False, 
        num_workers=num_workers,
        pin_memory=True,  # 对CUDA更有效
        sampler=sampler
    )


    
    
# 使用K折交叉验证的训练和评估函数 - 增加性能优化
def cross_validate_model(model_type, feature_combination, train_sequences, train_labels, test_sequences, test_labels, 
                       num_classes, k_folds=5, epochs=30, batch_size=128, patience=5):
    """使用K折交叉验证训练和评估模型（优化版本）"""
    start_time = time.time()
    
    # 提取特征 - 使用并行处理
    print(f"使用特征组合 {feature_combination} 提取特征...")
    X_train = extract_features_parallel(train_sequences, feature_combination)
    X_test = extract_features_parallel(test_sequences, feature_combination)
    print(f"训练特征形状: {X_train.shape}，测试特征形状: {X_test.shape}")
    print(f"特征提取完成，耗时: {time.time() - start_time:.2f}秒")
    
    # 创建数据集 - 直接使用numpy数组
    train_dataset = SequenceDataset(X_train, np.array(train_labels))
    test_dataset = SequenceDataset(X_test, np.array(test_labels))
    
    # 创建K折交叉验证
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    # 初始化存储结果的列表
    fold_val_metrics = []
    fold_test_metrics = []
    
    # 特殊处理XGBoost模型
    if model_type.lower() == "xgboost":
        from xgboost import XGBClassifier
        print(f"训练 {model_type} 模型...")
        
        # XGBoost模型的超参数 - 添加加速参数
        params = {
            'n_estimators': 200, 
            'max_depth': 8,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'use_label_encoder': False,
            'eval_metric': 'logloss',
            'n_jobs': -1,  # 使用全部CPU核心
            'tree_method': 'hist'  # 'hist'方法更快
        }
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
            fold_start = time.time()
            print(f"开始第 {fold+1}/{k_folds} 折...")
            
            # 创建模型
            model = XGBClassifier(**params)
            
            # 准备训练和验证数据
            X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
            y_train_fold, y_val_fold = np.array(train_labels)[train_idx], np.array(train_labels)[val_idx]
            
            # 训练模型
            model.fit(X_train_fold, y_train_fold)
            
            # 创建验证和测试数据加载器
            val_dataset = SequenceDataset(X_val_fold, y_val_fold)
            val_loader = create_dataloader(val_dataset, batch_size, num_workers=2)
            test_loader = create_dataloader(test_dataset, batch_size, num_workers=2)
            
            # 评估模型
            val_metrics = evaluate_model(model, val_loader)
            test_metrics = evaluate_model(model, test_loader)
            
            fold_val_metrics.append(val_metrics)
            fold_test_metrics.append(test_metrics)
            
            print(f"第 {fold+1} 折验证F1分数: {val_metrics['f1']:.4f}, 耗时: {time.time() - fold_start:.2f}秒")
        
    # PyTorch模型的交叉验证
    else:
        print(f"训练 {model_type} 模型...")
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(range(len(train_dataset)))):
            fold_start = time.time()
            print(f"开始第 {fold+1}/{k_folds} 折...")
            
            # 创建数据加载器
            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)
            
            train_loader = create_dataloader(train_dataset, batch_size, sampler=train_sampler, num_workers=2)
            val_loader = create_dataloader(train_dataset, batch_size, sampler=val_sampler, num_workers=2)
            test_loader = create_dataloader(test_dataset, batch_size, num_workers=2)
            
            # 创建模型
            input_dim = X_train.shape[1]
            model = create_model(model_type, input_dim, hidden_dim=512, num_layers=2, dropout=0.3, num_classes=num_classes)
            model.to(device)
            
            # 训练设置
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            best_val_acc, no_improve = 0.0, 0
            best_model_path = f"best_model_fold{fold}.pth"
            
            # 训练循环
            for epoch in range(epochs):
                model.train()
                
                # 批量处理训练样本
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                
                # 验证
                val_metrics = evaluate_model(model, val_loader, criterion)
                if val_metrics["acc"] > best_val_acc:
                    print(f"第 {epoch + 1} 轮验证准确率提升: {val_metrics['acc']:.4f} -> {best_val_acc:.4f}")
                    best_val_acc = val_metrics["acc"]
                    torch.save(model.state_dict(), best_model_path)
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= patience:
                    print(f"提前停止训练，在第 {epoch + 1} 轮")
                    break
            
            # 加载最佳模型
            model.load_state_dict(torch.load(best_model_path))
            
            # 最终评估
            val_metrics = evaluate_model(model, val_loader)
            test_metrics = evaluate_model(model, test_loader)
            
            fold_val_metrics.append(val_metrics)
            fold_test_metrics.append(test_metrics)
            
            print(f"第 {fold+1} 折验证F1分数: {val_metrics['f1']:.4f}, 耗时: {time.time() - fold_start:.2f}秒")
            
            # 删除临时模型文件
            if os.path.exists(best_model_path):
                os.remove(best_model_path)
    
    # 计算平均指标
    avg_val_metrics = {metric: np.mean([fold[metric] for fold in fold_val_metrics]) for metric in fold_val_metrics[0]}
    avg_test_metrics = {metric: np.mean([fold[metric] for fold in fold_test_metrics]) for metric in fold_test_metrics[0]}
    
    # 计算标准差
    std_val_metrics = {metric: np.std([fold[metric] for fold in fold_val_metrics]) for metric in fold_val_metrics[0]}
    std_test_metrics = {metric: np.std([fold[metric] for fold in fold_test_metrics]) for metric in fold_test_metrics[0]}
    
    print(f"总耗时: {time.time() - start_time:.2f}秒")
    
    return {
        'avg_val': avg_val_metrics,
        'std_val': std_val_metrics,
        'avg_test': avg_test_metrics,
        'std_test': std_test_metrics
    }

def print_cv_results(results, feature_combination):
    """打印交叉验证结果"""
    print(f"\n特征组合: {feature_combination}")
    print(f"验证集平均F1分数: {results['avg_val']['f1']:.4f} ± {results['std_val']['f1']:.4f}")
    print(f"验证集平均准确率: {results['avg_val']['acc']:.4f} ± {results['std_val']['acc']:.4f}")
    print(f"测试集平均F1分数: {results['avg_test']['f1']:.4f} ± {results['std_test']['f1']:.4f}")
    print(f"测试集平均准确率: {results['avg_test']['acc']:.4f} ± {results['std_test']['acc']:.4f}")

if __name__ == "__main__":
    # 设置随机种子以确保结果可复现
    seed_everything(666)
    
    # 记录开始时间
    start_time = time.time()
    
    # 模型列表
    model_types = ['bilstm','cnn','xgboost']  # 你可以添加 'cnn', 'xgboost' 等
    
    # FASTA文件路径 - 使用元组使其可哈希，以便缓存
    train_fasta_file = ("/root/VF-pred/raw_data/train_ba.fasta",)
    test_fasta_file = ("/root/VF-pred/raw_data/test.fasta",)
    
    # 可用的特征提取方法
    feature_methods = ['kmer', 'paac', 'qso', 'dde', 'dpc', 'aac', 'seqsim']
    
    num_classes = 2
    
    # 读取数据
    print("读取FASTA文件...")
    train_sequences, train_labels, _ = read_fasta(train_fasta_file)
    test_sequences, test_labels, _ = read_fasta(test_fasta_file)
    print(f"训练序列数: {len(train_sequences)}")
    print(f"测试序列数: {len(test_sequences)}")
    
    # 存储所有结果
    all_results = []
    
    # 每个模型的最佳特征组合
    best_combinations = {model: [] for model in model_types}
    
    # 遍历每个模型
    for model_type in model_types:
        model_start = time.time()
        print(f"\n=== 评估模型: {model_type} ===")
        model_results = []
        
        # 遍历特征组合（1到3个特征的组合）
        for r in range(1, 4):
            for combination in combinations(feature_methods, r):
                feature_combination = list(combination)
                print(f"\n使用特征组合 {feature_combination} 训练 {model_type}...")
                
                # 执行五折交叉验证
                cv_results = cross_validate_model(
                    model_type, 
                    feature_combination,
                    train_sequences, 
                    train_labels,
                    test_sequences, 
                    test_labels,
                    num_classes,
                    k_folds=5,
                    batch_size=256  # 增大批处理大小
                )
                
                # 打印结果
                print_cv_results(cv_results, feature_combination)
                
                # 保存结果
                result_entry = {
                    'Model': model_type,
                    'Feature Combination': feature_combination,
                    'Validation F1 Score': cv_results['avg_val']['f1'],
                    'Validation F1 Std': cv_results['std_val']['f1'],
                    'Validation Accuracy': cv_results['avg_val']['acc'],
                    'Validation Accuracy Std': cv_results['std_val']['acc'],
                    'Validation SN': cv_results['avg_val']['sn'],
                    'Validation SP': cv_results['avg_val']['sp'],
                    'Validation MCC': cv_results['avg_val']['mcc'],
                    'Test F1 Score': cv_results['avg_test']['f1'],
                    'Test F1 Std': cv_results['std_test']['f1'],
                    'Test Accuracy': cv_results['avg_test']['acc'],
                    'Test Accuracy Std': cv_results['std_test']['acc'],
                    'Test SN': cv_results['avg_test']['sn'],
                    'Test SP': cv_results['avg_test']['sp'],
                    'Test MCC': cv_results['avg_test']['mcc']
                }
                
                # 将结果添加到列表和评估列表
                all_results.append(result_entry)
                model_results.append((feature_combination, cv_results['avg_val']['f1']))
        
        # 按照验证集F1分数降序排序
        model_results.sort(key=lambda x: x[1], reverse=True)
        
        # 获取前三名最佳组合（如果有的话）
        top_combinations = model_results[:3]
        best_combinations[model_type] = [(comb, score) for comb, score in top_combinations]
        
        print(f"\n{model_type} 模型的前三名特征组合:")
        for i, (comb, score) in enumerate(best_combinations[model_type]):
            print(f"第 {i+1} 名: {comb}，F1分数: {score:.4f}")
        
        print(f"{model_type} 模型评估完成，耗时: {time.time() - model_start:.2f}秒")
    
    # 将所有结果保存到CSV
    df = pd.DataFrame(all_results)
    output_file = "/root/VF-pred/summary/trad_cv_feature_results_optimized.csv"
    df.to_csv(output_file, index=False)
    print(f"\n所有结果已保存到 {output_file}")
    
    # 将每个模型的最佳特征组合保存到单独的CSV
    best_results = []
    for model, combinations in best_combinations.items():
        for i, (combination, score) in enumerate(combinations):
            best_results.append({
                'Model': model,
                'Rank': i + 1,
                'Feature Combination': str(combination),
                'Validation F1 Score': score
            })
    
    best_output_file = "/root/VF-pred/summary/best_feature_combinations_optimized.csv"
    best_df = pd.DataFrame(best_results)
    best_df.to_csv(best_output_file, index=False)
    print(f"最佳特征组合已保存到 {best_output_file}")
    
    # 打印总耗时
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\n总执行时间: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")