import os
import warnings
import torch
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.air import session
from train import *

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", DeprecationWarning)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import json

# ==== 读取最优超参 ====
# delta  mamba  transformer cnn lstm
best_result_path = "/root/autodl-tmp/.autodl/prot5_tune_result/result/deltaattfusion/best_config.json"
with open(best_result_path, "r") as f:
    best_config = json.load(f)
    # 1
best_model_path = "/root/runs/esm_prot5_fusion/fold_10/best_model.pth"

# ==== 路径配置 ====
test_esm_path = '/root/autodl-tmp/.autodl/embedding_data/test_esm2_t33_650M_UR50D_mean.h5'
test_prot5_path = "/root/autodl-tmp/test_prot_features_modified.h5"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier_type = best_config['classifier_type'] # 使用融合模型类型
classifier_type="deltaattfusion"  # delta  deltaattfusion
is_fusion_model = classifier_type.lower() == "deltaattfusion"

# ==== 加载测试集 ====
if is_fusion_model:
    # 使用DualEmbeddingDataset加载双特征
    test_dataset = DualEmbeddingDataset(
        esm_h5_path=test_esm_path,
        prot5_h5_path=test_prot5_path,
        split='all'  # 加载所有测试数据
    )
    collate_function = dual_features_collate_fn
    
    # 获取ESM和ProtT5特征维度
    sample = test_dataset[0]
    esm_features, prot5_features = sample[0]
    esm_dim = esm_features.shape[-1]
    prot5_dim = prot5_features.shape[-1]
    print(f"ESM特征维度: {esm_dim}, ProtT5特征维度: {prot5_dim}")
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=best_config["batch_size"], 
        shuffle=False,
        collate_fn=collate_function
    )

    model = ImprovedDualPathwayFusion(
        esm_dim=esm_dim,
        prot5_dim=prot5_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=2,
        rank=rank,
        steps=step,
        dropout=dropout
    )
else:
    # 使用原来的单特征加载方式
    test_dataset = H5Dataset(
        h5_path=test_esm_path,
        feature_type="esm2+prot5",
        prot5_path=test_prot5_path,
        csv_path="/root/VF-pred/raw_data/test_seqsim_features.csv",
    )
    collate_function = collate_fn
    test_loader = DataLoader(
        test_dataset, 
        batch_size=best_config["batch_size"], 
        shuffle=False,
        collate_fn=collate_function
    )
    
    # 获取输入维度
    first_batch = next(iter(DataLoader(test_dataset, batch_size=1, collate_fn=collate_function)))
    input_dim = first_batch[0].shape[-1]
    print(f"输入特征维度: {input_dim}")
    if classifier_type.lower() == "delta":
    # 创建普通模型
        model = create_model(
            classifier_type=classifier_type,
            input_dim=input_dim,  # 根据加载的特征自动确定维度
            hidden_dim=best_config["hidden_dim"],
            num_layers=best_config["num_layers"],
            dropout=best_config["dropout"],
            rank=best_config["rank"],
            steps=best_config["steps"],
        )
    model = create_model(
    classifier_type=classifier_type,
    input_dim=input_dim,  # 根据加载的特征自动确定维度
    hidden_dim=best_config["hidden_dim"],
    num_layers=best_config["num_layers"],
    dropout=best_config["dropout"],
    rank=best_config["rank"],
    steps=best_config["steps"],
)

# 加载模型权重
model.to(device)
model.load_state_dict(torch.load(best_model_path))
model.eval()

# ==== 推理 ====
all_labels = []
all_preds = []
all_scores = []

with torch.no_grad():
    for batch in test_loader:
        x, y, lengths = batch
        
        if is_fusion_model:
            # 处理双特征输入
            esm_features, prot5_features = x
            esm_features = esm_features.to(device)
            prot5_features = prot5_features.to(device)
            x = (esm_features, prot5_features)
        else:
            # 处理单特征输入
            x = x.to(device)
        
        y = y.to(device)
        lengths = lengths.to(device)
        
        # 前向传播
        logits = model(x, lengths)
        probs = torch.nn.functional.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        
        # 收集结果
        all_labels.extend(y.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_scores.extend(probs.cpu().numpy())

all_scores = np.array(all_scores)

from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_curve, auc

# ==== 指标计算函数 ====
def calculate_metrics(labels, predictions, scores=None):
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    sn = tp / (tp + fn) if (tp + fn) > 0 else 0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * precision * sn / (precision + sn) if (precision + sn) > 0 else 0
    mcc_numerator = tp * tn - fp * fn
    mcc_denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = mcc_numerator / mcc_denominator if mcc_denominator > 0 else 0
    
    # Calculate AUC and AUPR if probability scores are provided
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

# ==== 输出最终指标 ====
sn, sp, acc, f1, mcc, auc_score, aupr_score = calculate_metrics(all_labels, all_preds, scores=all_scores)
print(f"Sensitivity (SN): {sn:.2f}%")
print(f"Specificity (SP): {sp:.2f}%")
print(f"Accuracy (ACC): {acc:.2f}%")
print(f"F1 Score       : {f1:.2f}%")
print(f"MCC            : {mcc:.2f}%")
print(f"AUC            : {auc_score:.2f}%")
print(f"AUPR           : {aupr_score:.2f}%")

