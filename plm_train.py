import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import h5py
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
import warnings
from model_type import *
import pandas as pd
import shutil

warnings.filterwarnings("ignore", category=UserWarning, message=".*?.*?")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
from Bio import SeqIO


# ------------------------------------------------------------------------------
# 1. Reproducibility
# ------------------------------------------------------------------------------
def seed_everything(seed: int = 42):
    # 设置随机种子以确保可重复性
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ------------------------------------------------------------------------------
# 2. Dataset Loader for HDF5 支持特征拼接
# ------------------------------------------------------------------------------

import numpy as np
from collections import Counter

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def extract_aac_feature(sequence: str) -> np.ndarray:
    """根据氨基酸序列提取 AAC 特征"""
    length = len(sequence)
    count = Counter(sequence)
    return np.array([count.get(aa, 0) / length for aa in AMINO_ACIDS], dtype=np.float32)


import h5py
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


from sklearn.preprocessing import MinMaxScaler


from trad_feature_train import *

class H5Dataset(Dataset):
    def __init__(
        self,
        h5_path: str,
        feature_type: str = "esm2",  # 可选值：esm2, esm2+csv, esm2+aac, prot5, prot5+csv, prot5+aac, csv_only, aac_only
        csv_path: str = None,
        prot5_path: str = None,  # ProtT5特征文件路径
        normalize_csv: bool = True,  # 是否对CSV特征进行标准化处理
        feature_combination: list = [],  # 新增的特征提取方式
    ):
        self.h5_path = h5_path
        self.feature_type = feature_type.lower()
        self.csv_path = csv_path
        self.prot5_path = prot5_path
        self.feature_combination = feature_combination  # 用于传入特征提取方式的组合
        self._keys = None
        self.sequence_to_csv_feat = {}
        self.normalize_csv = normalize_csv
        self.prot5_embeddings = {}
        self.prot5_sequences = {}
        self.prot5_labels = {}
        self.sequence_to_id_map = {}

        # 加载CSV特征
        if "csv" in self.feature_type and self.csv_path:
            self.sequence_to_csv_feat = self._load_csv_features()

        # 加载ProtT5特征
        if "prot5" in self.feature_type and self.prot5_path:
            self._load_prot5_features()

    def _load_csv_features(self):
        df = pd.read_csv(self.csv_path)
        feature_cols = df.columns.drop("Sequence")
        
        if self.normalize_csv:
            print("正在对CSV特征进行标准化处理...")
            scaler = MinMaxScaler()
            df[feature_cols] = scaler.fit_transform(df[feature_cols])
            print(f"CSV特征标准化完成: 包含 {len(feature_cols)} 个特征列")
        
        return {
            row["Sequence"]: row[feature_cols].values.astype(np.float32)
            for _, row in df.iterrows()
        }

    def _load_prot5_features(self):
        print(f"正在加载ProtT5特征: {self.prot5_path}")
        with h5py.File(self.prot5_path, 'r') as hf:
            for seq_id in hf['embeddings'].keys():
                embedding = hf['embeddings'][seq_id][()]
                sequence = hf['sequences'][seq_id][()].decode('ascii')
                label = int(hf['labels'][seq_id][()])
                
                self.prot5_embeddings[seq_id] = embedding
                self.prot5_sequences[seq_id] = sequence
                self.prot5_labels[seq_id] = label
                self.sequence_to_id_map[sequence] = seq_id
        
        print(f"ProtT5特征加载完成: 包含 {len(self.prot5_embeddings)} 个序列")
        
        if len(self.prot5_embeddings) > 0:
            sample_id = next(iter(self.prot5_embeddings))
            print(f"ProtT5特征维度: {self.prot5_embeddings[sample_id].shape}")

    def _get_prot5_feature_by_sequence(self, sequence):
        seq_id = self.sequence_to_id_map.get(sequence)
        if seq_id is None:
            for stored_seq, stored_id in self.sequence_to_id_map.items():
                if sequence in stored_seq or stored_seq in sequence:
                    return self.prot5_embeddings[stored_id]
            return None
        return self.prot5_embeddings[seq_id]

    def __len__(self):
        if self._keys is None:
            if self.feature_type == "prot5" and self.prot5_path:
                self._keys = list(self.prot5_embeddings.keys())
            else:
                with h5py.File(self.h5_path, "r") as f:
                    self._keys = list(f.keys())
        return len(self._keys)

    def __getitem__(self, idx):
        if self.feature_type == "prot5" and self.prot5_path:
            sid = self._keys[idx]
            feats = self.prot5_embeddings[sid]
            label = self.prot5_labels[sid]
            sequence = self.prot5_sequences[sid]
        else:
            with h5py.File(self.h5_path, "r") as f:
                sid = self._keys[idx]
                sequence = f[sid].attrs["sequence"]
                label = f[sid].attrs["label"]
                feats = f[sid]["features"][:] if "esm2" in self.feature_type else None
    
        # 加载ESM2+ProtT5组合 - 直接拼接特征
        if "esm2+prot5" in self.feature_type and self.prot5_path:
            prot5_feat = self._get_prot5_feature_by_sequence(sequence)
            if prot5_feat is not None:
                # 直接拼接两种特征，不再返回元组
                feats = np.concatenate((feats, prot5_feat), axis=-1)
            else:
                print(f"警告: 找不到序列'{sequence[:20]}...'的ProtT5特征")
        # 正常处理prot5+其他特征的组合
        elif "prot5" in self.feature_type and self.prot5_path and self.feature_type != "prot5":
            prot5_feat = self._get_prot5_feature_by_sequence(sequence)
            if prot5_feat is not None:
                feats = prot5_feat if feats is None else np.concatenate((feats, prot5_feat), axis=-1)
            else:
                print(f"警告: 找不到序列'{sequence[:20]}...'的ProtT5特征")
    
        # 只使用CSV特征或拼接CSV特征
        if "csv" in self.feature_type:
            csv_feat = self.sequence_to_csv_feat.get(sequence)
            if csv_feat is None:
                raise ValueError(f"CSV feature not found for sequence: {sequence}")
            feats = (
                csv_feat
                if feats is None
                else np.concatenate((feats, csv_feat), axis=-1)
            )
    
        # 只使用AAC特征或拼接AAC特征 (保留以兼容现有代码)
        if "aac" in self.feature_type:
            aac_feat = extract_aac_feature(sequence)
            feats = (
                aac_feat
                if feats is None
                else np.concatenate((feats, aac_feat), axis=-1)
            )
    
        # 处理新的特征提取方式
        if self.feature_combination:
            new_feats = extract_features_parallel([sequence], self.feature_combination)
            feats = (
                new_feats[0]
                if feats is None
                else np.concatenate((feats, new_feats[0]), axis=-1)
            )
    
        feats = torch.from_numpy(feats).float()
        label = torch.tensor(label, dtype=torch.long)
        return feats, label

# ------------------------------------------------------------------------------
# 3. Collate Function
# ------------------------------------------------------------------------------
def dual_features_collate_fn(batch):
    """处理包含两种特征的批次数据"""
    features_tuple, labels = zip(*batch)
    esm_features, prot5_features = zip(*features_tuple)
    
    # 特征处理
    esm_features = torch.stack(esm_features)
    prot5_features = torch.stack(prot5_features)
    
    # 标签处理
    labels = torch.stack(labels)
    
    # 由于是固定长度向量，lengths都是1
    lengths = torch.ones(len(batch), dtype=torch.long)
    
    return (esm_features, prot5_features), labels, lengths


# ------------------------------------------------------------------------------
# 4. Evaluation
# ------------------------------------------------------------------------------
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    accuracy_score,
    f1_score,
)
import numpy as np
from sklearn.model_selection import KFold


def evaluate(model, dataloader, criterion, device, is_fusion_model=False):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    all_attention_weights = []

    with torch.no_grad():
        for x, y, lengths in dataloader:
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
            
            logits = model(x, lengths)
            loss = criterion(logits, y)
            total_loss += loss.item() * (y.size(0) if isinstance(y, torch.Tensor) else len(y))
            
            preds = logits.argmax(dim=1).cpu().tolist()
            all_preds += preds
            all_labels += y.cpu().tolist()
            
            # 记录注意力权重
            if is_fusion_model and hasattr(model, "last_attention_weights"):
                all_attention_weights.append(model.last_attention_weights.cpu())

    avg_loss = total_loss / len(dataloader.dataset)

    # 分类报告
    report = classification_report(all_labels, all_preds, output_dict=True)
    cm = confusion_matrix(all_labels, all_preds)

    # 计算 SN, SP, ACC, F1, MCC
    if cm.shape == (2, 2):
        TN, FP, FN, TP = cm.ravel()
        SN = TP / (TP + FN + 1e-10)
        SP = TN / (TN + FP + 1e-10)
    else:
        SN = SP = 0.0

    ACC = accuracy_score(all_labels, all_preds)
    F1 = f1_score(all_labels, all_preds, average="binary")
    MCC = matthews_corrcoef(all_labels, all_preds)

    result = {
        "loss": avg_loss,
        "sn": round(SN * 100, 2),
        "sp": round(SP * 100, 2),
        "acc": round(ACC * 100, 2),
        "f1": round(F1 * 100, 2),
        "mcc": round(MCC * 100, 2),
        "classification_report": report,
    }
    
    # 添加注意力权重信息
    if all_attention_weights:
        all_weights = torch.cat(all_attention_weights, dim=0)
        mean_weights = all_weights.mean(dim=0)
        result["attention_weights"] = mean_weights.tolist()
        print(f"平均注意力权重: ESM={mean_weights[0]:.4f}, ProtT5={mean_weights[1]:.4f}")

    return result


def create_model(
    classifier_type,
    input_dim,
    hidden_dim,
    num_layers,
    dropout,
    rank=None,
    steps=None,
):
    num_classes=2
    if classifier_type == "cnn":
        model = BalancedCNN(input_dim=input_dim, num_classes=2)
    elif classifier_type == "gru":
        model = BalancedGRU(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=2)
    elif classifier_type == "transformer":
        model = BalancedTransformer(input_dim=input_dim, d_model=128, nhead=4, num_layers=num_layers, num_classes=2,dropout=dropout)
    elif classifier_type.lower() == "mlp":
        model = BalancedMLP(input_dim,  num_classes=2, dropout=dropout)
    elif classifier_type.lower() == "mamba":
        model = BalancedMamba(
            input_dim=input_dim,
            d_model=hidden_dim,
            d_state=16,
            num_layers=num_layers,
            num_classes=2,
            dropout=dropout,
        )
    elif classifier_type.lower() == "bilstm":
        model = BalancedBiLSTM(input_dim, hidden_dim, num_classes=2, dropout=dropout)
    elif classifier_type.lower() == "lstm":
        model = BalancedLSTM(input_dim, hidden_dim, num_classes=2, dropout=dropout)

    elif classifier_type.lower() == "delta":
        model = VFIter(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            rank=rank,
            steps=steps,
            dropout=dropout,
        )
    elif classifier_type.lower() == "deltaworank":
        model = Deltaworank(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            rank=rank,
            steps=steps,
            dropout=dropout,
        )
    elif classifier_type.lower() == "deltawonorm":
        model=Deltawonorm(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            rank=rank,
            steps=steps,
            dropout=dropout,
        )
    elif classifier_type.lower() == "deltawostep":
        model=Deltawostep(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            rank=rank,
            steps=steps,
            dropout=dropout,
        )
    elif classifier_type.lower() == "deltawores":
        model=Deltawores(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            rank=rank,
            steps=steps,
            dropout=dropout,
        )

    elif classifier_type.lower() == "dtvf":
        model =DTVFModel(
            input_size=input_dim,
            hidden_size_cnn=hidden_dim,
            hidden_size_lstm=hidden_dim,
            num_layers_cnn=num_layers,
            num_layers_lstm=num_layers,
            num_classes=num_classes,
            drop_cnn=dropout,
            drop_lstm=dropout
        )
        



        
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")
    return model

# 同时支持ESM和ProtT5特征的数据集
class DualEmbeddingDataset(Dataset):
    """同时加载ESM和ProtT5两种特征的数据集"""
    
    def __init__(self, esm_h5_path, prot5_h5_path, split='train', test_size=0.2, val_size=0.1, 
                 random_state=42):
        """
        初始化双特征数据集
        
        Args:
            esm_h5_path: ESM特征的H5文件路径
            prot5_h5_path: ProtT5特征的H5文件路径
            split: 数据集划分，可选'train', 'val', 'test'
            test_size: 测试集比例
            val_size: 验证集比例
            random_state: 随机种子
        """
        self.esm_dataset = H5Dataset(esm_h5_path, feature_type="esm2")
        self.prot5_dataset = H5Dataset(prot5_h5_path, feature_type="prot5", prot5_path=prot5_h5_path)
        
        # 获取ESM和ProtT5数据集中的序列和ID映射
        self.esm_seq_to_id = {}
        self.prot5_seq_to_id = {}
        
        # 从ESM数据集获取序列到ID的映射
        with h5py.File(esm_h5_path, "r") as f:
            for key in f.keys():
                sequence = f[key].attrs["sequence"]
                self.esm_seq_to_id[sequence] = key
        
        # 从ProtT5数据集获取序列到ID的映射
        for key, sequence in self.prot5_dataset.prot5_sequences.items():
            self.prot5_seq_to_id[sequence] = key
        
        # 找到两个数据集共有的序列
        esm_sequences = set(self.esm_seq_to_id.keys())
        prot5_sequences = set(self.prot5_seq_to_id.keys())
        common_sequences = list(esm_sequences.intersection(prot5_sequences))
        
        # 获取共有序列对应的ID对
        self.common_keys = []
        for seq in common_sequences:
            esm_id = self.esm_seq_to_id[seq]
            prot5_id = self.prot5_seq_to_id[seq]
            self.common_keys.append((esm_id, prot5_id))
        
        print(f"找到 {len(self.common_keys)} 个ESM和ProtT5共有的样本序列")
        
        # 划分训练、验证和测试集
        all_keys = self.common_keys
        train_keys, test_keys = train_test_split(
            all_keys, test_size=test_size, random_state=random_state
        )
        
        if val_size > 0:
            train_keys, val_keys = train_test_split(
                train_keys, test_size=val_size/(1-test_size), random_state=random_state
            )
        else:
            val_keys = []
        
        # 根据split参数选择相应的键集
        if split == 'train':
            self.keys = train_keys
        elif split == 'val':
            self.keys = val_keys
        elif split == 'test':
            self.keys = test_keys
        else:
            self.keys = all_keys  # 加载全部数据
            
        print(f"加载了 {len(self.keys)} 条 {split} 集数据")
    
    def _find_common_keys(self):
        """找出ESM和ProtT5数据集中共有的键"""
        esm_set = set(self.esm_keys)
        prot5_set = set(self.prot5_keys)
        common = list(esm_set.intersection(prot5_set))
        print(f"找到 {len(common)} 个ESM和ProtT5共有的样本")
        return common
    
    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        # 获取样本ID对 - 解包元组为两个独立的ID
        esm_id, prot5_id = self.keys[idx]
        
        # 使用esm_id访问ESM特征和标签
        with h5py.File(self.esm_dataset.h5_path, "r") as f:
            esm_feat = f[esm_id]["features"][:]
            label = f[esm_id].attrs["label"]
        
        # 使用prot5_id访问ProtT5特征
        prot5_feat = self.prot5_dataset.prot5_embeddings[prot5_id]
        
        # 转换为张量
        esm_feat = torch.tensor(esm_feat, dtype=torch.float32)
        prot5_feat = torch.tensor(prot5_feat, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        
        return (esm_feat, prot5_feat), label
# ------------------------------------------------------------------------------
# 5. 每一折的训练函数
# ------------------------------------------------------------------------------
def train_one_fold(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    log_dir,
    num_epochs,
    patience,
    device,
    classifier_type,
    is_fusion_model=False,
):
    writer = SummaryWriter(log_dir)
    best_f1 = 0.0
    no_improve = 0
    
    for epoch in range(1, num_epochs + 1):
        # 更新训练进度
        current_percentage = epoch / num_epochs
        if hasattr(model, "update_training_percentage"):
            model.update_training_percentage(current_percentage)
        
        model.train()
        total_loss = 0.0
        pbar = tqdm(
            train_loader, desc=f"[{classifier_type}] Epoch {epoch}", leave=False
        )
        
        for x, y, lengths in pbar:
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
            
            optimizer.zero_grad()
            logits = model(x, lengths)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item() * y.size(0)
            pbar.set_postfix(loss=loss.item())

        train_loss = total_loss / len(train_loader.dataset)
        val_report = evaluate(model, val_loader, criterion, device, is_fusion_model=is_fusion_model)
        val_loss = val_report["loss"]
        val_f1 = val_report["f1"]
        
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("F1/val", val_f1, epoch)
        
        # 记录注意力权重
        if is_fusion_model and "attention_weights" in val_report:
            writer.add_scalar("Attention/ESM", val_report["attention_weights"][0], epoch)
            writer.add_scalar("Attention/ProtT5", val_report["attention_weights"][1], epoch)
            
        if val_f1 > best_f1:
            best_f1 = val_f1
            no_improve = 0
            print(f"[{classifier_type}] Best F1: {best_f1}")
            torch.save(model.state_dict(), os.path.join(log_dir, "best_model.pth"))
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[{classifier_type}] Early stopping.")
                break

        if scheduler is not None:
            scheduler.step()

    writer.close()
    
    # 加载最佳模型进行最终评估
    best_model_path = os.path.join(log_dir, "best_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print(f"模型文件 {best_model_path} 不存在，将使用当前模型状态")
    
    # 返回验证集的最终评估报告
    final_val_report = evaluate(model, val_loader, criterion, device, is_fusion_model=is_fusion_model)
    return final_val_report
def collate_fn(batch):
    """处理批次数据"""
    features, labels = zip(*batch)
    
    # 特征处理
    features = torch.stack(features)
    
    # 标签处理
    labels = torch.stack(labels)
    
    # 由于是固定长度向量，lengths都是1
    lengths = torch.ones(len(batch), dtype=torch.long)
    
    return features, labels, lengths
# ==== 加权损失函数，增强特异性 ====
class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, sp_weight=1.0):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.sp_weight = sp_weight  # 特异性权重
        
    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 计算focal loss
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        
        # 对不同类别样本赋予不同权重，提高SP
        weights = torch.ones_like(targets, dtype=torch.float)
        # 为负类样本(标签为0)增加权重，鼓励模型正确预测负例
        weights[targets==0] = self.sp_weight
        
        return (F_loss * weights).mean()
# ------------------------------------------------------------------------------
# 5. 交叉验证训练
def train_one_experiment(
    train_path,
    classifier_type,
    log_dir,
    hidden_dim,
    num_layers,
    dropout,
    batch_size,
    num_epochs,
    patience,
    learning_rate,
    weight_decay,
    optimizer,
    scheduler,
    device,
    rank=None,
    steps=None,
    n_folds=10,
    prot5_path=None,  # 新增参数，用于指定ProtT5特征文件路径
):
    # 检查是否为融合模型
    is_fusion_model = classifier_type.lower() == "deltaattfusion"
    
    if is_fusion_model and prot5_path:
        # 使用双特征数据集
        dataset = DualEmbeddingDataset(
            esm_h5_path='/root/autodl-tmp/.autodl/embedding_data/train_ba_esm2_t33_650M_UR50D_mean.h5',
            prot5_h5_path=prot5_path,
            split='all'  # 加载所有数据，后续使用KFold分割
        )
        collate = dual_features_collate_fn
    else:
        # 使用单特征数据集
        dataset = H5Dataset(
            h5_path=train_path,
            feature_type='esm2+prot5',
            prot5_path=prot5_path
        )
        collate = collate_fn

    # 设置K折交叉验证
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    all_indices = np.arange(len(dataset))
    
    fold_results = []
    best_fold_f1 = 0
    best_fold = 0
    
    # 获取输入维度 - 对于融合模型，需要分别获取两种特征的维度
    if is_fusion_model:
        sample = dataset[0]
        esm_features, prot5_features = sample[0]
        esm_dim = esm_features.shape[-1]
        prot5_dim = prot5_features.shape[-1]
        print(f"ESM 特征维度: {esm_dim}, ProtT5 特征维度: {prot5_dim}")
        input_dim = None  # 融合模型不使用统一的input_dim
    else:
        first_batch = next(iter(DataLoader(dataset, batch_size=1, collate_fn=collate)))
        input_dim = first_batch[0].shape[-1]
        print(f"输入特征维度: {input_dim}")
    
    # 设置损失函数
    criterion = WeightedFocalLoss(sp_weight=1.05)  # 负类权重更高，提高特异性
    
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(all_indices)):
        fold_num = fold + 1
        print(f"\n====== 开始训练第 {fold_num}/{n_folds} 折 ======")
        
        # 创建数据加载器
        train_loader = DataLoader(
            Subset(dataset, train_idx),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate,
            num_workers=4,
        )
        
        val_loader = DataLoader(
            Subset(dataset, val_idx),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate,
        )
        
        # 为当前折创建模型
        if is_fusion_model:
            model = NoFinalFusionLayer(
                esm_dim=esm_dim,
                prot5_dim=prot5_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_classes=2,
                rank=rank,
                steps=steps,
                dropout=dropout,
            ) # DualPathwayFusion  DeltaAttentionFusion
            # model = DualPathwayFusion(
            #     esm_dim=esm_dim,
            #     prot5_dim=prot5_dim,
            #     hidden_dim=hidden_dim,
            #     num_layers=num_layers,
            #     num_classes=2,
            #     rank=rank,
            #     steps=steps,
            #     dropout=dropout,
            # )
        else:
            model = create_model(
                classifier_type=classifier_type,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                rank=rank,
                steps=steps,
            )
        model.to(device)
        
        # 创建优化器
        if optimizer == "Adam":
            opt = optim.Adam(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
        elif optimizer == "SGD":
            opt = optim.SGD(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=0.9,
            )
        elif optimizer == "RMSprop":
            opt = optim.RMSprop(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
        elif optimizer == "AdamW":
            opt = optim.AdamW(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
        
        # 创建学习率调度器
        if scheduler == "step":
            sched = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)
        elif scheduler == "cosine":
            sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs)
        elif scheduler == "none":
            sched = None
        
        # 为每个折创建单独的日志目录
        fold_log_dir = os.path.join(log_dir, f"fold_{fold_num}")
        os.makedirs(fold_log_dir, exist_ok=True)
        
        # 训练当前折
        fold_report = train_one_fold(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=opt,
            scheduler=sched,
            log_dir=fold_log_dir,
            num_epochs=num_epochs,
            patience=patience,
            device=device,
            classifier_type=classifier_type,
            is_fusion_model=is_fusion_model,
        )
        
        fold_results.append(fold_report)
        
        # 记录最佳折
        if fold_report["f1"] > best_fold_f1:
            best_fold_f1 = fold_report["f1"]
            best_fold = fold_num
            
            # 复制最佳模型到主日志目录
            best_model_source = os.path.join(fold_log_dir, "best_model.pth")
            best_model_target = os.path.join(log_dir, "best_model.pth")
            if os.path.exists(best_model_source):
                shutil.copy2(best_model_source, best_model_target)
            else:
                print(f"警告: 找不到最佳模型文件: {best_model_source}")
        
        print(f"第 {fold_num} 折结果: F1={fold_report['f1']:.4f}, ACC={fold_report['acc']:.4f}")

    # 后续代码不变...
    # 计算并返回所有折的平均结果
    avg_results = {
        "loss": np.mean([r["loss"] for r in fold_results]),
        "sn": np.mean([r["sn"] for r in fold_results]),
        "sp": np.mean([r["sp"] for r in fold_results]),
        "acc": np.mean([r["acc"] for r in fold_results]),
        "f1": np.mean([r["f1"] for r in fold_results]),
        "mcc": np.mean([r["mcc"] for r in fold_results]),
    }
    
    # 计算标准差
    std_results = {
        "loss_std": np.std([r["loss"] for r in fold_results]),
        "sn_std": np.std([r["sn"] for r in fold_results]),
        "sp_std": np.std([r["sp"] for r in fold_results]),
        "acc_std": np.std([r["acc"] for r in fold_results]),
        "f1_std": np.std([r["f1"] for r in fold_results]),
        "mcc_std": np.std([r["mcc"] for r in fold_results]),
    }
    
    # 合并平均结果和标准差
    final_results = {**avg_results, **std_results}
    
    # 记录每一折的结果
    final_results["fold_results"] = fold_results
    final_results["best_fold"] = best_fold
    final_results["best_fold_f1"] = best_fold_f1
    
    print(f"\n====== 交叉验证完成 ======")
    print(f"最佳折: {best_fold}/{n_folds}, F1={best_fold_f1:.4f}")
    print(f"平均结果: F1={avg_results['f1']:.4f} ± {std_results['f1_std']:.4f}, "
          f"ACC={avg_results['acc']:.4f} ± {std_results['acc_std']:.4f}")
    
    return final_results


# ------------------------------------------------------------------------------
# 6. Main Loop
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    seed_everything(1314) #3407 42 114514     1314
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Global settings
    hidden_dim = 256
    num_layers = 3
    dropout = 0.48733897240055324
    batch_size = 64
    num_epochs = 44
    patience = 4
    learning_rate = 0.0008660128292465401
    weight_decay = 0.0003360640432255342
    rank = 16
    step = 2

    esm_variants = [
        # "esm2_t12_35M_UR50D",
        # "esm2_t30_150M_UR50D",
        "esm2_t33_650M_UR50D",
        # "esm2_t36_3B_UR50D",
    ]
    
    classifier_types = [
        # "deltaatt",
        # "delta",
        # 'dtvf',
        'deltaattfusion'
        # "gru",
        # "transformer",
        # "mlp",
        # "bilstm",
        # "mamba",
        # 'biomamaba'
        # "lstm",
        # "test",
    ]
    
    base_embed_dir = "/root/autodl-tmp/.autodl/embedding_data"
    results = []

    for esm in esm_variants:
        for clf in classifier_types:
            exp_name = f"{esm}_{clf}"
            log_dir = os.path.join("runs", exp_name)
            os.makedirs(log_dir, exist_ok=True)

            train_path = os.path.join(base_embed_dir, f"train_ba_{esm}_mean.h5")
            
            print(f"\n>>> Running experiment: {exp_name}")
            if "deltaattfusion" == clf:
                print("\n>>> Running fusion model experiment")
                fusion_exp_name = "esm_prot5_fusion"
                fusion_log_dir = os.path.join("runs", fusion_exp_name)
                os.makedirs(fusion_log_dir, exist_ok=True)
                
                # 训练融合模型
                fusion_report = train_one_experiment(
                    train_path='/root/autodl-tmp/.autodl/embedding_data/train_ba_esm2_t33_650M_UR50D_mean.h5',
                    prot5_path='/root/autodl-tmp/train_ba_prot_features_modified.h5',
                    classifier_type="deltaattfusion",
                    log_dir=fusion_log_dir,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    dropout=dropout,
                    batch_size=batch_size,
                    num_epochs=num_epochs,
                    patience=patience,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    optimizer="AdamW",
                    scheduler="cosine",
                    device=device,
                    rank=rank,
                    steps=step,
                    n_folds=10,
                )
                
                print(f"\n>>> 融合模型实验结果:")
                print(f"平均损失: {fusion_report['loss']:.4f} ± {fusion_report['loss_std']:.4f}")
                print(f"平均 SN: {fusion_report['sn']:.4f} ± {fusion_report['sn_std']:.4f}")
                print(f"平均 SP: {fusion_report['sp']:.4f} ± {fusion_report['sp_std']:.4f}")
                print(f"平均 ACC: {fusion_report['acc']:.4f} ± {fusion_report['acc_std']:.4f}")
                print(f"平均 MCC: {fusion_report['mcc']:.4f} ± {fusion_report['mcc_std']:.4f}")
                print(f"平均 F1: {fusion_report['f1']:.4f} ± {fusion_report['f1_std']:.4f}")
                
                results.append({
                    "model": "ESM+ProtT5 Fusion",
                    "f1": fusion_report["f1"],
                    "f1_std": fusion_report["f1_std"],
                    "acc": fusion_report["acc"],
                    "acc_std": fusion_report["acc_std"],
                    "sn": fusion_report["sn"],
                    "sn_std": fusion_report["sn_std"],
                    "sp": fusion_report["sp"],
                    "sp_std": fusion_report["sp_std"],
                    "mcc": fusion_report["mcc"],
                    "mcc_std": fusion_report["mcc_std"]
                })
            else:

                report = train_one_experiment(
                    train_path=train_path,
                    classifier_type=clf,
                    log_dir=log_dir,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    dropout=dropout,
                    batch_size=batch_size,
                    num_epochs=num_epochs,
                    patience=patience,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    optimizer="AdamW",
                    scheduler="none",
                    device=device,
                    rank=rank,
                    steps=step,
                    n_folds=10,
                    prot5_path='/root/autodl-tmp/train_ba_prot_features_modified.h5',
                )
                
            print(f"\n>>> 实验 {exp_name} 最终结果:")
            print(f"平均 SN: {report['sn']:.4f} ± {report['sn_std']:.4f}")
            print(f"平均 SP: {report['sp']:.4f} ± {report['sp_std']:.4f}")
            print(f"平均 ACC: {report['acc']:.4f} ± {report['acc_std']:.4f}")
            print(f"平均 MCC: {report['mcc']:.4f} ± {report['mcc_std']:.4f}")
            print(f"平均 F1: {report['f1']:.4f} ± {report['f1_std']:.4f}")
            
            # 保存实验结果
            results_path = os.path.join(log_dir, "cv_results.json")
            with open(results_path, "w") as f:
                import json
                json.dump(report, f, indent=4)
            print(f"结果已保存到: {results_path}")
            
            results.append({
                "esm": esm,
                "classifier": clf,
                "f1": report["f1"],
                "f1_std": report["f1_std"],
                "acc": report["acc"],
                "acc_std": report["acc_std"]
            })
                        # 配置和训练融合模型
            
    print("\n=== Summary ===")
    for r in results:
        print(f"{r['esm']} + {r['classifier']}: "
              f"F1 = {r['f1']:.4f} ± {r['f1_std']:.4f}, "
              f"ACC = {r['acc']:.4f} ± {r['acc_std']:.4f}"
              f"Sn = {r['sn']:.4f} ± {r['sn_std']:.4f}"
              f"Sp = {r['sp']:.4f} ± {r['sp_std']:.4f}"
              f"MCC = {r['mcc']:.4f} ± {r['mcc_std']:.4f}")
        