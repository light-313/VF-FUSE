import os
import torch
import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class PROT5(Dataset):
    """蛋白质嵌入向量数据集
    
    用于加载和处理从ProtT5或ESM生成的蛋白质嵌入向量数据。
    支持从H5文件加载数据，并提供标准的PyTorch Dataset接口。
    """
    
    def __init__(self, h5_file_path, split='train', test_size=0.2, val_size=0.1, 
                 random_state=42, transform=None):
        """初始化数据集
        
        参数:
            h5_file_path (str): H5数据文件的路径
            split (str): 'train', 'val', 或 'test'之一，指定加载哪个数据子集
            test_size (float): 测试集占总数据的比例
            val_size (float): 验证集占训练数据的比例
            random_state (int): 随机种子，用于数据集划分
            transform (callable, optional): 应用于样本的可选转换函数
        """
        self.h5_file_path = h5_file_path
        self.split = split
        self.transform = transform
        
        # 加载数据
        self.embeddings, self.sequences, self.labels = self._load_data()
        
        # 获取所有键并转换为列表
        all_keys = list(self.embeddings.keys())
        
        # 划分训练、验证和测试集
        train_keys, test_keys = train_test_split(
            all_keys, test_size=test_size, random_state=random_state, stratify=[self.labels[k] for k in all_keys]
        )
        
        if val_size > 0:
            train_keys, val_keys = train_test_split(
                train_keys, test_size=val_size/(1-test_size), random_state=random_state,
                stratify=[self.labels[k] for k in train_keys]
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
        
        # 计算类别分布
        self.class_counts = {}
        for key in self.keys:
            label = int(self.labels[key])
            self.class_counts[label] = self.class_counts.get(label, 0) + 1
            
        print(f"加载了 {len(self.keys)} 条 {split} 集数据")
        print(f"类别分布: {self.class_counts}")

    def _load_data(self):
        """从H5文件加载数据"""
        with h5py.File(self.h5_file_path, 'r') as hf:
            # 读取嵌入向量、序列和标签
            embeddings = {key: hf['embeddings'][key][()] for key in hf['embeddings']}
            sequences = {key: hf['sequences'][key][()].decode('ascii') for key in hf['sequences']}
            labels = {key: int(hf['labels'][key][()]) for key in hf['labels']}
            
        print(f"从 {self.h5_file_path} 加载了 {len(embeddings)} 个嵌入向量")
        
        # 检查数据一致性
        if not all(k in labels for k in embeddings.keys()):
            missing = [k for k in embeddings.keys() if k not in labels]
            print(f"警告: 有 {len(missing)} 个嵌入向量没有对应的标签")
            
        return embeddings, sequences, labels

    def __len__(self):
        """返回数据集大小"""
        return len(self.keys)

    def __getitem__(self, idx):
        """获取指定索引的样本"""
        # 获取对应的键
        key = self.keys[idx]
        
        # 提取样本
        embedding = self.embeddings[key].astype(np.float32)
        label = self.labels[key]
        
        # 应用变换（如果有）
        if self.transform:
            embedding = self.transform(embedding)
            
        return torch.tensor(embedding), torch.tensor(label, dtype=torch.long)
    
    def get_sample_with_metadata(self, idx):
        """获取包含元数据的样本，用于分析和可视化"""
        key = self.keys[idx]
        return {
            'id': key,
            'embedding': self.embeddings[key],
            'sequence': self.sequences[key],
            'label': self.labels[key]
        }
    
    @property
    def embedding_dim(self):
        """返回嵌入向量的维度"""
        key = self.keys[0]  # 使用第一个样本获取维度
        return self.embeddings[key].shape[0]
    
    @staticmethod
    def get_dataloaders(h5_file_path, batch_size=32, test_size=0.2, val_size=0.1, 
                       random_state=42, transform=None, num_workers=4):
        """
        创建训练、验证和测试数据加载器
        
        参数:
            h5_file_path (str): H5数据文件的路径
            batch_size (int): 每个batch的大小
            test_size (float): 测试集占总数据的比例
            val_size (float): 验证集占训练数据的比例
            random_state (int): 随机种子，用于数据集划分
            transform (callable, optional): 应用于样本的可选转换函数
            num_workers (int): DataLoader使用的工作进程数
            
        返回:
            tuple: (训练数据加载器, 验证数据加载器, 测试数据加载器)
        """
        # 创建数据集
        train_dataset = PROT5(
            h5_file_path, split='train', test_size=test_size, 
            val_size=val_size, random_state=random_state, transform=transform
        )
        
        val_dataset = PROT5(
            h5_file_path, split='val', test_size=test_size, 
            val_size=val_size, random_state=random_state, transform=transform
        )
        
        test_dataset = PROT5(
            h5_file_path, split='test', test_size=test_size, 
            val_size=val_size, random_state=random_state, transform=transform
        )
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, 
            num_workers=num_workers, pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        
        return train_loader, val_loader, test_loader


# 使用示例
if __name__ == "__main__":
    # 文件路径
    h5_file_path = "/root/autodl-tmp/test_prot_features_modified.h5"
    
    # 创建数据集
    dataset = PROT5(h5_file_path, split='train')
    
    # 打印数据集信息
    print(f"数据集大小: {len(dataset)}")
    print(f"嵌入向量维度: {dataset.embedding_dim}")
    
    # 获取一个样本
    embedding, label = dataset[0]
    print(f"样本嵌入向量形状: {embedding.shape}")
    print(f"样本标签: {label}")
    
    # 使用便捷函数创建数据加载器
    train_loader, val_loader, test_loader = PROT5.get_dataloaders(
        h5_file_path, batch_size=1
    )
    
    # 显示数据加载器信息
    print(f"训练集批次数: {len(train_loader)}")
    print(f"验证集批次数: {len(val_loader)}")
    print(f"测试集批次数: {len(test_loader)}")
    
    # 获取并显示一个批次
    for batch_X, batch_y in train_loader:
        print(f"批次X形状: {batch_X.shape}")
        print(f"批次y形状: {batch_y.shape}")
        break