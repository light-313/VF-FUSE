# kmer(默认k=1,2)   dde  aac  dpc
import numpy as np
from sklearn.calibration import LabelEncoder
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
amino_acids = "ACDEFGHIKLMNPQRSTVWY"
from Bio import SeqIO
from itertools import product
def read_fasta(file_paths, max_length=200000):
    sequences, labels, ids = [], [], []
    for file_path in file_paths:
        for record in SeqIO.parse(file_path, "fasta"):
            seq = str(record.seq)
            if len(seq) > max_length:
                continue
            label = int(record.description.split("|")[1])  # Assumes label is in the second part
            sequences.append(seq)
            labels.append(label)
            ids.append(record.description)
    return sequences, labels, ids
# k-mer 特征提取函数
def kmer_feature_extraction(sequences, k_list):
    feature_all=[]
    for k in k_list:
        # 生成所有可能的 k-mers
        all_kmers = [''.join(p) for p in product(amino_acids, repeat=k)]
        kmer_dict = {kmer: i for i, kmer in enumerate(all_kmers)}

        kmer_features = []
        for seq in sequences:
            kmers = [seq[i:i + k] for i in range(len(seq) - k + 1)]
            kmer_counts = np.zeros(len(all_kmers))
            for kmer in kmers:
                if kmer in kmer_dict:
                    kmer_counts[kmer_dict[kmer]] += 1
            kmer_features.append(kmer_counts)
        
        # 归一化 k-mer 特征
        kmer_features = np.array(kmer_features)
        # print(f"k={k},feature shape:{kmer_features.shape}")
        
        kmer_features = kmer_features / (len(seq) - k + 1)
        feature_all.append(kmer_features)
    
    return np.concatenate(feature_all, axis=1)    
def aac_feature_extraction(sequences):
    """
    提取氨基酸组成特征
    
    该函数接受一个蛋白质序列列表作为输入，并为每个序列计算氨基酸组成（AAC）特征。
    氨基酸组成特征表示蛋白质序列中每种氨基酸的相对频率。
    
    参数:
    sequences (list of str): 蛋白质序列列表，每个序列是一个字符串
    
    返回:
    numpy.ndarray: 一个二维数组，其中每行对应一个序列的氨基酸组成特征
    """
    # 初始化氨基酸组成特征列表
    aac_features = []
    # 遍历每个序列
    for seq in sequences:
        # 计算并添加每个氨基酸的相对频率
        aac = [seq.count(aa) / len(seq) for aa in "ACDEFGHIKLMNPQRSTVWY"]
        aac_features.append(aac)
    # 将列表转换为numpy数组并返回
    return np.array(aac_features)


def dpc_feature_extraction(sequences):
    """
    计算序列的二联体频率（DPC）特征

    """
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    dipeptides = [''.join(p) for p in product(amino_acids, repeat=2)]
    
    dpc_features = []
    for seq in sequences:
        dpc = {dp: 0 for dp in dipeptides}
        for i in range(len(seq) - 1):
            dipeptide = seq[i:i+2]
            if dipeptide in dpc:
                dpc[dipeptide] += 1
        
        total_dipeptides = sum(dpc.values())
        if total_dipeptides > 0:
            dpc = {k: v / total_dipeptides for k, v in dpc.items()}
        
        dpc_features.append(list(dpc.values()))
    
    return np.array(dpc_features)


def dde_feature_extraction(sequences):
    """
    计算序列的二肽对频率（DDE）特征
    """

    # 定义氨基酸及其对应的密码子数量
    codon_counts = {
        'A': 4, 'C': 2, 'D': 2, 'E': 2, 'F': 2,
        'G': 4, 'H': 2, 'I': 3, 'K': 2, 'L': 6,
        'M': 1, 'N': 2, 'P': 4, 'Q': 2, 'R': 6,
        'S': 6, 'T': 4, 'V': 4, 'W': 1, 'Y': 2
    }

    # 总可能的密码子数（不包括终止密码子）
    total_codons = 61

    # 生成所有可能的二肽
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    dipeptides = [''.join(p) for p in product(amino_acids, repeat=2)]

    # 计算DDE特征
    dde_features = []
    for seq in sequences:
        # 计算每个二肽的实际频率
        dpc = {dp: 0 for dp in dipeptides}
        for i in range(len(seq) - 1):
            dipeptide = seq[i:i+2]
            if dipeptide in dpc:
                dpc[dipeptide] += 1
        
        total_dipeptides = len(seq) - 1
        if total_dipeptides > 0:
            dpc = {k: v / total_dipeptides for k, v in dpc.items()}
        
        # 计算每个二肽的理论均值和方差
        dde = []
        for dp in dipeptides:
            r, s = dp[0], dp[1]
            Cr = codon_counts[r]
            Cs = codon_counts[s]
            Tm = (Cr / total_codons) * (Cs / total_codons)
            Tv = Tm * (1 - Tm) / total_dipeptides
            if Tv > 0:
                dde_value = (dpc[dp] - Tm) / np.sqrt(Tv)
            else:
                dde_value = 0.0
            dde.append(dde_value)
        
        dde_features.append(dde)
    
    return np.array(dde_features)
# paac   qso
def paac_feature_extraction(sequences, lambda_value=30, weight=0.05):
    """
    提取伪氨基酸组成（Pseudo Amino Acid Composition, PAAC）特征
    
    参数:
    sequences (list of str): 蛋白质序列列表
    lambda_value (int): λ 值，表示考虑的相邻氨基酸对的最大距离
    weight (float): 权重因子，用于平衡序列信息和相邻氨基酸对信息
    
    返回:
    numpy.ndarray: 一个二维数组，其中每行对应一个序列的PAAC特征
    """
    # 定义氨基酸的物理化学性质
    physicochemical_properties = {
        'A': [0.358, 0.544, 0.353, -0.042, 1.330],
        'C': [0.246, 0.134, -0.466, 0.647, -1.664],
        'D': [0.105, 0.293, -1.623, 1.444, -1.142],
        'E': [0.151, 0.159, -0.346, 0.745, -0.344],
        'F': [0.397, -0.006, -0.342, 1.717, 0.035],
        'G': [0.562, 0.526, 1.330, -0.112, 0.707],
        'H': [0.299, 0.113, 0.335, 0.694, -0.272],
        'I': [0.441, -0.203, -0.389, 1.328, 0.549],
        'K': [0.219, 0.255, 0.816, 0.243, -1.497],
        'L': [0.369, -0.133, -0.501, 1.214, 0.471],
        'M': [0.344, -0.022, -0.229, 1.345, 0.244],
        'N': [0.160, 0.378, -1.122, 0.846, -0.663],
        'P': [0.501, 0.443, 0.945, -0.039, 1.708],
        'Q': [0.267, 0.124, 0.069, 0.538, -0.690],
        'R': [0.237, 0.260, 1.538, -0.207, -1.942],
        'S': [0.232, 0.305, -0.089, 0.670, -0.184],
        'T': [0.243, 0.389, -0.028, 0.580, -0.292],
        'V': [0.387, -0.184, -0.344, 1.242, 0.406],
        'W': [0.318, -0.083, -0.194, 2.323, 0.347],
        'Y': [0.292, -0.014, -0.125, 1.497, 0.013]
    }

    # 计算PAAC特征
    paac_features = []
    for seq in sequences:
        # 计算氨基酸组成特征
        aac = [seq.count(aa) / len(seq) for aa in "ACDEFGHIKLMNPQRSTVWY"]
        
        # 计算相邻氨基酸对的相关系数
        correlations = []
        for l in range(1, lambda_value + 1):
            correlation = 0.0
            for i in range(len(seq) - l):
                aa1 = seq[i]
                aa2 = seq[i + l]
                if aa1 in physicochemical_properties and aa2 in physicochemical_properties:
                    correlation += sum((physicochemical_properties[aa1][j] - physicochemical_properties[aa2][j]) ** 2 for j in range(5))
            
            # 防止除以零
            if len(seq) - l > 0:
                correlation /= len(seq) - l
            else:
                correlation = 0.0
            
            correlations.append(correlation)
        
        # 计算PAAC特征向量
        paac = aac + [weight * correlation / (1 + weight * sum(correlations)) for correlation in correlations]
        paac_features.append(paac)
    
    return np.array(paac_features)



def qso_feature_extraction(sequences, max_distance=30, weight=0.1):
    """
    提取准序列顺序（QSO）特征
    
    该函数接受一个蛋白质序列列表作为输入，并为每个序列计算QSO特征。
    QSO特征不仅包括氨基酸的组成，还包括氨基酸之间的距离分布。
    
    参数:
    sequences (list of str): 蛋白质序列列表，每个序列是一个字符串
    max_distance (int): 最大距离，默认为30
    weight (float): 权重因子，默认为0.1
    
    返回:
    numpy.ndarray: 一个二维数组，其中每行对应一个序列的QSO特征
    """
    # 氨基酸类型
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    
    # 初始化QSO特征列表
    qso_features = []
    
    # 遍历每个序列
    for seq in sequences:
        # 计算氨基酸组成
        aac = [seq.count(aa) / len(seq) for aa in amino_acids]
        
        # 计算距离分布特征
        distance_features = []
        for d in range(1, max_distance + 1):
            distance_count = 0
            for i in range(len(seq) - d):
                aa1 = seq[i]
                aa2 = seq[i + d]
                if aa1 in amino_acids and aa2 in amino_acids:
                    distance_count += 1
            
            # 防止除以零
            if len(seq) - d > 0:
                distance_features.append(distance_count / (len(seq) - d))
            else:
                distance_features.append(0.0)
        
        # 组合AAC和距离分布特征
        qso = aac + [weight * df for df in distance_features]
        
        # 归一化QSO特征
        qso = np.array(qso) / (1 + weight * max_distance)
        
        qso_features.append(qso)
    
    return np.array(qso_features)

if __name__ == "__main__":
    # 提取paac
    train_fasta_file = ["/root/VF-pred/raw_data/train_ba.fasta"]
    test_fasta_file = ["/root/VF-pred/raw_data/test.fasta"]
    sequences, labels, ids = read_fasta(train_fasta_file)
    test_sequences, test_labels, test_ids = read_fasta(test_fasta_file)
    feature_train=paac_feature_extraction(sequences)
    feature_test=paac_feature_extraction(test_sequences)
    # 保存csv
    import pandas as pd
    train_df = pd.DataFrame(feature_train)
    train_df['label'] = labels
    train_df['id'] = ids
    train_df['feature'] = feature_train.tolist()
    train_df.to_csv('/root/VF-pred/raw_data/paac_train_ba.csv', index=False)
    test_df = pd.DataFrame(feature_test)
    test_df['label'] = test_labels
    test_df['id'] = test_ids
    test_df['feature'] = feature_test.tolist()
    test_df.to_csv('/root/VF-pred/raw_data/paac_test.csv', index=False)
    