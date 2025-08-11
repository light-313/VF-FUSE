import torch
import torch.nn as nn
import torch.nn.functional as F
# from mamba_ssm import Mamba
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
import torch.nn as nn
import torch.nn.functional as F


# 消融实验用于单个特征
class ImprovedDualPathwayFusion(nn.Module):
    """
    改进的双路径融合模型
    """
    def __init__(
        self,
        esm_dim: int = 1280,
        prot5_dim: int = 1024,
        hidden_dim: int = 128,
        num_layers: int = 1,
        num_classes: int = 2,
        rank: int = 4,
        steps: int = 2,
        dropout: float = 0.1,
    ):
        super(ImprovedDualPathwayFusion, self).__init__()
        
        # 特征投影层 - 添加批归一化
        self.esm_proj = nn.Sequential(
            nn.Linear(esm_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )
        self.prot5_proj = nn.Sequential(
            nn.Linear(prot5_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # 改进的自适应门控机制
        self.adaptive_gate = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),  # 考虑两种特征的全局信息
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 4),  # 为每个路径输出两个权重
            nn.Sigmoid()
        )
        
        # 特征重要性注意力
        self.feature_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # 增强的路径处理
        self.sn_path = nn.ModuleList([
            ImprovedDeltaProductBlock(hidden_dim, rank=rank, steps=steps, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        self.sp_path = nn.ModuleList([
            ImprovedDeltaProductBlock(hidden_dim, rank=rank, steps=steps, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # 路径交互层
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=2,
            dropout=dropout,
            batch_first=True
        )
        
        # 自适应特征融合
        self.adaptive_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 增强的分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, num_classes)
        )
        
        # 可学习的路径平衡因子
        self.balance_factor = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x, lengths=None):
        esm_features, prot5_features = x
        
        # 增强的特征投影
        esm_full = self.esm_proj(esm_features)
        prot5_full = self.prot5_proj(prot5_features)
        
        # 拆分特征用于不同路径
        esm_sn, esm_sp = torch.chunk(esm_full, 2, dim=-1)
        prot5_sn, prot5_sp = torch.chunk(prot5_full, 2, dim=-1)
        
        # 全局上下文感知的门控
        global_context = torch.cat([esm_full, prot5_full], dim=-1)
        gate_weights = self.adaptive_gate(global_context)
        
        # 分配门控权重
        sn_esm_weight, sn_prot5_weight, sp_esm_weight, sp_prot5_weight = torch.chunk(gate_weights, 4, dim=-1)
        
        # 加权融合
        sn_input = sn_esm_weight * esm_sn + sn_prot5_weight * prot5_sn
        sp_input = sp_esm_weight * esm_sp + sp_prot5_weight * prot5_sp
        
        # 特征注意力增强
        sn_input = sn_input.unsqueeze(1)
        sp_input = sp_input.unsqueeze(1)
        
        sn_enhanced, _ = self.feature_attention(sn_input, sn_input, sn_input)
        sp_enhanced, _ = self.feature_attention(sp_input, sp_input, sp_input)
        
        sn_input = sn_enhanced.squeeze(1)
        sp_input = sp_enhanced.squeeze(1)
        
        # 通过各自路径处理
        sn_features = sn_input
        for block in self.sn_path:
            sn_features = block(sn_features)
            
        sp_features = sp_input
        for block in self.sp_path:
            sp_features = block(sp_features)
        
        # 路径交互 - 让两条路径互相学习
        sn_query = sn_features.unsqueeze(1)
        sp_key_value = sp_features.unsqueeze(1)
        
        sn_cross, _ = self.cross_attention(sn_query, sp_key_value, sp_key_value)
        sn_features = sn_features + 0.1 * sn_cross.squeeze(1)  # 小幅度交互
        
        sp_query = sp_features.unsqueeze(1)
        sn_key_value = sn_features.unsqueeze(1)
        
        sp_cross, _ = self.cross_attention(sp_query, sn_key_value, sn_key_value)
        sp_features = sp_features + 0.1 * sp_cross.squeeze(1)
        
        # 自适应融合
        combined_features = torch.cat([sn_features, sp_features], dim=-1)
        fused_features = self.adaptive_fusion(combined_features)
        
        # 动态路径平衡
        balance = torch.sigmoid(self.balance_factor)
        sn_part, sp_part = torch.chunk(fused_features, 2, dim=-1)
        final_features = balance * sn_part + (1 - balance) * sp_part
        final_features = torch.cat([final_features, final_features], dim=-1)  # 维持维度
        
        # 分类
        logits = self.classifier(final_features)
        
        return logits

class ImprovedDeltaProductBlock(nn.Module):
    """
    改进版 DeltaProductBlock：
    在每个迭代更新步骤中引入低秩投影和非线性激活，增强特征交互能力。
    保留原有的多步迭代更新和残差连接结构。
    """
    def __init__(self, dim: int, rank: int = 4, steps: int = 2, dropout: float = 0.1):
        """
        Args:
            dim: 输入和输出特征的维度。
            rank: 低秩投影的中间维度 (rank < dim)。
            steps: 迭代更新的步数。
            dropout: 最终应用的 Dropout 比率。
        """
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.steps = steps

        if rank >= dim:
             # 警告：rank >= dim 会导致低秩瓶颈失效
             print(f"Warning: rank ({rank}) >= dim ({dim}). Low-rank bottleneck may not be effective.")

        # 定义每一步迭代计算 delta 的模块序列
        # 每一步的 delta 计算都包含一个低秩投影和非线性
        self.delta_calculations = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, rank, bias=False), # 投影到低秩空间
                nn.GELU(),                       # 非线性激活 (使用 GeLU)
                nn.Linear(rank, dim, bias=False), # 从低秩空间投影回原维度 (得到 delta)
                # 可选：如果需要在每一步内部也加 dropout，可以在这里添加
                # nn.Dropout(dropout)
            )
            for _ in range(steps)
        ])

        # 所有迭代步完成后应用的归一化层和最终 Dropout 层
        self.norm = nn.LayerNorm(dim)
        # 使用单独的属性名，避免与 nn.Dropout 类名冲突
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量，形状为 (batch, dim)。

        Returns:
            输出张量，形状为 (batch, dim)。
        """
        # x: (batch, dim)

        # 在迭代开始前保存初始输入，用于最终的残差连接
        initial_residual = x

        # 应用迭代更新
        # 根据你原代码的逻辑，状态 'x' 在循环内部是原地更新的
        # 每一步计算 delta 是基于当前的状态值
        current_state = x # 使用一个独立的变量来表示当前状态

        for step_layer in self.delta_calculations:
            # 基于当前状态计算这一步的 delta
            delta = step_layer(current_state) # (batch, dim) -> (batch, dim)

            # 将计算出的 delta 应用到当前状态，更新状态
            current_state = current_state + delta

        # 在所有迭代步完成后，应用归一化和最终的 Dropout
        processed_x = self.norm(current_state)
        processed_x = self.dropout_layer(processed_x) # 应用 Dropout 层

        # 添加最初的残差连接
        return processed_x + initial_residual


# --------------------------------------------------------------
# --- IterLowRankBlock 定义 ---
# --------------------------------------------------------------
class IterLowRankBlock(nn.Module):
    """
    增强特征交互能力。
    保留原有的多步迭代更新和残差连接结构。
    """
    def __init__(self, dim: int, rank: int = 4, steps: int = 2, dropout: float = 0.1):
        """
        Args:
            dim: 输入和输出特征的维度。
            rank: 低秩投影的中间维度 (rank < dim)。
            steps: 迭代更新的步数。
            dropout: 最终应用的 Dropout 比率。
        """
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.steps = steps

        if rank >= dim:
             # 警告：rank >= dim 会导致低秩瓶颈失效
             print(f"Warning: rank ({rank}) >= dim ({dim}). Low-rank bottleneck may not be effective.")

        # 定义每一步迭代计算 delta 的模块序列
        # 每一步的 delta 计算都包含一个低秩投影和非线性
        self.delta_calculations = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, rank, bias=False), # 投影到低秩空间
                nn.GELU(),                       # 非线性激活 (使用 GeLU)
                nn.Linear(rank, dim, bias=False), # 从低秩空间投影回原维度 (得到 delta)
                # 可选：如果需要在每一步内部也加 dropout，可以在这里添加
                # nn.Dropout(dropout)
            )
            for _ in range(steps)
        ])

        # 所有迭代步完成后应用的归一化层和最终 Dropout 层
        self.norm = nn.LayerNorm(dim)
        # 使用单独的属性名，避免与 nn.Dropout 类名冲突
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量，形状为 (batch, dim)。

        Returns:
            输出张量，形状为 (batch, dim)。
        """
        # x: (batch, dim)

        # 在迭代开始前保存初始输入，用于最终的残差连接
        initial_residual = x

        # 应用迭代更新
        # 根据你原代码的逻辑，状态 'x' 在循环内部是原地更新的
        # 每一步计算 delta 是基于当前的状态值
        current_state = x # 使用一个独立的变量来表示当前状态

        for step_layer in self.delta_calculations:
            # 基于当前状态计算这一步的 delta
            delta = step_layer(current_state) # (batch, dim) -> (batch, dim)

            # 将计算出的 delta 应用到当前状态，更新状态
            current_state = current_state + delta

        # 在所有迭代步完成后，应用归一化和最终的 Dropout
        processed_x = self.norm(current_state)
        processed_x = self.dropout_layer(processed_x) # 应用 Dropout 层

        # 添加最初的残差连接
        return processed_x + initial_residual

# --------------------------------------------------------------
class VFIter(nn.Module):
    def __init__(
        self,
        input_dim: int = 1280,
        hidden_dim: int = 256,
        num_layers: int = 2, # 堆叠多少个 ImprovedDeltaProductBlock 层
        num_classes: int = 2,
        rank: int = 4,       # ImprovedDeltaProductBlock 内部的低秩维度
        steps: int = 2,      # ImprovedDeltaProductBlock 内部的迭代步数
        dropout: float = 0.1, # 用于 feature_proj, Delta blocks 内部和 classifier 的 Dropout 比率
    ):
        """
        Args:
            input_dim: 原始输入特征的维度。
            hidden_dim: 模型内部使用的特征维度 (隐藏层维度)。
            num_layers: 要堆叠的 ImprovedDeltaProductBlock 层的数量。
            num_classes: 分类任务的输出类别数。
            rank: 每个 ImprovedDeltaProductBlock 内部低秩投影的维度。
            steps: 每个 ImprovedDeltaProductBlock 内部迭代更新的步数。
            dropout: 在模型不同部分应用的 Dropout 比率。
        """
        super().__init__()

        # 初始特征投影层
        # 将输入特征映射到模型的隐藏维度
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(), # 初始投影后通常使用激活函数
            nn.Dropout(dropout), # 初始投影后应用 Dropout
        )

        self.delta_blocks = nn.ModuleList([
            # 每个 ImprovedDeltaProductBlock 内部包含了迭代更新、残差连接、归一化和 Dropout
            IterLowRankBlock(hidden_dim, rank=rank, steps=steps, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim), # 可选但通常有益的最终归一化层
            nn.Linear(hidden_dim, 64), # 在最终输出前可以加一个小的隐藏层
            nn.ReLU(),
            nn.Dropout(dropout), # 分类器内部应用 Dropout
            nn.Linear(64, num_classes) # 最终的全连接层输出类别得分
        )

    def forward(self, x: torch.Tensor, lengths=None) -> torch.Tensor:
        """
        ImprovedDeltamlp 模型的前向传播。

        Args:
            x: 输入张量，形状为 (batch, input_dim)。
            lengths: 在这个处理固定向量的模型中未使用。

        Returns:
            logits 张量，形状为 (batch, num_classes)。
        """
        # 通过初始投影层
        x = self.feature_proj(x)  # (batch, hidden_dim)

        for block in self.delta_blocks:
            x = block(x) # (batch, hidden_dim) -> (batch, hidden_dim)，每一层都进行这种变换

        # 通过最终分类器层
        logits = self.classifier(x)  # (batch, num_classes)

        return logits
# 消融实验和特征融合模型

# 辅助函数：计算参数量
# ------------------------------------------------------------------------------
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ------------------------------------------------------------------------------
# 输入转换辅助函数
# ------------------------------------------------------------------------------
def convert_x_shape(x):
    if x.dim() == 2:
        x = x.unsqueeze(1)  # [B, D] → [B, 1, D]
    elif x.dim() == 3:
        x = x.permute(0, 2, 1)
    else:
        raise ValueError(f"Unexpected input shape: {x.shape}")
    return x
