import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
import torch.nn as nn
import torch.nn.functional as F

# 消融实验用于单个特征
class DeltaProductBlockNoNorm(nn.Module):
    """
    去掉归一化层的 DeltaProductBlock，用于消融实验。
    """
    def __init__(self, dim: int, rank: int = 4, steps: int = 2, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.steps = steps

        self.delta_calculations = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, rank, bias=False),
                nn.GELU(),
                nn.Linear(rank, dim, bias=False),
            )
            for _ in range(steps)
        ])
        
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        current_state = x

        for step_layer in self.delta_calculations:
            delta = step_layer(current_state)
            current_state = current_state + delta

        processed_x = self.dropout_layer(current_state)
        return processed_x  # 没有归一化

class DeltaProductBlockNoResidual(nn.Module):
    """
    去掉残差连接的 DeltaProductBlock，用于消融实验。
    """
    def __init__(self, dim: int, rank: int = 4, steps: int = 2, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.steps = steps

        self.delta_calculations = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, rank, bias=False),
                nn.GELU(),
                nn.Linear(rank, dim, bias=False),
            )
            for _ in range(steps)
        ])
        
        self.norm = nn.LayerNorm(dim)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        current_state = x

        for step_layer in self.delta_calculations:
            delta = step_layer(current_state)
            current_state = current_state + delta  # 注意：这里不再使用残差连接

        processed_x = self.norm(current_state)
        processed_x = self.dropout_layer(processed_x)
        return processed_x  # 不加上初始输入
class Deltawores(nn.Module):
    """
    模型堆叠多个 ImprovedDeltaProductBlock 层，用于固定向量分类。
    该模型在初始投影后，主要依靠堆叠的改进版 DeltaProductBlock 进行特征处理。
    """
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

        # 堆叠多层 ImprovedDeltaProductBlock
        # 每一层处理前一层的输出
        self.delta_blocks = nn.ModuleList([
            # 每个 ImprovedDeltaProductBlock 内部包含了迭代更新、残差连接、归一化和 Dropout
            DeltaProductBlockNoResidual(hidden_dim, rank=rank,steps=steps, dropout=dropout)
            for _ in range(num_layers)
        ])

        # 最终的分类器层
        # 将堆叠层输出的特征映射到各类别得分 (logits)
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

        # 依次通过堆叠的 ImprovedDeltaProductBlock 层
        for block in self.delta_blocks:
            x = block(x) # (batch, hidden_dim) -> (batch, hidden_dim)，每一层都进行这种变换

        # 通过最终分类器层
        logits = self.classifier(x)  # (batch, num_classes)

        return logits

class DeltaProductBlockSingleStep(nn.Module):
    """
    单步迭代的 DeltaProductBlock，用于消融实验。
    """
    def __init__(self, dim: int, rank: int = 4, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.rank = rank

        # 只保留一步迭代
        self.delta_calculation = nn.Sequential(
            nn.Linear(dim, rank, bias=False), # 投影到低秩空间
            nn.GELU(),                       # 非线性激活
            nn.Linear(rank, dim, bias=False), # 从低秩空间投影回原维度
        )

        self.norm = nn.LayerNorm(dim)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        initial_residual = x
        delta = self.delta_calculation(x)  # 只计算一次 delta
        processed_x = self.norm(x + delta)  # 加上残差并归一化
        processed_x = self.dropout_layer(processed_x)
        return processed_x + initial_residual
class Deltawostep(nn.Module):
    """
    模型堆叠多个 ImprovedDeltaProductBlock 层，用于固定向量分类。
    该模型在初始投影后，主要依靠堆叠的改进版 DeltaProductBlock 进行特征处理。
    """
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

        # 堆叠多层 ImprovedDeltaProductBlock
        # 每一层处理前一层的输出
        self.delta_blocks = nn.ModuleList([
            # 每个 ImprovedDeltaProductBlock 内部包含了迭代更新、残差连接、归一化和 Dropout
            DeltaProductBlockSingleStep(hidden_dim, rank=rank, dropout=dropout)
            for _ in range(num_layers)
        ])

        # 最终的分类器层
        # 将堆叠层输出的特征映射到各类别得分 (logits)
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

        # 依次通过堆叠的 ImprovedDeltaProductBlock 层
        for block in self.delta_blocks:
            x = block(x) # (batch, hidden_dim) -> (batch, hidden_dim)，每一层都进行这种变换

        # 通过最终分类器层
        logits = self.classifier(x)  # (batch, num_classes)

        return logits

class Deltawonorm(nn.Module):
    """
    模型堆叠多个 ImprovedDeltaProductBlock 层，用于固定向量分类。
    该模型在初始投影后，主要依靠堆叠的改进版 DeltaProductBlock 进行特征处理。
    """
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

        # 堆叠多层 ImprovedDeltaProductBlock
        # 每一层处理前一层的输出
        self.delta_blocks = nn.ModuleList([
            # 每个 ImprovedDeltaProductBlock 内部包含了迭代更新、残差连接、归一化和 Dropout
            DeltaProductBlockNoNorm(hidden_dim, rank=rank,steps=steps, dropout=dropout)
            for _ in range(num_layers)
        ])

        # 最终的分类器层
        # 将堆叠层输出的特征映射到各类别得分 (logits)
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

        # 依次通过堆叠的 ImprovedDeltaProductBlock 层
        for block in self.delta_blocks:
            x = block(x) # (batch, hidden_dim) -> (batch, hidden_dim)，每一层都进行这种变换

        # 通过最终分类器层
        logits = self.classifier(x)  # (batch, num_classes)

        return logits
class DeltaProductBlockNoRank(nn.Module):
    """
    消除低秩投影的 DeltaProductBlock，用于消融实验。
    """
    def __init__(self, dim: int, steps: int = 2, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.steps = steps

        # 定义每一步迭代计算 delta 的模块序列，消除低秩投影
        self.delta_calculations = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim, bias=False),  # 无低秩投影，直接映射到原维度
                nn.GELU(),  # 非线性激活
                nn.Linear(dim, dim, bias=False),  # 从原维度返回
            )
            for _ in range(steps)
        ])
        
        self.norm = nn.LayerNorm(dim)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        initial_residual = x
        current_state = x

        for step_layer in self.delta_calculations:
            delta = step_layer(current_state)
            current_state = current_state + delta

        processed_x = self.norm(current_state)
        processed_x = self.dropout_layer(processed_x)

        return processed_x + initial_residual

class Deltaworank(nn.Module):
    """
    模型堆叠多个 ImprovedDeltaProductBlock 层，用于固定向量分类。
    该模型在初始投影后，主要依靠堆叠的改进版 DeltaProductBlock 进行特征处理。
    """
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

        # 堆叠多层 ImprovedDeltaProductBlock
        # 每一层处理前一层的输出
        self.delta_blocks = nn.ModuleList([
            # 每个 ImprovedDeltaProductBlock 内部包含了迭代更新、残差连接、归一化和 Dropout
            DeltaProductBlockNoRank(hidden_dim, steps=steps, dropout=dropout)
            for _ in range(num_layers)
        ])

        # 最终的分类器层
        # 将堆叠层输出的特征映射到各类别得分 (logits)
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

        # 依次通过堆叠的 ImprovedDeltaProductBlock 层
        for block in self.delta_blocks:
            x = block(x) # (batch, hidden_dim) -> (batch, hidden_dim)，每一层都进行这种变换

        # 通过最终分类器层
        logits = self.classifier(x)  # (batch, num_classes)

        return logits
    
# 消融实验用于单个特征

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

class DeltaAttentionFusion(nn.Module):
    """
    使用注意力机制融合ESM和ProtT5特征的Delta模型
    """
    def __init__(
        self,
        esm_dim: int = 1280,   # ESM特征维度
        prot5_dim: int = 1024,  # ProtT5特征维度
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 2,
        rank: int = 4,
        steps: int = 2,
        dropout: float = 0.1,
    ):
        super(DeltaAttentionFusion, self).__init__()
        
        # 特征投影层：将不同维度的特征投影到相同维度
        self.esm_proj = nn.Linear(esm_dim, hidden_dim)
        self.prot5_proj = nn.Linear(prot5_dim, hidden_dim)
        
        # 注意力融合层
        self.query_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Delta模型的处理块
        self.delta_blocks = nn.ModuleList([
            IterLowRankBlock(hidden_dim, rank=rank, steps=steps, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # 最终分类层
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x, lengths=None):
        """
        前向传播函数
        
        Args:
            x: 输入特征元组 (esm_features, prot5_features)
                esm_features: 形状为(batch, esm_dim)
                prot5_features: 形状为(batch, prot5_dim)
            lengths: 在这个处理固定向量的模型中未使用
            
        Returns:
            logits张量，形状为(batch, num_classes)
        """
        # 解包输入特征
        esm_features, prot5_features = x
        
        # 特征投影
        esm_proj = self.esm_proj(esm_features)  # (batch, hidden_dim)
        prot5_proj = self.prot5_proj(prot5_features)  # (batch, hidden_dim)
        
        # 拼接特征用于生成查询向量
        concat_features = torch.cat([esm_proj, prot5_proj], dim=-1)  # (batch, hidden_dim*2)
        
        # 计算注意力权重
        query = self.query_proj(concat_features).unsqueeze(1)  # (batch, 1, hidden_dim)
        
        # 准备key和value矩阵
        keys = torch.stack([
            self.key_proj(esm_proj),
            self.key_proj(prot5_proj)
        ], dim=1)  # (batch, 2, hidden_dim)
        
        values = torch.stack([
            self.value_proj(esm_proj),
            self.value_proj(prot5_proj)
        ], dim=1)  # (batch, 2, hidden_dim)
        
        # 计算注意力分数
        attn_scores = torch.matmul(query, keys.transpose(-2, -1)) / math.sqrt(keys.size(-1))  # (batch, 1, 2)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch, 1, 2)
        
        # 加权融合特征
        fused_features = torch.matmul(attn_weights, values).squeeze(1)  # (batch, hidden_dim)
        
        # 通过Delta块处理融合特征
        x = fused_features
        for block in self.delta_blocks:
            x = block(x)
        
        # 分类
        logits = self.classifier(x)
        
        # 在训练和评估时保存注意力权重
        self.last_attention_weights = attn_weights.squeeze(1)
        
        return logits

class DualPathwayFusion(nn.Module):
    """
    为SN和SP创建两条独立的处理路径，然后在最后阶段融合
    """
    def __init__(
        self,
        esm_dim: int = 1280,
        prot5_dim: int = 1024,
        hidden_dim: int = 128,  # 每条路径的维度减半，总参数量保持不变
        num_layers: int = 4,    # 每条路径层数减半
        num_classes: int = 2,
        rank: int = 8,
        steps: int = 1,
        dropout: float = 0.1,
    ):
        super(DualPathwayFusion, self).__init__()
        
        # 特征投影层
        self.esm_proj = nn.Linear(esm_dim, hidden_dim * 2)
        self.prot5_proj = nn.Linear(prot5_dim, hidden_dim * 2)
        
        # SN优化路径（偏向ProtT5特征）
        self.sn_path = nn.ModuleList([
            IterLowRankBlock(hidden_dim, rank=rank, steps=steps, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # SP优化路径（偏向ESM特征）
        self.sp_path = nn.ModuleList([
            IterLowRankBlock(hidden_dim, rank=rank, steps=steps, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # SN特征选择门控
        self.sn_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # SP特征选择门控
        self.sp_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # 路径融合层
        self.fusion_layer = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.fusion_norm = nn.LayerNorm(hidden_dim * 2)
        
        # 最终分类层
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x, lengths=None):
        """前向传播函数"""
        # 解包输入特征
        esm_features, prot5_features = x
        
        # 特征投影
        esm_full = self.esm_proj(esm_features)  # (batch, hidden_dim*2)
        prot5_full = self.prot5_proj(prot5_features)  # (batch, hidden_dim*2)
        
        # 拆分特征用于不同路径
        esm_sn, esm_sp = torch.chunk(esm_full, 2, dim=-1)
        prot5_sn, prot5_sp = torch.chunk(prot5_full, 2, dim=-1)
        
        # SN路径（ProtT5在SN表现更好）
        sn_concat = torch.cat([esm_sn, prot5_sn], dim=-1)
        sn_gate_value = self.sn_gate(sn_concat)
        # 给ProtT5更高的初始权重
        sn_input = sn_gate_value * esm_sn + (1 - sn_gate_value) * prot5_sn
        
        # SP路径（ESM在SP表现更好）
        sp_concat = torch.cat([esm_sp, prot5_sp], dim=-1)
        sp_gate_value = self.sp_gate(sp_concat)
        # 给ESM更高的初始权重
        sp_input = (1 - sp_gate_value) * esm_sp + sp_gate_value * prot5_sp
        
        # 通过各自的路径处理
        sn_features = sn_input
        for block in self.sn_path:
            sn_features = block(sn_features)
        
        sp_features = sp_input
        for block in self.sp_path:
            sp_features = block(sp_features)
        
        # 融合两条路径的输出
        combined_features = torch.cat([sn_features, sp_features], dim=-1)
        fused_features = self.fusion_layer(combined_features)
        fused_features = self.fusion_norm(fused_features)
        
        # 分类
        logits = self.classifier(fused_features)
        
        # 保存门控值以便分析
        self.sn_gate_values = sn_gate_value
        self.sp_gate_values = sp_gate_value
        
        return logits
   
class FixedWeightFusion(nn.Module):
    """消融实验：使用固定权重而非门控机制融合特征"""
    def __init__(
        self,
        esm_dim=1280,
        prot5_dim=1024,
        hidden_dim=128,
        num_layers=1,
        num_classes=2,
        rank=4,
        steps=2,
        dropout=0.1,
        fixed_weight=0.5  # 固定权重参数
    ):
        super(FixedWeightFusion, self).__init__()
        
        # 沿用原模型的特征投影层
        self.esm_proj = nn.Linear(esm_dim, hidden_dim * 2)
        self.prot5_proj = nn.Linear(prot5_dim, hidden_dim * 2)
        
        # 沿用原模型的路径处理
        self.sn_path = nn.ModuleList([
            IterLowRankBlock(hidden_dim, rank=rank, steps=steps, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        self.sp_path = nn.ModuleList([
            IterLowRankBlock(hidden_dim, rank=rank, steps=steps, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        self.fixed_weight = fixed_weight  # 存储固定权重
        
        # 沿用原模型的融合层和分类器
        self.fusion_layer = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.fusion_norm = nn.LayerNorm(hidden_dim * 2)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x, lengths=None):
        esm_features, prot5_features = x
        
        # 特征投影
        esm_full = self.esm_proj(esm_features)
        prot5_full = self.prot5_proj(prot5_features)
        
        # 拆分特征用于不同路径
        esm_sn, esm_sp = torch.chunk(esm_full, 2, dim=-1)
        prot5_sn, prot5_sp = torch.chunk(prot5_full, 2, dim=-1)
        
        # 使用固定权重而非门控值融合特征
        sn_input = self.fixed_weight * esm_sn + (1 - self.fixed_weight) * prot5_sn
        sp_input = (1 - self.fixed_weight) * esm_sp + self.fixed_weight * prot5_sp
        
        # 通过各自的路径处理
        sn_features = sn_input
        for block in self.sn_path:
            sn_features = block(sn_features)
        
        sp_features = sp_input
        for block in self.sp_path:
            sp_features = block(sp_features)
        
        # 融合两条路径的输出
        combined_features = torch.cat([sn_features, sp_features], dim=-1)
        fused_features = self.fusion_layer(combined_features)
        fused_features = self.fusion_norm(fused_features)
        
        # 分类
        logits = self.classifier(fused_features)
        
        return logits
    
    
class SharedGateFusion(nn.Module):
    """消融实验：两条路径使用相同的门控网络"""
    def __init__(
        self,
        esm_dim=1280,
        prot5_dim=1024,
        hidden_dim=128,
        num_layers=1,
        num_classes=2,
        rank=4,
        steps=2,
        dropout=0.1,
    ):
        super(SharedGateFusion, self).__init__()
        
        self.esm_proj = nn.Linear(esm_dim, hidden_dim * 2)
        self.prot5_proj = nn.Linear(prot5_dim, hidden_dim * 2)
        
        self.sn_path = nn.ModuleList([
            IterLowRankBlock(hidden_dim, rank=rank, steps=steps, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        self.sp_path = nn.ModuleList([
            IterLowRankBlock(hidden_dim, rank=rank, steps=steps, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # 共享的门控网络
        self.shared_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        self.fusion_layer = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.fusion_norm = nn.LayerNorm(hidden_dim * 2)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x, lengths=None):
        esm_features, prot5_features = x
        
        # 特征投影
        esm_full = self.esm_proj(esm_features)
        prot5_full = self.prot5_proj(prot5_features)
        
        # 拆分特征用于不同路径
        esm_sn, esm_sp = torch.chunk(esm_full, 2, dim=-1)
        prot5_sn, prot5_sp = torch.chunk(prot5_full, 2, dim=-1)
        
        # 共享门控计算
        sn_concat = torch.cat([esm_sn, prot5_sn], dim=-1)
        gate_value = self.shared_gate(sn_concat)
        
        # 使用相同的门控值，但维持原始偏好
        sn_input = gate_value * esm_sn + (1 - gate_value) * prot5_sn
        sp_input = (1 - gate_value) * esm_sp + gate_value * prot5_sp
        
        # 其余处理保持不变
        sn_features = sn_input
        for block in self.sn_path:
            sn_features = block(sn_features)
        
        sp_features = sp_input
        for block in self.sp_path:
            sp_features = block(sp_features)
        
        combined_features = torch.cat([sn_features, sp_features], dim=-1)
        fused_features = self.fusion_layer(combined_features)
        fused_features = self.fusion_norm(fused_features)
        
        logits = self.classifier(fused_features)
        
        # 保存门控值以便分析
        self.gate_values = gate_value
        
        return logits
    
class SinglePathFusion(nn.Module):
    """消融实验：将双路径合并为单一路径"""
    def __init__(
        self,
        esm_dim=1280,
        prot5_dim=1024,
        hidden_dim=256,  # 增大hidden_dim以保持参数量相当
        num_layers=1,
        num_classes=2,
        rank=4,
        steps=2,
        dropout=0.1,
    ):
        super(SinglePathFusion, self).__init__()
        
        # 特征投影层
        self.esm_proj = nn.Linear(esm_dim, hidden_dim)
        self.prot5_proj = nn.Linear(prot5_dim, hidden_dim)
        
        # 特征融合门控
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # 单一处理路径
        self.path = nn.ModuleList([
            IterLowRankBlock(hidden_dim, rank=rank, steps=steps, dropout=dropout)
            for _ in range(num_layers * 2)  # 加倍层数以保持复杂度
        ])
        
        # 最终分类层
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x, lengths=None):
        esm_features, prot5_features = x
        
        # 特征投影
        esm_proj = self.esm_proj(esm_features)  # (batch, hidden_dim)
        prot5_proj = self.prot5_proj(prot5_features)  # (batch, hidden_dim)
        
        # 计算融合门控值
        concat_features = torch.cat([esm_proj, prot5_proj], dim=-1)
        gate_value = self.fusion_gate(concat_features)
        
        # 融合特征
        fused_features = gate_value * esm_proj + (1 - gate_value) * prot5_proj
        
        # 通过单一路径处理
        x = fused_features
        for block in self.path:
            x = block(x)
        
        # 分类
        logits = self.classifier(x)
        
        return logits
    

class NoFinalFusionLayer(nn.Module):
    """消融实验：移除最终融合层，直接拼接两条路径的输出"""
    def __init__(
        self,
        esm_dim=1280,
        prot5_dim=1024,
        hidden_dim=128,
        num_layers=1,
        num_classes=2,
        rank=4,
        steps=2,
        dropout=0.1,
    ):
        super(NoFinalFusionLayer, self).__init__()
        
        # 特征投影层
        self.esm_proj = nn.Linear(esm_dim, hidden_dim * 2)
        self.prot5_proj = nn.Linear(prot5_dim, hidden_dim * 2)
        
        # SN和SP路径的门控
        self.sn_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        self.sp_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # 路径处理层
        self.sn_path = nn.ModuleList([
            IterLowRankBlock(hidden_dim, rank=rank, steps=steps, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        self.sp_path = nn.ModuleList([
            IterLowRankBlock(hidden_dim, rank=rank, steps=steps, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # 直接使用归一化层但没有融合变换
        self.fusion_norm = nn.LayerNorm(hidden_dim * 2)
        
        # 最终分类层
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x, lengths=None):
        esm_features, prot5_features = x
        
        # 特征投影
        esm_full = self.esm_proj(esm_features)
        prot5_full = self.prot5_proj(prot5_features)
        
        # 拆分特征用于不同路径
        esm_sn, esm_sp = torch.chunk(esm_full, 2, dim=-1)
        prot5_sn, prot5_sp = torch.chunk(prot5_full, 2, dim=-1)
        
        # SN路径（ProtT5在SN表现更好）
        sn_concat = torch.cat([esm_sn, prot5_sn], dim=-1)
        sn_gate_value = self.sn_gate(sn_concat)
        sn_input = sn_gate_value * esm_sn + (1 - sn_gate_value) * prot5_sn
        
        # SP路径（ESM在SP表现更好）
        sp_concat = torch.cat([esm_sp, prot5_sp], dim=-1)
        sp_gate_value = self.sp_gate(sp_concat)
        sp_input = (1 - sp_gate_value) * esm_sp + sp_gate_value * prot5_sp
        
        # 通过各自的路径处理
        sn_features = sn_input
        for block in self.sn_path:
            sn_features = block(sn_features)
        
        sp_features = sp_input
        for block in self.sp_path:
            sp_features = block(sp_features)
        
        # 简单拼接两条路径的输出，不做融合变换
        combined_features = torch.cat([sn_features, sp_features], dim=-1)
        normalized_features = self.fusion_norm(combined_features)
        
        # 分类
        logits = self.classifier(normalized_features)
        
        return logits

class PathInteractionFusion(nn.Module):
    """消融实验：添加路径之间的信息交流"""
    def __init__(
        self,
        esm_dim=1280,
        prot5_dim=1024,
        hidden_dim=128,
        num_layers=1,
        num_classes=2,
        rank=4,
        steps=2,
        dropout=0.1,
    ):
        super(PathInteractionFusion, self).__init__()
        
        # 特征投影层
        self.esm_proj = nn.Linear(esm_dim, hidden_dim * 2)
        self.prot5_proj = nn.Linear(prot5_dim, hidden_dim * 2)
        
        # SN和SP路径的门控
        self.sn_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        self.sp_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # 路径处理层
        self.sn_path = nn.ModuleList([
            IterLowRankBlock(hidden_dim, rank=rank, steps=steps, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        self.sp_path = nn.ModuleList([
            IterLowRankBlock(hidden_dim, rank=rank, steps=steps, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # 路径交互层 - 新增
        self.path_interaction = nn.ModuleList([
            nn.Linear(hidden_dim * 2, hidden_dim * 2)
            for _ in range(num_layers)
        ])
        
        # 路径融合层
        self.fusion_layer = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.fusion_norm = nn.LayerNorm(hidden_dim * 2)
        
        # 最终分类层
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x, lengths=None):
        esm_features, prot5_features = x
        
        # 特征投影
        esm_full = self.esm_proj(esm_features)
        prot5_full = self.prot5_proj(prot5_features)
        
        # 拆分特征用于不同路径
        esm_sn, esm_sp = torch.chunk(esm_full, 2, dim=-1)
        prot5_sn, prot5_sp = torch.chunk(prot5_full, 2, dim=-1)
        
        # SN路径（ProtT5在SN表现更好）
        sn_concat = torch.cat([esm_sn, prot5_sn], dim=-1)
        sn_gate_value = self.sn_gate(sn_concat)
        sn_input = sn_gate_value * esm_sn + (1 - sn_gate_value) * prot5_sn
        
        # SP路径（ESM在SP表现更好）
        sp_concat = torch.cat([esm_sp, prot5_sp], dim=-1)
        sp_gate_value = self.sp_gate(sp_concat)
        sp_input = (1 - sp_gate_value) * esm_sp + sp_gate_value * prot5_sp
        
        # 通过各自的路径处理，并在每一层后添加路径交互
        sn_features = sn_input
        sp_features = sp_input
        
        for i in range(len(self.sn_path)):
            # 处理每条路径
            sn_features_new = self.sn_path[i](sn_features)
            sp_features_new = self.sp_path[i](sp_features)
            
            # 路径交互 - 两条路径的特征互相影响
            combined = torch.cat([sn_features_new, sp_features_new], dim=-1)
            interaction = self.path_interaction[i](combined)
            sn_inter, sp_inter = torch.chunk(interaction, 2, dim=-1)
            
            # 加入交互信息
            sn_features = sn_features_new + sn_inter * 0.1  # 控制交互强度
            sp_features = sp_features_new + sp_inter * 0.1
        
        # 融合两条路径的输出
        combined_features = torch.cat([sn_features, sp_features], dim=-1)
        fused_features = self.fusion_layer(combined_features)
        fused_features = self.fusion_norm(fused_features)
        
        # 分类
        logits = self.classifier(fused_features)
        
        return logits    
# ------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------
# 模型1: 平衡版CNN (~300K参数)
# ------------------------------------------------------------------------------
class BalancedCNN(nn.Module):
    def __init__(self, input_dim=1280, num_classes=2):
        super(BalancedCNN, self).__init__()
        # 使用投影层减少输入维度，这是参数量最大的来源
        self.input_proj = nn.Linear(input_dim, 128)
        
        # 使用较小的卷积核和通道数
        self.conv1 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 96, kernel_size=3, padding=1)
        
        self.dropout = nn.Dropout(0.3)
        self.pool = nn.AdaptiveMaxPool1d(1)
        
        # 使用简单的分类器
        self.classifier = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, lengths=None):
        # 首先通过投影层减少维度
        if x.dim() == 3:
            # 处理序列输入 [B, L, D]
            batch_size, seq_len, _ = x.shape
            x = x.reshape(-1, x.size(-1))  # [B*L, D]
            x = self.input_proj(x)  # [B*L, 128]
            x = x.view(batch_size, seq_len, -1)  # [B, L, 128]
        else:
            # 处理单向量输入 [B, D]
            x = self.input_proj(x)  # [B, 128]
            x = x.unsqueeze(1)  # [B, 1, 128]
        
        # 将输入调整为CNN期望的格式 [B, C, L]
        x = x.transpose(1, 2)
            
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.dropout(x)
        
        x = self.pool(x).squeeze(-1)  # [B, 96]
        out = self.classifier(x)
        return out

# ------------------------------------------------------------------------------
# 模型2: 平衡版MLP (~300K参数)
# ------------------------------------------------------------------------------
class BalancedMLP(nn.Module):
    def __init__(self, input_dim=1280, num_classes=2, dropout=0.3):
        super(BalancedMLP, self).__init__()
        # 使用投影层减少输入维度
        self.input_proj = nn.Linear(input_dim, 256)
        
        # MLP结构，减少层数和宽度
        self.mlp = nn.Sequential(
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(256, 192),
            nn.LayerNorm(192),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(192, 96),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(96, num_classes)
        )

    def forward(self, x, lengths=None):
        if x.dim() == 3:
            x = x.mean(dim=1)  # 如果是序列，对序列取平均
        
        x = self.input_proj(x)
        return self.mlp(x)

# ------------------------------------------------------------------------------
# 模型3: 平衡版GRU (~300K参数)
# ------------------------------------------------------------------------------
class BalancedGRU(nn.Module):
    def __init__(self, input_dim=1280, hidden_dim=64, num_classes=2):
        super(BalancedGRU, self).__init__()
        # 使用较小的投影维度
        self.input_proj = nn.Linear(input_dim, 96) 
        
        # 减少GRU的隐藏维度和层数
        self.gru = nn.GRU(96, hidden_dim, batch_first=True, 
                          bidirectional=True, num_layers=2)
        
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, lengths=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, D] -> [B, 1, D]
            
        # 通过投影层
        batch_size, seq_len, _ = x.shape
        x = x.reshape(-1, x.size(-1))  # [B*L, D]
        x = self.input_proj(x)  # [B*L, 96]
        x = x.view(batch_size, seq_len, -1)  # [B, L, 96]
        
        if lengths is not None:
            x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            
        output, _ = self.gru(x)
        
        if lengths is not None:
            output, _ = pad_packed_sequence(output, batch_first=True)
            
        # 获取最后一个时间步的隐藏状态
        batch_size = output.size(0)
        
        if lengths is not None:
            # 使用真实长度获取每个序列的最后一个输出
            indices = (lengths - 1).view(-1, 1).expand(batch_size, output.size(2))
            indices = indices.unsqueeze(1).to(output.device)
            last_output = output.gather(1, indices).squeeze(1)
        else:
            # 如果没有长度信息，就使用最后一个时间步
            last_output = output[:, -1, :]
            
        last_output = self.dropout(last_output)
        out = self.fc(last_output)
        return out

# ------------------------------------------------------------------------------
# 模型4: 平衡版LSTM (~300K参数)
# ------------------------------------------------------------------------------
class BalancedLSTM(nn.Module):
    def __init__(self, input_dim=1280, hidden_dim=64, num_classes=2, dropout=0.3):
        super(BalancedLSTM, self).__init__()
        # 使用较小的投影维度
        self.input_proj = nn.Linear(input_dim, 96)
        
        # 减小LSTM的隐藏维度和层数
        self.lstm = nn.LSTM(96, hidden_dim, batch_first=True, 
                           bidirectional=True, num_layers=1)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, lengths=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, D] -> [B, 1, D]
            
        # 通过投影层
        batch_size, seq_len, _ = x.shape
        x = x.reshape(-1, x.size(-1))  # [B*L, D]
        x = self.input_proj(x)  # [B*L, 96]
        x = x.view(batch_size, seq_len, -1)  # [B, L, 96]
        
        if lengths is not None:
            x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            
        output, _ = self.lstm(x)
        
        if lengths is not None:
            output, _ = pad_packed_sequence(output, batch_first=True)
            
        # 使用最后一个时间步
        last_output = output[:, -1, :]
        last_output = self.norm(last_output)
        last_output = self.dropout(last_output)
        
        out = self.fc(last_output)
        return out

# ------------------------------------------------------------------------------
# 模型5: 平衡版Transformer (~300K参数)
# ------------------------------------------------------------------------------
class BalancedTransformer(nn.Module):
    def __init__(self, input_dim=1280, d_model=128, nhead=4, num_layers=1, num_classes=2, dropout=0.1):
        super(BalancedTransformer, self).__init__()
        # 使用投影层减少输入维度
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 减小Transformer的层数
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,  # 减小前馈网络大小
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # 分类层
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, lengths=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, D] -> [B, 1, D]
            
        batch_size, seq_len, _ = x.shape
        x = x.reshape(-1, x.size(-1))  # [B*L, D]
        x = self.input_proj(x)  # [B*L, 128]
        x = x.view(batch_size, seq_len, -1)  # [B, L, 128]
        
        # 创建padding mask
        if lengths is not None:
            mask = torch.zeros(x.shape[0], x.shape[1], dtype=torch.bool, device=x.device)
            for i, length in enumerate(lengths):
                mask[i, length:] = True
        else:
            mask = None
            
        x = self.transformer(x, src_key_padding_mask=mask)
        
        # 使用序列的平均值进行分类
        if mask is not None:
            # 创建有效元素的掩码
            valid_tokens = (~mask).float().unsqueeze(-1)
            # 计算序列的有效平均值
            x = (x * valid_tokens).sum(dim=1) / valid_tokens.sum(dim=1)
        else:
            # 没有掩码时取平均值
            x = x.mean(dim=1)
            
        out = self.classifier(x)
        return out

# ------------------------------------------------------------------------------
# 模型6: 平衡版Mamba (~300K参数) - 参数量已经接近300K，基本不需调整
# ------------------------------------------------------------------------------
class BalancedMamba(nn.Module):
    def __init__(self, input_dim=1280, d_model=128, d_state=16, num_layers=2, num_classes=2, dropout=0.1):
        super(BalancedMamba, self).__init__()
        # 使用投影层减少输入维度
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 设置Mamba层
        self.mamba_layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=2, expand=1)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
        # 分类层
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, lengths=None):
        # 处理单个向量的情况
        if x.dim() == 2:
            x = self.input_proj(x)
            x = x.unsqueeze(1)  # (batch, d_model) -> (batch, 1, d_model)
        else:
            # 处理序列的情况
            batch_size, seq_len, _ = x.shape
            x = x.reshape(-1, x.size(-1))  # [B*L, D]
            x = self.input_proj(x)  # [B*L, 128]
            x = x.view(batch_size, seq_len, -1)  # [B, L, 128]
            
        for mamba in self.mamba_layers:
            x = x + mamba(x)  # 添加残差连接
        
        x = self.norm(x)
        
        # 如果是单个向量，直接取特征
        if x.size(1) == 1:
            x = x.squeeze(1)
        else:
            # 否则取平均
            x = x.mean(dim=1)
        
        logits = self.classifier(x)
        return logits


# ------------------------------------------------------------------------------
# 模型8: 平衡版BiLSTM (~300K参数)
# ------------------------------------------------------------------------------
class BalancedBiLSTM(nn.Module):
    def __init__(self, input_dim=1280, hidden_dim=64, num_classes=2, dropout=0.3):
        super(BalancedBiLSTM, self).__init__()
        # 使用较小的投影维度
        self.input_proj = nn.Linear(input_dim, 96)
        
        # 双向LSTM，减少隐藏维度和层数
        self.bilstm = nn.LSTM(96, hidden_dim, batch_first=True, 
                             bidirectional=True, num_layers=1)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, lengths=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, D] -> [B, 1, D]
            
        # 通过投影层
        batch_size, seq_len, _ = x.shape
        x = x.reshape(-1, x.size(-1))  # [B*L, D]
        x = self.input_proj(x)  # [B*L, 96]
        x = x.view(batch_size, seq_len, -1)  # [B, L, 96]
        
        if lengths is not None:
            x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            
        output, _ = self.bilstm(x)
        
        if lengths is not None:
            output, _ = pad_packed_sequence(output, batch_first=True)
            
        # 使用最后一个时间步
        last_output = output[:, -1, :]
        last_output = self.norm(last_output)
        last_output = self.dropout(last_output)
        
        out = self.fc(last_output)
        return out
# ------------------------------------------------------------------------------
# dtvf
# ------------------------------------------------------------------------------



class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, lstm_out):
        out = self.linear(lstm_out)
        
        score = torch.bmm(out, out.transpose(1, 2))
        attn = self.softmax(score)
        context = torch.bmm(attn, lstm_out)
        return context

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, drop):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(drop)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = Attention(hidden_size)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.attention(out)
        out = out.permute(0, 2, 1)
        out = self.batch_norm(out)
        out = out.permute(0, 2, 1)
        out = self.fc(out[:, -1, :])
        return out

class CNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, drop):
        super(CNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(drop)
        self.conv = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1)
        self.attention = Attention(hidden_size)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        x=convert_x_shape(x)
        out = x.permute(0, 2, 1)
        out = self.conv(out)
        out = out.permute(0, 2, 1)
        out = self.attention(out)
        out = out.permute(0, 2, 1)
        out = self.batch_norm(out)
        out = out.permute(0, 2, 1)
        out = self.fc(out[:, -1, :])
        return out

class DTVFModel(nn.Module):
    def __init__(self, input_size, hidden_size_cnn, hidden_size_lstm, num_layers_cnn, num_layers_lstm, num_classes, drop_cnn, drop_lstm):
        super(DTVFModel, self).__init__()
        self.cnn = CNNModel(input_size, hidden_size_cnn, num_layers_cnn, num_classes, drop_cnn)
        self.lstm = LSTMModel(input_size, hidden_size_lstm, num_layers_lstm, num_classes, drop_lstm)
        self.weight = nn.Parameter(torch.tensor(0.8))

    def forward(self, x,lengths=None):
        out_cnn = self.cnn(x)
        x=convert_x_shape(x)
        out_lstm = self.lstm(x)
        out = self.weight * out_cnn + (1 - self.weight) * out_lstm
        return out

# ------------------------------------------------------------------------------
# 参数量信息函数
# ------------------------------------------------------------------------------
def print_model_params():
    input_dim = 1280
    models = {
        "BalancedCNN": BalancedCNN(input_dim=input_dim),
        "BalancedMLP": BalancedMLP(input_dim=input_dim),
        "BalancedGRU": BalancedGRU(input_dim=input_dim),
        "BalancedLSTM": BalancedLSTM(input_dim=input_dim),
        "BalancedTransformer": BalancedTransformer(input_dim=input_dim),
        "BalancedMamba": BalancedMamba(input_dim=input_dim),
        "BalancedDelta": VFIter(input_dim=input_dim),
        "BalancedBiLSTM": BalancedBiLSTM(input_dim=input_dim),
    }
    
    print("Model parameter counts:")
    for name, model in models.items():
        print(f"{name}: {count_parameters(model):,} parameters")

if __name__ == "__main__":
    import torch
    print_model_params()
    # import io
    # import netron
    # import os
    
    # # 实例化模型
    # model = Delta(input_dim=1280)
    
    # # 创建示例输入
    # dummy_input = torch.randn(1, 1280)
    
    # # 导出为ONNX格式
    # onnx_path = "delta_model.onnx"
    # torch.onnx.export(model, dummy_input, onnx_path, verbose=True)
    
    # # 使用netron可视化
    # # 这会在浏览器中打开可视化界面
    # netron.start(onnx_path)
    
    # # 打印所有模型的参数量
    # print_model_params()
    
    # # 等待用户输入以保持可视化界面开启
    # print("Visualization is open in browser. Press Enter to close...")
    # input()
    
    # # 清理ONNX文件
    # if os.path.exists(onnx_path):
    #     os.remove(onnx_path)