import torch
import torch.nn as nn
from torch import Tensor


class MLP(nn.Module):
    """
    Transformer中的前馈神经网络（Feed Forward Network，FFN）。

    作用：
        在自注意力（Self-Attention）完成信息聚合后，
        对每个Token独立进行非线性特征变换。

    结构：
        Linear(C -> 4C)
            ↓
        ReLU
            ↓
        Linear(4C -> C)

    说明：
        Transformer中的Attention负责：
            “不同Token之间的信息交互”

        MLP负责：
            “对每个Token自身进行深度特征提取”

    输入：
        (B, T, C)

    输出：
        (B, T, C)

    其中：
        B = Batch Size（批次大小）
        T = Sequence Length（序列长度）
        C = Embedding Dimension（嵌入维度）
    """

    def __init__(self, n_embed: int) -> None:
        """
        初始化MLP模块。

        参数：
            n_embed:
                输入Embedding维度。
        """
        super().__init__()

        # 第一层全连接
        #
        # 将Embedding维度扩大4倍
        #
        # 例如：
        # 768 -> 3072
        #
        # GPT系列模型通常采用4倍扩展
        self.hidden = nn.Linear(
            n_embed,
            4 * n_embed
        )

        # 非线性激活函数
        #
        # Transformer原论文使用ReLU
        #
        # 后续模型：
        # GPT2 -> GELU
        # GPT3 -> GELU
        # LLaMA -> SwiGLU
        #
        self.relu = nn.ReLU()

        # 投影层
        #
        # 将维度压缩回原始Embedding大小
        #
        # 3072 -> 768
        #
        self.proj = nn.Linear(
            4 * n_embed,
            n_embed
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播。

        数据流：

            输入
              ↓
        hidden层扩展
              ↓
            ReLU
              ↓
        proj层压缩
              ↓
            输出

        参数：
            x:
                输入张量
                形状：(B, T, C)

        返回：
            输出张量
            形状：(B, T, C)
        """

        # 特征扩展
        x = self.forward_embedding(x)

        # 投影回原始维度
        x = self.project_embedding(x)

        return x

    def forward_embedding(self, x: Tensor) -> Tensor:
        """
        特征扩展阶段。

        执行过程：

            Linear(C → 4C)
                    ↓
                 ReLU

        例如：

            输入：
                (2, 10, 768)

            hidden层后：
                (2, 10, 3072)

        参数：
            x:
                输入张量

        返回：
            激活后的高维特征
        """

        x = self.hidden(x)

        # 引入非线性能力
        x = self.relu(x)

        return x

    def project_embedding(self, x: Tensor) -> Tensor:
        """
        特征压缩阶段。

        执行过程：

            Linear(4C → C)

        例如：

            输入：
                (2, 10, 3072)

            输出：
                (2, 10, 768)

        参数：
            x:
                输入张量

        返回：
            投影后的输出张量
        """

        x = self.proj(x)

        return x


if __name__ == '__main__':

    # ====================================
    # 测试代码
    # ====================================

    batch_size = 2
    sequence_length = 3
    embedding_dim = 16

    # 构造随机输入
    #
    # B=2
    # T=3
    # C=16
    #
    input_tensor = torch.randn(
        batch_size,
        sequence_length,
        embedding_dim
    )

    # 创建MLP模块
    mlp_module = MLP(
        n_embed=embedding_dim
    )

    # 前向计算
    output_tensor = mlp_module(input_tensor)

    print("输入形状:", input_tensor.shape)
    print("输出形状:", output_tensor.shape)

    # 预期输出：
    #
    # 输入：
    # torch.Size([2, 3, 16])
    #
    # 输出：
    # torch.Size([2, 3, 16])