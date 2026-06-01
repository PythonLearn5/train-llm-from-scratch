import torch
import torch.nn as nn

from src.models.attention import MultiHeadAttention
from src.models.mlp import MLP


class Block(nn.Module):
    """
    Transformer Block（Transformer基础模块）。

    一个标准Transformer Block由以下部分组成：

        输入
          │
          ▼
      LayerNorm
          │
          ▼
    Multi-Head Attention
          │
          ▼
      Residual Add
          │
          ▼
      LayerNorm
          │
          ▼
          MLP
          │
          ▼
      Residual Add
          │
          ▼
         输出

    作用：

        Attention：
            负责不同Token之间的信息交互。

        MLP：
            负责对每个Token进行非线性特征提取。

        LayerNorm：
            保持训练稳定。

        Residual：
            缓解深层网络梯度消失问题。

    当前实现属于：

        Pre-LN Transformer

    即：

        LayerNorm
            ↓
        子模块

    而不是：

        子模块
            ↓
        LayerNorm

    GPT-2、GPT-3、LLaMA等现代模型均采用Pre-LN结构。
    """

    def __init__(
        self,
        n_head: int,
        n_embed: int,
        context_length: int
    ) -> None:
        """
        初始化Transformer Block。

        参数：

            n_head:
                注意力头数量

            n_embed:
                Embedding维度

            context_length:
                最大上下文长度
        """
        super().__init__()

        # ==========================
        # Attention前LayerNorm
        # ==========================
        self.ln1 = nn.LayerNorm(n_embed)

        # ==========================
        # 多头注意力模块
        # ==========================
        self.attn = MultiHeadAttention(
            n_head,
            n_embed,
            context_length
        )

        # ==========================
        # MLP前LayerNorm
        # ==========================
        self.ln2 = nn.LayerNorm(n_embed)

        # ==========================
        # 前馈神经网络
        # ==========================
        self.mlp = MLP(n_embed)

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Transformer Block标准前向传播。

        数据流：

            x
            │
            ▼
        LayerNorm
            │
            ▼
        Attention
            │
            ▼
        Residual Add
            │
            ▼
        LayerNorm
            │
            ▼
            MLP
            │
            ▼
        Residual Add
            │
            ▼
           输出

        参数：

            x:
                输入特征

                (B,T,C)

        返回：

            输出特征

                (B,T,C)
        """

        # ====================================
        # Attention子层
        # ====================================
        #
        # Pre-LN结构：
        #
        # x
        #  ↓
        # LN
        #  ↓
        # Attention
        #  ↓
        # Add
        #
        # 公式：
        #
        # x = x + Attention(LN(x))
        #
        x = x + self.attn(
            self.ln1(x)
        )

        # ====================================
        # MLP子层
        # ====================================
        #
        # x
        #  ↓
        # LN
        #  ↓
        # MLP
        #  ↓
        # Add
        #
        # 公式：
        #
        # x = x + MLP(LN(x))
        #
        x = x + self.mlp(
            self.ln2(x)
        )

        return x

    def forward_embedding(
        self,
        x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        用于提取Transformer中间层Embedding。

        常用于：

            1. 特征可视化
            2. Attention分析
            3. Embedding导出
            4. 模型调试

        参数：

            x:
                输入特征

                (B,T,C)

        返回：

            x:
                MLP隐藏层输出

            res:
                Attention残差输出
        """

        # ====================================
        # Attention阶段
        # ====================================

        res = x + self.attn(
            self.ln1(x)
        )

        # ====================================
        # MLP隐藏层特征
        # ====================================
        #
        # 注意：
        # 此处没有执行proj层
        #
        # 即：
        #
        # Linear(C→4C)
        #      ↓
        #    ReLU
        #
        # 返回的是高维隐藏特征
        #
        x = self.mlp.forward_embedding(
            self.ln2(res)
        )

        return x, res


if __name__ == '__main__':

    # ====================================
    # 测试代码
    # ====================================

    batch_size = 2

    sequence_length = 5

    embedding_dim = 32

    num_heads = 4

    context_len = 5

    # 构造随机输入
    #
    # (B,T,C)
    #
    input_tensor = torch.randn(
        batch_size,
        sequence_length,
        embedding_dim
    )

    # 创建Transformer Block
    transformer_block = Block(
        n_head=num_heads,
        n_embed=embedding_dim,
        context_length=context_len
    )

    # 前向传播
    output_tensor = transformer_block(
        input_tensor
    )

    print("输入形状:", input_tensor.shape)
    print("输出形状:", output_tensor.shape)

    # 预期：
    #
    # 输入：
    # torch.Size([2, 5, 32])
    #
    # 输出：
    # torch.Size([2, 5, 32])