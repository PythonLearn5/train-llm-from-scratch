import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Head(nn.Module):
    """
    单个自注意力头（Self-Attention Head）。

    作用：
        1. 将输入分别映射为 Query（查询）、Key（键）、Value（值）。
        2. 计算 Query 与 Key 的相似度，得到注意力权重。
        3. 使用注意力权重对 Value 进行加权求和。
        4. 使用因果掩码（Causal Mask）防止当前位置看到未来信息。

    输入：
        x -> (B, T, C)
            B：Batch Size（批次大小）
            T：Token 数量（序列长度）
            C：Embedding 维度

    输出：
        (B, T, head_size)
    """

    def __init__(self, head_size: int, n_embed: int, context_length: int) -> None:
        """
        初始化单个注意力头。

        参数：
            head_size:
                当前注意力头的输出维度。

            n_embed:
                输入Embedding维度。

            context_length:
                最大上下文长度，用于生成因果掩码矩阵。
        """
        super().__init__()

        # Key投影层
        self.key = nn.Linear(n_embed, head_size, bias=False)

        # Query投影层
        self.query = nn.Linear(n_embed, head_size, bias=False)

        # Value投影层
        self.value = nn.Linear(n_embed, head_size, bias=False)

        # 构造下三角矩阵
        #
        # 例如 context_length=5:
        #
        # 1 0 0 0 0
        # 1 1 0 0 0
        # 1 1 1 0 0
        # 1 1 1 1 0
        # 1 1 1 1 1
        #
        # 用于实现 GPT 的因果注意力（Causal Attention）
        self.register_buffer(
            'tril',
            torch.tril(torch.ones(context_length, context_length))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        参数：
            x:
                输入张量，形状为 (B, T, C)

        返回：
            注意力输出：
                (B, T, head_size)
        """
        B, T, C = x.shape

        # 当前注意力头维度
        head_size = self.key.out_features

        # 生成 Key
        # (B,T,C) -> (B,T,head_size)
        k = self.key(x)

        # 生成 Query
        # (B,T,C) -> (B,T,head_size)
        q = self.query(x)

        # 缩放因子
        #
        # Transformer论文中的：
        # Attention(Q,K,V) = softmax(QK^T / sqrt(dk))V
        #
        # 防止维度过大导致Softmax梯度过小
        scale_factor = 1 / math.sqrt(head_size)

        # 计算注意力分数（Attention Score）
        #
        # q : (B,T,H)
        # k : (B,T,H)
        #
        # k.transpose(-2,-1)
        #      (B,H,T)
        #
        # 结果：
        #      (B,T,T)
        #
        # 表示每个Token与其它Token之间的相关性
        attn_weights = q @ k.transpose(-2, -1) * scale_factor

        # 应用因果掩码
        #
        # 将未来位置填充为 -inf
        #
        # 例如：
        #
        # 原始：
        # [1.2 0.5 0.8 2.0]
        #
        # Mask后：
        # [1.2 0.5 -inf -inf]
        #
        # Softmax后未来位置概率变为0
        attn_weights = attn_weights.masked_fill(
            self.tril[:T, :T] == 0,
            float('-inf')
        )

        # 转换为概率分布
        #
        # (B,T,T)
        attn_weights = F.softmax(attn_weights, dim=-1)

        # 生成 Value
        #
        # (B,T,C)
        # -> (B,T,head_size)
        v = self.value(x)

        # 根据注意力权重对 Value 加权求和
        #
        # (B,T,T)
        # @
        # (B,T,H)
        #
        # =>
        # (B,T,H)
        out = attn_weights @ v

        return out


class MultiHeadAttention(nn.Module):
    """
    多头注意力（Multi-Head Attention）。

    核心思想：
        将Embedding拆分到多个Attention Head中并行计算。

    优点：
        不同Head可以学习不同类型的信息：

        Head1 -> 语法关系
        Head2 -> 位置关系
        Head3 -> 长距离依赖
        Head4 -> 语义关系

    最终将所有Head输出拼接（Concat）起来。
    """

    def __init__(
        self,
        n_head: int,
        n_embed: int,
        context_length: int
    ) -> None:
        """
        初始化多头注意力。

        参数：
            n_head:
                注意力头数量

            n_embed:
                Embedding维度

            context_length:
                最大上下文长度
        """
        super().__init__()

        # 创建多个Attention Head
        #
        # 假设：
        # n_embed = 32
        # n_head = 4
        #
        # 则每个Head:
        # head_size = 32 / 4 = 8
        #
        self.heads = nn.ModuleList([
            Head(
                n_embed // n_head,
                n_embed,
                context_length
            )
            for _ in range(n_head)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        参数：
            x:
                输入张量
                (B,T,C)

        返回：
            多头拼接结果
                (B,T,C)
        """

        # 并行执行所有Attention Head
        #
        # 假设：
        # Head输出：
        # (B,T,8)
        #
        # 共4个Head：
        # [(B,T,8),
        #  (B,T,8),
        #  (B,T,8),
        #  (B,T,8)]
        #
        # 拼接后：
        # (B,T,32)
        x = torch.cat(
            [head(x) for head in self.heads],
            dim=-1
        )

        return x


if __name__ == '__main__':

    # ==========================
    # 测试代码
    # ==========================

    batch_size = 2
    sequence_length = 5
    embedding_dim = 32
    num_heads = 4
    context_len = 5

    # 模拟输入：
    #
    # B=2
    # T=5
    # C=32
    #
    input_tensor = torch.randn(
        batch_size,
        sequence_length,
        embedding_dim
    )

    # 创建多头注意力模块
    multihead_attn = MultiHeadAttention(
        n_head=num_heads,
        n_embed=embedding_dim,
        context_length=context_len
    )

    # 前向计算
    output_tensor = multihead_attn(input_tensor)

    print("输入形状:", input_tensor.shape)
    print("输出形状:", output_tensor.shape)

    # 预期结果：
    #
    # 输入：
    # torch.Size([2, 5, 32])
    #
    # 输出：
    # torch.Size([2, 5, 32])