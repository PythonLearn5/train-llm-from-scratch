import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.transformer_block import Block


class Transformer(nn.Module):
    """
    Transformer语言模型。

    模型结构：

        输入Token
             ↓
        Token Embedding
             +
        Position Embedding
             ↓
      Transformer Block × N
             ↓
          LayerNorm
             ↓
           LM Head
             ↓
           Logits
             ↓
          Softmax
             ↓
         下一个Token

    作用：
        1. 将Token转换为向量表示
        2. 加入位置信息
        3. 通过多层Transformer提取上下文特征
        4. 预测下一个最可能出现的Token

    输入：
        idx:
            (B,T)

    输出：
        logits:
            (B,T,vocab_size)

    其中：

        B = Batch Size（批次大小）
        T = Sequence Length（序列长度）
    """

    def __init__(
        self,
        n_head: int,
        n_embed: int,
        context_length: int,
        vocab_size: int,
        N_BLOCKS: int
    ) -> None:
        """
        初始化Transformer模型。

        参数：

            n_head:
                每个Transformer Block中的注意力头数量

            n_embed:
                Embedding维度

            context_length:
                最大上下文长度

            vocab_size:
                词表大小

            N_BLOCKS:
                Transformer Block层数
        """
        super().__init__()

        self.context_length = context_length
        self.N_BLOCKS = N_BLOCKS

        # ==========================
        # Token Embedding
        # ==========================
        #
        # 将词ID映射成向量
        #
        # vocab_size -> n_embed
        #
        self.token_embed = nn.Embedding(
            vocab_size,
            n_embed
        )

        # ==========================
        # Position Embedding
        # ==========================
        #
        # 给每个位置一个可学习的位置向量
        #
        # 例如：
        #
        # token:
        # 我 爱 学习 AI
        #
        # position:
        # 0  1  2  3
        #
        self.position_embed = nn.Embedding(
            context_length,
            n_embed
        )

        # ==========================
        # Transformer Blocks
        # ==========================
        #
        # 多层Transformer堆叠
        #
        self.attn_blocks = nn.ModuleList([
            Block(
                n_head,
                n_embed,
                context_length
            )
            for _ in range(N_BLOCKS)
        ])

        # 最终LayerNorm
        self.layer_norm = nn.LayerNorm(
            n_embed
        )

        # ==========================
        # Language Model Head
        # ==========================
        #
        # 将Embedding映射到词表空间
        #
        # n_embed -> vocab_size
        #
        self.lm_head = nn.Linear(
            n_embed,
            vocab_size
        )

        # 保存位置索引
        #
        # 例如：
        #
        # context_length=8
        #
        # [0,1,2,3,4,5,6,7]
        #
        self.register_buffer(
            'pos_idxs',
            torch.arange(context_length)
        )

    def _pre_attn_pass(
        self,
        idx: torch.Tensor
    ) -> torch.Tensor:
        """
        构造Transformer输入Embedding。

        执行过程：

            Token Embedding
                    +
            Position Embedding

        参数：

            idx:
                Token索引

                (B,T)

        返回：

            输入Embedding

                (B,T,C)
        """

        B, T = idx.shape

        # Token向量
        #
        # (B,T)
        # ->
        # (B,T,C)
        #
        tok_embedding = self.token_embed(idx)

        # 位置向量
        #
        # (T)
        # ->
        # (T,C)
        #
        pos_embedding = self.position_embed(
            self.pos_idxs[:T]
        )

        # 广播相加
        #
        # (B,T,C)
        # +
        # (T,C)
        #
        return tok_embedding + pos_embedding

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        标准前向传播。

        参数：

            idx:
                输入Token序列

            targets:
                目标Token序列

        返回：

            logits:
                模型预测结果

            loss:
                交叉熵损失
        """

        # Token + Position Embedding
        x = self._pre_attn_pass(idx)

        # ==========================
        # Transformer Blocks
        # ==========================
        #
        # 不断提取上下文信息
        #
        for block in self.attn_blocks:
            x = block(x)

        # 最终归一化
        x = self.layer_norm(x)

        # ==========================
        # 输出层
        # ==========================
        #
        # (B,T,C)
        # ->
        # (B,T,vocab_size)
        #
        logits = self.lm_head(x)

        loss = None

        # ==========================
        # 训练阶段计算Loss
        # ==========================
        #
        if targets is not None:

            B, T, C = logits.shape

            # 拉平成二维
            #
            # (B*T,vocab_size)
            #
            flat_logits = logits.view(
                B * T,
                C
            )

            # (B*T)
            targets = targets.view(
                B * T
            ).long()

            # CrossEntropy
            loss = F.cross_entropy(
                flat_logits,
                targets
            )

        return logits, loss

    def forward_embedding(
        self,
        idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        仅执行Embedding与Attention部分。

        常用于：
            特征提取
            可视化分析
            中间层调试

        参数：

            idx:
                Token索引

        返回：

            x:
                最终特征

            residual:
                残差特征
        """

        x = self._pre_attn_pass(idx)

        residual = x

        for block in self.attn_blocks:
            x, residual = block.forward_embedding(x)

        return x, residual

    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int
    ) -> torch.Tensor:
        """
        自回归生成文本。

        原理：

            输入：
                今天天气

            预测：
                很

            变成：
                今天天气很

            再预测：
                好

            变成：
                今天天气很好

            循环直到生成结束。

        参数：

            idx:
                初始Token序列

            max_new_tokens:
                最大生成Token数量

        返回：

            完整生成结果
        """

        for _ in range(max_new_tokens):

            # 截取最近context_length个Token
            #
            # GPT只能看到固定长度上下文
            #
            idx_cond = idx[:, -self.context_length:]

            # 前向推理
            logits, _ = self(idx_cond)

            # 取最后一个Token位置
            #
            # (B,vocab_size)
            #
            logits = logits[:, -1, :]

            # 转概率
            probs = F.softmax(
                logits,
                dim=-1
            )

            # 按概率随机采样
            idx_next = torch.multinomial(
                probs,
                num_samples=1
            )

            # 拼接到序列尾部
            idx = torch.cat(
                (idx, idx_next),
                dim=1
            )

        return idx


if __name__ == '__main__':

    # ====================================
    # 测试代码
    # ====================================

    batch_size = 2
    sequence_length = 5

    vocab_size = 100

    embedding_dim = 32

    num_heads = 4

    num_blocks = 2

    context_len = 5

    # 随机生成Token序列
    #
    # (B,T)
    #
    input_indices = torch.randint(
        0,
        vocab_size,
        (batch_size, sequence_length)
    )

    # 创建Transformer模型
    transformer_model = Transformer(
        n_head=num_heads,
        n_embed=embedding_dim,
        context_length=context_len,
        vocab_size=vocab_size,
        N_BLOCKS=num_blocks
    )

    # 训练模式测试
    logits, loss = transformer_model(
        input_indices,
        targets=input_indices
    )

    print("Logits形状:", logits.shape)
    print("Loss:", loss)

    # ====================================
    # 文本生成测试
    # ====================================

    # 取第一个Token作为起始输入
    start_indices = input_indices[:, :1]

    generated_tokens = transformer_model.generate(
        start_indices,
        max_new_tokens=5
    )

    print("生成结果形状:", generated_tokens.shape)