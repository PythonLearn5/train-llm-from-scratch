# =========================
# 模型参数量估算公式
# =========================

# 词表大小（Vocabulary Size）
# 表示模型能够识别的 Token 数量。
# 50304 是 GPT-2 常用词表大小。
VOCAB_SIZE = 50304

# 隐藏层维度（Embedding Dimension）
# Transformer 的核心特征维度。
# 该值越大，模型表达能力越强，同时参数量和显存占用也会显著增加。
N_EMBED = 384

# Transformer Block 数量
# 即 Transformer 层数。
# 层数越多，模型学习能力越强，但训练时间更长。
N_BLOCKS = 6


def estimate_params(
    vocab_size,
    n_embed,
    n_blocks
):
    embedding = vocab_size * n_embed

    transformer = (
        12 *
        n_blocks *
        n_embed *
        n_embed
    )

    total = embedding + transformer

    print(f"Embedding : {embedding:,}")
    print(f"Transformer : {transformer:,}")
    print(f"Total : {total:,}")

    return total

estimate_params(
    VOCAB_SIZE,
    N_EMBED,
    N_BLOCKS
)