# =========================
# 模型基础配置
# =========================

# 词表大小（Vocabulary Size）
# 表示模型能够识别的 Token 数量。
# 50304 是 GPT-2 常用词表大小。
VOCAB_SIZE = 50304

# 模型最大上下文长度（Context Length）
# 即模型一次能够看到多少个 Token。
# 数值越大，模型能理解更长的上下文，但显存消耗也越高。
CONTEXT_LENGTH = 256

# 隐藏层维度（Embedding Dimension）
# Transformer 的核心特征维度。
# 该值越大，模型表达能力越强，同时参数量和显存占用也会显著增加。
N_EMBED = 384

# 多头注意力头数（Attention Heads）
# 通常要求 N_EMBED 能被 N_HEAD 整除。
# 每个 Head 学习不同的注意力模式。
N_HEAD = 6

# Transformer Block 数量
# 即 Transformer 层数。
# 层数越多，模型学习能力越强，但训练时间更长。
N_BLOCKS = 6


# =========================
# 数据集配置
# =========================

# 训练集路径
# 训练时使用的数据文件。
TRAIN_PATH = "data/train/pile_train.h5"

# 验证集路径
# 用于评估模型效果，不参与参数更新。
DEV_PATH = "data/val/pile_dev.h5"


# =========================
# 训练参数配置
# =========================

# Batch Size
# 每次训练送入模型的样本数量。
#
# 当前设置：
#   batch_size = 2
#
# 优点：
#   显存占用低，适合 16GB 显卡。
#
# 缺点：
#   梯度波动较大，训练速度相对较慢。
T_BATCH_SIZE = 2


# 训练时使用的上下文长度
#
# 通常应与 CONTEXT_LENGTH 保持一致。
#
# 例如：
#   CONTEXT_LENGTH = 256
#   T_CONTEXT_LENGTH = 256
#
# 表示训练时每个样本长度为 256 Token。
T_CONTEXT_LENGTH = 256


# 总训练步数（Training Steps）
#
# 每一步执行：
#   前向传播
#   反向传播
#   参数更新
#
# 200000 步属于较长训练。
T_TRAIN_STEPS = 200000


# 每隔多少步进行一次验证
#
# 例如：
#   每训练 1000 步
#   在验证集上评估一次 Loss
T_EVAL_STEPS = 1000


# 每次验证时抽取多少个 Batch
#
# 数值越大：
#   验证结果越稳定
#
# 但验证耗时也会增加。
T_EVAL_ITERS = 100


# 学习率衰减开始的步数
#
# 例如：
#   前 50000 步使用较大学习率
#   之后降低学习率进行精细训练
T_LR_DECAY_STEP = 50000


# 初始学习率
#
# 对于当前约 1,000 万~2,000 万参数模型：
#   3e-4 是较常见的选择
T_LR = 3e-4


# 学习率衰减后的值
#
# 用于后期稳定收敛。
T_LR_DECAYED = 5e-5


# 模型保存路径
#
# 训练完成后将保存：
#   模型权重
#   优化器状态
#   Loss 曲线数据
T_OUT_PATH = "models/transformer_B.pt"


# =========================
# 设备配置
# =========================

# 使用 CUDA GPU 训练
#
# 如果需要自动判断：
#
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
#
DEVICE = 'cuda'


# =========================
# 汇总配置字典
# =========================
#
# 所有训练代码统一从此字典读取配置，
# 后续修改参数只需要改上面的配置项即可。
#
default_config = {
    'vocab_size': VOCAB_SIZE,
    'context_length': CONTEXT_LENGTH,
    'n_embed': N_EMBED,
    'n_head': N_HEAD,
    'n_blocks': N_BLOCKS,
    'train_path': TRAIN_PATH,
    'dev_path': DEV_PATH,
    't_batch_size': T_BATCH_SIZE,
    't_context_length': T_CONTEXT_LENGTH,
    't_train_steps': T_TRAIN_STEPS,
    't_eval_steps': T_EVAL_STEPS,
    't_eval_iters': T_EVAL_ITERS,
    't_lr_decay_step': T_LR_DECAY_STEP,
    't_lr': T_LR,
    't_lr_decayed': T_LR_DECAYED,
    't_out_path': T_OUT_PATH,
    'device': DEVICE,
}