import torch
import torch.nn.functional as F
import os
from tqdm import tqdm
import numpy as np

from config.config import default_config as config
from src.models.transformer import Transformer
from data_loader.data_loader import get_batch_iterator

from typing import Dict


# ==========================
# 1. 初始化模型
# ==========================

model = Transformer(
    n_head=config['n_head'],
    n_embed=config['n_embed'],
    context_length=config['context_length'],
    vocab_size=config['vocab_size'],
    N_BLOCKS=config['n_blocks']
).to(config['device'])

# 统计参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"模型总参数量: {total_params:,}")


# ==========================
# 2. 优化器
# ==========================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config['t_lr']
)

# 记录训练loss
losses = []

# 平滑窗口（用于显示loss趋势）
AVG_WINDOW = 64


# ==========================
# 3. 验证/评估函数
# ==========================

@torch.no_grad()
def estimate_loss(steps: int) -> Dict[str, float]:
    """
    在 train / dev 集上评估模型平均loss。

    用途：
        - 监控是否过拟合
        - 判断训练是否收敛
    """

    out = {}
    model.eval()

    for split in ['train', 'dev']:

        # 选择数据路径
        data_path = (
            config['train_path']
            if split == 'train'
            else config['dev_path']
        )

        # 构造数据迭代器
        batch_iterator_eval = get_batch_iterator(
            data_path,
            config['t_batch_size'],
            config['t_context_length'],
            device=config['device']
        )

        losses_eval = torch.zeros(steps)

        for k in range(steps):

            try:
                xb, yb = next(batch_iterator_eval)

                # 前向计算loss
                _, loss = model(xb, yb)

                losses_eval[k] = loss.item()

            except StopIteration:
                print(f"{split}数据提前结束")
                break

        out[split] = losses_eval[:k + 1].mean()

    model.train()
    return out


# ==========================
# 4. 训练数据迭代器
# ==========================

batch_iterator = get_batch_iterator(
    config['train_path'],
    config['t_batch_size'],
    config['t_context_length'],
    device=config['device']
)


# ==========================
# 5. 训练主循环
# ==========================

pbar = tqdm(range(config['t_train_steps']))

for step in pbar:

    try:
        # --------------------------
        # 取一个batch
        # --------------------------
        xb, yb = next(batch_iterator)

        # --------------------------
        # 前向传播 + loss
        # --------------------------
        _, loss = model(xb, yb)

        losses.append(loss.item())

        # 显示滑动平均loss
        pbar.set_description(
            f"Train loss: {np.mean(losses[-AVG_WINDOW:]):.4f}"
        )

        # --------------------------
        # 反向传播
        # --------------------------
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # --------------------------
        # 定期评估
        # --------------------------
        if step % config['t_eval_steps'] == 0:

            evaluation_losses = estimate_loss(
                config['t_eval_iters']
            )

            print(
                f"Step {step} | "
                f"Train: {evaluation_losses['train']:.4f} | "
                f"Dev: {evaluation_losses['dev']:.4f}"
            )

        # --------------------------
        # 学习率衰减
        # --------------------------
        if step == config['t_lr_decay_step']:

            print("学习率衰减")

            for g in optimizer.param_groups:
                g['lr'] = config['t_lr_decayed']

    except StopIteration:
        print("训练数据迭代结束")
        break


# ==========================
# 6. 保存模型
# ==========================

os.makedirs(
    config['t_out_path'].split('/')[0],
    exist_ok=True
)

# 最终评估
evaluation_losses = estimate_loss(200)

train_loss = evaluation_losses['train']
dev_loss = evaluation_losses['dev']

# 防止覆盖模型
modified_model_out_path = config['t_out_path']
save_tries = 0

while os.path.exists(modified_model_out_path):

    save_tries += 1

    base = os.path.splitext(config['t_out_path'])[0]

    modified_model_out_path = (
        base + f"_{save_tries}.pt"
    )

# ==========================
# 保存checkpoint
# ==========================
torch.save(
    {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses,
        'train_loss': train_loss,
        'dev_loss': dev_loss,
        'steps': len(losses),
    },
    modified_model_out_path
)

print(f"模型已保存: {modified_model_out_path}")

print(
    f"训练完成 | Train loss: {train_loss:.4f} | Dev loss: {dev_loss:.4f}"
)