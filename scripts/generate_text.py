import torch
import tiktoken
import argparse

from config.config import default_config as config
from src.models.transformer import Transformer


def generate_text(
    model_path: str,
    input_text: str,
    max_new_tokens: int = 100,
    device: str = 'cuda'
) -> str:
    """
    使用训练好的 Transformer 模型进行文本生成（推理阶段）。

    整体流程：

        输入文本
            ↓
        Tokenizer编码
            ↓
        Transformer预测下一个token
            ↓
        采样生成
            ↓
        循环扩展序列
            ↓
        Decoder还原文本

    参数：

        model_path:
            模型权重路径（checkpoint）

        input_text:
            生成的起始文本（prompt）

        max_new_tokens:
            最多生成多少个token

        device:
            cuda / cpu

    返回：

        生成的完整文本
    """

    # ==========================
    # 加载模型权重
    # ==========================
    checkpoint = torch.load(
        model_path,
        map_location=torch.device(device)
    )

    # ==========================
    # 构建Transformer模型
    # ==========================
    model = Transformer(
        n_head=config['n_head'],
        n_embed=config['n_embed'],
        context_length=config['context_length'],
        vocab_size=config['vocab_size'],
        N_BLOCKS=config['n_blocks']
    )

    model.load_state_dict(
        checkpoint['model_state_dict']
    )

    model.eval().to(device)

    # ==========================
    # 初始化Tokenizer
    # ==========================
    enc = tiktoken.get_encoding("r50k_base")

    # ==========================
    # 输入文本编码
    # ==========================
    start_ids = enc.encode_ordinary(input_text)

    context = torch.tensor(
        start_ids,
        dtype=torch.long,
        device=device
    ).unsqueeze(0)

    # ==========================
    # 生成阶段（Autoregressive）
    # ==========================
    with torch.no_grad():

        generated_tokens = model.generate(
            context,
            max_new_tokens=max_new_tokens
        )[0].tolist()

    # ==========================
    # 解码Token
    # ==========================
    output_text = enc.decode(generated_tokens)

    return output_text


def main() -> None:
    """
    命令行入口：

    示例：

        python generate.py \
            --model_path model.pt \
            --input_text "你好世界" \
            --max_new_tokens 100
    """

    parser = argparse.ArgumentParser(
        description="使用训练好的Transformer生成文本"
    )

    parser.add_argument(
        '--model_path',
        type=str,
        help='模型checkpoint路径'
    )

    parser.add_argument(
        '--input_text',
        type=str,
        help='生成起始文本'
    )

    parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=100,
        help='最多生成token数量'
    )

    args = parser.parse_args()

    # ==========================
    # 调用生成函数
    # ==========================
    generated = generate_text(
        args.model_path,
        args.input_text,
        args.max_new_tokens,
        config['device']
    )

    print("\n===== 生成结果 =====\n")
    print(generated)


if __name__ == "__main__":
    main()