import os
import json
import zstandard as zstd
import tiktoken
import h5py
from tqdm import tqdm
import argparse
from typing import Optional


def process_files(
    input_dir: str,
    output_file: str,
    tokenizer_name: str,
    max_data: Optional[int] = None
) -> None:
    """
    数据预处理核心函数：

    功能：
        将 .jsonl.zst 文本数据：
            → 解压
            → 提取 text 字段
            → tokenizer编码
            → 写入 HDF5（二进制Token流）

    最终输出：

        HDF5:
            tokens: [int, int, int, ...]

    这是 GPT 训练前的标准数据格式转换步骤。

    参数：

        input_dir:
            输入数据目录（train / val）

        output_file:
            输出 HDF5 文件路径

        tokenizer_name:
            tiktoken tokenizer 名称（如 r50k_base）

        max_data:
            每个文件最多处理多少条样本（调试用）
    """

    if max_data is not None:
        print(
            f"限制模式: max_data={max_data}，每个文件只处理前 {max_data} 条数据"
        )
    else:
        print("完整模式: 处理所有 JSON 数据")

    # ==========================
    # 初始化 tokenizer
    # ==========================
    enc = tiktoken.get_encoding(tokenizer_name)

    # ==========================
    # 创建 HDF5 输出文件
    # ==========================
    with h5py.File(output_file, 'w') as out_f:

        # 一维动态增长的Token数组
        dataset = out_f.create_dataset(
            'tokens',
            (0,),
            maxshape=(None,),
            dtype='i'
        )

        # 当前写入位置
        start_index = 0

        # ==========================
        # 遍历所有压缩文件
        # ==========================
        for filename in sorted(os.listdir(input_dir)):

            if not filename.endswith(".jsonl.zst"):
                continue

            in_file = os.path.join(input_dir, filename)

            print(f"\n处理文件: {in_file}")

            processed_lines = 0

            # ==========================
            # 解压读取 zst 文件
            # ==========================
            with zstd.open(in_file, 'rt', encoding='utf-8') as in_f:

                for line in tqdm(
                    in_f,
                    desc=f"Processing {filename}",
                    total=max_data if max_data else None
                ):

                    try:
                        # JSON解析
                        data = json.loads(line)

                        # 提取文本字段
                        text = data.get('text')

                        if text:

                            # ==========================
                            # Tokenization
                            # ==========================
                            #
                            # 添加 endoftext 标记
                            #
                            encoded = enc.encode(
                                text + "<|endoftext|>",
                                allowed_special={'<|endoftext|>'}
                            )

                            encoded_len = len(encoded)

                            # ==========================
                            # 扩展HDF5空间
                            # ==========================
                            end_index = start_index + encoded_len

                            dataset.resize(
                                dataset.shape[0] + encoded_len,
                                axis=0
                            )

                            # 写入token
                            dataset[start_index:end_index] = encoded

                            start_index = end_index

                        else:
                            print(
                                f"警告: {filename} 中缺少 text 字段"
                            )

                    except json.JSONDecodeError:
                        print(
                            f"JSON解析失败: {filename}"
                        )

                    except Exception as e:
                        print(
                            f"处理异常: {filename} -> {e}"
                        )

                    processed_lines += 1

                    # ==========================
                    # 调试模式限制数据量
                    # ==========================
                    if max_data and processed_lines >= max_data:
                        break


def main():
    """
    主函数：

    功能：
        1. 解析命令行参数
        2. 检查目录
        3. 处理 train / val 数据
    """

    parser = argparse.ArgumentParser(
        description="将 Pile 数据集转换为 HDF5 Token格式"
    )

    parser.add_argument(
        "--train_dir",
        type=str,
        default="data/train",
        help="训练数据目录"
    )

    parser.add_argument(
        "--val_dir",
        type=str,
        default="data/val",
        help="验证数据目录"
    )

    parser.add_argument(
        "--out_train_file",
        type=str,
        default="data/train/pile_train.h5",
        help="训练集输出HDF5"
    )

    parser.add_argument(
        "--out_val_file",
        type=str,
        default="data/val/pile_dev.h5",
        help="验证集输出HDF5"
    )

    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="r50k_base",
        help="tiktoken tokenizer类型"
    )

    parser.add_argument(
        "--max_data",
        type=int,
        default=1000,
        help="每个文件最大处理条数（调试用）"
    )

    args = parser.parse_args()

    # ==========================
    # 校验目录
    # ==========================
    if not os.path.isdir(args.train_dir):
        print(f"错误: 训练目录不存在 {args.train_dir}")
        return

    if not os.path.isdir(args.val_dir):
        print(f"错误: 验证目录不存在 {args.val_dir}")
        return

    # ==========================
    # 处理训练集
    # ==========================
    print("\n===== 开始处理训练集 =====")
    process_files(
        args.train_dir,
        args.out_train_file,
        args.tokenizer_name,
        args.max_data
    )

    print("训练集处理完成")

    # ==========================
    # 处理验证集
    # ==========================
    print("\n===== 开始处理验证集 =====")
    process_files(
        args.val_dir,
        args.out_val_file,
        args.tokenizer_name,
        args.max_data
    )

    print("验证集处理完成")


if __name__ == "__main__":
    main()