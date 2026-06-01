import os
import argparse
import requests
from tqdm import tqdm
from typing import List


# ==========================
# 数据集基础地址
# ==========================
#
# 使用 HuggingFace 上的 Pile 子集（无版权版本）
#
BASE_URL = "https://huggingface.co/datasets/monology/pile-uncopyrighted/resolve/main"

# 验证集地址
VAL_URL = f"{BASE_URL}/val.jsonl.zst"

# 训练集文件列表（共65个分片）
TRAIN_URLS = [
    f"{BASE_URL}/train/{i:02d}.jsonl.zst"
    for i in range(65)
]


def download_file(url: str, file_name: str) -> None:
    """
    从指定URL下载文件，并保存到本地。

    特点：
        - 流式下载（避免内存占用过高）
        - tqdm显示进度条

    参数：

        url:
            文件下载地址

        file_name:
            本地保存路径
    """

    print(f"开始下载: {file_name} ...")

    # 发起HTTP请求（流式）
    response = requests.get(url, stream=True)

    # 文件总大小（用于进度条）
    total_size = int(response.headers.get('content-length', 0))

    # 每次读取1KB
    block_size = 1024

    # 写入文件
    with open(file_name, 'wb') as f:

        # tqdm进度条
        for chunk in tqdm(
            response.iter_content(block_size),
            total=total_size // block_size,
            desc="Downloading",
            leave=True
        ):
            f.write(chunk)


def download_dataset(
    val_url: str,
    train_urls: List[str],
    val_dir: str,
    train_dir: str,
    max_train_files: int
) -> None:
    """
    下载完整数据集（训练集 + 验证集）。

    数据结构：

        data/
          ├── train/
          │     ├── 00.jsonl.zst
          │     ├── 01.jsonl.zst
          │     └── ...
          └── val/
                └── val.jsonl.zst

    参数：

        val_url:
            验证集下载地址

        train_urls:
            训练集分片URL列表

        val_dir:
            验证集存储目录

        train_dir:
            训练集存储目录

        max_train_files:
            最多下载多少个训练分片
    """

    # ==========================
    # 下载验证集
    # ==========================
    val_file_path = os.path.join(val_dir, "val.jsonl.zst")

    if not os.path.exists(val_file_path):

        print(f"未找到验证集，开始下载: {val_url}")

        download_file(val_url, val_file_path)

    else:
        print("验证集已存在，跳过下载")

    # ==========================
    # 下载训练集
    # ==========================
    for idx, url in enumerate(train_urls[:max_train_files]):

        file_name = f"{idx:02d}.jsonl.zst"

        file_path = os.path.join(train_dir, file_name)

        if not os.path.exists(file_path):

            print(f"未找到训练文件 {file_name}，开始下载...")

            download_file(url, file_path)

        else:
            print(f"训练文件 {file_name} 已存在，跳过")


def main() -> None:
    """
    主函数：

    负责：
        1. 解析命令行参数
        2. 创建目录
        3. 启动数据下载流程
    """

    parser = argparse.ArgumentParser(
        description="下载 Pile 数据集（HuggingFace版本）"
    )

    # 最多下载多少训练文件
    parser.add_argument(
        '--train_max',
        type=int,
        default=1,
        help="最多下载多少个训练分片"
    )

    # 训练集目录
    parser.add_argument(
        '--train_dir',
        default="data/train",
        help="训练数据存储目录"
    )

    # 验证集目录
    parser.add_argument(
        '--val_dir',
        default="data/val",
        help="验证数据存储目录"
    )

    args = parser.parse_args()

    # ==========================
    # 创建目录（如果不存在）
    # ==========================
    os.makedirs(args.train_dir, exist_ok=True)
    os.makedirs(args.val_dir, exist_ok=True)

    # ==========================
    # 开始下载数据
    # ==========================
    download_dataset(
        VAL_URL,
        TRAIN_URLS,
        args.val_dir,
        args.train_dir,
        args.train_max
    )

    print("数据集下载完成")


if __name__ == "__main__":
    main()