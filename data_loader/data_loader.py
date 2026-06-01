import torch
import numpy as np
import h5py

from typing import Iterator, Tuple


def get_batch_iterator(
    data_path: str,
    batch_size: int,
    context_length: int,
    device: str = "cpu"
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """
    GPT训练数据迭代器。
    功能：
        从HDF5文件中读取Token数据，
        按照固定长度切分成训练样本，
        并持续生成Batch。
    数据格式：
        HDF5
          └── tokens
    例如：
        tokens:
        [1,2,3,4,5,6,7,8,9...]

    context_length=4
    生成：
        X=[1,2,3,4]
        Y=[2,3,4,5]

        X=[5,6,7,8]
        Y=[6,7,8,9]
    参数：
        data_path:
            HDF5文件路径
        batch_size:
            每个Batch包含多少条样本
        context_length:
            上下文长度
        device:
            cpu 或 cuda
    返回：
        生成器
        每次返回：
            xb:
                输入Token
            yb:
                目标Token
    """

    # ==========================
    # 打开HDF5文件
    # ==========================
    with h5py.File(data_path, 'r') as hdf5_file:

        # 获取Token数据集
        #
        # HDF5结构：
        #
        # root
        #  └── tokens
        #
        dataset = hdf5_file['tokens']

        # 数据总长度
        #
        # 例如：
        #
        # [1,2,3,...100000]
        #
        dataset_size = dataset.shape[0]

        # ==========================
        # 计算可生成样本数
        # ==========================
        #
        # 每个样本长度：
        #
        # context_length + 1
        #
        # 因为：
        #
        # X需要context_length
        # Y需要向后偏移1位
        #
        n_examples = (
            dataset_size - 1
        ) // context_length

        # ==========================
        # 构造样本索引
        # ==========================
        #
        # [0,1,2,3,4...]
        #
        example_idxs = np.arange(
            n_examples
        )

        # 打乱顺序
        np.random.shuffle(
            example_idxs
        )

        # Epoch计数器
        epochs = 0

        # 当前样本位置
        counter = 0

        # 无限循环
        #
        # Trainer不断next()
        #
        while True:

            # ==========================
            # Epoch结束检查
            # ==========================
            if counter + batch_size > n_examples:

                np.random.shuffle(
                    example_idxs
                )

                counter = 0

                print(
                    f"Finished epoch {epochs}"
                )

                epochs += 1

            # ==========================
            # 获取Batch索引
            # ==========================
            #
            # 例如：
            #
            # [5,20,31,8]
            #
            random_indices = (
                example_idxs[
                    counter:counter + batch_size
                ]
                * context_length
            )

            # ==========================
            # 读取数据
            # ==========================
            #
            # 每个样本长度：
            #
            # context_length + 1
            #
            # 例如：
            #
            # [10,20,30,40,50]
            #
            random_samples = torch.tensor(
                np.array([
                    dataset[
                        idx:
                        idx + context_length + 1
                    ]
                    for idx in random_indices
                ])
            )

            # ==========================
            # 构造输入X
            # ==========================
            #
            # [10,20,30,40]
            #
            xb = random_samples[
                :,
                :context_length
            ].to(device)

            # ==========================
            # 构造目标Y
            # ==========================
            #
            # [20,30,40,50]
            #
            yb = random_samples[
                :,
                1:context_length + 1
            ].to(device)

            # 移动到下一个Batch
            counter += batch_size

            yield xb, yb


if __name__ == '__main__':

    # ====================================
    # 测试代码
    # ====================================

    import os

    dummy_data_path = "dummy_data.h5"

    # 如果测试文件不存在
    if not os.path.exists(dummy_data_path):

        with h5py.File(
            dummy_data_path,
            'w'
        ) as f:

            # 创建测试Token
            #
            # [0,1,2...999]
            #
            f.create_dataset(
                'tokens',
                data=np.arange(1000)
            )

    batch_size = 4

    context_length = 10

    # 获取Batch生成器
    data_iter = get_batch_iterator(
        dummy_data_path,
        batch_size,
        context_length
    )

    # 读取一个Batch
    xb, yb = next(data_iter)

    print("输入形状:", xb.shape)
    print("目标形状:", yb.shape)

    print("\n输入样本:")
    print(xb[0])

    print("\n目标样本:")
    print(yb[0])