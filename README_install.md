### 环境安装

py -m venv .venv

.\.venv\Scripts\Activate.ps1

测试：
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.version.cuda)"
python -c "import torch;print(torch.cuda.is_available())"
python -c "import torch;print(torch.cuda.get_device_name(0))"

torch 版本问题 cpu 没有cuda 需要重新安装
2.12.0+cpu

安装 CUDA 版 PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

查看显卡：
nvidia-smi


下载训练数据，请运行：
python scripts/data_download.py
该脚本支持以下参数：
--train_max：要下载的最大训练文件数量。默认值为 1（最大值为 30）。每个文件大小约为 11 GB。
--train_dir：用于存储训练数据的目录。默认值为data/train.
--val_dir：用于存储验证数据的目录。默认值为data/val.

预处理下载的数据，请运行：
python scripts/data_preprocess.py
--train_dir：存储训练数据文件的目录（默认为data/train）。
--val_dir：存储验证数据文件的目录（默认为空data/val）。
--out_train_file：存储已处理训练数据的 HDF5 格式路径（默认值为data/train/pile_train.h5）。
--out_val_file：存储已处理的验证数据的 HDF5 格式路径（默认为data/val/pile_dev.h5）。
--tokenizer_name：用于处理数据的分词器的名称（默认值为r50k_base）。
--max_data每个数据集（包括训练集和验证集）要处理的最大 JSON 对象（行）数。默认值为 1000。


训练模型，请运行：
python scripts/train_transformer.py

要使用训练好的模型生成文本，请运行：

python scripts/generate_text.py --model_path models/your_model.pth --input_text hi
该脚本支持以下参数：
--model_path：通往已训练模型的路径。
--input_text：用于生成新文本的初始文本提示。
--max_new_tokens：要生成的最大令牌数（默认值为 100）。
它将使用训练好的模型，根据输入提示生成文本。