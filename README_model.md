# 参数量估算公式

# 一、变量含义

```
VOCAB_SIZE      # 词表大小
CONTEXT_LENGTH  # 最大上下文长度

N_EMBED         # 隐藏层维度(d_model)
N_HEAD          # 注意力头数

N_BLOCKS        # Transformer层数(L)
```

其中：

```
d = N_EMBED
L = N_BLOCKS
V = VOCAB_SIZE
```

# 二、Embedding层参数

Token Embedding：

```text
VOCAB_SIZE × N_EMBED
```

公式：

V × d

对应代码：

```
embedding_params = VOCAB_SIZE * N_EMBED
```

# 三、Attention参数

每层包含：

```text
Q
K
V
O
```

四个线性层。

每个矩阵：

```text
N_EMBED × N_EMBED
```

代码：

```
attention_params = 4 * N_EMBED * N_EMBED
```

---

# 四、FFN参数

GPT标准FFN：

```text
N_EMBED
↓
4 × N_EMBED
↓
N_EMBED
```

代码：

```
ffn_params = 8 * N_EMBED * N_EMBED
```

# 五、单层Block参数

Attention：

```text
4d²
```

FFN：

```text
8d²
```

合计：

12d^2

代码：

```
block_params = 12 * N_EMBED * N_EMBED
```

---

# 六、全部Transformer参数

共有：N_BLOCKS 层。
公式：

12Ld^2

代码：
```
transformer_params = (
    12 *
    N_BLOCKS *
    N_EMBED *
    N_EMBED
)
```

# 七、总参数量估算

GPT类模型常用近似公式：

Vd + 12Ld^2

代码：

```
total_params = (
    VOCAB_SIZE * N_EMBED +
    12 * N_BLOCKS * N_EMBED * N_EMBED
)
```


# 八、代入你当前配置

```python
VOCAB_SIZE = 50304
N_EMBED = 384
N_BLOCKS = 6
```

Embedding：

```
VOCAB_SIZE * N_EMBED
50304 * 384
```
约等于
```text
19.3M
```

Transformer：

```
12 * 6 * 384 * 384
```
约等于

```text
10.6M
```

---

总计：

```
19.3M + 10.6M
```
约等于
```text
29.9M
```

约：

```text
3000万参数
```

# 九、快速估算函数

你可以直接放到项目里：

```python
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
```

调用：

```python
estimate_params(
    VOCAB_SIZE,
    N_EMBED,
    N_BLOCKS
)
```

---

# 十、显存估算公式（训练时）

如果使用：

```python
optimizer = AdamW(...)
```

FP32训练经验公式：

```python
memory_bytes = total_params * 16
memory_gb = memory_bytes / 1024**3
```

经验值：

| 参数量  | AdamW训练显存 |
| ---- | --------- |
| 30M  | 约0.5~1GB  |
| 100M | 约2GB      |
| 300M | 约5~8GB    |
| 1B   | 约16~24GB  |
| 3B   | 约48GB+    |

所以你现在这套：

```python
N_EMBED = 384
N_BLOCKS = 6
```

