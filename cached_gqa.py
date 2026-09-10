"""
本文件结构：技术说明 → 实现 → 最小 print 示例。
在此目录运行 python 本文件名即可查看示例；示例不是完整正确性测试。
学习实现，使用 AI 辅助整理；不代表原创算法或生产实现。

## 1. 统一符号和 Mask 约定

| 符号 | 含义 |
| --- | --- |
| B | batch 中样本数 |
| T | 当前序列长度；有缓存时为新 chunk 长度 |
| M | 模型特征维度 |
| Hq / Hkv | Query 头数 / Key、Value 头数 |
| d | 每头维度，M/Hq；Q、K 的每头维度必须一致 |
| L | 历史缓存的物理时间长度 |
| V | 词表大小；不要与 Value 张量混淆 |

输入 `mask` 的 1/True 表示有效位置；`blocked_mask` 的 True 表示禁止关注。变量名保留语义，避免对同一个 mask 一会儿解释为有效、一会儿解释为禁止。

### 什么时候 unsqueeze？

从右向左对齐维度：尺寸必须相同或其中一个为 1。

| 操作 | 被操作张量 | mask 需要的形状 | 目的 |
| --- | --- | --- | --- |
| 帧特征清零 | [B,T,M] | [B,T,1] | 同一 token 的 M 个数使用同一标记 |
| 单头屏蔽 Key | [B,Tq,Tk] | [B,1,Tk] | 所有 Query 共用相同 Key 有效性 |
| 多头屏蔽 Key | [B,H,Tq,Tk] | [B,1,1,Tk] | 所有头、所有 Query 共用 Key 标记 |
| 多头屏蔽 Query 权重 | [B,H,Tq,Tk] | [B,1,Tq,1] | 将这个 Query 对全部 Key 的权重清零 |
| 帧级 Loss | [B,T] | [B,T] | 已经一帧一个数，无需扩维 |

`keepdim=True` 保留被归约的轴，长度改成 1。例如 [B,H,Tq,Tk] 沿最后轴 `all` 后得到 [B,H,Tq,1]，才能直接广播回每一行的所有 Key。

布尔 mask 可以与浮点张量相乘，但 `NaN * 0` 仍然是 NaN。本仓库首先避免在 softmax 中制造 NaN，而不是寄希望于最后乘零。

## 3. CachedGQA（attention.py）

### 为什么 KV 的输出维度更小？

以 M=128、Hq=8、Hkv=2 为例：每头 d=16。

| 张量 | 形状 |
| --- | --- |
| Q 投影 | [B,T,128] |
| K/V 投影 | [B,T,32] |
| q | [B,8,T,16] |
| 当前 k/v | [B,2,T,16] |
| 缓存合并后 k/v | [B,2,L+T,16] |
| 本次计算展开后的 k/v | [B,8,L+T,16] |
| scores/weights | [B,8,T,L+T] |
| 合并头后的输出 | [B,T,128] |

共享的是 KV 头，不能让 K 的每头维度变成 64。每个 KV 头连续复制 Hq/Hkv 次，沿头轴 dim=1 复制，不能沿最后的特征轴复制。Hq=Hkv 就退化为 MHA；Hkv=1 为 MQA。

### 因果位置与缓存

没有缓存：Query 位置为 0…T-1。已有 L 列缓存：Query 位置为 L…L+T-1；Key 位置为 0…L+T-1。

`q_pos=[T,1]` 表示每一行 Query 的位置，`k_pos=[1,L+T]` 表示每一列 Key 的位置。`k_pos > q_pos` 标记未来，True 的位置不能看。首次 Prefill 也必须用这个 mask。

`future | key_padding | query_padding` 是逻辑或：任意一个原因禁止，就禁止。masked_fill 不需要分辨“因为什么禁止”，只按最后的 bool 矩阵操作。

新 K 先应用带 L 偏移的 RoPE，再与已经旋转过的旧 K 拼接，不能对历史 K 每次重复旋转。缓存返回未展开的 Hkv 个头。本例缓存不自动 detach：在训练时保留计算图；纯推理调用者使用 `torch.no_grad()`。

外部必须接收更新：

```python
output, weights, new_cache = model(chunk, chunk_mask, cache)
cache = new_cache
```

也可直接 `output, weights, cache = model(...)`。如果只用 new_cache 接收，却一直把旧 cache 传回去，缓存不会自动更新。

### 全屏蔽行

单个 `-inf` 在一行仍有有效分数时，其 softmax 权重为 0。但整行都是 `-inf` 时，稳定 softmax 中的“减最大值”会出现 `-inf-(-inf)`，从而 NaN。

代码先找出全屏蔽行，将其临时分数置为 0，执行 softmax，再把所有禁止位置权重归零。这样普通有效行和为 1，完全屏蔽行和为 0，且反传不会先经过 NaN。

最后还要在 out_proj 后屏蔽 Query，因为 Linear bias 可以把零向量变成非零。

### 验证重点与边界

测试检查完整/分块输出一致、未来权重为零、padding 权重与输出为零、改变未来不影响过去、反向梯度有限。使用含左 padding、空洞、全 padding 的 batch。

位置定义是物理列编号，不是每条样本有效 token 的累计数量。跨 chunk 保持同一批样本、同一列顺序才能比较；没有实现动态批调度、PagedAttention、KV 淘汰或生成循环。
"""
import torch
from torch import nn
from torch.nn import functional as F
from rotary_embedding import RotaryEmbedding
from masked_softmax import masked_softmax


class CachedGQA(nn.Module):
    def __init__(self, model_dim=128, num_query_heads=8, num_kv_heads=2):
        super().__init__()
        if min(model_dim, num_query_heads, num_kv_heads) <= 0:
            raise ValueError('dimensions and head counts must be positive')
        if model_dim % num_query_heads or num_query_heads % num_kv_heads:
            raise ValueError('model_dim % Hq == 0 and Hq % Hkv == 0 required')
        self.model_dim = model_dim
        self.hq, self.hkv = num_query_heads, num_kv_heads
        self.head_dim = model_dim // num_query_heads
        self.q_proj = nn.Linear(model_dim, model_dim)
        self.k_proj = nn.Linear(model_dim, self.hkv * self.head_dim)
        self.v_proj = nn.Linear(model_dim, self.hkv * self.head_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)
        self.rope = RotaryEmbedding(self.head_dim)

    def split_heads(self, x, heads):
        b, t, _ = x.shape
        return x.reshape(b, t, heads, self.head_dim).transpose(1, 2)

    def forward(self, x, token_mask, cache=None):
        # cache=(rotated_k, v, mask)，缓存始终只有 Hkv 个头。
        b, t, _ = x.shape
        if t == 0:
            raise ValueError('empty chunks are not supported')
        mask = token_mask.bool()
        length = 0 if cache is None else cache[0].size(2)
        q = self.split_heads(self.q_proj(x), self.hq)
        k = self.split_heads(self.k_proj(x), self.hkv)
        v = self.split_heads(self.v_proj(x), self.hkv)
        positions = (torch.arange(t, device=x.device) + length).expand(b, t)
        q, k = self.rope(q, k, positions)
        if cache is None:
            k_all, v_all, all_mask = k, v, mask
        else:
            k_all = torch.cat([cache[0], k], dim=2)
            v_all = torch.cat([cache[1], v], dim=2)
            all_mask = torch.cat([cache[2], mask], dim=1)
        new_cache = (k_all, v_all, all_mask)

        # 展开仅用于本次计算；不要将展开后的 KV 返回为缓存。
        repeat = self.hq // self.hkv
        keys = k_all.repeat_interleave(repeat, dim=1)
        values = v_all.repeat_interleave(repeat, dim=1)
        q_pos = torch.arange(t, device=x.device)[:, None] + length
        k_pos = torch.arange(length + t, device=x.device)[None, :]
        future = (k_pos > q_pos)[None, None, :, :]
        blocked = future | (~all_mask[:, None, None, :])
        # 包含 query mask，使 padding query 的权重本身也为零。
        blocked = blocked | (~mask[:, None, :, None])
        scores = (q @ keys.transpose(-1, -2)) / self.head_dim**0.5
        weights = masked_softmax(scores, blocked)
        attended = weights @ values
        attended = attended.transpose(1, 2).reshape(b, t, self.model_dim)
        output = self.out_proj(attended)
        # out_proj 的 bias 可重新产生非零值，故在投影后屏蔽。
        output = output.masked_fill(~mask.unsqueeze(-1), 0.0)
        return output, weights, new_cache


if __name__ == "__main__":
    model = CachedGQA(32, 4, 2).eval()
    x = torch.randn(2, 5, 32)
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
    cache = None
    with torch.no_grad():
        for start, end in [(0, 2), (2, 5)]:
            output, weights, cache = model(x[:, start:end], mask[:, start:end], cache)
            print('output / weights / cache:', output.shape, weights.shape, cache[0].shape)
