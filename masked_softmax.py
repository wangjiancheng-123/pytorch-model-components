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

### 全屏蔽行

单个 `-inf` 在一行仍有有效分数时，其 softmax 权重为 0。但整行都是 `-inf` 时，稳定 softmax 中的“减最大值”会出现 `-inf-(-inf)`，从而 NaN。

代码先找出全屏蔽行，将其临时分数置为 0，执行 softmax，再把所有禁止位置权重归零。这样普通有效行和为 1，完全屏蔽行和为 0，且反传不会先经过 NaN。

最后还要在 out_proj 后屏蔽 Query，因为 Linear bias 可以把零向量变成非零。
"""
import torch
from torch import nn
from torch.nn import functional as F


def masked_softmax(scores, blocked_mask):
    """True 表示禁止关注；全屏蔽行返回全零，且避免 softmax(-inf,...,-inf)。"""
    scores = scores.masked_fill(blocked_mask, float('-inf'))
    fully_blocked = blocked_mask.all(-1, keepdim=True)
    scores = scores.masked_fill(fully_blocked, 0.0)
    return scores.softmax(-1).masked_fill(blocked_mask, 0.0)


if __name__ == "__main__":
    scores = torch.tensor([[1., 2., 3.], [1., 2., 3.]])
    blocked = torch.tensor([[False, False, True], [True, True, True]])
    weights = masked_softmax(scores, blocked)
    print('weights:', weights)
    print('row sums:', weights.sum(-1))
