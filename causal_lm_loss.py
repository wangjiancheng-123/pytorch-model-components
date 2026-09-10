"""
本文件结构：技术说明 → 实现 → 最小 print 示例。
在此目录运行 python 本文件名即可查看示例；示例不是完整正确性测试。
学习实现，使用 AI 辅助整理；不代表原创算法或生产实现。

### Response-only Causal LM Loss

`logits[:,t]` 预测 `input_ids[:,t+1]`，因此同时切片：预测取 `:-1`，标签取 `1:`。不先 argmax，不先 softmax；CrossEntropy 内部处理 log-softmax。

有效位置需同时满足：目标不是 padding、前驱不是 padding、目标属于 response。response mask 只检查目标，不要求前驱也属于 response，因为第一个 response token 可以由最后一个 prompt token 预测。

无效标签通过 masked_fill 设为 -100；`ignore_index=-100` 表示忽略“标签值等于 -100”的条目，不是第 -100 列。全无效时直接返回 `logits[:, :0, :].sum()`：空切片求和为 0，但仍连接原计算图，可以 backward。

loss、accuracy、valid_count 使用同一组过滤条件。你之前先索引标签，后过滤 batch/time 索引，会导致预测与标签数量不一致；统一切片后再构造 valid 可以避免这个问题。
"""
import torch
from torch import nn
from torch.nn import functional as F


def causal_lm_loss(logits, input_ids, attention_mask, response_mask):
    # logits[:,t] 预测 input_ids[:,t+1]。目标和前驱必须均有效。
    shifted_logits = logits[:, :-1, :]
    labels = input_ids[:, 1:]
    valid = (attention_mask[:, :-1].bool() & attention_mask[:, 1:].bool()
             & response_mask[:, 1:].bool())
    labels = labels.masked_fill(~valid, -100)
    count = valid.sum()
    if count.item() == 0:
        return logits[:, :0, :].sum(), 0, 0.0
    loss = F.cross_entropy(shifted_logits.reshape(-1, logits.size(-1)),
                           labels.reshape(-1), ignore_index=-100, reduction='sum') / count
    correct = ((shifted_logits.argmax(-1) == labels) & valid).sum()
    return loss, count.item(), (correct.float() / count).item()


if __name__ == "__main__":
    logits = torch.randn(2, 5, 10, requires_grad=True)
    ids = torch.randint(0, 10, (2, 5))
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
    response = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 0, 0]])
    loss, count, accuracy = causal_lm_loss(logits, ids, mask, response)
    print('loss / count / accuracy:', loss.item(), count, accuracy)
