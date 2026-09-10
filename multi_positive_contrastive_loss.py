"""
本文件结构：技术说明 → 实现 → 最小 print 示例。
在此目录运行 python 本文件名即可查看示例；示例不是完整正确性测试。
学习实现，使用 AI 辅助整理；不代表原创算法或生产实现。

### 多正例对比学习

各 embedding 沿最后维 L2 归一化，再计算 `S = audio @ text.T / temperature`，结果 [B,B]，第 i 行第 j 列表示音频 i 与文本 j 的相似度。

`pair_ids[:,None] == pair_ids[None,:]` 生成正例矩阵；不要额外与单位矩阵取交集，否则会丢掉非对角线正例。假设音频和文本各 B 条、共用对应的 pair_ids，每行至少有一个同 ID 正例。

每行 loss 为 `-sum(positive * log_softmax(S)) / positive_count`，然后对样本平均；交换方向后再平均。这里是“正例 log 概率均值”，不是“正例概率求和后再取 log”，二者优化目标不同。

temperature 越小，分布越尖锐；必须大于 0。labels/pair_ids 是离散类别或组别，不与 logits 做数值距离回归。
"""
import torch
from torch import nn
from torch.nn import functional as F


def multi_positive_contrastive_loss(audio, text, pair_ids, temperature=0.1):
    # 每行对所有正例的 log probability 求平均，再对两方向平均。
    if temperature <= 0 or audio.size(0) == 0:
        raise ValueError('positive temperature and nonempty batch required')
    audio, text = F.normalize(audio, dim=-1), F.normalize(text, dim=-1)
    scores = audio @ text.T / temperature
    positive = pair_ids[:, None] == pair_ids[None, :]

    def direction(s, p):
        log_probs = F.log_softmax(s, dim=-1)
        return (-(log_probs * p).sum(-1) / p.sum(-1).clamp_min(1)).mean()

    return (direction(scores, positive) + direction(scores.T, positive.T)) / 2


if __name__ == "__main__":
    audio, text = torch.randn(4, 8), torch.randn(4, 8)
    ids = torch.tensor([0, 0, 1, 2])
    print('positive pairs:', ids[:, None] == ids[None, :])
    print('loss:', multi_positive_contrastive_loss(audio, text, ids).item())
