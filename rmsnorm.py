"""
本文件结构：技术说明 → 实现 → 最小 print 示例。
在此目录运行 python 本文件名即可查看示例；示例不是完整正确性测试。
学习实现，使用 AI 辅助整理；不代表原创算法或生产实现。

### RMSNorm

公式：`y = x / sqrt(mean(x², last_dim) + eps) * weight`。

1. `square().mean(-1, keepdim=True)`：对每个 token 的特征维统计平方均值，[B,T,M] → [B,T,1]。
2. `rsqrt`：计算平方根的倒数，再乘 x，相当于除以均方根。
3. `weight`：[M] 可训练缩放，逐元素相乘，不是 [M,M] 矩阵投影。

初始化为 1，让初始输出等于归一化结果。初始化为 0 会使该分支初始输出为 0，也会阻断此刻通过该缩放传向 x 的梯度。RMSNorm 不减均值；LayerNorm 会减均值并按方差归一化。

代码在半精度输入时以 float32 统计平方均值，再转回输入类型；这不代表整个模型自动获得混合精度数值安全保证。
"""
import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, model_dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(model_dim))
        self.eps = eps

    def forward(self, x):
        # 半精度输入在 float32 中计算平方均值，避免溢出。
        work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        normalized = work * torch.rsqrt(work.square().mean(-1, keepdim=True) + self.eps)
        return normalized.to(x.dtype) * self.weight.to(x.dtype)


if __name__ == "__main__":
    x = torch.randn(2, 5, 32)
    y = RMSNorm(32)(x)
    print('output:', y.shape)
