"""
本文件结构：技术说明 → 实现 → 最小 print 示例。
在此目录运行 python 本文件名即可查看示例；示例不是完整正确性测试。
学习实现，使用 AI 辅助整理；不代表原创算法或生产实现。

## 4. LoRALinear（lora.py）

PyTorch Linear 权重是 [out,in]。本例 `A=[rank,in]`、`B=[out,rank]`：

`y = x Wᵀ + bias + (alpha/rank) x Aᵀ Bᵀ`。

先经过 A 降维，再经过 B 升维。A 随机初始化、B 全零，使新建 LoRA 输出与基础层完全相同。基础层冻结；不使用 `no_grad` 包住基础前向，否则也可能阻断对输入所需的梯度。

### B 全零为什么能学习？

B 的梯度由上游梯度与 A(x) 决定，不需要 B 自己非零。第一步 A 的梯度包含 B，因而为零；更新 B 后，A 就有机会收到非零梯度。这是初始化设计，不是梯度断了。

### 合并

`W_merged = W + (alpha/rank) B A`。本例复制基础 Linear 后原地增加增量，保留原 bias；两条 LoRA 分支没有独立 bias。

这样得到一个只做一次 Linear 的导出模型，原模块不变，重复导出不会重复叠加。不使用 `base.weight = ordinary_tensor`；那会破坏 Parameter 注册或报错。

这里有意不使用原地 merge/unmerge：否则还要保存合并标记、防止重复加权重、防止合并状态下继续更新 A/B，并保证 checkpoint 状态一致。若需要恢复训练，使用原 LoRALinear；合并后的 Linear 已经不包含可恢复的独立 A/B。
"""
import torch
from torch import nn
from torch.nn import functional as F
import copy
import math


class LoRALinear(nn.Module):
    def __init__(self, base_layer, rank=8, alpha=16):
        super().__init__()
        if rank <= 0:
            raise ValueError('rank must be positive')
        self.base = base_layer
        self.base.requires_grad_(False)
        w = base_layer.weight
        self.A = nn.Parameter(w.new_empty(rank, base_layer.in_features))
        self.B = nn.Parameter(w.new_zeros(base_layer.out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.A), self.B) * self.scale

    @torch.no_grad()
    def to_merged_linear(self):
        merged = copy.deepcopy(self.base)
        merged.weight.add_((self.B @ self.A) * self.scale)
        return merged


if __name__ == "__main__":
    model = LoRALinear(nn.Linear(8, 6), rank=2, alpha=4)
    x = torch.randn(2, 5, 8)
    print('initial difference:', (model(x) - model.base(x)).abs().max().item())
    model(x).square().mean().backward()
    print('first-step A / B grad:', model.A.grad.norm().item(), model.B.grad.norm().item())
    print('merged difference:', (model(x) - model.to_merged_linear()(x)).abs().max().item())
