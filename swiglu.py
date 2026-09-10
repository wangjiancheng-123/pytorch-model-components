"""
本文件结构：技术说明 → 实现 → 最小 print 示例。
在此目录运行 python 本文件名即可查看示例；示例不是完整正确性测试。
学习实现，使用 AI 辅助整理；不代表原创算法或生产实现。

### SwiGLU

公式：`down(silu(gate(x)) * up(x))`。

两条并行分支都把 [B,T,M] 映射到 [B,T,F]，逐元素相乘后再投影回 M。gate 不是先后串联的普通 Linear，也不是一个全局标量。

`SiLU(x)=x*sigmoid(x)`，保留平滑的负值区域；ReLU 在负半轴输出 0。这里采用无 bias Linear，属于本例设计选择。
"""
import torch
from torch import nn
from torch.nn import functional as F


class SwiGLU(nn.Module):
    def __init__(self, model_dim, hidden_dim):
        super().__init__()
        self.gate = nn.Linear(model_dim, hidden_dim, bias=False)
        self.up = nn.Linear(model_dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, model_dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


if __name__ == "__main__":
    x = torch.randn(2, 5, 32)
    print('output:', SwiGLU(32, 64)(x).shape)
