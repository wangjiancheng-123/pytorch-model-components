"""
本文件结构：技术说明 → 实现 → 最小 print 示例。
在此目录运行 python 本文件名即可查看示例；示例不是完整正确性测试。
学习实现，使用 AI 辅助整理；不代表原创算法或生产实现。

速度场预测器的最小接口

输入 x=[B,T,D]、t=[B]、condition=[B,C]、mask=[B,T]；输出 [B,T,D]。
时间和条件扩展到每一帧，与 x 拼接，再经过两层 MLP。最后屏蔽 padding。
此模块只用于跑通训练与采样，不含 self-attention、位置编码或时序卷积，不能建模跨帧依赖，也不是 DiT。
全零 condition 是本例约定的无条件输入，必须配合条件丢弃训练。
"""
import torch
from torch import nn
from torch.nn import functional as F


class TinyVelocityModel(nn.Module):
    """可运行接口示例：逐帧 MLP，不是完整 DiT 或 ACE-Step 复现。"""
    def __init__(self, audio_dim=80, condition_dim=32, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(audio_dim + condition_dim + 1, hidden_dim),
                                 nn.SiLU(), nn.Linear(hidden_dim, audio_dim))

    def forward(self, x, t, condition, mask):
        length = x.size(1)
        cond = condition[:, None, :].expand(-1, length, -1)
        time = t[:, None, None].expand(-1, length, 1)
        out = self.net(torch.cat([x, time, cond], dim=-1))
        return out.masked_fill(~mask.bool().unsqueeze(-1), 0.0)


if __name__ == "__main__":
    model = TinyVelocityModel(4, 3, 16)
    x, t, cond = torch.randn(2, 5, 4), torch.rand(2), torch.randn(2, 3)
    print('velocity:', model(x, t, cond, torch.ones(2, 5)).shape)
