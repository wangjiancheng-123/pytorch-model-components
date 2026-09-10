"""
本文件结构：技术说明 → 实现 → 最小 print 示例。
在此目录运行 python 本文件名即可查看示例；示例不是完整正确性测试。
学习实现，使用 AI 辅助整理；不代表原创算法或生产实现。

### RoPE

第 j 个二维特征对使用固定频率 `base^(-2j/d)`，位置 p 的角度为 `p * frequency[j]`。

对相邻 even、odd 通道旋转：

```python
rotated_even = even * cos - odd * sin
rotated_odd = even * sin + odd * cos
```

- 频率来自通道编号，不能根据 Q/K 的数值生成。
- `stack(..., dim=-1).flatten(-2)` 将旋转后的偶数、奇数通道交错还原。
- cos/sin 为 [B,1,T,d/2]，共享到各头；所以 Q 与 K 的头数可以不同。
- head_dim 必须是偶数；这里是“相邻通道配对”版本，不能直接混用另一种“前后半区配对”的权重约定。
- 旋转保持向量长度，这是测试的数学依据。
- `register_buffer(..., persistent=False)`：随模块移动设备，不作为优化参数，也不写入 state_dict。重新初始化可从 head_dim/base 重建；保存时仍需要记录这些构造配置。
- 先用局部变量 `inv_freq` 再注册，避免先赋 `self.inv_freq` 后同名注册引发属性冲突。
"""
import torch
from torch import nn
from torch.nn import functional as F


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, base=10000.0):
        super().__init__()
        if head_dim <= 0 or head_dim % 2:
            raise ValueError('RoPE head_dim must be positive and even')
        inv_freq = base ** (-torch.arange(0, head_dim, 2).float() / head_dim)
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    def forward(self, q, k, position_ids):
        # positions [B,T]；Q/K [B,H,T,d]，二者的 H 可以不同。
        angles = position_ids.float().unsqueeze(-1) * self.inv_freq.float()
        cos, sin = angles.cos().unsqueeze(1), angles.sin().unsqueeze(1)

        def rotate(x):
            c, s = cos.to(x.dtype), sin.to(x.dtype)
            even, odd = x[..., 0::2], x[..., 1::2]
            return torch.stack((even*c - odd*s, even*s + odd*c), -1).flatten(-2)

        return rotate(q), rotate(k)


if __name__ == "__main__":
    q, k = torch.randn(2, 4, 5, 8), torch.randn(2, 2, 5, 8)
    qr, kr = RotaryEmbedding(8)(q, k, torch.arange(5).expand(2, 5))
    print('Q / K:', qr.shape, kr.shape)
    print('norm difference:', (q.norm(dim=-1) - qr.norm(dim=-1)).abs().max().item())
